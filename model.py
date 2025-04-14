import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# Root Mean Square Layer Normalization
# 后面在每个Transformer子层的输入上进行RMS归一化，而不是在输出上
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

# 想了解RoPE可以看RoPE作者博客https://kexue.fm/archives/8265
# 预计算相位复数形式位置编码，旋转位置编码（RoPE）
# RoPE是q和k的相对位置编码,相位形式点积自动反映n-m相对位置
# dim--attention_dim end--最大长度两倍
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    # 预计算旋转频率，维度高（靠近向量头部）旋转更快，维度低（尾部）旋转慢
    # (0, 2, ..., d - 2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 设置位置(0~end-1)
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # 求t和freqs的外积（t转置乘freqs），得到频率矩阵
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # torch.polar(abs, angle)，相位角形式表示复数
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # (1024, 32)
    return pos_cis

# 应用旋转位置编码（RoPE）
# xq、xk--(batch_size, seq_len, num_head, head_size)，这里的pos_cis是前一个函数pos_cis截取一段
def apply_rotary_emb(xq, xk, pos_cis):
    # 统一pos_cis与xq，xk的形状
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # pos_cis的shape需要为(512, 32)
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        # shape变为(1, 512, 1, 32)
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)
    # 2是为了将q和k转化为复数a+bi的形式
    # 原shape为(2, 512, 12, 64)--(batch_size, seq_len, atten_head, head_dim)
    # reshape后变为(2, 512, 12, 32, 2)
    # 然后从实数转为复数，shape变为(2, 512, 12, 32)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    # (2, 512, 12, 32, 2)--flatten-->(2, 512, 12, 64)
    # 将q和k乘频率矩阵，逐元素旋转
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# 使得用于多头注意力的k和v的头匹配q的头的数量
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
# 采用GQA分组查询注意力机制，多头q分成多组，每组q共享同一个k和v矩阵
# 中间计算采用了flash attention
class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        # GQA分组k和v矩阵最后和每个q相乘需要复制到Q的维度，表示k和v需要复制的次数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 检测是否支持flash_attn
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # 上三角设置为-inf是因为，softmax最后会变为0，而且Q乘K矩阵最后的结果只有对角线及左下角（所以令为0，softmax之后为1）
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1) #将主对角线及下方，下三角元素设置为0，其他保留为-inf
        self.register_buffer("mask", mask, persistent=False)

    # q, k参与RoPE位置编码，然后k-v进行kvcache继承前面的k和v，多头的attn，其次多头q分成多组，共享每组的kv进行GQA
    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # q和k应用旋转位置编码(RoPE)
        # q和k的作用是捕捉序列中元素的关系（包括相对位置信息）
        # v的作用是存储元素的实际信息，不需要直接参与位置关系的计算
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        # kv_cache实现(推理时使用)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # k和v复制到和q相同的维度，（GQA）
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash and seq_len != 1:
            # 设置dropout概率
            dropout_p = self.dropout if self.training else 0.0 # self.training是继承的module类变量
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True # 启用因果掩码，确保只能看到过去的信息
            )
        else:
            # 除以一个头维度平方根是为了防止梯度消失
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) 
            # 使上三角为-inf，保证看不到未来的
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            # 4 * 512 = 2048
            hidden_dim = 4 * config.dim
            # hidden_dim = 2048 * 2 / 3 = 1365
            hidden_dim = int(2 * hidden_dim / 3)
            # 使得config.hidden_dim为config.multiple_of的倍数,并又大于hidden_dim，默认config.multiple_of设置为64
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 激活的和不激活的逐元素相乘，激活层使用SwiGLU
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# Mixture of Experts(MoE)混合专家模型的门控机制
class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:                                                        
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h) # (bsz*seq_len, hidden_size)

        # hidden_state * self.weight转置
        # hidden_state--(bsz * seq, hidden_size) self.weight--(n_routed_experts, gating_dim)
        # score--(bsz * seq, n_routed_experts)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        # topk_weight--(bsz * seq, k), topk_idx--(bsz * seq, k)--记录原index
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # compute auxiliary loss for expert routing
        if self.training and self.alpha > 0.0:
            # (bsz * seq, n_routed_experts)
            scores_for_aux = scores
            aux_topk = self.top_k
            # (bsz, seq * k)
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            # ture 则执行序列级的辅助损失计算，否则执行批次级别的辅助损失计算
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)

                # 创建全0张量ce(count_experts)--用于存储每个批次样本的专家使用计数
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # topk_idx_for_aux_loss存储的是原n_routed_experts的index
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # score_for_seq_aux在seq_len这一维度上平均，然后转置，ce乘转之后的score_for_seq_aux
                # (batch_size, batch_size) -总和-> (batch_size) -平均-> aux_loss(一个标量)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            # 批次级别辅助损失计算
            else:
                # (bsz * seq * k) --> (bsz * seq * k, n_routed_experts)
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # ce指每个专家被选择的平均频率
                ce = mask_ce.float().mean(0)
                # Pi指每个expert被选择的平均概率
                Pi = scores_for_aux.mean(0)
                # fi指每个专家的使用频率
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        # 当过度使用某些experts时，aux_loss会增大
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # (bsz * seq, dim)
        x = x.view(-1, x.shape[-1])
        # (bsz * seq * k)
        # 某个batch的3个token： [0, 1, 0, 2, 1, 3] -- 有3个token，k为2
        # [[0, 1], [0, 2], [1, 3]] --> [0, 1, 0, 2, 1, 3]
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练模式下，重复输入数据
            # (bsz * seq, dim) --> (bsz * seq * k, dim)
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                # y -- 选择了的k个expert的输出
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            # topk_weight--(bsz * seq, k)
            # 输出与权重相乘，unsqueeze是增加1个维度，方便相乘
            # 最后得到(bsz * seq, k, dim) --> (bsz * seq, dim)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # (bsz, seq, dim)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        # cumsum是累加前面的元素，所以可以使得end_idx接着前面的累加
        # 假设flat_expert_indices = [0, 1, 0, 2, 1, 3]
        # flat_expert_indices.bincount()为[2, 2, 1, 1]表示exp0~3各自出现的次数（按exp0到exp3的顺序排）
        # 然后进行累加也就是flat_expert_indices.bincount()然后再.cumsum(0)
        # 变为[2, 4, 5, 6]表示0到2-1即[0:2]切片属于exp1, 2到4-1即[2:4]切片属于,...以此类推
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 除以num_experts_per_tok把idxs还原为原token编号
        # 为什么可以还原，因为0, 1是token1的, 2, 3是tokens2的,以此类推
        # 这里举个例子：
        # flat_expert_indices = [0, 1, 0, 2, 1, 3] 
        # 表示token0（索引0和1）分给exp0和exp1，token1（索引2和3）分给exp0和exp2，token2（索引4和5）分给exp1和exp3
        # 然后按exp的顺序排，从exp0排到exp3，即idxs = flat_expert_indices.argsort()
        # 此处idxs = [0, 2, 1, 4, 3, 5]这里的数字表示flat_expert_indices里面index
        # 此处我们明白index为0,2的属于exp0, 1,4的属于exp1, 3的属于exp2, 5的属于exp3，那是因为我们推出来的，但是程序不清楚
        # 所以前面的tokens_per_expert就是用来记录哪个属于哪个exp
        # 然后因为idxs对应的是索引，每个token对应两个exp（num_experts_per_tok为2），所以把索引的值除以2对应原token的位置
        # 从而token_idxs = idxs // self.config.num_experts_per_tok
        # token_idxs为[0, 1, 0, 2, 1, 2]里面的数字即代表了token位置例如数字0表示是token0
        # 然后后面的for循环就是把token_idxs和tokens_per_expert结合起来确认哪些token是输入哪些experts
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # idxs按exp排序，idxs里的元素表示flat_expert_weights中的index
            # flat_expert_weights每个index对应一个权重，结合flat_expert_indices解释，就是同一个index对应的weight和exp
            # 比如flat_expert_weights[3] = 0.7和flat_expert_indices[3] = 5
            # 表示这个index对应的是token1(3 / 2 = 1),并且对应的exp是exp5，在expert[5](token1)之后需要乘的权重是0.7
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            # scatter_add_是将expert_out按照exp_token_idx.view(-1, 1).repeat(1, x.shape[-1])的索引位置填充到expert_cache
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    # layer_id表示每个block是第几个block，比如layers.7.attention Attention(
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        # residual的形式
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class MiniMindLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        # embedding vocab_size映射为config里面的dim
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)
        # dim --> vocab_size
        logits = self.output(self.norm(h))
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        # 流式生成
        # 给出yield生成的过程，然后从每个str里面取最后一个，比如聊天中一个一个字生成就是这样做的
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成
        # 不直接给出yield生成的过程，而是提前处理yield生成的结果，然后最后给出一个最后处理完的句子
        generated = []
        for i in range(input_ids.size(0)):
            # (seq_len) --> (1, seq_len)
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            # 这里yield input_ids[:, start:]返回多个过程字符串组成的列表，[[[3]], [[3, 4]], [[3, 4, 5]], ..., [[3, 4, 5, ..., 2]]]
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            # 取每个过程串最后一个字符
            # e.g. tokens_list = [[[3]], [[4]], [[5]], ..., [[2]]]
            # 注意，这里最外层是列表，内层是(1,1)形状的张量
            tokens_list = [tokens[:, -1:] for tokens in out]
            # 内层(1,1)形状拼接完后，为[[3, 4, 5, ..., 2]] (1, n)
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            # 拼完后，是(1, len)
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        # generated形状是(batch_size, 1, len)
        # 取最长seq的length
        max_length = max(seq.size(1) for seq in generated)
        # seq的形状是(1, len)
        # torch.cat(...)形状是(1, max_len)
        # generated形状是(batch_size, 1, max_len)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        # (batch_size, max_len)
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        # 当inputs_ids的seq_len超过最大长度则截止
        while input_ids.shape[1] < max_new_tokens - 1:
            # 流式生成，输入第一个seq时也就是生成第一个token时，直接输入整个句子做完整的qkv
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                # 当当前已经不是第一个句子时，就只需要传入input_ids[:, -1:]也就是用最后一个token的qkv就够了，之前的q不需要，之前的k和v经kv-cache存放cat
                # 在这里用了kv-cache， 在stream的过程中记录past_kvs
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            # rp -- repetition penalty
            # 将logits中的inputs_ids重复部分降低概率，避免生成重复内容
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            # top-p采样，nucleus sampling
            if top_p is not None and top_p < 1.0:
                # sorted_logits对应排序后的概率，sorted_indices对应排序后的原索引记录原始位置
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                # 累加
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # 右移一位，并让第0位为false，强制保留
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                # 把排完的位置还原为索引位置
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            # 生成下一个词
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1) # 根据概率权重选择下一个词
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            # 迭代返回生成器
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
