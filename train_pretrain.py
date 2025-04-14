import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):# used to print information when the parallel is running
    if not ddp or dist.get_rank() == 0: # not ddp means Distributed Data Parallel is off, 
        print(content)# dist.get_rank() is the parallel rank when ddp is running


def get_lr(current_step, total_steps, lr): #余弦退火，在训练过程中逐渐降低学习率
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr # 参数分组调整学习率，不过这里是统一调整为lr，lr是全局学习率，AdamW能够自适应调整学习率，lr相当于学习率基准，AdamW动态调整

        with ctx:
            # res有三个属性：logits、aux_loss、past_key_values
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)), # (batch_size * sequence_len, cls)
                Y.view(-1) # (batch_size * sequence_len)
            ).view(Y.size()) # 将loss恢复为target原来的形状
            loss = (loss * loss_mask).sum() / loss_mask.sum() # 将loss除去无效部分（比如padding）然后取平均
            # 如果是MoE则带有aux_loss，避免一直使用几个experts，如果倾向于一直使用某些experts，aux_loss会增大
            loss += res.aux_loss
            loss = loss / args.accumulation_steps # 使用梯度累计，小批次要缩小

        scaler.scale(loss).backward() 
        # backward后grad（梯度）存放在model.parameters().grad中
        # 不同精度的进行梯度缩放后反向传播，为什么要梯度缩放呢？因为f16和bf16可能存在下溢导致损失为0，放大loss来累计就不会出现这种情况，反向传播相当于求链式法则的那个导数

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer) # 前面求梯度累计的时候是梯度缩放后的，现在要进行更新要对梯度取消缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 避免grad梯度爆炸而进行剪枝

            scaler.step(optimizer) # 梯度更新
            scaler.update() # 根据前面更新时是否出现梯度溢出

            optimizer.zero_grad(set_to_none=True) # 梯度清零

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict() # 如果被并行封装，则得到内层模型参数
            else:
                state_dict = model.state_dict() # 如果没有被并行封装，则直接得到参数

            torch.save(state_dict, ckp)
            print("save successfully! ,{}".format(ckp))
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # 先读取tokenizer_config.json其中包含一些config信息如char_template、如最大长度信息等等，
    # 然后读取tokenizer.json文件，其中包含merges.txt和vocab.json的内容，以及分词器的基本信息（如added_token）
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    #wandb(weight and bias)，记录模型训练过程的损失，学习率，准确率等
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    #如果是用cuda训练则使用混合精度的上下文管理器
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # means is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,#当 pin_memory 设置为 True 时，数据会被加载到主机内存中，并锁定在内存页中，这样可以加快 CPU 到 GPU 的数据传输速度。
        drop_last=False,#是否丢弃最后一个不完整批次
        shuffle=False,# shuffle设置为False时数据集顺序不变，是否打乱数据集
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16'])) # 如果args.dtype为f16或bf16，则启动混合精度的梯度缩放
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate) # 

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
