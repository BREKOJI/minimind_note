import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        # module.weight.shape[0] == module.weight.shape[1]这个的意思是去wq wk wv wo这四个加lora（如果用了GQA就只应用于wq wo）
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            # 为wo wq添加一个lora属性（用了GQA）
            '''
            layers.7.attention Attention(
            (wq): Linear(
                in_features=512, out_features=512, bias=False
                (lora): LoRA(
                (A): Linear(in_features=512, out_features=16, bias=False)
                (B): Linear(in_features=16, out_features=512, bias=False)
                )
            )
            (wk): Linear(in_features=512, out_features=128, bias=False)
            (wv): Linear(in_features=512, out_features=128, bias=False)
            (wo): Linear(
                in_features=512, out_features=512, bias=False
                (lora): LoRA(
                (A): Linear(in_features=512, out_features=16, bias=False)
                (B): Linear(in_features=16, out_features=512, bias=False)
                )
            )
            (attn_dropout): Dropout(p=0.0, inplace=False)
            (resid_dropout): Dropout(p=0.0, inplace=False)
            )
            '''
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 'layers.7.attention.wq.lora.A.weight': tensor(...)
            # 对应'{name}.lora.{k}': v
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
