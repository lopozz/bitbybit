import torch
from torch import nn


class LoRAAdapter(nn.Module):
    """
    Wraps an existing nn.Linear (or nn.Embedding) and adds low-rank adapters.
    """

    def __init__(
        self, module: nn.Module, r: int = 4, alpha: int = None, init_scale: float = 1e-3
    ):
        super().__init__()
        self.module = module  # e.g., nn.Linear
        self.r = r
        if alpha is None:
            alpha = r
        self.scaling = alpha / r

        # For Linear: out_features x in_features -> we create A: r x in, B: out x r
        if isinstance(module, nn.Linear):
            in_f = module.in_features
            out_f = module.out_features
            self.A = nn.Parameter(torch.zeros(r, in_f))
            self.B = nn.Parameter(torch.zeros(out_f, r))
            nn.init.kaiming_uniform_(self.A, a=5**0.5)
            nn.init.zeros_(self.B)
        else:
            raise NotImplementedError
        # freeze the base
        for p in self.module.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.module(x)
        delta = (x @ self.A.T) @ self.B.T
        return base + delta * self.scaling


def find_parent_and_attr(model, module_name: str):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        # handle list indices in module names like blocks.3.attn
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora_to_linear_modules(
    model: nn.Module, target_modules: list[str], r=4, alpha=None
):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any([tm in name for tm in target_modules]):
            print(name)
            parent, attr = find_parent_and_attr(model, name)
            wrapped = LoRAAdapter(mod, r=r, alpha=alpha)
            wrapped.to(mod.weight.device)
            setattr(parent, attr, wrapped)
