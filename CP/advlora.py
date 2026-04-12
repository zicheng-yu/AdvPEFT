import torch
from torch import nn
import torch.nn.functional as F


class AdvLoRALinear(nn.Module):
    def __init__(
        self,
        linear,
        rank,
        alpha_init=1e-3,
        kmeans_iters=10,
        kmeans_tol=1e-4,
    ):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("AdvLoRALinear expects an nn.Linear module")

        self.base = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = max(1, min(rank, self.in_features, self.out_features))
        self.kmeans_iters = kmeans_iters
        self.kmeans_tol = kmeans_tol

        self.lora_A = nn.Parameter(torch.empty(self.in_features, self.rank))
        self.lora_B = nn.Parameter(torch.empty(self.rank, self.out_features))
        self.lora_alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.reset_parameters()
        self.freeze_base()
        self.cluster_initialize()

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def reset_parameters(self):
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def freeze_base(self):
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    @torch.no_grad()
    def cluster_initialize(self):
        weight = self.base.weight.detach().float().t().contiguous()
        centers_t, assignment = _kmeans_columns(
            weight,
            rank=self.rank,
            max_iters=self.kmeans_iters,
            tol=self.kmeans_tol,
        )
        self.lora_A.copy_(centers_t)
        self.lora_B.zero_()
        self.lora_B.scatter_(0, assignment.unsqueeze(0), 1.0)

    def align_loss(self):
        weight = self.base.weight.t()
        approx = self.lora_A @ self.lora_B
        return torch.sum((weight - approx) ** 2)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A) @ self.lora_B
        return base_out + self.lora_alpha * lora_out


@torch.no_grad()
def _kmeans_columns(weight_t, rank, max_iters=10, tol=1e-4):
    data = weight_t.t().contiguous()
    num_cols = data.size(0)
    rank = min(rank, num_cols)

    if rank == num_cols:
        assignment = torch.arange(num_cols, device=data.device)
        return data.t().contiguous(), assignment

    indices = torch.linspace(0, num_cols - 1, steps=rank, device=data.device)
    indices = indices.round().long().unique(sorted=True)
    if indices.numel() < rank:
        extra = torch.arange(num_cols, device=data.device)
        keep_mask = torch.ones(num_cols, dtype=torch.bool, device=data.device)
        keep_mask[indices] = False
        extra = extra[keep_mask][: rank - indices.numel()]
        indices = torch.cat([indices, extra], dim=0)

    centers = data[indices].clone()
    assignment = torch.zeros(num_cols, dtype=torch.long, device=data.device)

    for _ in range(max_iters):
        distances = torch.cdist(data, centers, p=2)
        new_assignment = distances.argmin(dim=1)

        new_centers = centers.clone()
        for idx in range(rank):
            mask = new_assignment == idx
            if mask.any():
                new_centers[idx] = data[mask].mean(dim=0)

        shift = torch.norm(new_centers - centers, dim=1).max()
        centers = new_centers
        assignment = new_assignment
        if shift.item() <= tol:
            break

    return centers.t().contiguous(), assignment


def wrap_linear_layer(module, name, rank, alpha_init=1e-3, kmeans_iters=10):
    linear = getattr(module, name)
    if isinstance(linear, AdvLoRALinear):
        return linear
    wrapped = AdvLoRALinear(
        linear,
        rank=rank,
        alpha_init=alpha_init,
        kmeans_iters=kmeans_iters,
    )
    setattr(module, name, wrapped)
    return wrapped


def iter_advlora_layers(module):
    for child in module.modules():
        if isinstance(child, AdvLoRALinear):
            yield child
