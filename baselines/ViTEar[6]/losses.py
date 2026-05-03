from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    def __init__(self, d: int, n: int, s: float = 64.0, m: float = 0.35):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, d))
        nn.init.xavier_uniform_(self.W)
        self.s = float(s)
        self.m = float(m)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(x, F.normalize(self.W))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine).scatter_(1, y[:, None], 1)
        return F.cross_entropy(self.s * (one_hot * phi + (1 - one_hot) * cosine), y)


class ElasticCosFace(CosFace):
    def __init__(self, d: int, n: int, std: float = 0.05, **kwargs):
        super().__init__(d, n, **kwargs)
        self.std = float(std)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(x, F.normalize(self.W))
        margin = torch.normal(self.m, self.std, y.size(), device=y.device)
        cosine[torch.arange(len(y), device=y.device), y] -= margin
        return F.cross_entropy(self.s * cosine, y)


class ElasticCosFacePlus(ElasticCosFace):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(x, F.normalize(self.W))
        margin = torch.abs(torch.normal(self.m, self.std, y.size(), device=y.device))
        cosine[torch.arange(len(y), device=y.device), y] -= margin
        return F.cross_entropy(self.s * cosine, y)
