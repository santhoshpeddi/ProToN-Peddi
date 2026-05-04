import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):

    def __init__(self, embedding_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # normalize class weights
        W = F.normalize(self.weight, dim=1)

        # cosine similarity
        cosine = F.linear(embeddings, W)  # (B, C)
        sine = torch.sqrt(1.0 - cosine**2).clamp(0, 1)

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits


class WeightedArcFaceLoss(nn.Module):
    def __init__(self, arcface_layer, demographic_weights=None):
        super().__init__()
        self.arcface = arcface_layer
        self.demographic_weights = demographic_weights  # dict or None

    def forward(self, embeddings, labels, demographic_groups=None):
        logits = self.arcface(embeddings, labels)

        # per-sample CE loss
        ce_loss = F.cross_entropy(
            logits, labels, reduction="none"
        )  # (B,)

        if self.demographic_weights is not None:
            weights = torch.tensor(
                [self.demographic_weights[g] for g in demographic_groups],
                device=ce_loss.device,
                dtype=ce_loss.dtype
            )
            loss = (ce_loss * weights).mean()
        else:
            loss = ce_loss.mean()

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, demographic_weights=None, max_weight=5.0):
        super().__init__()
        self.demographic_weights = demographic_weights
        self.max_weight = max_weight

    def forward(self, logits, labels, demographic_groups=None):
        ce = F.cross_entropy(logits, labels, reduction="none")

        if self.demographic_weights is not None and demographic_groups is not None:
            weights = [
                self.demographic_weights.get(g, 1.0)
                for g in demographic_groups
            ]

            weights = torch.tensor(
                weights,
                device=ce.device,
                dtype=ce.dtype
            )

            weights = torch.clamp(weights, min=0.5, max=self.max_weight)

            ce = ce * weights

        return ce.mean()
