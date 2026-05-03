from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_val_embeddings(models, aligner, loader, device):
    if not isinstance(models, (list, tuple)):
        models = [models]

    for model in models:
        model.eval()

    all_embeddings = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if aligner is not None:
            imgs = aligner(imgs)

        embeddings = [model(imgs) for model in models]
        emb = torch.cat(embeddings, dim=1)
        emb = F.normalize(emb, dim=1)

        all_embeddings.append(emb.cpu())
        all_labels.append(labels.to(torch.long).cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def evaluate_ensemble(model, aligner, val_loader, path, device):
    return extract_val_embeddings(model, aligner, val_loader, device)
