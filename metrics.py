# metrics.py
# MAD (Mean Average Distance). Basically just checking how far apart 
# node embeddings are. If this drops to zero, the nodes are identical (over-smoothed).

import torch

def compute_mad(embeddings):
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    dist = torch.cdist(embeddings, embeddings)
    mask = ~torch.eye(n, dtype=torch.bool, device=dist.device)
    return dist[mask].mean().item()

def mad_per_layer(layer_embeddings):
    return [compute_mad(e) for e in layer_embeddings]
