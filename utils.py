import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

def info_nce(u: torch.Tensor, v: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Compute symmetric InfoNCE loss for positive pairs (u_i, v_i).
    Shapes:
      u: [N, d]
      v: [N, d]
    Returns:
      scalar loss tensor
    """
    u = torch.nn.functional.normalize(u, dim=-1)
    v = torch.nn.functional.normalize(v, dim=-1)
    logits = u @ v.t()  # [N, N]
    logits = logits / temperature
    labels = torch.arange(u.size(0), device=u.device)
    loss_u = torch.nn.functional.cross_entropy(logits, labels)
    loss_v = torch.nn.functional.cross_entropy(logits.t(), labels)
    return (loss_u + loss_v) * 0.5

def correlation_score(u: torch.Tensor, d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Correlation-like decoder score in [-1,1] per sample.
    """
    u = u - u.mean(dim=-1, keepdim=True)
    d = d - d.mean(dim=-1, keepdim=True)
    u = u / (u.std(dim=-1, keepdim=True) + eps)
    d = d / (d.std(dim=-1, keepdim=True) + eps)
    return (u * d).sum(dim=-1) / u.size(-1)
