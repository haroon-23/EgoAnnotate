"""Vendored minimal ACT policy (Phase B) — no dependency on the lerobot package.

Faithful to the ACT paper (Zhao et al.), shrunk for CPU smoke-training on a
2017 Intel Mac:

- CVAE encoder: MLP ``q(z | action_chunk, joint_state)`` -> ``mu, logvar``,
  latent dim 16 (paper: 32).
- Policy decoder: Transformer, 2 layers, ``d_model=128``, 4 heads, ff=256
  (paper: 4 layers / 512 — too heavy for old Intel at 224px).
- Visual backbone: tiny 4-layer CNN on 96x96 frames (not ResNet) -> 128-d.
- Predicts an **absolute** action chunk of length ``k`` (matches our
  ``absolute_joint_positions`` export convention).
- Loss: ``L1(recon) + beta * KL`` with ``beta=10`` (ACT paper).

``torch`` is optional at import time; constructing any class without torch
raises a helpful error via :func:`src.training.require_torch`.
"""

from __future__ import annotations

from . import require_torch

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in torch-free envs
    _TORCH_AVAILABLE = False

_BaseModule = nn.Module if _TORCH_AVAILABLE else object


class TinyCNNEncoder(_BaseModule):
    """4-layer CNN: (B,3,96,96) -> (B, out_dim). ~300k params."""

    def __init__(self, out_dim: int = 128):
        if not _TORCH_AVAILABLE:
            require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(),   # 48
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # 24
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 12
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),# 6
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, images):
        return self.net(images)


class CVAEEncoder(_BaseModule):
    """MLP q(z | action_chunk, state) -> (mu, logvar). Latent dim 16."""

    def __init__(self, action_dim: int, state_dim: int, chunk_size: int,
                 latent_dim: int = 16, hidden_dim: int = 256):
        if not _TORCH_AVAILABLE:
            require_torch()
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = chunk_size * action_dim + state_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * latent_dim),
        )

    def forward(self, action_chunk, state):
        # action_chunk: (B, k, D), state: (B, D)
        x = torch.cat([action_chunk.flatten(1), state], dim=1)
        mu_logvar = self.mlp(x)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar, deterministic: bool = False):
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


class TinyACTPolicy(_BaseModule):
    """Minimal ACT: CVAE latent + transformer decoder -> absolute action chunk.

    forward(state, images, action_chunk=None):
      - with action_chunk (training): returns (pred, mu, logvar)
      - without (inference/rollout):  returns (pred, None, None) with z = 0
    """

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int = 16,
                 latent_dim: int = 16, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256):
        if not _TORCH_AVAILABLE:
            require_torch()
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        self.image_encoder = TinyCNNEncoder(out_dim=d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.cvae_encoder = CVAEEncoder(action_dim, state_dim, chunk_size,
                                        latent_dim=latent_dim)
        # Learned positional action queries, one per chunk step.
        self.query_embed = nn.Parameter(torch.randn(1, chunk_size, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, state, images, action_chunk=None):
        B = state.shape[0]
        img_feat = self.image_encoder(images)            # (B, d)
        state_tok = self.state_proj(state).unsqueeze(1)  # (B, 1, d)

        if action_chunk is not None:
            mu, logvar = self.cvae_encoder(action_chunk, state)
            z = self.cvae_encoder.reparameterize(
                mu, logvar, deterministic=not self.training)
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=state.device)
        z_tok = self.z_proj(z).unsqueeze(1)              # (B, 1, d)

        memory = torch.cat([z_tok, state_tok, img_feat.unsqueeze(1)], dim=1)  # (B,3,d)
        queries = self.query_embed.expand(B, -1, -1)     # (B, k, d)
        h = self.decoder(queries, memory)                # (B, k, d)
        pred = self.action_head(h)                       # (B, k, D)
        return pred, mu, logvar


def act_loss(pred, target, mu, logvar, kl_beta: float = 10.0):
    """L1 reconstruction + beta * KL. Returns (total, {"l1": ..., "kl": ...})."""
    if not _TORCH_AVAILABLE:
        require_torch()
    l1 = F.l1_loss(pred, target)
    # KL(N(mu, sigma) || N(0, I)), summed over latent dim, mean over batch.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total = l1 + kl_beta * kl
    return total, {"l1": float(l1.detach()), "kl": float(kl.detach())}


def count_parameters(model) -> int:
    if not _TORCH_AVAILABLE:
        require_torch()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["TinyCNNEncoder", "CVAEEncoder", "TinyACTPolicy", "act_loss",
           "count_parameters"]
