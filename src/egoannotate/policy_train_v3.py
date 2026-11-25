"""End-to-end open-source PyTorch action-chunking policy training smoke test over LeRobot v3 Parquet exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class LeRobotV3ChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Loads exported LeRobot v3 Parquet (state + tactile -> action chunk) for direct policy training."""

    def __init__(self, parquet_path: Path, chunk_size: int = 8) -> None:
        table = pq.read_table(parquet_path)
        self.state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        self.tactile = np.asarray(table["observation.tactile"].to_pylist(), dtype=np.float32)
        self.action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        self.chunk_size = int(chunk_size)

        self.obs = np.concatenate([self.state, self.tactile], axis=1)
        self.obs_mean = self.obs.mean(axis=0, keepdims=True)
        self.obs_std = np.maximum(self.obs.std(axis=0, keepdims=True), 1e-4)
        self.act_mean = self.action.mean(axis=0, keepdims=True)
        self.act_std = np.maximum(self.action.std(axis=0, keepdims=True), 1e-4)

        self.obs_norm = ((self.obs - self.obs_mean) / self.obs_std).astype(np.float32)
        self.act_norm = ((self.action - self.act_mean) / self.act_std).astype(np.float32)
        self.valid_len = max(1, len(self.obs_norm) - self.chunk_size)

    def __len__(self) -> int:
        return self.valid_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        obs_t = torch.from_numpy(self.obs_norm[idx])
        act_chunk = torch.from_numpy(self.act_norm[idx : idx + self.chunk_size])
        return obs_t, act_chunk


class ActionChunkPolicyNet(nn.Module):
    """Lightweight open-source PyTorch action-chunking policy network (obs -> [chunk_size, action_dim])."""

    def __init__(self, obs_dim: int, action_dim: int, chunk_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        return out.view(-1, self.chunk_size, self.action_dim)


def run_policy_training_smoke(
    lerobot_dir: Path,
    *,
    chunk_size: int = 8,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 2e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train ActionChunkPolicyNet on exported LeRobot v3 data and verify real loss convergence."""
    parquet_path = lerobot_dir / "data" / "chunk-000" / "file-000.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"LeRobot v3 parquet not found: {parquet_path}")

    torch.manual_seed(seed)
    ds = LeRobotV3ChunkDataset(parquet_path, chunk_size=chunk_size)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    obs_dim = ds.obs_norm.shape[1]
    action_dim = ds.act_norm.shape[1]
    model = ActionChunkPolicyNet(obs_dim=obs_dim, action_dim=action_dim, chunk_size=chunk_size, hidden_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    epoch_losses: list[float] = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        steps = 0
        for obs_b, act_b in loader:
            optimizer.zero_grad()
            pred = model(obs_b)
            loss = criterion(pred, act_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        epoch_losses.append(round(total_loss / max(steps, 1), 6))

    model.eval()
    with torch.no_grad():
        sample_obs, _ = ds[0]
        sample_pred = model(sample_obs.unsqueeze(0))
        finite_pred = bool(torch.isfinite(sample_pred).all().item())

    return {
        "parquet": str(parquet_path),
        "samples": len(ds),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "initial_loss": epoch_losses[0],
        "final_loss": epoch_losses[-1],
        "loss_decreased": bool(epoch_losses[-1] < epoch_losses[0]),
        "finite_predictions": finite_pred,
        "epoch_losses": epoch_losses,
    }
