#!/usr/bin/env python3
"""ACT CPU smoke-train (Phase B): wiring proof, not a good policy.

Real merged bytes -> normalized tensors -> decreasing imitation loss ->
saved checkpoint. Success criteria (all must hold, else exit 1):

  1. mean loss of last 10 steps < 0.9 * mean loss of first 10 steps
     (for runs < 20 steps, second-half mean < 0.9 * first-half mean);
  2. no NaN/Inf in loss or parameters;
  3. a no-grad rollout forward pass returns action chunks of shape (B, k, D).

CPU only: CUDA is refused even if present, to keep results comparable.

Usage:
    python scripts/train_act_smoke.py --data <merged_lerobot_v3> [--steps 50]
        [--batch-size 8] [--chunk-size 16] [--lr 1e-4] [--seed 0]
        [--out runs/smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training import require_torch  # noqa: E402
from src.training.act_policy import TinyACTPolicy, act_loss, count_parameters  # noqa: E402
from src.training.lerobot_dataset import LeRobotParquetDataset  # noqa: E402

logger = logging.getLogger(__name__)


def check_success(losses: list, model, rollout_shape_ok: bool,
                  kl_beta: float = 10.0) -> dict:
    """Evaluate the three smoke-train success criteria."""
    torch = require_torch()
    n = len(losses)
    if n >= 20:
        first = float(np.mean(losses[:10]))
        last = float(np.mean(losses[-10:]))
    else:
        half = max(1, n // 2)
        first = float(np.mean(losses[:half]))
        last = float(np.mean(losses[half:]))
    decreasing = last < 0.9 * first

    finite_loss = all(float(v) == v and abs(v) != float("inf") for v in losses)
    params_finite = all(
        bool(torch.isfinite(p).all()) for p in model.parameters())

    checks = {
        "loss_decreasing": bool(decreasing),
        "first_mean": first,
        "last_mean": last,
        "loss_finite": bool(finite_loss),
        "params_finite": bool(params_finite),
        "rollout_shape_ok": bool(rollout_shape_ok),
        "kl_beta": kl_beta,
    }
    checks["success"] = all([
        checks["loss_decreasing"], checks["loss_finite"],
        checks["params_finite"], checks["rollout_shape_ok"],
    ])
    return checks


def run_training(data: str | Path, steps: int = 50, batch_size: int = 8,
                 chunk_size: int = 16, lr: float = 1e-4, seed: int = 0,
                 out: str | Path = "runs/smoke", image_size: int = 96,
                 kl_beta: float = 10.0) -> dict:
    """Run the smoke train. Returns a result dict (also written to out/)."""
    torch = require_torch()
    device = torch.device("cpu")
    assert device.type == "cpu", "smoke train is CPU-only by design"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ds = LeRobotParquetDataset(data, chunk_size=chunk_size, image_size=image_size)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = TinyACTPolicy(ds.state_dim, ds.action_dim,
                          chunk_size=chunk_size).to(device)
    n_params = count_parameters(model)
    logger.info("TinyACTPolicy: %d params (state_dim=%d, action_dim=%d)",
                n_params, ds.state_dim, ds.action_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    losses, l1s, kls = [], [], []
    it = iter(loader)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        state = batch["state"].to(device)
        images = batch["images"].to(device)
        action_chunk = batch["action_chunk"].to(device)

        pred, mu, logvar = model(state, images, action_chunk)
        loss, parts = act_loss(pred, action_chunk, mu, logvar, kl_beta=kl_beta)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(float(loss.detach()))
        l1s.append(parts["l1"])
        kls.append(parts["kl"])
        if (step + 1) % 10 == 0 or step == 0:
            logger.info("step %d/%d loss=%.4f (l1=%.4f kl=%.4f)",
                        step + 1, steps, losses[-1], l1s[-1], kls[-1])

    # No-grad rollout shape check.
    model.eval()
    rollout_ok = False
    with torch.no_grad():
        probe = next(iter(loader))
        pred, mu, logvar = model(probe["state"].to(device),
                                 probe["images"].to(device))
        rollout_ok = (mu is None and logvar is None
                      and tuple(pred.shape) == (probe["state"].shape[0],
                                                chunk_size, ds.action_dim))
    ds.close()

    checks = check_success(losses, model, rollout_ok, kl_beta=kl_beta)
    wall_s = time.time() - t0
    config = {
        "data": str(data), "steps": steps, "batch_size": batch_size,
        "chunk_size": chunk_size, "lr": lr, "seed": seed,
        "image_size": image_size, "kl_beta": kl_beta,
        "n_params": n_params, "device": "cpu",
        "torch_version": torch.__version__,
        "wall_seconds": round(wall_s, 1),
    }
    with open(out_dir / "loss_curve.json", "w") as f:
        json.dump({"loss": losses, "l1": l1s, "kl": kls}, f)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**config, "checks": checks}, f, indent=2)
    torch.save({"model_state": model.state_dict(), "config": config},
               out_dir / "policy.pt")
    logger.info("Smoke train done in %.1fs: success=%s %s",
                wall_s, checks["success"], checks)
    return {"losses": losses, "checks": checks, "config": config,
            "out_dir": str(out_dir)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ACT CPU smoke-train (Phase B).")
    ap.add_argument("--data", required=True, help="merged lerobot_v3 root")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--kl-beta", type=float, default=10.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        result = run_training(
            args.data, steps=args.steps, batch_size=args.batch_size,
            chunk_size=args.chunk_size, lr=args.lr, seed=args.seed,
            out=args.out, image_size=args.image_size, kl_beta=args.kl_beta)
    except ImportError as e:
        print(f"Cannot train: {e}", file=sys.stderr)
        return 2
    if not result["checks"]["success"]:
        print(f"SMOKE TRAIN FAILED: {result['checks']}", file=sys.stderr)
        return 1
    print(f"SMOKE TRAIN PASSED in {result['config']['wall_seconds']}s "
          f"-> {result['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
