#!/usr/bin/env python3
"""Sparse-clone MuJoCo Menagerie at a pinned commit into data/robots/mujoco_menagerie.

Idempotent: an existing checkout at the pinned commit is left untouched. The
source may be the GitHub URL (default) or a local clone (``--source``) that
contains the pinned commit.

    python scripts/fetch_robot_models.py [--dest DIR] [--source URL_OR_PATH]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"
MODELS = {
    "franka_emika_panda": "Apache-2.0",
    "unitree_g1": "BSD-3-Clause",
    "unitree_h1": "BSD-3-Clause",
    "wonik_allegro": "BSD-2-Clause",
}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "data" / "robots" / "mujoco_menagerie"


class FetchError(RuntimeError):
    pass


def _git(args: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise FetchError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def current_commit(dest: Path) -> str | None:
    if not (dest / ".git").exists():
        return None
    try:
        return _git(["rev-parse", "HEAD"], cwd=dest)
    except FetchError:
        return None


def models_present(dest: Path) -> bool:
    return all((dest / m).is_dir() for m in MODELS)


def fetch(dest: Path = DEFAULT_DEST, source: str = MENAGERIE_URL) -> Path:
    """Make the pinned Menagerie checkout available at ``dest``; return it."""
    dest = Path(dest)
    if current_commit(dest) == MENAGERIE_COMMIT and models_present(dest):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        _git(["init", "-q"], cwd=dest)
        _git(["remote", "add", "origin", source], cwd=dest)
    else:
        _git(["remote", "set-url", "origin", source], cwd=dest)
    _git(["sparse-checkout", "init", "--cone"], cwd=dest)
    _git(["sparse-checkout", "set", *MODELS], cwd=dest)
    _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_COMMIT], cwd=dest)
    _git(["checkout", "-q", "--force", "FETCH_HEAD"], cwd=dest)
    head = current_commit(dest)
    if head != MENAGERIE_COMMIT:
        raise FetchError(f"checked out {head}, expected pinned commit {MENAGERIE_COMMIT}")
    if not models_present(dest):
        raise FetchError(f"missing models after checkout: {[m for m in MODELS if not (dest / m).is_dir()]}")
    return dest


def licences(dest: Path) -> dict[str, str]:
    out = {}
    for model, spdx in MODELS.items():
        text = (dest / model / "LICENSE").read_text(errors="replace")
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        out[model] = f"{spdx} ({first})"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=os.environ.get("EGOANNOTATE_MENAGERIE", str(DEFAULT_DEST)))
    ap.add_argument("--source", default=MENAGERIE_URL, help="GitHub URL or local clone path")
    args = ap.parse_args(argv)
    try:
        dest = fetch(Path(args.dest), args.source)
    except (FetchError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"MuJoCo Menagerie @ {MENAGERIE_COMMIT} in {dest}")
    for model, lic in licences(dest).items():
        print(f"  {model}: {lic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
