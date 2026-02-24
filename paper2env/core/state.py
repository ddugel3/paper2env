from __future__ import annotations

import os
import shutil

from .models import RunConfig


FILES_TO_SNAPSHOT = ["Dockerfile", "requirements.txt", "environment.yml", "env.json"]


def snapshot_state(cfg: RunConfig, attempt_idx: int) -> str:
    snap_dir = os.path.join(cfg.workdir, ".snapshots", f"attempt_{attempt_idx:02d}")
    os.makedirs(snap_dir, exist_ok=True)
    for fn in FILES_TO_SNAPSHOT:
        src = os.path.join(cfg.workdir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(snap_dir, fn))
    return snap_dir


def rollback_state(cfg: RunConfig, snap_dir: str) -> None:
    for fn in FILES_TO_SNAPSHOT:
        src = os.path.join(snap_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(cfg.workdir, fn))
