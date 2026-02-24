from __future__ import annotations

import platform
import re
import shutil
import subprocess
from typing import Optional, Tuple


_PYTORCH_CUDA_TAGS = [
    ("13.0", "cu130"),
    ("12.8", "cu128"),
    ("12.6", "cu126"),
    ("11.8", "cu118"),
]


def _run_command(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        return ""
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return out.strip()


def detect_host_cuda_version() -> Tuple[Optional[str], Optional[str]]:
    if shutil.which("nvidia-smi") is None:
        return None, None

    out = _run_command(["nvidia-smi"])
    if not out:
        return None, None

    cuda_match = re.search(r"CUDA Version:\s*([\d.]+)", out)
    driver_match = re.search(r"Driver Version:\s*([\d.]+)", out)
    cuda_version = cuda_match.group(1) if cuda_match else None
    driver_version = driver_match.group(1) if driver_match else None
    return cuda_version, driver_version


def list_gpus() -> list[tuple[int, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = _run_command(["nvidia-smi", "-L"])
    if not out:
        return []
    gpus: list[tuple[int, str]] = []
    for line in out.splitlines():
        m = re.match(r"GPU\s+(\d+):\s*(.+?)\s*\(UUID:", line)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        name = m.group(2).strip()
        gpus.append((idx, name))
    return gpus


def get_platform_info() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def check_docker() -> Tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker not found"
    out = _run_command(["docker", "version", "--format", "{{.Server.Version}}"])
    if not out:
        return False, "docker not running or permission denied"
    return True, out.strip()


def try_run_command(args: list[str]) -> Tuple[int, str]:
    if shutil.which(args[0]) is None:
        return 127, ""
    try:
        proc = subprocess.run(args, check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        return 1, ""
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, out.strip()


def select_pytorch_cuda_tag(cuda_version: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not cuda_version:
        return None, None

    def _to_float(val: str) -> float:
        try:
            parts = val.split(".")
            return float(f"{int(parts[0])}.{int(parts[1])}")
        except Exception:
            return 0.0

    target = _to_float(cuda_version)
    best = None
    for ver, tag in _PYTORCH_CUDA_TAGS:
        if _to_float(ver) <= target:
            best = (ver, tag)
            break
    if not best:
        return None, None
    return best


def pytorch_extra_index_url(cuda_version: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    ver, tag = select_pytorch_cuda_tag(cuda_version)
    if not tag:
        return None, None
    return ver, f"https://download.pytorch.org/whl/{tag}"
