from __future__ import annotations

import json
import re
import shlex
import subprocess
from typing import Any, Dict, List, Optional


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def call_ollama(model: str, prompt: str, timeout_s: int = 120) -> Optional[Dict[str, Any]]:
    try:
        p = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except Exception:
        return None

    if p.returncode != 0:
        return None

    return _extract_json(p.stdout)


def _clip(text: Optional[str], max_chars: int) -> str:
    if not text:
        return "(none)"
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _summarize_requirements(requirements_text: Optional[str], max_lines: int = 30) -> str:
    if not requirements_text:
        return "(none)"
    lines: List[str] = []
    for raw in requirements_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
        if len(lines) >= max_lines:
            break
    if not lines:
        return "(none)"
    text = "\n".join(lines)
    total_nonempty = sum(1 for ln in requirements_text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    if total_nonempty > len(lines):
        text += f"\n... [{total_nonempty - len(lines)} more requirement lines]"
    return text


def _summarize_dockerfile(dockerfile_text: Optional[str], max_lines: int = 25) -> str:
    if not dockerfile_text:
        return "(none)"
    picked: List[str] = []
    for raw in dockerfile_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith(("FROM ", "ARG ", "ENV ", "WORKDIR ", "RUN ", "COPY ", "ADD ", "CMD ", "ENTRYPOINT ")):
            picked.append(line)
        if len(picked) >= max_lines:
            break
    if not picked:
        picked = [ln.strip() for ln in dockerfile_text.splitlines() if ln.strip()][:max_lines]
    text = "\n".join(picked)
    total_nonempty = sum(1 for ln in dockerfile_text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    if total_nonempty > len(picked):
        text += f"\n... [{total_nonempty - len(picked)} more Dockerfile lines]"
    return text


def build_patch_prompt(
    logs: str,
    *,
    run_cmd: Optional[List[str]] = None,
    base_image: Optional[str] = None,
    python_version: Optional[str] = None,
    requirements_text: Optional[str] = None,
    dockerfile_text: Optional[str] = None,
) -> str:
    run_cmd_text = shlex.join(run_cmd) if run_cmd else "(unknown)"
    return (
        "You are an expert at fixing Python/CUDA/Docker build failures.\n"
        "Given the execution context and error logs, propose minimal changes to fix the environment.\n"
        "Prefer the smallest patch that matches the provided run command and dependency files.\n"
        "Return ONLY JSON in this schema:\n"
        "{\n"
        "  \"actions\": [\n"
        "    {\"type\": \"pin\", \"package\": \"torch\", \"version\": \"2.0.1\"},\n"
        "    {\"type\": \"add_requirement\", \"package\": \"x\"},\n"
        "    {\"type\": \"reorder_requirements\", \"packages\": [\"torch\", \"torchvision\"]},\n"
        "    {\"type\": \"set_base_image\", \"value\": \"nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04\"},\n"
        "    {\"type\": \"set_python\", \"version\": \"3.9\"}\n"
        "  ]\n"
        "}\n"
        "Execution context:\n"
        f"- run_cmd: {run_cmd_text}\n"
        f"- base_image: {base_image or '(unknown)'}\n"
        f"- python_version: {python_version or '(unknown)'}\n"
        "requirements.txt summary:\n"
        f"{_clip(_summarize_requirements(requirements_text), 2200)}\n"
        "Dockerfile summary:\n"
        f"{_clip(_summarize_dockerfile(dockerfile_text), 2200)}\n"
        "Error logs:\n"
        f"{logs[-6000:]}\n"
    )
