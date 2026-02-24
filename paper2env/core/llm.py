from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Dict, Optional


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


def build_patch_prompt(logs: str) -> str:
    return (
        "You are an expert at fixing Python/CUDA/Docker build failures.\n"
        "Given the error logs, propose minimal changes to fix the environment.\n"
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
        "Error logs:\n"
        f"{logs[-6000:]}\n"
    )
