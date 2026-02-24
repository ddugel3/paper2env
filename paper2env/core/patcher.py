from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .llm import build_patch_prompt, call_ollama

from .models import RunConfig


_LAST_LOGS = ""
_TORCH_TV_COMPAT: Optional[Dict[str, str]] = None


def _clean_url_suffix(url: str) -> str:
    return url.rstrip(").,;]}>'\"")


def _clean_urls_in_text(text: str) -> tuple[str, bool]:
    if not text:
        return text, False
    changed = False
    lines = []
    for line in text.splitlines():
        if "http://" not in line and "https://" not in line:
            lines.append(line)
            continue
        parts = line.split()
        new_parts = []
        for part in parts:
            if part.startswith(("http://", "https://")):
                cleaned = _clean_url_suffix(part)
                if cleaned != part:
                    changed = True
                new_parts.append(cleaned)
            else:
                new_parts.append(part)
        lines.append(" ".join(new_parts))
    return "\n".join(lines), changed


def _reorder_requirements_text(text: str, packages: list[str]) -> str:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    pkgs = [p.lower() for p in packages if p]
    if not pkgs:
        return text
    matched = []
    remaining = []
    used = set()
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith(("-", "--")) or stripped.startswith(("git+", "http://", "https://")):
            remaining.append(ln)
            continue
        name = re.split(r"[=<>!~]", stripped, 1)[0].strip().lower()
        if name in pkgs and name not in used:
            matched.append(ln)
            used.add(name)
        else:
            remaining.append(ln)
    if not matched:
        return text
    return "\n".join(matched + remaining) + "\n"


def _remove_requirement_text(text: str, package: str) -> str:
    lines = []
    target = package.lower()
    for ln in text.splitlines():
        val = ln.strip()
        if not val or val.startswith("#"):
            lines.append(ln)
            continue
        name = re.split(r"[=<>!~]", val, 1)[0].strip().lower()
        if name == target:
            continue
        lines.append(ln)
    return "\n".join(lines) + ("\n" if lines else "")


def _replace_requirement_text(text: str, package: str, new_spec: str) -> str:
    lines = []
    target = package.lower()
    replaced = False
    for ln in text.splitlines():
        val = ln.strip()
        if not val or val.startswith("#"):
            lines.append(ln)
            continue
        name = re.split(r"[=<>!~]", val, 1)[0].strip().lower()
        if name == target:
            lines.append(new_spec)
            replaced = True
        else:
            lines.append(ln)
    if not replaced:
        lines.append(new_spec)
    return "\n".join(lines) + ("\n" if lines else "")


def _set_python_version(cfg: RunConfig, version: str) -> None:
    cfg.python_version = version
    # normalize system packages to match python version
    keep_py3 = {
        "python3-dev",
        "python3-distutils",
        "python3-setuptools",
        "python3-venv",
        "python3-wheel",
    }
    new_pkgs = []
    for pkg in cfg.system_packages:
        if pkg.startswith("python3.") or pkg.startswith("python3-"):
            if pkg in keep_py3:
                new_pkgs.append(pkg)
            continue
        new_pkgs.append(pkg)
    new_pkgs.append(f"python{version}")
    if "python3-pip" not in new_pkgs:
        new_pkgs.append("python3-pip")
    cfg.system_packages = new_pkgs
    # bump ubuntu base for newer python if needed
    if version.startswith("3.10") or version.startswith("3.11"):
        if cfg.base_image == "ubuntu:20.04":
            cfg.base_image = "ubuntu:22.04"
        elif "ubuntu20.04" in cfg.base_image:
            cfg.base_image = cfg.base_image.replace("ubuntu20.04", "ubuntu22.04")


def normalize_python_packages(cfg: RunConfig) -> bool:
    before = list(cfg.system_packages)
    _set_python_version(cfg, cfg.python_version)
    return before != cfg.system_packages


def set_last_logs(logs: str) -> None:
    global _LAST_LOGS
    _LAST_LOGS = logs


def _normalize_version(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    m = re.match(r"(\d+\.\d+\.\d+)", val)
    return m.group(1) if m else val


def _load_torch_compat() -> Dict[str, str]:
    global _TORCH_TV_COMPAT
    if _TORCH_TV_COMPAT is not None:
        return _TORCH_TV_COMPAT
    try:
        with open("paper2env/core/torch_compat.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _TORCH_TV_COMPAT = {str(k): str(v) for k, v in data.items()}
        else:
            _TORCH_TV_COMPAT = {}
    except Exception:
        _TORCH_TV_COMPAT = {}
    return _TORCH_TV_COMPAT


def _align_torch_vision(cfg: RunConfig) -> Optional[str]:
    compat = _load_torch_compat()
    torch_ver = _normalize_version(cfg.requirements.get("torch"))
    tv_ver = _normalize_version(cfg.requirements.get("torchvision"))
    if torch_ver and torch_ver in compat:
        pair = compat[torch_ver]
        if tv_ver != pair:
            cfg.requirements["torchvision"] = pair
            return f"align torchvision=={pair} for torch=={torch_ver}"
    if tv_ver:
        for tver, tpair in compat.items():
            if tpair == tv_ver:
                if torch_ver != tver:
                    cfg.requirements["torch"] = tver
                    return f"align torch=={tver} for torchvision=={tv_ver}"
    return None


def apply_patch(cfg: RunConfig, error_type: str) -> List[str]:
    patches: List[str] = []

    # ensure torch is present before building common torch-dependent packages
    if cfg.requirements_text:
        torch_deps = ("xformers", "flash-attn", "bitsandbytes", "triton")
        if any(dep in cfg.requirements_text for dep in torch_deps) and "torch" not in cfg.requirements_text:
            cfg.requirements_text = "torch\n" + cfg.requirements_text
            patches.append("prepend torch to requirements.txt for torch-dependent packages")
    else:
        torch_deps = ("xformers", "flash-attn", "bitsandbytes", "triton")
        if any(dep in cfg.requirements for dep in torch_deps) and "torch" not in cfg.requirements:
            cfg.requirements["torch"] = None
            patches.append("add torch for torch-dependent packages")

    if error_type == "dockerfile_parse_error":
        if cfg.dockerfile_text:
            cfg.dockerfile_text = None
            patches.append("discard invalid Dockerfile (fallback to generated template)")

    elif error_type == "torch_cuda_mismatch":
        if cfg.requirements.get("torch") is None:
            cfg.requirements["torch"] = "2.0.1"
            patches.append("pin torch==2.0.1")
        if cfg.requirements.get("torchvision") is None:
            cfg.requirements["torchvision"] = "0.15.2"
            patches.append("pin torchvision==0.15.2")
        aligned = _align_torch_vision(cfg)
        if aligned:
            patches.append(aligned)

    elif error_type == "python_requires":
        if cfg.dockerfile_text:
            updated = False
            df = cfg.dockerfile_text
            if "ubuntu18.04" in df:
                df = df.replace("ubuntu18.04", "ubuntu20.04")
                updated = True
            if "pip/3.6/get-pip.py" in df:
                df = df.replace("pip/3.6/get-pip.py", "pip/3.8/get-pip.py")
                updated = True
            if updated:
                cfg.dockerfile_text = df
                cfg.python_version = "3.8"
                patches.append("update repo Dockerfile to python 3.8 (ubuntu20.04 + get-pip 3.8)")
            else:
                _set_python_version(cfg, "3.9")
                patches.append("bump python to meet requires-python")
        else:
            if cfg.python_version == "3.8":
                _set_python_version(cfg, "3.9")
                patches.append("bump python 3.8->3.9")
            elif cfg.python_version == "3.9":
                _set_python_version(cfg, "3.10")
                patches.append("bump python 3.9->3.10")
    elif error_type == "python_too_old_numpy2":
        if cfg.python_version in ("3.8", "3.9"):
            prev = cfg.python_version
            _set_python_version(cfg, "3.10")
            patches.append(f"bump python {prev}->3.10 (numpy>=2)")
    elif error_type == "missing_sqlite_dev":
        if "libsqlite3-dev" not in cfg.system_packages:
            cfg.system_packages.append("libsqlite3-dev")
            patches.append("add libsqlite3-dev for sqlite headers")

    elif error_type == "cuda_driver_insufficient":
        if "11.8" in cfg.base_image:
            cfg.base_image = cfg.base_image.replace("11.8", "11.7")
            patches.append("downgrade cuda base 11.8->11.7")
        elif "11.7" in cfg.base_image:
            cfg.base_image = cfg.base_image.replace("11.7", "11.6")
            patches.append("downgrade cuda base 11.7->11.6")

    elif error_type == "module_not_found":
        m = re.search(r"(?:No module named|ModuleNotFoundError:\s+No module named)\s+['\"]([a-zA-Z0-9_\-\.]+)['\"]", _LAST_LOGS)
        if m:
            missing = m.group(1).split(".")[0]
            if cfg.requirements_text:
                if missing not in cfg.requirements_text:
                    cfg.requirements_text = cfg.requirements_text.rstrip() + f"\n{missing}\n"
                    patches.append(f"add missing requirement to requirements.txt: {missing}")
            else:
                if missing not in cfg.requirements:
                    cfg.requirements[missing] = None
                    patches.append(f"add missing requirement: {missing}")
    elif error_type == "missing_build_tools":
        for pkg in ["build-essential", "cmake", "ninja-build", "pkg-config"]:
            if pkg not in cfg.system_packages:
                cfg.system_packages.append(pkg)
        patches.append("add build tools: build-essential cmake ninja-build pkg-config")
    elif error_type == "missing_python_dev":
        if "python3-dev" not in cfg.system_packages:
            cfg.system_packages.append("python3-dev")
            patches.append("add python3-dev for Python.h")
    elif error_type == "missing_gl_libs":
        for pkg in ["libgl1", "libglib2.0-0", "libsm6", "libxrender1", "libxext6"]:
            if pkg not in cfg.system_packages:
                cfg.system_packages.append(pkg)
        patches.append("add GL runtime libs (libgl1/libglib2.0-0/libsm6/libxrender1/libxext6)")
    elif error_type == "missing_cuda_nvcc":
        if "nvidia/cuda" in cfg.base_image and "runtime" in cfg.base_image:
            cfg.base_image = cfg.base_image.replace("runtime", "devel")
            patches.append("switch cuda base to devel (nvcc)")
    elif error_type == "cuda_home_missing":
        if "nvidia/cuda" in cfg.base_image and "runtime" in cfg.base_image:
            cfg.base_image = cfg.base_image.replace("runtime", "devel")
            patches.append("switch cuda base to devel (CUDA_HOME)")
        elif "nvidia/cuda" not in cfg.base_image:
            cfg.base_image = "nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04"
            patches.append("switch base image to cuda devel (CUDA_HOME)")
        for pkg in ["build-essential", "cmake", "ninja-build", "pkg-config"]:
            if pkg not in cfg.system_packages:
                cfg.system_packages.append(pkg)
        patches.append("add build tools for CUDA extensions")
    elif error_type == "torchvision_requires_torch":
        aligned = _align_torch_vision(cfg)
        if aligned:
            patches.append(aligned)
    elif error_type == "apt_repo_key":
        for pkg in ["gnupg", "ca-certificates"]:
            if pkg not in cfg.system_packages:
                cfg.system_packages.append(pkg)
        patches.append("add gnupg/ca-certificates for apt repo keys")
    elif error_type == "apt_fetch_failed":
        if "ca-certificates" not in cfg.system_packages:
            cfg.system_packages.append("ca-certificates")
        patches.append("add ca-certificates for apt fetch issues")
    elif error_type == "pip_tls":
        if "ca-certificates" not in cfg.system_packages:
            cfg.system_packages.append("ca-certificates")
        patches.append("add ca-certificates for pip TLS errors")
    elif error_type == "no_space_left":
        patches.append("no space left on device (manual cleanup required)")
    elif error_type == "process_killed":
        patches.append("process killed (possibly OOM); consider fewer packages or more memory")
    elif error_type == "glibcxx_missing":
        if "libstdc++6" not in cfg.system_packages:
            cfg.system_packages.append("libstdc++6")
        patches.append("add libstdc++6 for GLIBCXX")
    elif error_type == "pip_dependency_conflict":
        if not cfg.constraints_text:
            # build a minimal constraints list from pinned requirements or versions in requirements_text
            constraints = []
            if cfg.requirements_text:
                for line in cfg.requirements_text.splitlines():
                    val = line.strip()
                    if not val or val == "." or val.startswith("git+"):
                        continue
                    if "==" in val or ">=" in val or "<=" in val:
                        constraints.append(val)
            else:
                for pkg, ver in cfg.requirements.items():
                    if ver:
                        constraints.append(f"{pkg}=={ver}")
            if constraints:
                cfg.constraints_text = "\n".join(constraints)
                patches.append("add constraints.txt from pinned requirements")

    elif error_type == "no_matching_distribution":
        m = re.search(
            r"(?:No matching distribution found for|Could not find a version that satisfies the requirement)\s+([A-Za-z0-9_.-]+)",
            _LAST_LOGS,
        )
        if re.search(r"Requires-Python\s*>=\s*[\d.]+,\s*<\s*3\.11", _LAST_LOGS):
            if cfg.python_version not in ("3.10", "3.9", "3.8"):
                _set_python_version(cfg, "3.10")
                patches.append("set python 3.10 (requires-python <3.11)")
        if m:
            pkg = m.group(1)
            if pkg.lower() == "decord" and "aarch64" in _LAST_LOGS and cfg.requirements_text:
                cfg.requirements_text = _remove_requirement_text(cfg.requirements_text, pkg)
                patches.append("remove decord (no aarch64 wheel)")
                return patches
            if pkg.lower() == "numpy" and "1.24.4" in _LAST_LOGS and cfg.requirements_text:
                cfg.requirements_text = _replace_requirement_text(cfg.requirements_text, "numpy", "numpy==1.24.4")
                patches.append("pin numpy==1.24.4 (no matching distribution for 1.26.0)")
                return patches
            if pkg in cfg.requirements and cfg.requirements.get(pkg) is not None:
                cfg.requirements[pkg] = None
                patches.append(f"unpin {pkg} (no matching distribution)")
        if not patches:
            if cfg.python_version in ("3.8", "3.9"):
                prev = cfg.python_version
                _set_python_version(cfg, "3.10")
                patches.append(f"bump python {prev}->3.10 (no matching distribution)")
            elif cfg.python_version == "3.10":
                _set_python_version(cfg, "3.11")
                patches.append("bump python 3.10->3.11 (no matching distribution)")

    elif error_type == "git_clone_failed":
        changed = False
        if cfg.dockerfile_text:
            cleaned, updated = _clean_urls_in_text(cfg.dockerfile_text)
            if updated:
                cfg.dockerfile_text = cleaned
                changed = True
        github_urls = cfg.source_hints.get("github_urls") or []
        if github_urls:
            new_urls = []
            for url in github_urls:
                cleaned = _clean_url_suffix(url)
                if cleaned != url:
                    changed = True
                new_urls.append(cleaned)
            cfg.source_hints["github_urls"] = new_urls
        if changed:
            patches.append("sanitize git clone URLs")

    elif error_type == "rust_compiler_missing":
        for pkg in ["rustc", "cargo"]:
            if pkg not in cfg.system_packages:
                cfg.system_packages.append(pkg)
        patches.append("add rustc/cargo for Rust-based builds")
    elif error_type == "tzdata_prompt":
        if "tzdata" not in cfg.system_packages:
            cfg.system_packages.append("tzdata")
            patches.append("add tzdata to avoid interactive timezone prompt")

    return patches


def apply_llm_patch(cfg: RunConfig, model: str) -> List[str]:
    prompt = build_patch_prompt(
        _LAST_LOGS,
        run_cmd=cfg.run_cmd,
        base_image=cfg.base_image,
        python_version=cfg.python_version,
        requirements_text=cfg.requirements_text,
        dockerfile_text=cfg.dockerfile_text,
    )
    data = call_ollama(model, prompt)
    if not data or "actions" not in data:
        return []

    patches: List[str] = []
    actions = data.get("actions") or []
    if not isinstance(actions, list):
        return []

    for act in actions:
        if not isinstance(act, dict):
            continue
        typ = act.get("type")
        if typ == "pin":
            pkg = act.get("package")
            ver = act.get("version")
            if pkg and ver:
                cfg.requirements[pkg] = str(ver)
                patches.append(f"pin {pkg}=={ver} (llm)")
        elif typ == "add_requirement":
            pkg = act.get("package")
            if pkg and pkg not in cfg.requirements:
                cfg.requirements[pkg] = None
                patches.append(f"add requirement {pkg} (llm)")
        elif typ == "set_base_image":
            val = act.get("value")
            if val:
                cfg.base_image = str(val)
                patches.append(f"set base image {val} (llm)")
        elif typ == "set_python":
            ver = act.get("version")
            if ver:
                _set_python_version(cfg, str(ver))
                patches.append(f"set python {ver} (llm)")
        elif typ == "reorder_requirements":
            if cfg.requirements_text:
                pkgs = act.get("packages") or []
                if isinstance(pkgs, list) and pkgs:
                    cfg.requirements_text = _reorder_requirements_text(cfg.requirements_text, [str(p) for p in pkgs])
                    patches.append("reorder requirements.txt (llm)")

    return patches
