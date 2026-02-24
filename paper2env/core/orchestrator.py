from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import List

from .classifier import classify_error
from .models import AttemptResult, RunConfig
from .patcher import apply_llm_patch, apply_patch, normalize_python_packages, set_last_logs
from .runner import docker_build, docker_run
from .state import rollback_state, snapshot_state
from .templates import render_dockerfile


def write_requirements(cfg: RunConfig) -> None:
    if cfg.requirements_text:
        content = _sanitize_requirements_text(cfg.requirements_text)
        if not cfg.requirements_is_conda and cfg.requirements:
            # ensure required packages (e.g., torch) are present in requirements.txt
            existing = set()
            for line in content.splitlines():
                val = line.strip()
                if not val or val.startswith("#"):
                    continue
                name = re.split(r"[=<>!~]", val, 1)[0].strip()
                if name:
                    existing.add(name.lower())
            injected = []
            for pkg, ver in cfg.requirements.items():
                if pkg.lower() in existing:
                    continue
                injected.append(f"{pkg}=={ver}" if ver else pkg)
            if injected:
                content = "\n".join(injected) + "\n" + content
        # ensure core packages install first (torch stack before xformers)
        content = _reorder_requirements_text(content, ["torch", "torchvision", "torchaudio"])
        # peel off xformers and VCS deps into a separate file to install without build isolation
        post_lines = []
        main_lines = []
        for line in content.splitlines():
            val = line.strip()
            if not val:
                continue
            name = re.split(r"[=<>!~]", val, 1)[0].strip().lower()
            if name == "xformers" or val.startswith(("git+", "-e ", "http://", "https://")):
                post_lines.append(line)
            else:
                main_lines.append(line)
        content = "\n".join(main_lines) + ("\n" if main_lines else "")
    else:
        lines = []
        if cfg.pip_extra_index_url:
            lines.append(f"--extra-index-url {cfg.pip_extra_index_url}")
        for pkg, ver in cfg.requirements.items():
            if ver:
                lines.append(f"{pkg}=={ver}")
            else:
                lines.append(pkg)
        content = "\n".join(lines) + "\n"
        post_lines = []
        for pkg, ver in list(cfg.requirements.items()):
            if pkg.lower() == "xformers":
                post_lines.append(f"{pkg}=={ver}" if ver else pkg)

    with open(os.path.join(cfg.workdir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(content)
    post_path = os.path.join(cfg.workdir, "requirements.post.txt")
    with open(post_path, "w", encoding="utf-8") as f:
        if post_lines:
            f.write("\n".join(post_lines) + "\n")
        else:
            f.write("")


def write_env_json(cfg: RunConfig) -> None:
    with open(os.path.join(cfg.workdir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)


def write_dockerfile(cfg: RunConfig) -> None:
    content = cfg.dockerfile_text or render_dockerfile(cfg)
    with open(os.path.join(cfg.workdir, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(content)


def write_environment_yml(cfg: RunConfig) -> None:
    if not cfg.environment_yml_text:
        return
    with open(os.path.join(cfg.workdir, "environment.yml"), "w", encoding="utf-8") as f:
        f.write(cfg.environment_yml_text.strip() + "\n")


def write_constraints(cfg: RunConfig) -> None:
    if not cfg.constraints_text:
        return
    with open(os.path.join(cfg.workdir, "constraints.txt"), "w", encoding="utf-8") as f:
        f.write(_normalize_requirements_text(cfg.constraints_text))


def write_log(workdir: str, attempt_idx: int, logs: str) -> None:
    log_dir = os.path.join(workdir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"attempt_{attempt_idx:02d}.log"), "w", encoding="utf-8") as f:
        f.write(logs)


def write_failure_report(workdir: str, attempt_idx: int, result: AttemptResult, cfg: RunConfig) -> None:
    report = {
        "attempt": attempt_idx,
        "success": result.success,
        "error_type": result.error_type,
        "exit_code": result.exit_code,
        "patches_applied": result.patches_applied,
        "log_tail": (result.logs or "")[-4000:],
        "env": cfg.to_dict(),
    }
    path = os.path.join(workdir, "failure_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _normalize_requirements_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        val = line.strip()
        if not val or val == ".":
            continue
        lines.append(val)
    return "\n".join(lines) + "\n"


def _sanitize_requirements_text(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == ".":
            continue
        if line.startswith("#"):
            continue
        # skip common conda export headers/entries
        if line.startswith(("_libgcc_mutex", "_openmp_mutex")):
            continue
        if line.startswith(("name:", "prefix:", "channels:", "dependencies:", "platform:")):
            continue
        if re.match(r"^[A-Za-z0-9_.-]+=\d", line) and "==" not in line:
            continue

        # keep common pip options and VCS URLs
        if line.startswith(("git+", "http://", "https://", "-e ")):
            lines.append(line)
            continue
        if line.startswith(("-f ", "--find-links", "--extra-index-url", "--index-url", "--trusted-host", "-i ")):
            lines.append(line)
            continue

        # keep valid pip specifiers or bare packages
        if any(op in line for op in ("==", ">=", "<=", "~=", "!=")):
            lines.append(line)
            continue
        if re.match(r"^[A-Za-z0-9_.-]+$", line):
            lines.append(line)
            continue

        # drop anything else that looks like conda metadata
        if "=" in line and "==" not in line:
            continue

    return "\n".join(lines) + "\n"


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


def _next_attempt_index(workdir: str) -> int:
    log_dir = os.path.join(workdir, "logs")
    if not os.path.isdir(log_dir):
        return 1
    max_idx = 0
    for name in os.listdir(log_dir):
        m = re.match(r"attempt_(\d+)\.log$", name)
        if m:
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except ValueError:
                continue
    return max_idx + 1


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def self_heal_loop(
    cfg: RunConfig,
    max_attempts: int = 5,
    do_docker: bool = True,
    llm_model: str | None = None,
) -> AttemptResult:
    os.makedirs(cfg.workdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.workdir, ".snapshots"), exist_ok=True)

    patches_applied_all: List[str] = []

    start_idx = _next_attempt_index(cfg.workdir)
    no_cache_next = False
    last_hash_path = os.path.join(cfg.workdir, ".last_dockerfile.sha256")
    for i in range(start_idx, start_idx + max_attempts):
        normalize_python_packages(cfg)
        # Ensure torch is installed before torch-dependent packages even without an error-triggered patch.
        if cfg.requirements_text:
            torch_deps = ("xformers", "flash-attn", "bitsandbytes", "triton")
            if any(dep in cfg.requirements_text for dep in torch_deps) and "torch" not in cfg.requirements_text:
                cfg.requirements_text = "torch\n" + cfg.requirements_text
        write_requirements(cfg)
        write_env_json(cfg)
        write_dockerfile(cfg)
        write_environment_yml(cfg)
        write_constraints(cfg)

        dockerfile_path = os.path.join(cfg.workdir, "Dockerfile")
        if os.path.exists(dockerfile_path):
            try:
                current_hash = _file_hash(dockerfile_path)
                prev_hash = None
                if os.path.exists(last_hash_path):
                    with open(last_hash_path, "r", encoding="utf-8") as f:
                        prev_hash = f.read().strip() or None
                if prev_hash != current_hash:
                    no_cache_next = True
                with open(last_hash_path, "w", encoding="utf-8") as f:
                    f.write(current_hash)
            except Exception:
                pass

        snap_dir = snapshot_state(cfg, i)

        if not do_docker:
            return AttemptResult(True, 0, "dry-run", "none", patches_applied_all)

        print(f"[attempt {i}] docker build 시작...")
        t0 = time.time()
        code, logs = docker_build(cfg.workdir, cfg.docker_tag, no_cache=no_cache_next, verbose=cfg.verbose)
        dt = time.time() - t0
        if no_cache_next:
            print(f"[attempt {i}] build cache disabled")
        print(f"[attempt {i}] docker build 종료 (t={dt:.1f}s)")
        set_last_logs(logs)
        write_log(cfg.workdir, i, logs)
        no_cache_next = False
        if code != 0:
            err = classify_error(logs)
            if err == "docker_permission":
                result = AttemptResult(False, code, logs, err, patches_applied_all)
                write_failure_report(cfg.workdir, i, result, cfg)
                if cfg.best_effort:
                    return AttemptResult(True, code, logs, err, patches_applied_all)
                return result
            patches = apply_patch(cfg, err)
            if llm_model and (err == "unknown" or not patches):
                llm_patches = apply_llm_patch(cfg, llm_model)
                if llm_patches:
                    patches.extend(llm_patches)
            if not patches:
                result = AttemptResult(False, code, logs, err, patches_applied_all)
                write_failure_report(cfg.workdir, i, result, cfg)
                if cfg.best_effort:
                    return AttemptResult(True, code, logs, err, patches_applied_all)
                return result
            patches_applied_all.extend([f"[build] {p}" for p in patches])
            no_cache_next = True
            continue

        print(f"[attempt {i}] docker run 시작...")
        t0 = time.time()
        gpu_indices = None
        if cfg.gpu_mode != "none":
            gpu_indices = cfg.gpu_indices
        code, logs = docker_run(
            cfg.docker_tag,
            cfg.run_cmd,
            cfg.container_name,
            gpu_indices,
            keep_container=cfg.keep_container,
            mount_host_dir=cfg.mount_host_dir,
            mount_container_dir=cfg.mount_container_dir,
            detach=cfg.detach,
        )
        dt = time.time() - t0
        print(f"[attempt {i}] docker run 종료 (t={dt:.1f}s)")
        set_last_logs(logs)
        write_log(cfg.workdir, i, logs)
        if code == 0:
            return AttemptResult(True, code, logs, "none", patches_applied_all)

        err = classify_error(logs)
        patches = apply_patch(cfg, err)
        if llm_model and (err == "unknown" or not patches):
            llm_patches = apply_llm_patch(cfg, llm_model)
            if llm_patches:
                patches.extend(llm_patches)
        if not patches:
            rollback_state(cfg, snap_dir)
            result = AttemptResult(False, code, logs, err, patches_applied_all)
            write_failure_report(cfg.workdir, i, result, cfg)
            if cfg.best_effort:
                return AttemptResult(True, code, logs, err, patches_applied_all)
            return result

        patches_applied_all.extend([f"[run] {p}" for p in patches])

    result = AttemptResult(False, 1, "", "max_attempts_exceeded", patches_applied_all)
    write_failure_report(cfg.workdir, start_idx + max_attempts - 1, result, cfg)
    return result
