from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Optional

from .core.host import (
    check_docker,
    detect_host_cuda_version,
    get_platform_info,
    list_gpus,
    pytorch_extra_index_url,
    try_run_command,
)
from .core.models import RunConfig
from .core.orchestrator import self_heal_loop
from .core.parser import parse_pdf
from .core.templates import choose_base_image
from .core.websearch import (
    derive_requirements_from_readme,
    derive_requirements_from_setup_cfg,
    derive_requirements_from_setup_py,
    extract_github_urls,
    extract_hf_urls,
    harvest_links_from_web_urls,
    scan_github_repo,
    scan_hf_model,
    search_all,
    search_web,
)


def _normalize_requirements_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    lines = []
    for line in text.splitlines():
        val = line.strip()
        if not val or val == ".":
            continue
        lines.append(val)
    return "\n".join(lines)


def _paper_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    name = re.sub(r"\.pdf$", "", base, flags=re.I)
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "-", name)
    return (name.strip("-") or "paper").lower()


def _prompt_or_default(label: str, default: str) -> str:
    try:
        val = input(f"{label} [{default}]: ").strip()
    except EOFError:
        return default
    return val or default


def _prompt_yes_no(label: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    try:
        val = input(f"{label} [{suffix}]: ").strip().lower()
    except EOFError:
        return default
    if not val:
        return default
    return val in ("y", "yes")


def _parse_gpu_indices(text: Optional[str]) -> Optional[list[int]]:
    if not text:
        return None
    indices: list[int] = []
    for part in text.split(","):
        val = part.strip()
        if not val:
            continue
        try:
            idx = int(val)
        except ValueError:
            continue
        if idx not in indices:
            indices.append(idx)
    return indices


def _parse_mount(text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    if ":" not in text:
        return text, "/workspace"
    host, container = text.split(":", 1)
    host = host.strip()
    container = container.strip()
    return (host or None), (container or None)


def _clean_url(url: str) -> str:
    return url.rstrip(").,;]}>")


def _clean_urls(urls: list[str]) -> list[str]:
    cleaned = []
    for u in urls:
        if not u:
            continue
        cu = _clean_url(u.strip())
        if cu and cu not in cleaned:
            cleaned.append(cu)
    return cleaned


def _clean_dockerfile_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    def _clean_in_line(line: str) -> str:
        if "http://" not in line and "https://" not in line:
            return line
        parts = line.split()
        new_parts = []
        for p in parts:
            if p.startswith("http://") or p.startswith("https://"):
                new_parts.append(_clean_url(p))
            else:
                new_parts.append(p)
        return " ".join(new_parts)

    lines = [_clean_in_line(line) for line in text.splitlines()]
    return "\n".join(lines)


def _patch_dockerfile_for_conda(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    if "pip install -r /workspace/requirements.txt" not in text:
        return text
    lines = []
    for line in text.splitlines():
        if "pip install -r /workspace/requirements.txt" in line:
            lines.append("RUN echo 'skip conda-style requirements.txt (use conda env or repo requirements)'")
        else:
            lines.append(line)
    return "\n".join(lines)


def _is_conda_export(text: Optional[str]) -> bool:
    if not text:
        return False
    head = text.strip().splitlines()[:5]
    joined = "\n".join(head)
    if "conda create --name" in joined or "platform: linux-64" in joined:
        return True
    if "_libgcc_mutex=" in text or "_openmp_mutex=" in text:
        return True
    return False


def _existing_container_names() -> list[str]:
    code, out = try_run_command(["docker", "ps", "-a", "--format", "{{.Names}}"])
    if code != 0 or not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _load_existing_artifacts(workdir: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    dockerfile = None
    requirements = None
    env_yml = None
    df_path = os.path.join(workdir, "Dockerfile")
    req_path = os.path.join(workdir, "requirements.txt")
    env_path = os.path.join(workdir, "environment.yml")
    if os.path.isfile(df_path):
        try:
            dockerfile = open(df_path, "r", encoding="utf-8").read()
        except Exception:
            dockerfile = None
    if os.path.isfile(req_path):
        try:
            requirements = open(req_path, "r", encoding="utf-8").read()
        except Exception:
            requirements = None
    if os.path.isfile(env_path):
        try:
            env_yml = open(env_path, "r", encoding="utf-8").read()
        except Exception:
            env_yml = None
    return dockerfile, requirements, env_yml


def _unique_container_name(base: str, interactive: bool) -> str:
    names = set(_existing_container_names())
    if base not in names:
        return base
    if interactive:
        return _prompt_or_default("Container name already exists. New name", f"{base}-2")
    # non-interactive: auto-append suffix
    idx = 2
    while f"{base}-{idx}" in names:
        idx += 1
    return f"{base}-{idx}"


def _requirements_from_parsed(framework: Optional[str], libs: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    reqs: Dict[str, Optional[str]] = {}
    if framework == "pytorch":
        reqs["torch"] = libs.get("torch")
        if "torchvision" in libs:
            reqs["torchvision"] = libs.get("torchvision")
    elif framework == "tensorflow":
        reqs["tensorflow"] = libs.get("tensorflow")
    elif framework == "jax":
        reqs["jax"] = libs.get("jax")

    for k, v in libs.items():
        if k not in reqs:
            reqs[k] = v

    if not reqs:
        reqs["numpy"] = None

    return reqs


def cmd_run(args: argparse.Namespace) -> int:
    parsed = parse_pdf(args.pdf)

    python_version = parsed.python_version or args.python or "3.8"
    cuda_version = parsed.cuda_version or args.cuda
    gpu_mode = args.gpu
    gpu_indices = _parse_gpu_indices(args.gpu_index) or []
    gpu_list = []
    if gpu_mode != "none":
        gpu_list = list_gpus()
        if args.interactive and gpu_mode == "auto":
            if gpu_list:
                print("Detected GPUs:")
                for idx, name in gpu_list:
                    print(f"  {idx}: {name}")
                if not _prompt_yes_no("Use GPU for this run?", True):
                    gpu_mode = "none"
                    gpu_indices = []
            else:
                print("No NVIDIA GPU detected (nvidia-smi not found). Proceeding with CPU.")
        if gpu_mode != "none" and gpu_list:
            if args.interactive and len(gpu_list) > 1 and not args.gpu_index:
                choices = ", ".join(f"{i}:{name}" for i, name in gpu_list)
                picked = _prompt_or_default(f"GPU indices (comma, empty=all). Available: {choices}", "")
                gpu_indices = _parse_gpu_indices(picked) or []
            if args.gpu_index:
                gpu_indices = _parse_gpu_indices(args.gpu_index) or []
            if not gpu_indices:
                gpu_indices = [i for i, _ in gpu_list]
    host_cuda_version = None
    host_driver_version = None
    if not cuda_version and not args.no_auto_cuda:
        host_cuda_version, host_driver_version = detect_host_cuda_version()
        if host_cuda_version:
            cuda_version = host_cuda_version
    web_hints = {"github_urls": [], "hf_urls": [], "web_urls": [], "docker_images": [], "dockerhub_urls": []}
    web_search_enabled = not args.no_web_search
    if args.web_search:
        web_search_enabled = True
    if web_search_enabled and not (parsed.github_urls or parsed.hf_urls or parsed.docker_images):
        query = parsed.title or _paper_id_from_path(args.pdf)
        web_hints = search_all(query, limit=args.web_limit)
        web_hints["web_urls"] = search_web(query, limit=args.web_limit)
        web_hints.setdefault("docker_images", [])
        web_hints.setdefault("dockerhub_urls", [])
        web_hints["github_urls"] += extract_github_urls(web_hints["web_urls"])
        web_hints["hf_urls"] += extract_hf_urls(web_hints["web_urls"])
        harvested = harvest_links_from_web_urls(web_hints["web_urls"])
        web_hints["github_urls"] += harvested.get("github_urls", [])
        web_hints["hf_urls"] += harvested.get("hf_urls", [])
        web_hints["dockerhub_urls"] += harvested.get("dockerhub_urls", [])
        web_hints["docker_images"] += harvested.get("docker_images", [])

    manual_github = [args.github] if args.github else []
    manual_hf = [args.hf] if args.hf else []
    manual_docker = [args.docker_image] if args.docker_image else []

    docker_images = manual_docker + list(parsed.docker_images) + web_hints["docker_images"]
    if not docker_images and web_search_enabled:
        docker_images = []

    base_image = docker_images[0] if docker_images else choose_base_image(cuda_version, python_version)

    requirements = _requirements_from_parsed(parsed.framework, parsed.libraries)
    pip_extra_index_url = None
    system_packages = [f"python{python_version}", "python3-pip", "build-essential"]
    for pkg in parsed.system_packages:
        if pkg not in system_packages:
            system_packages.append(pkg)

    github_urls = _clean_urls(manual_github + parsed.github_urls + web_hints["github_urls"])
    hf_urls = _clean_urls(manual_hf + parsed.hf_urls + web_hints["hf_urls"])

    repo_dockerfile = None
    repo_requirements = None
    repo_env_yml = None
    repo_req_source = None
    if github_urls:
        repo_data = scan_github_repo(github_urls[0])
        repo_dockerfile = repo_data.get("dockerfile")
        repo_requirements = repo_data.get("requirements")
        repo_env_yml = repo_data.get("environment_yml")
        repo_readme = repo_data.get("readme")
        repo_setup_py = repo_data.get("setup_py")
        repo_setup_cfg = repo_data.get("setup_cfg")
        if repo_dockerfile:
            m = re.search(r"^FROM\s+([A-Za-z0-9./:_-]+)", repo_dockerfile, re.M)
            if m:
                base_image = m.group(1)
        if not repo_requirements and repo_readme:
            repo_requirements = derive_requirements_from_readme(repo_readme, github_urls[0])
            if repo_requirements:
                repo_req_source = "readme"
        if not repo_requirements and repo_setup_cfg:
            repo_requirements = derive_requirements_from_setup_cfg(repo_setup_cfg)
            if repo_requirements:
                repo_req_source = "setup.cfg"
        if not repo_requirements and repo_setup_py:
            repo_requirements = derive_requirements_from_setup_py(repo_setup_py)
            if repo_requirements:
                repo_req_source = "setup.py"
        if repo_requirements and not repo_req_source:
            repo_req_source = "requirements.txt"

    hf_requirements = None
    hf_env_yml = None
    hf_req_source = None
    if hf_urls and not repo_requirements:
        hf_data = scan_hf_model(hf_urls[0])
        hf_requirements = hf_data.get("requirements")
        hf_env_yml = hf_data.get("environment_yml")
        if hf_requirements:
            hf_req_source = "huggingface"

    if parsed.framework == "pytorch":
        run_cmd = ["python", "-c", "import torch; print(torch.__version__)"]
    elif parsed.framework == "tensorflow":
        run_cmd = ["python", "-c", "import tensorflow as tf; print(tf.__version__)"]
    else:
        run_cmd = ["python", "-c", "print('ok')"]

    paper_id = _paper_id_from_path(args.pdf)
    workdir = args.workdir or os.path.join(os.getcwd(), "workspace", paper_id)
    docker_tag = f"paper2env/{paper_id}:local"
    container_name = args.container_name
    if args.tag:
        docker_tag = f"paper2env/{paper_id}:{args.tag}"
    if args.image_name:
        docker_tag = args.image_name if ":" in args.image_name else f"{args.image_name}:local"
    mount_host_dir, mount_container_dir = _parse_mount(args.mount)
    mount_enabled = not args.no_mount
    if args.interactive:
        docker_tag = _prompt_or_default("Docker image tag (enter to use default)", docker_tag)
        container_name = _prompt_or_default(
            "Docker container name (enter to use default)",
            container_name or f"paper2env-{paper_id}",
        )
        if mount_enabled and not args.mount:
            if _prompt_yes_no("Mount local workspace into container?", True):
                mount_host_dir = workdir
                mount_container_dir = "/workspace"
            else:
                mount_enabled = False
    if not container_name:
        container_name = f"paper2env-{paper_id}"
    container_name = _unique_container_name(container_name, args.interactive)
    if mount_enabled and not mount_host_dir:
        mount_host_dir = workdir
    if mount_enabled and not mount_container_dir:
        mount_container_dir = "/workspace"

    notes = []
    if docker_images:
        notes.append(f"base_image from docker image hint: {docker_images[0]}")
    else:
        notes.append(f"base_image inferred from python/cuda: {base_image}")
    if repo_dockerfile:
        notes.append("Dockerfile sourced from GitHub repo")
    if repo_requirements:
        src = repo_req_source or "github"
        notes.append(f"requirements sourced from GitHub ({src})")
    if hf_requirements:
        src = hf_req_source or "huggingface"
        notes.append(f"requirements sourced from HuggingFace ({src})")
    if repo_env_yml or hf_env_yml:
        notes.append("environment.yml sourced from external repo")
    if parsed.raw_text_excerpt:
        notes.append("paper text parsed for framework/python/cuda hints")
    if host_cuda_version:
        if host_driver_version:
            notes.append(f"host CUDA detected via nvidia-smi: {host_cuda_version} (driver {host_driver_version})")
        else:
            notes.append(f"host CUDA detected via nvidia-smi: {host_cuda_version}")
    elif not args.no_auto_cuda and not parsed.cuda_version and not args.cuda:
        notes.append("host CUDA auto-detect unavailable (nvidia-smi not found); falling back to CPU base unless paper hints exist")
    if gpu_mode == "none":
        notes.append("GPU disabled by user")
    elif gpu_list:
        chosen = ",".join(str(i) for i in gpu_indices) if gpu_indices else "all"
        notes.append(f"GPU enabled: {chosen}")
    else:
        notes.append("GPU requested but not detected (nvidia-smi not found)")

    if parsed.framework == "pytorch" and not (repo_requirements or hf_requirements):
        chosen_cuda, extra_idx = pytorch_extra_index_url(cuda_version)
        if extra_idx:
            pip_extra_index_url = extra_idx
            notes.append(f"pytorch extra-index set for CUDA {chosen_cuda}")
        elif cuda_version:
            notes.append("pytorch CUDA wheel not selected; falling back to CPU wheels")

    keep_container = not args.rm
    keep_alive = not args.no_keep_alive and keep_container
    detach = keep_alive
    if keep_alive:
        if mount_enabled and github_urls:
            repo = github_urls[0]
            run_cmd = ["bash", "-lc", f"git clone {repo} /workspace/repo || true; tail -f /dev/null"]
        else:
            run_cmd = ["tail", "-f", "/dev/null"]

    repo_requirements = _normalize_requirements_text(repo_requirements)
    hf_requirements = _normalize_requirements_text(hf_requirements)
    requirements_is_conda = _is_conda_export(repo_requirements or hf_requirements)

    existing_dockerfile = None
    existing_requirements = None
    existing_env_yml = None
    if args.resume and os.path.isdir(workdir):
        existing_dockerfile, existing_requirements, existing_env_yml = _load_existing_artifacts(workdir)
        if existing_dockerfile:
            existing_dockerfile = _clean_dockerfile_text(existing_dockerfile)
            notes.append("using existing Dockerfile from workspace")
        if existing_requirements:
            notes.append("using existing requirements.txt from workspace")
        if existing_env_yml:
            notes.append("using existing environment.yml from workspace")

    if requirements_is_conda:
        notes.append("conda-style requirements detected; skipping pip install for requirements.txt")
        existing_dockerfile = _patch_dockerfile_for_conda(existing_dockerfile)

    if github_urls and "git" not in system_packages:
        system_packages.append("git")
    if (repo_requirements or hf_requirements) and "git+" in (repo_requirements or hf_requirements):
        if "git" not in system_packages:
            system_packages.append("git")

    cfg = RunConfig(
        paper_id=paper_id,
        workdir=workdir,
        base_image=base_image,
        python_version=python_version,
        requirements=requirements,
        system_packages=system_packages,
        source_hints={
            "docker_images": docker_images,
            "github_urls": github_urls,
            "hf_urls": hf_urls,
            "dockerhub_urls": parsed.dockerhub_urls + web_hints["dockerhub_urls"],
            "web_urls": web_hints["web_urls"],
        },
        dockerfile_text=existing_dockerfile or repo_dockerfile,
        requirements_text=existing_requirements or repo_requirements or hf_requirements,
        environment_yml_text=existing_env_yml or repo_env_yml or hf_env_yml,
        constraints_text=None,
        pip_extra_index_url=pip_extra_index_url,
        gpu_mode=gpu_mode,
        gpu_indices=gpu_indices,
        notes=notes,
        run_cmd=run_cmd,
        docker_tag=docker_tag,
        container_name=container_name,
        keep_container=not args.rm,
        mount_host_dir=mount_host_dir if mount_enabled else None,
        mount_container_dir=mount_container_dir if mount_enabled else None,
        keep_alive=keep_alive,
        detach=detach,
        verbose=args.verbose,
        requirements_is_conda=requirements_is_conda,
        best_effort=args.best_effort,
    )

    result = self_heal_loop(
        cfg,
        max_attempts=args.max_attempts,
        do_docker=not args.dry_run,
        llm_model=args.llm_model,
    )

    print("SUCCESS:", result.success)
    print("ERROR_TYPE:", result.error_type)
    print("PATCHES:", result.patches_applied)
    if result.logs:
        print("LOG_TAIL:", result.logs[-2000:])
    if result.success and result.error_type != "none":
        print("BEST_EFFORT:", True)
    if result.success and result.error_type == "none" and keep_alive and container_name:
        print("CONTAINER:", container_name)
        print("NEXT:", f"docker exec -it {container_name} /bin/bash")

    return 0 if result.success else 2


def cmd_export(args: argparse.Namespace) -> int:
    parsed = parse_pdf(args.pdf)
    python_version = parsed.python_version or args.python or "3.8"
    cuda_version = parsed.cuda_version or args.cuda
    host_cuda_version = None
    host_driver_version = None
    if not cuda_version and not args.no_auto_cuda:
        host_cuda_version, host_driver_version = detect_host_cuda_version()
        if host_cuda_version:
            cuda_version = host_cuda_version
    web_hints = {"github_urls": [], "hf_urls": [], "web_urls": [], "docker_images": [], "dockerhub_urls": []}
    web_search_enabled = not args.no_web_search
    if args.web_search:
        web_search_enabled = True
    if web_search_enabled and not (parsed.github_urls or parsed.hf_urls or parsed.docker_images):
        query = parsed.title or _paper_id_from_path(args.pdf)
        web_hints = search_all(query, limit=args.web_limit)
        web_hints["web_urls"] = search_web(query, limit=args.web_limit)
        web_hints.setdefault("docker_images", [])
        web_hints.setdefault("dockerhub_urls", [])
        web_hints["github_urls"] += extract_github_urls(web_hints["web_urls"])
        web_hints["hf_urls"] += extract_hf_urls(web_hints["web_urls"])
        harvested = harvest_links_from_web_urls(web_hints["web_urls"])
        web_hints["github_urls"] += harvested.get("github_urls", [])
        web_hints["hf_urls"] += harvested.get("hf_urls", [])
        web_hints["dockerhub_urls"] += harvested.get("dockerhub_urls", [])
        web_hints["docker_images"] += harvested.get("docker_images", [])

    manual_github = [args.github] if args.github else []
    manual_hf = [args.hf] if args.hf else []
    manual_docker = [args.docker_image] if args.docker_image else []

    docker_images = manual_docker + list(parsed.docker_images) + web_hints["docker_images"]
    if not docker_images and web_search_enabled:
        docker_images = []

    base_image = docker_images[0] if docker_images else choose_base_image(cuda_version, python_version)

    requirements = _requirements_from_parsed(parsed.framework, parsed.libraries)
    pip_extra_index_url = None
    system_packages = [f"python{python_version}", "python3-pip", "build-essential"]
    for pkg in parsed.system_packages:
        if pkg not in system_packages:
            system_packages.append(pkg)

    github_urls = _clean_urls(manual_github + parsed.github_urls + web_hints["github_urls"])
    hf_urls = _clean_urls(manual_hf + parsed.hf_urls + web_hints["hf_urls"])

    repo_dockerfile = None
    repo_requirements = None
    repo_env_yml = None
    repo_req_source = None
    if github_urls:
        repo_data = scan_github_repo(github_urls[0])
        repo_dockerfile = repo_data.get("dockerfile")
        repo_requirements = repo_data.get("requirements")
        repo_env_yml = repo_data.get("environment_yml")
        repo_readme = repo_data.get("readme")
        repo_setup_py = repo_data.get("setup_py")
        repo_setup_cfg = repo_data.get("setup_cfg")
        if repo_dockerfile:
            m = re.search(r"^FROM\s+([A-Za-z0-9./:_-]+)", repo_dockerfile, re.M)
            if m:
                base_image = m.group(1)
        if not repo_requirements and repo_readme:
            repo_requirements = derive_requirements_from_readme(repo_readme, github_urls[0])
            if repo_requirements:
                repo_req_source = "readme"
        if not repo_requirements and repo_setup_cfg:
            repo_requirements = derive_requirements_from_setup_cfg(repo_setup_cfg)
            if repo_requirements:
                repo_req_source = "setup.cfg"
        if not repo_requirements and repo_setup_py:
            repo_requirements = derive_requirements_from_setup_py(repo_setup_py)
            if repo_requirements:
                repo_req_source = "setup.py"
        if repo_requirements and not repo_req_source:
            repo_req_source = "requirements.txt"

    hf_requirements = None
    hf_env_yml = None
    hf_req_source = None
    if hf_urls and not repo_requirements:
        hf_data = scan_hf_model(hf_urls[0])
        hf_requirements = hf_data.get("requirements")
        hf_env_yml = hf_data.get("environment_yml")
        if hf_requirements:
            hf_req_source = "huggingface"

    paper_id = _paper_id_from_path(args.pdf)
    workdir = args.out or os.path.join(os.getcwd(), "workspace", paper_id)
    os.makedirs(workdir, exist_ok=True)

    docker_tag = f"paper2env/{paper_id}:local"
    if args.tag:
        docker_tag = f"paper2env/{paper_id}:{args.tag}"
    if args.image_name:
        docker_tag = args.image_name if ":" in args.image_name else f"{args.image_name}:local"
    notes = []
    if docker_images:
        notes.append(f"base_image from docker image hint: {docker_images[0]}")
    else:
        notes.append(f"base_image inferred from python/cuda: {base_image}")
    if repo_dockerfile:
        notes.append("Dockerfile sourced from GitHub repo")
    if repo_requirements:
        src = repo_req_source or "github"
        notes.append(f"requirements sourced from GitHub ({src})")
    if hf_requirements:
        src = hf_req_source or "huggingface"
        notes.append(f"requirements sourced from HuggingFace ({src})")
    if repo_env_yml or hf_env_yml:
        notes.append("environment.yml sourced from external repo")
    if parsed.raw_text_excerpt:
        notes.append("paper text parsed for framework/python/cuda hints")
    if host_cuda_version:
        if host_driver_version:
            notes.append(f"host CUDA detected via nvidia-smi: {host_cuda_version} (driver {host_driver_version})")
        else:
            notes.append(f"host CUDA detected via nvidia-smi: {host_cuda_version}")
    elif not args.no_auto_cuda and not parsed.cuda_version and not args.cuda:
        notes.append("host CUDA auto-detect unavailable (nvidia-smi not found); falling back to CPU base unless paper hints exist")

    if parsed.framework == "pytorch" and not (repo_requirements or hf_requirements):
        chosen_cuda, extra_idx = pytorch_extra_index_url(cuda_version)
        if extra_idx:
            pip_extra_index_url = extra_idx
            notes.append(f"pytorch extra-index set for CUDA {chosen_cuda}")
        elif cuda_version:
            notes.append("pytorch CUDA wheel not selected; falling back to CPU wheels")
    repo_requirements = _normalize_requirements_text(repo_requirements)
    hf_requirements = _normalize_requirements_text(hf_requirements)

    if github_urls and "git" not in system_packages:
        system_packages.append("git")
    if (repo_requirements or hf_requirements) and "git+" in (repo_requirements or hf_requirements):
        if "git" not in system_packages:
            system_packages.append("git")

    cfg = RunConfig(
        paper_id=paper_id,
        workdir=workdir,
        base_image=base_image,
        python_version=python_version,
        requirements=requirements,
        system_packages=system_packages,
        source_hints={
            "docker_images": docker_images,
            "github_urls": github_urls,
            "hf_urls": hf_urls,
            "dockerhub_urls": parsed.dockerhub_urls + web_hints["dockerhub_urls"],
            "web_urls": web_hints["web_urls"],
        },
        dockerfile_text=repo_dockerfile,
        requirements_text=repo_requirements or hf_requirements,
        environment_yml_text=repo_env_yml or hf_env_yml,
        constraints_text=None,
        pip_extra_index_url=pip_extra_index_url,
        gpu_mode="none",
        gpu_indices=[],
        notes=notes,
        run_cmd=["python", "-c", "print('ok')"],
        docker_tag=docker_tag,
        container_name=None,
        keep_container=False,
        mount_host_dir=None,
        mount_container_dir=None,
        keep_alive=False,
        detach=False,
        verbose=args.verbose,
        requirements_is_conda=False,
        best_effort=False,
    )

    from .core.orchestrator import write_dockerfile, write_env_json, write_requirements

    write_requirements(cfg)
    write_env_json(cfg)
    write_dockerfile(cfg)

    print("Exported to:", workdir)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    info = get_platform_info()
    print("System:", f"{info['system']} {info['release']} ({info['machine']})")
    print("Python:", info["python"])

    ok, docker_msg = check_docker()
    print("Docker:", "OK" if ok else "NOT OK", f"({docker_msg})")

    cuda_version, driver_version = detect_host_cuda_version()
    if cuda_version:
        if driver_version:
            print("CUDA:", f"{cuda_version} (driver {driver_version})")
        else:
            print("CUDA:", cuda_version)
    else:
        print("CUDA: not detected (nvidia-smi not found)")

    gpus = list_gpus()
    if gpus:
        print("GPUs:")
        for idx, name in gpus:
            print(f"  {idx}: {name}")
    else:
        print("GPUs: none detected")
        if info["system"].lower() == "darwin":
            print("Hint: macOS에서는 NVIDIA GPU/driver를 사용할 수 없습니다.")
        else:
            print("Hint: GPU가 있어도 NVIDIA 드라이버가 없으면 nvidia-smi가 없습니다.")
            print("Hint: driver + nvidia-container-toolkit 설치 후 재확인하세요.")

    run_deep = args.deep
    if not run_deep and args.interactive and not gpus:
        run_deep = _prompt_yes_no("Run extra GPU diagnostics (lspci/lsmod/nvidia-container-cli)?", False)

    if run_deep:
        code, out = try_run_command(["lspci"])
        if code == 0 and out:
            found = any("nvidia" in line.lower() for line in out.splitlines())
            print("lspci: NVIDIA GPU detected" if found else "lspci: no NVIDIA GPU")
        elif code == 127:
            print("lspci: not available")
        else:
            print("lspci: failed")

        code, out = try_run_command(["lsmod"])
        if code == 0 and out:
            loaded = any(line.startswith("nvidia") for line in out.splitlines())
            print("lsmod: nvidia module loaded" if loaded else "lsmod: nvidia module not loaded")
        elif code == 127:
            print("lsmod: not available")
        else:
            print("lsmod: failed")

        code, out = try_run_command(["nvidia-container-cli", "info"])
        if code == 0 and out:
            print("nvidia-container-cli: OK")
        elif code == 127:
            print("nvidia-container-cli: not available")
        else:
            print("nvidia-container-cli: failed")
    return 0


def cmd_help(_: argparse.Namespace) -> int:
    print("paper2env quick help")
    print("  doctor:     paper2env doctor")
    print("  run:        paper2env run <paper.pdf>")
    print("  run auto:   paper2env run <paper.pdf> --yes")
    print("  run cpu:    paper2env run <paper.pdf> --gpu none")
    print("  run mount:  paper2env run <paper.pdf> --mount <host_dir>:/workspace")
    print("  export:     paper2env export <paper.pdf> --out <dir>")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paper2env", description="Paper PDF to reproducible Docker env")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="parse paper and build/run docker env")
    run.add_argument("pdf", help="paper pdf path")
    run.add_argument("--workdir", help="workspace dir")
    run.add_argument("--max-attempts", type=int, default=5)
    run.add_argument("--dry-run", action="store_true", help="only generate files")
    run.add_argument("--python", help="override python version")
    run.add_argument("--cuda", help="override cuda version")
    run.add_argument("--no-auto-cuda", action="store_true", help="disable host CUDA auto-detection")
    run.add_argument("--llm-model", help="use local ollama model for unknown errors")
    run.add_argument("--image-name", help="docker image name (optionally with :tag)")
    run.add_argument("--tag", help="docker image tag (used with default image name)")
    run.add_argument("--container-name", help="docker container name")
    run.add_argument("--rm", action="store_true", help="remove container after run")
    run.add_argument("--mount", help="mount host_dir:container_dir (default: workdir:/workspace)")
    run.add_argument("--no-mount", action="store_true", help="disable mounting local workspace")
    run.add_argument("--no-keep-alive", action="store_true", help="do not keep container running after run")
    run.add_argument("--best-effort", action="store_true", help="do not fail hard on unknown errors (record report and exit 0)")
    run.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=True,
        help="stream docker build/run logs (default on)",
    )
    run.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="disable streaming logs",
    )
    run.add_argument("--resume", action="store_true", help="reuse existing workspace artifacts")
    run.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=True,
        help="prompt for image/container names (default on)",
    )
    run.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="disable interactive prompts",
    )
    run.add_argument(
        "--yes",
        dest="interactive",
        action="store_false",
        help="non-interactive (assume defaults)",
    )
    run.add_argument("--gpu", choices=["auto", "none"], default="auto", help="GPU usage policy (default: auto)")
    run.add_argument("--gpu-index", help="comma-separated GPU indices to use")
    run.add_argument("--github", help="explicit github repo url to prioritize")
    run.add_argument("--hf", help="explicit huggingface model url to prioritize")
    run.add_argument("--docker-image", help="explicit docker image to use as base")
    run.add_argument("--web-search", action="store_true", help="fallback to web search for repo hints (default on)")
    run.add_argument("--no-web-search", action="store_true", help="disable web search fallback")
    run.add_argument("--web-limit", type=int, default=3, help="max web search results per source")
    run.set_defaults(func=cmd_run)

    exp = sub.add_parser("export", help="only generate Dockerfile/requirements/env.json")
    exp.add_argument("pdf", help="paper pdf path")
    exp.add_argument("--out", help="output dir")
    exp.add_argument("--python", help="override python version")
    exp.add_argument("--cuda", help="override cuda version")
    exp.add_argument("--no-auto-cuda", action="store_true", help="disable host CUDA auto-detection")
    exp.add_argument("--image-name", help="docker image name (optionally with :tag)")
    exp.add_argument("--tag", help="docker image tag (used with default image name)")
    exp.add_argument("--github", help="explicit github repo url to prioritize")
    exp.add_argument("--hf", help="explicit huggingface model url to prioritize")
    exp.add_argument("--docker-image", help="explicit docker image to use as base")
    exp.add_argument("--web-search", action="store_true", help="fallback to web search for repo hints (default on)")
    exp.add_argument("--no-web-search", action="store_true", help="disable web search fallback")
    exp.add_argument("--web-limit", type=int, default=3, help="max web search results per source")
    exp.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=True,
        help="stream docker build/run logs (default on)",
    )
    exp.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="disable streaming logs",
    )
    exp.set_defaults(func=cmd_export)

    doc = sub.add_parser("doctor", help="check local environment (docker/gpu/python)")
    doc.add_argument("--deep", action="store_true", help="run extra GPU diagnostics (lspci/lsmod)")
    doc.add_argument("--yes", dest="interactive", action="store_false", help="non-interactive (assume defaults)")
    doc.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=True,
        help="interactive prompts (default on)",
    )
    doc.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="disable interactive prompts",
    )
    doc.set_defaults(func=cmd_doctor)

    h = sub.add_parser("help", help="show quick usage tips")
    h.set_defaults(func=cmd_help)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
