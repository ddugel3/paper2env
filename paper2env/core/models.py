from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class ParsedEnv:
    title: Optional[str]
    framework: Optional[str]
    python_version: Optional[str]
    cuda_version: Optional[str]
    libraries: Dict[str, Optional[str]]
    system_packages: List[str]
    docker_images: List[str]
    github_urls: List[str]
    hf_urls: List[str]
    dockerhub_urls: List[str]
    raw_text_excerpt: str


@dataclass
class RunConfig:
    paper_id: str
    workdir: str
    base_image: str
    python_version: str
    requirements: Dict[str, Optional[str]]
    system_packages: List[str]
    source_hints: Dict[str, List[str]]
    dockerfile_text: Optional[str]
    requirements_text: Optional[str]
    environment_yml_text: Optional[str]
    constraints_text: Optional[str]
    pip_extra_index_url: Optional[str]
    gpu_mode: str
    gpu_indices: List[int]
    notes: List[str]
    run_cmd: List[str]
    docker_tag: str
    container_name: Optional[str]
    keep_container: bool
    mount_host_dir: Optional[str]
    mount_container_dir: Optional[str]
    keep_alive: bool
    detach: bool
    verbose: bool
    requirements_is_conda: bool
    best_effort: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttemptResult:
    success: bool
    exit_code: int
    logs: str
    error_type: str
    patches_applied: List[str]
