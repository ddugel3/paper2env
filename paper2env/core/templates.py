from __future__ import annotations

from .models import RunConfig


def choose_base_image(cuda_version: str | None, python_version: str) -> str:
    if not cuda_version:
        if python_version.startswith("3.10"):
            return "ubuntu:22.04"
        return "ubuntu:20.04"

    ubuntu = "20.04"
    if python_version.startswith("3.10"):
        ubuntu = "22.04"
    return f"nvidia/cuda:{cuda_version}.0-cudnn8-runtime-ubuntu{ubuntu}"


def render_dockerfile(cfg: RunConfig) -> str:
    if not cfg.system_packages:
        install_line = ""
        pkg_line = ""
    else:
        pkg_line = " \\\n    ".join(cfg.system_packages)
        install_line = f"RUN apt-get update && apt-get install -y \\\n    {pkg_line}"

    python_link = ""
    if any(pkg.startswith("python") for pkg in cfg.system_packages):
        py = cfg.python_version
        python_link = (
            f" && \\\n    ln -sf /usr/bin/python{py} /usr/bin/python && \\\n"
            "    python -m pip install --upgrade pip"
        )

    if cfg.requirements_is_conda:
        pip_line = "RUN echo 'skip conda-style requirements.txt (use conda env or repo requirements)'"
    else:
        pip_line = "RUN python -m pip install -r /workspace/requirements.txt"
        if cfg.constraints_text:
            pip_line = "RUN python -m pip install -r /workspace/requirements.txt -c /workspace/constraints.txt"
        pip_line += "\nRUN if [ -f /workspace/requirements.post.txt ]; then PIP_NO_BUILD_ISOLATION=1 python -m pip install -r /workspace/requirements.post.txt; fi"

    maybe_clone = ""
    maybe_build = ""
    maybe_repo_requirements = ""
    if cfg.source_hints.get("github_urls"):
        repo = cfg.source_hints["github_urls"][0]
        if repo:
            maybe_clone = f"\nRUN git clone {repo} /workspace/repo"
            maybe_repo_requirements = (
                "\nRUN bash -lc \"if [ -f /workspace/repo/requirements.txt ]; then "
                "awk 'BEGIN{OFS=\"\"} "
                "{line=$0; sub(/\\r$/,\"\",line); gsub(/^ +| +$/, \"\", line); "
                "if(line==\"\" || line==\".\" || line ~ /^#/) next; "
                "if(line ~ /^(_libgcc_mutex|_openmp_mutex)/) next; "
                "if(line ~ /^(name:|prefix:|channels:|dependencies:|platform:)/) next; "
                "if(line ~ /^[A-Za-z0-9_.-]+=\\\\d/ && line !~ /==/) next; "
                "if(line ~ /^(git\\+|https?:\\/\\/|-e )/) {print line; next;} "
                "if(line ~ /^(-f |--find-links|--extra-index-url|--index-url|--trusted-host|-i )/) {print line; next;} "
                "if(line ~ /(==|>=|<=|~=|!=)/) {print line; next;} "
                "if(line ~ /^[A-Za-z0-9_.-]+$/) {print line; next;} "
                "if(line ~ /=/ && line !~ /==/) next; "
                "}' /workspace/repo/requirements.txt > /tmp/requirements.pip.txt; "
                "if [ -s /tmp/requirements.pip.txt ]; then python -m pip install -r /tmp/requirements.pip.txt; fi; "
                "fi\""
            )
            if "word2vec" in repo.lower():
                maybe_build = "\nRUN cd /workspace/repo && make"

    cuda_env = ""
    if "nvidia/cuda" in cfg.base_image:
        cuda_env = "ENV CUDA_HOME=/usr/local/cuda\\nENV PATH=/usr/local/cuda/bin:$PATH\\nENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\\n"

    return f"""FROM {cfg.base_image}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
{cuda_env}

{install_line}{python_link}

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
COPY requirements.post.txt /workspace/requirements.post.txt
{pip_line}
{maybe_clone}{maybe_repo_requirements}{maybe_build}
"""
