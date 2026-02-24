from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple


def _get_json(url: str, headers: Dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except Exception:
        return {}


def _get_text(url: str, headers: Dict[str, str] | None = None) -> str:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        # Fallback to curl if urllib fails (e.g., SSL/proxy issues)
        try:
            out = subprocess.check_output(
                ["curl", "-fsSL", url],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
            return out
        except Exception:
            return ""


def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in seq:
        val = item.strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def search_github(query: str, limit: int = 3) -> List[str]:
    if not query:
        return []
    q = urllib.parse.quote_plus(query)
    url = f"https://api.github.com/search/repositories?q={q}&per_page={limit}"
    data = _get_json(url, headers={"User-Agent": "paper2env"})
    items = data.get("items") or []
    urls = []
    for it in items:
        u = it.get("html_url")
        if u:
            urls.append(u)
    return urls


def search_hf(query: str, limit: int = 3) -> List[str]:
    if not query:
        return []
    q = urllib.parse.quote_plus(query)
    url = f"https://huggingface.co/api/models?search={q}&limit={limit}"
    data = _get_json(url, headers={"User-Agent": "paper2env"})
    urls = []
    if isinstance(data, list):
        for it in data:
            mid = it.get("modelId")
            if mid:
                urls.append(f"https://huggingface.co/{mid}")
    return urls


def search_web(query: str, limit: int = 5) -> List[str]:
    if not query:
        return []
    q = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={q}"
    html = _get_text(url, headers={"User-Agent": "paper2env"})
    if not html:
        return []
    links = re.findall(r'href="(https?://[^"]+)"', html, re.I)
    out: List[str] = []
    for link in links:
        if "duckduckgo.com" in link:
            continue
        out.append(link)
        if len(out) >= limit:
            break
    return out


def _extract_links_from_html(html: str) -> Dict[str, List[str]]:
    github = re.findall(r"https?://github\.com/[^\s\"'<>\)\]]+", html, re.I)
    hf = re.findall(r"https?://huggingface\.co/[^\s\"'<>\)\]]+", html, re.I)
    dockerhub = re.findall(r"https?://hub\.docker\.com/r/[^\s\"'<>\)\]]+", html, re.I)
    return {
        "github": _unique(github),
        "hf": _unique(hf),
        "dockerhub": _unique(dockerhub),
    }


def _extract_docker_images_from_html(html: str) -> List[str]:
    images = []
    for pat in [
        r"\bFROM\s+([A-Za-z0-9./:_-]+)",
        r"\bdocker\s+pull\s+([A-Za-z0-9./:_-]+)",
        r"\bimage:\s*([A-Za-z0-9./:_-]+)",
    ]:
        images.extend(re.findall(pat, html, re.I))
    return _unique(images)


def harvest_links_from_web_urls(urls: List[str], per_url_limit: int = 2) -> Dict[str, List[str]]:
    github: List[str] = []
    hf: List[str] = []
    dockerhub: List[str] = []
    docker_images: List[str] = []

    for url in urls:
        html = _get_text(url, headers={"User-Agent": "paper2env"})
        if not html:
            continue
        links = _extract_links_from_html(html)
        github += links["github"][:per_url_limit]
        hf += links["hf"][:per_url_limit]
        dockerhub += links["dockerhub"][:per_url_limit]
        docker_images += _extract_docker_images_from_html(html)[:per_url_limit]

    return {
        "github_urls": _unique(github),
        "hf_urls": _unique(hf),
        "dockerhub_urls": _unique(dockerhub),
        "docker_images": _unique(docker_images),
    }


def _parse_github_owner_repo(url: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"github\.com/([^/\s]+)/([^/\s\)#]+)", url)
    if not m:
        return None
    owner = m.group(1)
    repo = m.group(2).replace(".git", "")
    return owner, repo


def _get_github_contents(owner: str, repo: str, path: str = "") -> List[dict]:
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}".rstrip("/")
    data = _get_json(api, headers={"User-Agent": "paper2env"})
    if isinstance(data, list):
        return data
    return []


def _fetch_github_file(download_url: str) -> str:
    if not download_url:
        return ""
    return _get_text(download_url, headers={"User-Agent": "paper2env"})


def _get_github_repo_meta(owner: str, repo: str) -> Dict[str, str]:
    api = f"https://api.github.com/repos/{owner}/{repo}"
    data = _get_json(api, headers={"User-Agent": "paper2env"})
    default_branch = data.get("default_branch") or "main"
    return {"default_branch": default_branch}


def _get_github_tree(owner: str, repo: str, branch: str) -> List[dict]:
    api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    data = _get_json(api, headers={"User-Agent": "paper2env"})
    tree = data.get("tree")
    if isinstance(tree, list):
        return tree
    return []


def _raw_github_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def scan_github_repo(repo_url: str) -> Dict[str, Optional[str]]:
    parsed = _parse_github_owner_repo(repo_url)
    if not parsed:
        return {
            "dockerfile": None,
            "requirements": None,
            "environment_yml": None,
        }
    owner, repo = parsed
    meta = _get_github_repo_meta(owner, repo)
    branch = meta["default_branch"]

    tree = _get_github_tree(owner, repo, branch)
    docker_path = None
    req_path = None
    env_path = None
    readme_path = None
    setup_py_path = None
    setup_cfg_path = None
    if tree:
        for item in tree:
            path = item.get("path") or ""
            if not docker_path and path.endswith("Dockerfile"):
                docker_path = path
            if not req_path and path.endswith("requirements.txt"):
                req_path = path
            if not env_path and (path.endswith("environment.yml") or path.endswith("environment.yaml")):
                env_path = path
            if not readme_path and re.search(r"^readme(\.md)?$", path.split("/")[-1], re.I):
                readme_path = path
            if not setup_py_path and path.endswith("setup.py"):
                setup_py_path = path
            if not setup_cfg_path and path.endswith("setup.cfg"):
                setup_cfg_path = path
            if docker_path and req_path and env_path:
                break

    dockerfile = _fetch_github_file(_raw_github_url(owner, repo, branch, docker_path)) if docker_path else ""
    requirements = _fetch_github_file(_raw_github_url(owner, repo, branch, req_path)) if req_path else ""
    environment_yml = _fetch_github_file(_raw_github_url(owner, repo, branch, env_path)) if env_path else ""
    readme = _fetch_github_file(_raw_github_url(owner, repo, branch, readme_path)) if readme_path else ""
    setup_py = _fetch_github_file(_raw_github_url(owner, repo, branch, setup_py_path)) if setup_py_path else ""
    setup_cfg = _fetch_github_file(_raw_github_url(owner, repo, branch, setup_cfg_path)) if setup_cfg_path else ""

    if not (dockerfile or requirements or environment_yml):
        # Fallback: try common paths on main/master without API tree
        for br in [branch, "main", "master"]:
            if not requirements:
                requirements = _fetch_github_file(_raw_github_url(owner, repo, br, "requirements.txt"))
            if not dockerfile:
                dockerfile = _fetch_github_file(_raw_github_url(owner, repo, br, "Dockerfile"))
            if not environment_yml:
                environment_yml = _fetch_github_file(_raw_github_url(owner, repo, br, "environment.yml"))
            if not readme:
                readme = _fetch_github_file(_raw_github_url(owner, repo, br, "README.md"))
            if not readme:
                readme = _fetch_github_file(_raw_github_url(owner, repo, br, "README"))
            if not setup_py:
                setup_py = _fetch_github_file(_raw_github_url(owner, repo, br, "setup.py"))
            if not setup_cfg:
                setup_cfg = _fetch_github_file(_raw_github_url(owner, repo, br, "setup.cfg"))
            if requirements or dockerfile or environment_yml:
                break

    return {
        "dockerfile": dockerfile or None,
        "requirements": requirements or None,
        "environment_yml": environment_yml or None,
        "readme": readme or None,
        "setup_py": setup_py or None,
        "setup_cfg": setup_cfg or None,
    }


def derive_requirements_from_readme(readme: str, repo_url: str) -> Optional[str]:
    if not readme:
        return None

    reqs: List[str] = []

    # capture pip install lines
    for m in re.finditer(r"pip\s+install\s+([^\n`]+)", readme, re.I):
        chunk = m.group(1).strip()
        parts = [p for p in re.split(r"\s+", chunk) if p]
        for p in parts:
            if p.startswith("-"):
                continue
            reqs.append(p)

    # if readme mentions pytorch/torchvision versions
    pt = re.search(r"pytorch\s*>=\s*([\d.]+)", readme, re.I)
    tv = re.search(r"torchvision\s*>=\s*([\d.]+)", readme, re.I)
    if pt:
        reqs.append(f"torch>={pt.group(1)}")
    if tv:
        reqs.append(f"torchvision>={tv.group(1)}")

    # If repo install is suggested but not captured, add VCS line
    if "git+https://github.com/" in readme.lower():
        if not any(s.startswith("git+https://github.com/") for s in reqs):
            reqs.append(f"git+{repo_url}")

    # de-duplicate while preserving order
    out: List[str] = []
    seen = set()
    for r in reqs:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)

    if not out:
        return None
    return "\n".join(out)


def derive_requirements_from_setup_cfg(setup_cfg: str) -> Optional[str]:
    if not setup_cfg:
        return None
    m = re.search(r"(?s)\[options\].*?install_requires\s*=\s*(.+?)(?:\n\[|\Z)", setup_cfg)
    if not m:
        return None
    block = m.group(1)
    reqs = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return "\n".join(_unique(reqs)) if reqs else None


def derive_requirements_from_setup_py(setup_py: str) -> Optional[str]:
    if not setup_py:
        return None
    # Very simple heuristic: find install_requires=[...]
    m = re.search(r"install_requires\s*=\s*\[(.*?)\]", setup_py, re.S)
    items: List[str] = []
    if m:
        inside = m.group(1)
        items += re.findall(r"['\"]([^'\"]+)['\"]", inside)

    # If install_requires is empty, try extras_require "all"
    if not items:
        m2 = re.search(r"extras_require\s*=\s*\{(.*?)\}\s*,", setup_py, re.S)
        if m2:
            block = m2.group(1)
            m_all = re.search(r"['\"]all['\"]\s*:\s*\[(.*?)\]", block, re.S)
            if m_all:
                inside_all = m_all.group(1)
                items += re.findall(r"['\"]([^'\"]+)['\"]", inside_all)

    return "\n".join(_unique(items)) if items else None


def _normalize_hf_id(url: str) -> Optional[str]:
    m = re.search(r"huggingface\.co/([^/\s]+)/([^/\s\)#]+)", url)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


def scan_hf_model(url: str) -> Dict[str, Optional[str]]:
    model_id = _normalize_hf_id(url)
    if not model_id:
        return {"requirements": None, "environment_yml": None}

    def fetch_on_branch(branch: str) -> Dict[str, str]:
        req_url = f"https://huggingface.co/{model_id}/raw/{branch}/requirements.txt"
        env_url = f"https://huggingface.co/{model_id}/raw/{branch}/environment.yml"
        return {
            "requirements": _get_text(req_url, headers={"User-Agent": "paper2env"}),
            "environment_yml": _get_text(env_url, headers={"User-Agent": "paper2env"}),
        }

    data = fetch_on_branch("main")
    if not data["requirements"] and not data["environment_yml"]:
        data = fetch_on_branch("master")
    requirements = data["requirements"]
    environment_yml = data["environment_yml"]

    return {
        "requirements": requirements or None,
        "environment_yml": environment_yml or None,
    }


def search_all(query: str, limit: int = 3) -> Dict[str, List[str]]:
    return {
        "github_urls": search_github(query, limit=limit),
        "hf_urls": search_hf(query, limit=limit),
    }


def extract_github_urls(urls: List[str]) -> List[str]:
    out = []
    for u in urls:
        if "github.com/" in u:
            out.append(u)
    return out


def extract_hf_urls(urls: List[str]) -> List[str]:
    out = []
    for u in urls:
        if "huggingface.co/" in u:
            out.append(u)
    return out
