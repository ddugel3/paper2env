from __future__ import annotations

import re
import shutil
import subprocess
from typing import Dict, List, Optional

from .models import ParsedEnv


_FRAMEWORK_PATTERNS = [
    ("pytorch", re.compile(r"pytorch", re.I)),
    ("tensorflow", re.compile(r"tensorflow", re.I)),
    ("jax", re.compile(r"\bjax\b", re.I)),
]

_LIBS = [
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "jax",
    "numpy",
    "opencv",
    "scipy",
    "pandas",
    "scikit-learn",
]


def _extract_text_with_pypdf(pdf_path: str, max_pages: int) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""

    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def _extract_text_with_pdftotext(pdf_path: str, max_pages: int) -> str:
    if not shutil.which("pdftotext"):
        return ""
    cmd = ["pdftotext", "-f", "1", "-l", str(max_pages), pdf_path, "-"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return ""
    return out


def _extract_text(pdf_path: str, max_pages: int = 12) -> str:
    text = _extract_text_with_pypdf(pdf_path, max_pages)
    if text.strip():
        return text
    text = _extract_text_with_pdftotext(pdf_path, max_pages)
    return text


def _find_system_packages(text: str) -> Dict[str, bool]:
    flags = {
        "needs_build_essential": False,
    }

    if re.search(r"\bC\+\+\b|C\+\+\s+code|C\s+code|C\+\+\s+implementation", text, re.I):
        flags["needs_build_essential"] = True
    if re.search(r"code\.google\.com/p/word2vec", text, re.I):
        flags["needs_build_essential"] = True

    return flags


def _guess_title(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:30]:
        if re.search(r"arxiv|\babstract\b|google inc|university|@|http", ln, re.I):
            continue
        if len(ln.split()) >= 4 and len(ln) <= 120:
            return ln
    return None


def _find_framework(text: str) -> Optional[str]:
    for name, pat in _FRAMEWORK_PATTERNS:
        if pat.search(text):
            return name
    return None


def _find_python_version(text: str) -> Optional[str]:
    m = re.search(r"Python\s*(\d+\.\d+)", text, re.I)
    if m:
        return m.group(1)
    return None


def _find_cuda_version(text: str) -> Optional[str]:
    m = re.search(r"CUDA\s*(\d+\.\d+)", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"cu(\d{2,3})", text, re.I)
    if m:
        val = m.group(1)
        if len(val) == 2:
            return f"{val[0]}.{val[1]}"
        if len(val) == 3:
            return f"{val[0:2]}.{val[2]}"
    return None


def _find_lib_versions(text: str) -> Dict[str, Optional[str]]:
    libs: Dict[str, Optional[str]] = {}
    for lib in _LIBS:
        m = re.search(rf"{re.escape(lib)}\s*(?:==|>=|<=|=|v)?\s*(\d+\.\d+(?:\.\d+)?)", text, re.I)
        if m:
            libs[lib] = m.group(1)
    return libs


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


def _find_urls(text: str) -> Dict[str, List[str]]:
    github = re.findall(r"https?://github\.com/[^\s\)\]]+", text, re.I)
    hf = re.findall(r"https?://huggingface\.co/[^\s\)\]]+", text, re.I)
    dockerhub = re.findall(r"https?://hub\.docker\.com/r/[^\s\)\]]+", text, re.I)
    return {
        "github": _unique(github),
        "hf": _unique(hf),
        "dockerhub": _unique(dockerhub),
    }


def _find_docker_images(text: str) -> List[str]:
    images: List[str] = []
    from_matches = re.findall(r"\bFROM\s+([A-Za-z0-9./:_-]+)", text)
    images.extend(from_matches)

    common_prefixes = ("nvidia/cuda", "pytorch/pytorch", "tensorflow/tensorflow", "nvcr.io", "ghcr.io")
    generic = re.findall(r"\b([A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?)\b", text)
    for cand in generic:
        if ":" in cand or cand.startswith(common_prefixes):
            images.append(cand)

    return _unique(images)


def parse_pdf(pdf_path: str) -> ParsedEnv:
    text = _extract_text(pdf_path)
    title = _guess_title(text)
    framework = _find_framework(text)
    python_version = _find_python_version(text)
    cuda_version = _find_cuda_version(text)
    libs = _find_lib_versions(text)
    sys_flags = _find_system_packages(text)
    urls = _find_urls(text)
    docker_images = _find_docker_images(text)
    system_packages = []
    if sys_flags.get("needs_build_essential"):
        system_packages.append("build-essential")

    excerpt = text[:2000]
    return ParsedEnv(
        title=title,
        framework=framework,
        python_version=python_version,
        cuda_version=cuda_version,
        libraries=libs,
        system_packages=system_packages,
        docker_images=docker_images,
        github_urls=urls["github"],
        hf_urls=urls["hf"],
        dockerhub_urls=urls["dockerhub"],
        raw_text_excerpt=excerpt,
    )
