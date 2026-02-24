import re

ERROR_RULES = [
    ("dockerfile_parse_error", re.compile(r"dockerfile parse error.*unknown instruction", re.I)),
    ("docker_permission", re.compile(r"permission denied while trying to connect to the docker API", re.I)),
    ("tzdata_prompt", re.compile(r"Please select the geographic area in which you live", re.I)),
    ("cuda_driver_insufficient", re.compile(r"CUDA driver version is insufficient", re.I)),
    ("cuda_missing_lib", re.compile(r"libcudart\.so|libcuda\.so|cannot open shared object file", re.I)),
    ("cuda_no_kernel_image", re.compile(r"no kernel image is available", re.I)),
    ("torch_cuda_mismatch", re.compile(r"compiled with different CUDA versions|torchvision.*CUDA", re.I)),
    ("pip_dependency_conflict", re.compile(r"ResolutionImpossible|conflicting dependencies", re.I)),
    ("no_matching_distribution", re.compile(r"No matching distribution found for|Could not find a version that satisfies the requirement", re.I)),
    ("python_too_old_numpy2", re.compile(r"numpy<3\.0\.0,>=2\.0\.0.*No matching distribution", re.I | re.S)),
    ("missing_sqlite_dev", re.compile(r"fatal error: sqlite3\.h: No such file or directory", re.I)),
    ("python_requires", re.compile(r"requires-python|Python version .* not supported|requires a different Python", re.I)),
    ("module_not_found", re.compile(r"No module named|ModuleNotFoundError", re.I)),
    ("missing_build_tools", re.compile(r"gcc: not found|g\+\+: not found|command not found: make|make: not found|ninja: not found|cmake: not found|pkg-config: not found", re.I)),
    ("missing_python_dev", re.compile(r"Python\.h: No such file or directory", re.I)),
    ("missing_gl_libs", re.compile(r"libGL\.so|libglib-2\.0\.so|libSM\.so|libXrender\.so|libXext\.so", re.I)),
    ("missing_cuda_nvcc", re.compile(r"nvcc: not found|CUDA_HOME.*not set|No CUDA runtime is found", re.I)),
    ("cuda_home_missing", re.compile(r"CUDA_HOME environment variable is not set", re.I)),
    ("torchvision_requires_torch", re.compile(r"torchvision.*requires.*torch|torchvision.*but torch==", re.I)),
    ("apt_repo_key", re.compile(r"NO_PUBKEY|The following signatures couldn't be verified|Release file is not valid", re.I)),
    ("apt_fetch_failed", re.compile(r"Failed to fetch|Temporary failure resolving|Could not resolve", re.I)),
    ("pip_tls", re.compile(r"CERTIFICATE_VERIFY_FAILED|TLSV1_ALERT|SSL: CERTIFICATE_VERIFY_FAILED", re.I)),
    ("git_clone_failed", re.compile(r"git clone .*exit code: 128|fatal: repository .* not found|fatal: unable to access", re.I)),
    ("rust_compiler_missing", re.compile(r"Rust compiler|cargo: not found|rustc: not found", re.I)),
    ("no_space_left", re.compile(r"No space left on device", re.I)),
    ("process_killed", re.compile(r"Kill(ed)? process|Killed\\s+\\d+|exit code 137", re.I)),
    ("glibcxx_missing", re.compile(r"GLIBCXX_\\d+\\.\\d+\\.\\d+ not found", re.I)),
]


def classify_error(logs: str) -> str:
    for name, pattern in ERROR_RULES:
        if pattern.search(logs):
            return name
    return "unknown"
