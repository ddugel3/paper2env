from __future__ import annotations

import subprocess
from typing import List, Optional, Tuple


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = p.communicate()
    return p.returncode, out


def run_cmd_stream(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []
    if p.stdout:
        for line in p.stdout:
            print(line.rstrip())
            lines.append(line)
    p.wait()
    return p.returncode, "".join(lines)


def docker_build(workdir: str, tag: str, no_cache: bool = False, verbose: bool = False) -> Tuple[int, str]:
    cmd = ["docker", "build", "-t", tag]
    if no_cache:
        cmd.append("--no-cache")
    cmd.append(".")
    if verbose:
        return run_cmd_stream(cmd, cwd=workdir)
    return run_cmd(cmd, cwd=workdir)


def docker_run(
    tag: str,
    run_cmd_list: List[str],
    container_name: Optional[str] = None,
    gpu_indices: Optional[List[int]] = None,
    keep_container: bool = True,
    mount_host_dir: Optional[str] = None,
    mount_container_dir: Optional[str] = None,
    detach: bool = False,
) -> Tuple[int, str]:
    cmd = ["docker", "run"]
    if detach:
        cmd.append("-d")
    if not keep_container:
        cmd.append("--rm")
    if container_name:
        cmd += ["--name", container_name]
    if mount_host_dir and mount_container_dir:
        cmd += ["-v", f"{mount_host_dir}:{mount_container_dir}"]
    if gpu_indices is not None:
        if len(gpu_indices) == 0:
            pass
        else:
            indices = ",".join(str(i) for i in gpu_indices)
            cmd += ["--gpus", f"device={indices}"]
    cmd += [tag] + run_cmd_list
    return run_cmd(cmd, cwd=None)
