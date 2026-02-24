# Paper2Env

논문 PDF 하나를 넣으면 실험 환경(Dockerfile/requirements/env.json)을 자동 생성하고, 필요 시 self-healing 루프로 build/run을 재시도하는 로컬 CLI입니다.

## 설치 (PyPI)

```bash
pipx install paper2env
```

또는

```bash
pip install paper2env
```

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

paper2env run paper.pdf
paper2env run paper.pdf --web-search
paper2env run paper.pdf --github <repo_url>
paper2env run paper.pdf --interactive
```

## CLI

```bash
paper2env run <paper.pdf> [--workdir <dir>] [--max-attempts 5] [--dry-run]
paper2env export <paper.pdf> --out <dir>
paper2env run <paper.pdf> --llm-model <ollama_model>
paper2env run <paper.pdf> --web-search
paper2env run <paper.pdf> --github <repo_url>
paper2env run <paper.pdf> --hf <model_url>
paper2env run <paper.pdf> --docker-image <image>
paper2env run <paper.pdf> --image-name <name[:tag]> --tag <tag>
paper2env run <paper.pdf> --container-name <name>
paper2env run <paper.pdf> --interactive
paper2env run <paper.pdf> --no-interactive
paper2env run <paper.pdf> --yes
paper2env run <paper.pdf> --rm
paper2env run <paper.pdf> --mount <host_dir>:/workspace
paper2env run <paper.pdf> --no-mount
paper2env run <paper.pdf> --no-keep-alive
paper2env run <paper.pdf> --best-effort
paper2env run <paper.pdf> --resume
paper2env run <paper.pdf> --verbose
paper2env run <paper.pdf> --no-verbose
paper2env doctor
paper2env doctor --deep
paper2env help
```

## 산출물

- `Dockerfile`
- `requirements.txt`
- `env.json`
- `logs/attempt_XX.log`
- `failure_report.json` (실패 시)

## 동작 규칙 (요약)

- 기본 입력은 `paper.pdf` 하나이며, 논문/레포/웹 힌트를 우선순위에 따라 처리합니다.
- GitHub/HF 힌트가 있으면 레포 파일을 우선 사용해 의존성을 추출합니다.
- Dockerfile은 논문/레포 힌트 기반으로 base image를 결정하고 필요한 시스템 패키지를 추가합니다.
- Self‑healing은 에러 로그를 규칙 기반으로 분류해 최소 수정으로 재시도합니다.

자세한 규칙은 `docs/rules.md` 참고.

## 주의

- 실제 Docker build/run을 수행하려면 로컬에 Docker가 설치되어 있어야 합니다.
- CUDA 기반 이미지는 호스트 드라이버/장비 제약에 따라 실패할 수 있습니다.
- `--llm-model`은 로컬 Ollama 설치가 필요합니다. (예: `ollama run llama3`)
- LLM 기능은 외부 `ollama` 바이너리를 호출하므로 `paper2env` 패키지 자체 용량은 거의 늘지 않습니다. 다만 로컬 모델 파일은 수백 MB~수 GB까지 커질 수 있습니다.
- `--web-search`는 인터넷 연결이 필요하며, 논문 내 링크가 없을 때 GitHub/HF/웹 힌트를 찾습니다.
- `--github`/`--hf`/`--docker-image`는 사용자가 직접 우선 참조를 지정하는 옵션입니다.
- 기본 실행은 대화형이며, Docker 이미지 태그/컨테이너 이름과 GPU 사용 여부 등을 프롬프트로 입력받습니다. (엔터=기본값)
- 기본적으로 컨테이너는 유지되며, 실행 후 삭제하려면 `--rm`을 사용합니다.
- 기본적으로 로컬 `workspace/`를 컨테이너 `/workspace`에 마운트합니다. 끄려면 `--no-mount`.
- 기본적으로 컨테이너를 백그라운드에서 계속 실행합니다. 끄려면 `--no-keep-alive`.
- 기본적으로 항상 새로 생성합니다. 기존 `workspace/`를 재사용하려면 `--resume`.
- 자동 진행은 `--yes` 또는 `--no-interactive`로 설정합니다.
- `--best-effort`는 알 수 없는 에러에서 실패를 중단하지 않고 `failure_report.json`만 기록한 뒤 종료 코드를 0으로 반환합니다.

## Ollama 사용 순서

1. Ollama 실행
2. 모델 받기
3. 실행할 때 `--llm-model` 옵션 넣기

예시:

```bash
ollama serve
ollama pull llama3
paper2env run papers/SimCLR.pdf --llm-model llama3
```
