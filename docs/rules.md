# 동작 규칙 상세

이 문서는 `paper2env`가 입력 논문을 처리해 실험 환경을 생성하고, 실패 시 self‑healing을 수행하는 전체 규칙을 자세히 정리한 것입니다.

## 1) 입력과 기본 우선순위

- 기본 입력은 `paper.pdf` 하나입니다.
- 추가 입력이 주어지면 우선순위는 다음과 같습니다.
  - 사용자 지정 입력: `--github`, `--hf`, `--docker-image`
  - 논문 내부 힌트: GitHub/HF/Docker 링크, 버전 문자열, 프레임워크 명시
  - 웹 검색 힌트: GitHub → HF → 일반 웹 순 (기본 ON, `--no-web-search`로 끔)
- `--cuda`가 없고 논문에서 CUDA 버전을 찾지 못하면, 호스트에서 `nvidia-smi`로 CUDA 버전을 자동 감지합니다. (`--no-auto-cuda`로 끔)
- `nvidia-smi`가 없거나 실행 실패 시에는 자동 감지를 건너뛰고, 논문 힌트/사용자 입력이 없으면 CPU 기반으로 진행합니다.
- `--gpu` 기본값은 `auto`이며, `--interactive`일 때는 GPU 사용 여부를 먼저 물어봅니다. GPU가 여러 개인 경우 `--gpu-index`로 선택하거나 프롬프트에서 선택할 수 있습니다.
- 기본 실행은 대화형이며, 질문 없이 전부 자동으로 진행하려면 `--yes` 또는 `--no-interactive`를 사용합니다.
  - 기본 대화형 모드에서는 GPU 사용 여부와 Docker 이미지 태그/컨테이너 이름 등을 반드시 확인합니다.
  - 기본적으로 로컬 `workspace/`를 컨테이너 `/workspace`에 마운트합니다. 끄려면 `--no-mount`.

## GPU 사용 방식 (상세)

### 기본 동작

- `--gpu`의 기본값은 `auto`입니다.
- GPU가 감지되면 가능한 한 GPU로 실행합니다.
- `--interactive`일 때는 GPU 사용 여부를 먼저 질문합니다.

### 사용 예시

- 기본 실행: `paper2env run paper.pdf`
- 대화형 실행: `paper2env run paper.pdf --interactive`
- 완전 자동: `paper2env run paper.pdf --yes`
- GPU 비활성화: `paper2env run paper.pdf --gpu none`
- 특정 GPU만 사용: `paper2env run paper.pdf --gpu-index 0`
- 여러 GPU 선택: `paper2env run paper.pdf --gpu-index 0,1`

### GPU 감지 실패 시

- `nvidia-smi`가 없거나 실행 실패 시 GPU를 감지할 수 없습니다.
- 이 경우 GPU 사용은 건너뛰고 CPU 기반으로 진행합니다.
- 필요하면 `--gpu-index`로 명시적으로 지정할 수 있지만, 호스트에서 GPU가 감지되지 않으면 실행 시 GPU가 연결되지 않습니다.

## 2) 논문에서 힌트 추출

논문 본문에서 다음 정보를 가능한 한 식별합니다.

- 프레임워크: PyTorch/TensorFlow/JAX 등
- Python 버전: 3.8/3.9/3.10 등
- CUDA 버전: 11.1/11.7/12.1 등
- 주요 라이브러리: `transformers`, `diffusers`, `opencv`, `torchvision` 등
- 링크: GitHub/HF/DockerHub/프로젝트 웹사이트

힌트 간 충돌이 있을 경우는 아래 원칙을 따릅니다.

- 논문에 명시된 버전이 가장 높은 우선순위를 가집니다.
- 코드/레포에서 직접 확인한 의존성이 2순위입니다.
- 웹 검색 결과는 마지막 보정에만 사용합니다.

## 3) GitHub/HF/웹 힌트 처리

### GitHub 레포 힌트가 있는 경우

- 의존성 추출 우선순위:
  - `Dockerfile` → `requirements.txt` → `setup.py` → `setup.cfg` → `README`
- `Dockerfile`이 있으면 base image와 시스템 패키지를 그대로 존중합니다.
- `requirements.txt`가 있으면 가능한 한 그대로 유지하되, self‑healing 단계에서 최소 수정합니다.

### HF(Hugging Face) 힌트가 있는 경우

- `requirements.txt`, `environment.yml`을 우선 사용합니다.
- HF 모델 카드에 CUDA/프레임워크 표기가 있으면 base image 선택에 반영합니다.

### 웹 검색 힌트가 있는 경우

- 논문 내 링크가 전혀 없을 때만 보강용으로 활용합니다.
- 검색 순서: GitHub → HF → 일반 웹
- 유사 프로젝트/코드가 확인되면 그 레포의 의존성을 2차 힌트로 반영합니다.

## 4) Dockerfile 생성 규칙

- base image 선택 우선순위:
  - 논문/레포에서 명시된 이미지
  - CUDA 버전 힌트가 있는 경우 `nvidia/cuda:<ver>-cudnn8-runtime-ubuntu20.04`
  - GPU 힌트가 없으면 `python:<ver>-slim` 또는 안정적인 LTS 기반
- GitHub URL이 있으면 `git` 설치 후 `git clone`을 수행합니다.
- 레포에 `requirements.txt`가 있으면 `pip install -r /workspace/repo/requirements.txt`를 추가로 수행합니다.
- `word2vec` 계열은 빌드가 필요하므로 자동으로 `make`를 수행합니다.
- 최소 실행 경로를 만들기 위해 `WORKDIR`, `COPY`, `RUN` 순서를 표준화합니다.

## 5) requirements/env.json 생성 규칙

- `requirements.txt`는 레포 파일을 최대한 유지합니다.
- PyTorch 프레임워크이고 외부 레포 requirements가 없을 때, 감지된 CUDA 버전에 맞는 PyTorch wheel 인덱스를 자동으로 추가합니다.
- `env.json`은 다음 정보를 포함합니다.
  - python/cuda 버전
  - base image 정보
  - 추출된 주요 라이브러리 목록
  - 사용된 힌트 출처 (paper/github/hf/web)

## 6) Self‑healing 동작 규칙

Self‑healing은 실패 로그를 규칙 기반으로 분류해 최소 수정으로 재시도합니다.

### 에러 로그 분류

- `ModuleNotFoundError`/`ImportError` → 누락된 패키지 설치
- `No matching distribution found` → python 버전 상향/하향 조정 또는 패키지 버전 핀
- `CUDA not found`/`libcudart.so` → CUDA base image로 전환
- `gcc`/`g++`/`cmake` 없음 → 빌드 툴 설치
- `OSError: libGL.so` 등 시스템 라이브러리 누락 → apt 패키지 추가
- `Python.h` 누락 → `python3-dev` 추가
- `nvcc` 없음 → CUDA `devel` 이미지로 전환
- `torch`/`torchvision` 버전 불일치 → 호환 버전으로 정렬
  - 호환 매핑은 `paper2env/core/torch_compat.json`에서 관리합니다.
- conda export 형식 `requirements.txt` 감지 시 pip 설치를 건너뜁니다. (레포 `requirements.txt` 또는 conda 환경 사용)
  - 레포 `requirements.txt`가 conda export 형식이면 `pip install`을 건너뜁니다.
- requirements 파일은 pip 호환 라인만 필터링해 설치합니다. (conda 메타데이터/빌드스트링 제거)
- `NO_PUBKEY`/`Release file is not valid` → `gnupg`, `ca-certificates` 추가
- `Failed to fetch`/DNS 실패 → `ca-certificates` 추가
- `CERTIFICATE_VERIFY_FAILED` → `ca-certificates` 추가
- `GLIBCXX_* not found` → `libstdc++6` 추가
- `No space left on device` → 사용자 정리 필요 안내
- `Killed`/`exit code 137` → 메모리 부족 가능성 안내

### 수정 전략

- Python 버전 조정: 논문/레포 힌트에 맞춰 상향 또는 하향
- 베이스 이미지 교체: CPU ↔ CUDA 전환
- 시스템 패키지 추가: 빌드 도구/런타임 라이브러리
- pip 충돌 시 `constraints.txt` 생성 후 재시도

## 7) 재시도 정책

- 기본 최대 시도 횟수: `--max-attempts` (기본 5)
- 각 시도는 `logs/attempt_XX.log`로 저장됩니다.
- 최종 실패 시 `failure_report.json`에 실패 원인과 시도 내역을 기록합니다.

## 8) 옵션별 동작 차이

- `--dry-run`: 파일만 생성하고 build/run은 하지 않습니다.
- `--interactive`: 이미지 태그/컨테이너 이름을 프롬프트로 입력받습니다.
- 기본적으로 컨테이너는 유지되며, 실행 후 삭제하려면 `--rm`을 사용합니다.
- 기본적으로 로컬 `workspace/`를 컨테이너 `/workspace`에 마운트합니다. 끄려면 `--no-mount`.
- 기본적으로 컨테이너를 백그라운드에서 계속 실행합니다. 끄려면 `--no-keep-alive`.
- `--no-web-search`: 외부 검색 힌트를 사용하지 않습니다.
- `--llm-model`: 로컬 Ollama 모델로 힌트 추출을 보강합니다.
- LLM 기능은 외부 `ollama` 바이너리를 호출하므로 `paper2env` 패키지 자체 용량은 거의 늘지 않습니다. 다만 로컬 모델 파일은 수백 MB~수 GB까지 커질 수 있습니다.
- `--best-effort`: 알 수 없는 에러에서 실패를 중단하지 않고 `failure_report.json`만 기록한 뒤 종료 코드를 0으로 반환합니다.
- 기본적으로 항상 새로 생성합니다. 기존 `workspace/`를 재사용하려면 `--resume`.
- 기본적으로 Docker build/run 로그를 실시간으로 출력합니다. 끄려면 `--no-verbose`.

### 빌드 캐시

- 패치가 적용된 다음 시도에서는 `docker build --no-cache`로 캐시를 비활성화합니다.

## 9) 경계 조건

- 논문에 명시된 버전이 구식이면, 최신화보다 재현성을 우선합니다.
- CUDA 기반 이미지는 호스트 환경 제약으로 실패할 수 있습니다.
- 외부 레포의 `Dockerfile`이 과도하게 무겁다면 self‑healing에서 축약할 수 있습니다.

## 10) 환경 진단 (`paper2env doctor`)

- 기본 진단 항목: OS/아키텍처, Python 버전, Docker 동작 여부, CUDA/GPU 감지 결과
- `nvidia-smi`가 없을 경우:
  - macOS: NVIDIA GPU/driver 사용 불가 안내
  - Linux/서버: NVIDIA 드라이버 또는 `nvidia-container-toolkit` 설치 필요 안내
- `--deep` 옵션을 주면 추가로 다음 항목을 검사합니다.
  - `lspci`: 하드웨어 수준에서 NVIDIA GPU 유무 확인
  - `lsmod`: `nvidia` 커널 모듈 로드 여부 확인
  - `nvidia-container-cli info`: 컨테이너 GPU 패스스루 환경 확인
- 기본적으로 `paper2env doctor`는 대화형이며, GPU 미감지 시 추가 진단 실행 여부를 묻습니다. (`--yes` 또는 `--no-interactive`로 생략 가능)
