# Paper2Env 발표용 슬라이드 원고

## 1. 제목
Paper2Env: 논문 PDF → 재현 가능한 Docker 환경 자동 생성

## 2. 한 줄 요약
논문 PDF 하나만 넣으면 Docker 환경을 만들고, 실패 로그를 분석해 자동으로 수정/재시도하는 재현 자동화 도구.

## 3. 문제 정의
- 논문 재현은 환경 셋업에 시간이 가장 많이 듦
- CUDA/Python/의존성 충돌이 빈번
- 문서에 적힌 환경이 실제로는 재현되지 않는 경우가 많음

## 4. 목표
- “논문 → 실행 가능 환경”까지 자동화
- 실패 시 자동 복구(Self‑healing)
- 재현 가능성과 이식성 확보

## 5. 핵심 아이디어
- PDF에서 프레임워크/버전/링크 힌트 추출
- GitHub/HF/웹 힌트로 requirements/Dockerfile 구성
- Docker build/run 실패 로그를 룰 기반 + LLM으로 수정

## 6. 시스템 흐름
1. PDF 파싱
2. 힌트 우선순위로 환경 구성
3. Docker build/run
4. 실패 시 로그 분류 → 패치 → 재시도

## 6-1. 기술 스택
- Python 3.8+
- Docker / Dockerfile
- PDF 파싱: `pypdf`
- 로그 분류: 정규식 룰 기반
- LLM 보강: `Ollama` (선택)
- 배포: PyPI (`build`, `twine`)

## 7. 자동 복구 전략 (예시)
- Missing module → requirements 자동 추가
- Python/CUDA 버전 불일치 → 자동 조정
- 빌드 도구/라이브러리 누락 → 시스템 패키지 추가
- VCS 패키지/확장 빌드 실패 → 분리 설치 + build isolation 제어
- tzdata 프롬프트 → noninteractive 설정

## 7-1. 구현 상세 (어떻게 만들었는지)
- PDF 파싱: `pypdf`로 본문 텍스트를 추출하고, 정규식으로 프레임워크/버전/CUDA/링크를 파악
- 힌트 우선순위: 사용자 입력 → 논문 내 링크/버전 → 웹 검색 결과 순으로 반영
- 환경 생성: `env.json`에 모든 힌트/결정 사항 기록, `requirements.txt`/`Dockerfile` 자동 생성
- 요구사항 정합:
  - `requirements.txt`에 누락된 핵심 패키지(예: `torch`) 자동 삽입
  - `torch/torchvision/torchaudio`를 앞에 배치해 설치 순서 보장
  - VCS 패키지/확장 빌드는 `requirements.post.txt`로 분리
  - 분리된 패키지는 `PIP_NO_BUILD_ISOLATION=1`로 설치하여 의존성 가시성 확보
- Dockerfile 생성:
  - CUDA 힌트가 있으면 `nvidia/cuda:*` 베이스 선택
  - `DEBIAN_FRONTEND=noninteractive`, `TZ=Etc/UTC`로 대화형 프롬프트 방지
  - CUDA 베이스일 경우 `CUDA_HOME` 및 PATH/LD_LIBRARY_PATH 설정
- Self‑healing 루프:
  - 실패 로그를 분류 → 패치 → 재시도 (최대 시도 횟수 제한)
  - 패치가 없으면 실패 리포트(`failure_report.json`) 기록
- LLM 보강:
  - 룰 매칭이 실패하면 LLM에게 최소 수정안을 요청
  - LLM은 `pin/add_requirement/set_base_image/set_python/reorder_requirements` 액션 제안 가능

## 8. LLM 보강
- 규칙에 없는 오류는 LLM이 패치를 제안
- `--llm-model` 옵션으로 사용 (기본은 룰 기반)

## 9. 데모
```bash
paper2env run papers/MINIMA.pdf --llm-model llama3 --yes
```

## 10. 결과
- 자동 환경 생성
- 실패 시 재시도/수정 기록
- Docker 기반으로 재현성 확보

## 11. 한계
- CUDA/드라이버/아키텍처 제약
- 일부 패키지 소스 빌드 필요
- 100% 자동 성공 보장 X (best‑effort 제공)

## 12. 레포 구조 (핵심)
- `paper2env/cli.py`: CLI 진입점
- `paper2env/core/parser.py`: PDF 파싱
- `paper2env/core/orchestrator.py`: self‑healing 루프
- `paper2env/core/patcher.py`: 룰/LLM 패치

## 13. 배포
- PyPI: `paper2env==0.1.1`
- 배포: `python -m build` → `python -m twine upload dist/*`

## 14. Q&A
필요하면 상세 데모/실패 로그/재시도 기록 공유 가능
