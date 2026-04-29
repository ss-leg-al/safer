# Privacy Guard Agent

영상에 담긴 개인정보를 AI가 자동으로 찾아 지워드립니다.

얼굴, 문서, 화면, 명패, 신분증 — 영상을 올리면 AI 에이전트가 씬을 파악하고,
픽셀 단위로 정밀하게 마스킹한 영상과 탐지 리포트를 돌려줍니다.

---

## 이런 상황에 씁니다

- 회의 녹화 영상을 외부에 공유하기 전에 얼굴·화면을 지워야 할 때
- 강의 영상에서 수강생 얼굴이나 개인 문서를 제거해야 할 때
- 인터뷰 영상의 신분증·명패를 자동으로 블랙박스 처리해야 할 때
- 영상 후처리를 사람이 프레임마다 직접 하기엔 너무 많을 때

---

## 어떻게 동작하나요

영상을 업로드하면 AI 에이전트가 7단계를 스스로 판단하며 처리합니다.

```
1. 프레임 추출        영상을 1fps로 분해
2. 씬 분석            GPT-4o가 "회의실이니까 얼굴·화면이 있을 것"을 판단
3. PII 탐지           SAM3가 텍스트 프롬프트로 픽셀 단위 마스크 생성
4. 신뢰도 검증        confidence 낮으면 프롬프트 보완 후 재탐지 (최대 2회)
5. 전체 추적          SAM3가 모든 프레임에 마스크 전파
6. 마스킹 적용        얼굴→흐림, 문서/신분증→검정, 화면→픽셀화
7. 영상 합성 + 리포트 마스킹된 mp4 + 탐지 요약 JSON/PDF 생성
```

처리 중 AI의 판단 과정이 화면에 실시간으로 표시됩니다.

---

## 결과물

| 파일 | 내용 |
|---|---|
| `output.mp4` | PII가 마스킹된 영상 |
| `report.json` | 탐지된 객체 수·종류·신뢰도·위치 |
| `report.pdf` | 요약 테이블 + 마스킹 썸네일 |

---

## 마스킹 방식

| PII 종류 | 처리 방식 |
|---|---|
| 얼굴 (face) | 가우시안 블러 |
| 문서 (document) | 검정 박스 |
| 화면 (screen) | 픽셀화 |
| 명패 (nameplate) | 검정 박스 |
| 신분증 (id_card) | 검정 박스 |

---

## 빠른 시작

### 필요한 것
- NVIDIA GPU (VRAM 10GB 이상 권장)
- CUDA 12.6+
- OpenAI API 키
- conda

### 1. 환경 설정

```bash
conda create -n sam3 -c conda-forge python=3.12 ffmpeg -y
conda activate sam3

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. SAM3 모델 다운로드

HuggingFace에서 [facebook/sam3](https://huggingface.co/facebook/sam3) access request 후:

```bash
huggingface-cli login

python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', local_dir='checkpoints')
"
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 4. 환경변수

```bash
cp .env.example .env
# OPENAI_API_KEY 입력
```

### 5. 실행

터미널 1:
```bash
conda activate sam3
uvicorn backend.main:app --reload
```

터미널 2:
```bash
cd frontend && npm run dev
```

브라우저에서 http://localhost:5173 접속 후 영상 업로드.

---

## 환경변수 목록

| 키 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | 필수 | GPT-4o 씬 분석 + supervisor |
| `MAX_VIDEO_DURATION` | 300 | 처리 가능한 최대 영상 길이 (초) |
| `MAX_VIDEO_SIZE_MB` | 200 | 최대 업로드 크기 (MB) |
| `SAMPLE_FPS` | 1 | 프레임 추출 밀도 |
| `CONFIDENCE_THRESHOLD` | 0.7 | 재탐지 트리거 신뢰도 임계값 |
| `MAX_RETRY_COUNT` | 2 | 탐지 재시도 최대 횟수 |

---

## API

외부 시스템에서 직접 호출하거나 CLI로 검증할 때:

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/health` | 서버 상태 및 SAM3 로드 여부 확인 |
| `POST` | `/api/jobs` | 영상 업로드 → `job_id` 반환 |
| `GET` | `/api/jobs/{job_id}/status` | `pending` / `running` / `done` / `failed` |
| `GET` | `/api/jobs/{job_id}/stream` | AI 판단 과정 실시간 스트림 (SSE) |
| `GET` | `/api/jobs/{job_id}/report` | 탐지 리포트 JSON |
| `GET` | `/api/jobs/{job_id}/download` | 마스킹된 mp4 다운로드 |

**CLI 예시:**

```bash
JOB=$(curl -s -F "file=@video.mp4" http://localhost:8000/api/jobs \
      | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

# 완료 대기
curl -s http://localhost:8000/api/jobs/$JOB/status   # → done

# 결과 확인
curl -s http://localhost:8000/api/jobs/$JOB/report
curl -O http://localhost:8000/api/jobs/$JOB/download
```

---

## 현재 제약

- 영상 트랙만 처리 — **오디오는 보존되지 않습니다**
- 동시에 1개 작업만 안전하게 처리됩니다 (GPU 단일 점유)
- 서버 재시작 시 처리 중이던 작업은 유실됩니다 (완료된 결과물은 보존)
- 인증 없음 — 내부 데모 전용입니다

---

## 기술 스택

| 영역 | 기술 |
|---|---|
| AI 에이전트 | LangGraph · GPT-4o |
| 탐지·추적 | Meta SAM3 (텍스트 프롬프트 기반) |
| 영상 처리 | ffmpeg · OpenCV |
| 백엔드 | FastAPI |
| 프론트엔드 | React 18 · Vite · Tailwind CSS |
