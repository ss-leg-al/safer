# Privacy Guard Agent

영상 PII 자동 비식별화 시스템. LangGraph 에이전트가 GPT-4o Vision으로 PII(얼굴/문서/스크린/명패/신분증)를 탐지하고 OpenCV CSRT로 추적, ffmpeg로 마스킹된 영상을 합성합니다.

상세 명세는 [CLAUDE.md](CLAUDE.md), 이 빌드의 개발 계획은 `~/.claude/plans/rippling-questing-flamingo.md` 참고.

## 스택

- **Backend**: FastAPI + LangGraph (`create_react_agent` supervisor) + LangChain-OpenAI (GPT-4o) + OpenCV (CSRT) + ffmpeg
- **Frontend**: React 18 + Vite + Tailwind CSS
- **비동기**: FastAPI `BackgroundTasks` (단일 프로세스, MVP용 — Celery/Redis 미사용)
- **상태/로그**: 파일시스템 (`uploads/{job_id}/logs.jsonl`, `status.txt`) + 인메모리 `JobStore`

## 에이전틱 디자인

이 프로젝트는 **고정 DAG가 아니라 supervisor + tool-calling 루프**입니다.

```
HumanMessage("anonymize job_id=...")
   ↓
[Supervisor (GPT-4o)] ← 매 스텝 다음 툴 결정
   ↓ tool_call
[Tool 실행 → 결과 요약 문자열]
   ↑ ToolMessage
[Supervisor] ← 결과 보고 다음 행동 판단
   ...
[Supervisor] → final_answer (DONE)
```

LLM이 7개 툴 중 무엇을 언제 호출할지 매 턴 결정합니다 — `extract_frames` 호출 후 `analyze_scene` 결과를 보고 어떤 PII 타입을 노릴지 정하고, `detect_pii`의 confidence가 낮으면 다른 `target_types`로 재시도하는 등의 판단을 LLM이 합니다. 큰 데이터(bbox 배열, 프레임별 추적 결과)는 per-job `JobStore`에 저장하고 LLM에는 짧은 요약만 전달해 토큰 비용을 억제합니다.

## 디렉토리

```
backend/
  main.py                 # FastAPI 앱 + 모든 엔드포인트 + startup에서 SAM3 로드
  config.py               # pydantic-settings로 .env 로드
  schemas.py              # Pydantic 응답 모델
  models/
    sam3_loader.py        # SAM3 싱글톤 (GPU 1회 로드, 재사용)
  agent/
    graph.py              # create_react_agent supervisor + 시스템 프롬프트
    runner.py             # BackgroundTasks 진입점, agent.stream으로 supervisor reasoning 캡처
    log_emitter.py        # JSONL 로그 + status 파일 헬퍼
    job_store.py          # per-job 인메모리 컨텍스트 (bbox/frames 경로/씬 등)
    tools/
      agentic.py          # 7개 @tool 함수: extract_frames, analyze_scene, detect_pii,
                          # track_objects, mask_frames, compose_video, generate_report
checkpoints/
  sam3.pt                 # SAM3 가중치 (HuggingFace에서 별도 다운로드)
frontend/
  src/
    App.jsx
    api/client.js
    hooks/{useAgentLog,useJobStatus}.js
    components/{UploadZone,AgentLog,ReportPanel,DownloadButton,VideoPreview}.jsx
uploads/                  # 입력 영상 + 추출 프레임 + 마스킹 프레임 + 로그/상태
outputs/                  # 마스킹된 mp4 + report.json
```

## 설치 (1회)

> **주의**: Meta SAM3는 Python 3.12 + PyTorch 2.7+ 필수. 새 conda env 사용.

### 1. conda env + ffmpeg + PyTorch

```bash
conda create -n sam3 -c conda-forge python=3.12 ffmpeg -y
conda activate sam3

# PyTorch (CUDA 12.6+ 드라이버 → cu128 빌드)
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('cuda:', torch.cuda.is_available())"  # → True
```

### 2. Meta SAM3 패키지 + 가중치 (HuggingFace access 필요)

```bash
# 2a. Meta sam3 repo 클론 + 설치
git clone https://github.com/facebookresearch/sam3.git ~/sam3_repo
cd ~/sam3_repo && pip install -e . && cd -

# 2b. 추가 deps (sam3가 setup.py에 안 적은 것들)
pip install "setuptools<81" "numpy<2" psutil pycocotools einops scipy hydra-core decord

# 2c. HF access request: https://huggingface.co/facebook/sam3 (즉시~수시간 승인)
#     read 토큰 발급: https://huggingface.co/settings/tokens
huggingface-cli login   # 토큰 붙여넣기 (또는 HF_TOKEN 환경변수)

# 2d. sam3.pt 다운로드 (3.45 GB)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', local_dir='checkpoints')
"

# 2e. 검증
python -c "from backend.models.sam3_loader import load_sam3; print('loaded:', load_sam3() is not None)"
```

### 3. 백엔드 + 에이전트 deps

```bash
pip install -r requirements.txt
# 만약 opencv 충돌 시:
pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python==4.10.0.84
```

### 4. 프론트엔드

```bash
cd frontend && npm install && cd ..
```

## .env

```bash
cp .env.example .env
# OPENAI_API_KEY 채우기
```

| 키 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | - | GPT-4o supervisor + scene_analyzer (필수) |
| `MAX_VIDEO_DURATION` | 300 | 최대 영상 길이(초) |
| `MAX_VIDEO_SIZE_MB` | 200 | 최대 업로드 크기 |
| `SAMPLE_FPS` | 1 | 프레임 추출 fps |
| `CONFIDENCE_THRESHOLD` | 0.7 | detect_pii 재탐지 트리거 임계값 |
| `MAX_RETRY_COUNT` | 2 | 최대 재탐지 횟수 |

## 실행 (2개 프로세스)

터미널 1 — 백엔드:
```bash
conda activate sam3
uvicorn backend.main:app --reload
```

터미널 2 — 프론트:
```bash
cd frontend
npm run dev
```

브라우저에서 http://localhost:5173 접속.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/health` | 헬스체크 (SAM3 로드 상태 포함) |
| `POST` | `/api/jobs` | 영상 업로드 (multipart `file`) → `{job_id}` |
| `GET` | `/api/jobs/{job_id}/status` | `pending` / `running` / `done` / `failed` |
| `GET` | `/api/jobs/{job_id}/stream` | SSE 로그 스트림 |
| `GET` | `/api/jobs/{job_id}/report` | 완료된 PII 리포트 JSON |
| `GET` | `/api/jobs/{job_id}/download` | 마스킹된 mp4 |

## Supervisor 시스템 프롬프트 (요지)

LLM이 매 스텝 결정을 내리는 기준:
- `extract_frames`는 항상 첫 호출
- `analyze_scene` 결과의 `expected_pii`를 다음 `detect_pii`의 `target_types`로 활용
- `detect_pii` 평균 confidence가 0.7 미만이면 `target_types`를 보완해 재시도 (최대 2회)
- 0개 탐지가 확정되면 track/mask는 건너뛰고도 compose+report는 항상 실행 → output.mp4와 report.json은 매번 생성
- `generate_report` 후 텍스트 답변으로 종료

전체 시스템 프롬프트는 [backend/agent/graph.py](backend/agent/graph.py) 참고.

## CLI 단독 실행 (UI 없이)

```python
from backend.agent.runner import run_agent_job
run_agent_job("test-job", "/path/to/video.mp4")
# 결과: outputs/test-job/output.mp4, outputs/test-job/report.json
```

## 알려진 제약 / Phase 4로 미룸

- **OpenAI API 키 필수**. supervisor가 LLM에 의존하므로 키가 없으면 `status=failed`로 종료. (이전 fixed-pipeline 구현은 키 없이도 fallback으로 완주 가능했지만 에이전틱으로 가면서 트레이드오프).
- 동시 job 1개만 안전 (BackgroundTasks 단일 프로세스). 프로덕션은 Celery+Redis로 전환.
- 인증/권한 없음 — 내부 데모 한정.
- 마스킹된 mp4는 영상 트랙만 (오디오 미보존).
- `detect_pii`는 첫 프레임만 GPT-4o로 호출 (비용 절감, tracker가 전파). 빠른 객체 변화에는 약할 수 있음.
- VideoPreview의 bbox 오버레이는 미구현 (스트레치).
- `ffmpeg/ffprobe`는 conda 환경 내 PATH에 있어야 함 — `conda activate privacy-guard` 후 실행.
- `JobStore`는 인메모리 — uvicorn 재시작 시 진행 중 job은 유실. 결과물(report.json, output.mp4)은 디스크에 남음.

## 검증 (E2E 스모크)

```bash
# 합성 테스트 영상
ffmpeg -f lavfi -i "testsrc2=size=320x240:rate=1:duration=3" -pix_fmt yuv420p -c:v libx264 /tmp/test.mp4 -y

# 업로드 → 그래프 실행 → done까지
JOB=$(curl -s -F "file=@/tmp/test.mp4" http://127.0.0.1:8000/api/jobs | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
sleep 3
curl -s http://127.0.0.1:8000/api/jobs/$JOB/status     # → done
curl -s http://127.0.0.1:8000/api/jobs/$JOB/report     # → JSON
ls outputs/$JOB/                                        # → output.mp4 + report.json
```

**OpenAI 키가 유효해야 done까지 완주합니다.** supervisor가 LLM 호출에 실패하면 `status=failed`. 키 없이 underlying tool 로직만 검증하려면:

```python
from backend.agent.tools.agentic import extract_frames, mask_frames, compose_video, generate_report
from backend.agent.job_store import get_store
job = "manual-test"
store = get_store(job)
store.video_path = "/tmp/test.mp4"
extract_frames.invoke({"job_id": job})
store.detected_objects = []  # skip detect_pii
mask_frames.invoke({"job_id": job})
compose_video.invoke({"job_id": job})
generate_report.invoke({"job_id": job})
```
