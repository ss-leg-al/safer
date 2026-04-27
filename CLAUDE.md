# Privacy Guard Agent — CLAUDE.md

## 프로젝트 개요

AI 기반 영상 개인정보 자동 비식별화 시스템.
영상을 업로드하면 LLM 에이전트가 GPT-4o로 씬을 분석하고,
SAM3로 PII 객체를 텍스트 프롬프트 기반으로 탐지·세그멘테이션·추적한 뒤 마스킹하고 리포트를 생성한다.

---

## SAM3 역할 (핵심)

SAM3는 이 프로젝트의 핵심 모델이다. 기존 SAM2와의 차이점:

| 기능 | SAM2 | SAM3 |
|------|------|------|
| 프롬프트 방식 | 클릭/bbox (시각적) | **텍스트 프롬프트** ("face", "document") |
| 탐지 | 수동으로 객체 지정 필요 | 텍스트만으로 zero-shot 탐지 |
| 추적 | 단일 객체, 수동 재시작 | **다중 객체 동시 추적 + 메모리뱅크** |
| 세그멘테이션 | bbox 수준 | **픽셀 단위 마스크** |

→ GPT-4o가 "이 영상엔 face, document가 있다"고 판단하면,
  SAM3가 그 텍스트를 직접 받아서 탐지 + 픽셀 마스크 생성 + 전체 영상 추적까지 처리한다.
  별도 tracker(CSRT 등) 불필요.

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| Frontend | React 18 + Vite + Tailwind CSS |
| Backend | FastAPI + Celery + Redis |
| Agent | LangGraph (Tool-calling loop) |
| Scene Analysis | GPT-4o Vision API |
| Segmentation + Tracking | **SAM3** (Ultralytics, 로컬 GPU) |
| Masking | OpenCV (마스크 위에 blur/blackbox/pixelate 적용) |
| Video | ffmpeg (프레임 추출/합성) |
| Storage | 로컬 파일시스템 (uploads/, outputs/) |

### GPU 요구사항

SAM3는 로컬 GPU에서 실행한다.

```
권장: RTX 3090 / A4000 이상 (VRAM 16GB+)
최소: RTX 3080 (VRAM 10GB, 해상도 제한 필요)
CUDA: 11.8 이상
모델 파일: sam3.pt (HuggingFace에서 access request 후 다운로드)
```

---

## 디렉토리 구조

```
privacy-guard-agent/
├── backend/
│   ├── main.py                  # FastAPI 앱 진입점
│   ├── agent/
│   │   ├── graph.py             # LangGraph 에이전트 그래프 정의
│   │   ├── state.py             # AgentState TypedDict
│   │   └── tools/
│   │       ├── frame_extractor.py    # ffmpeg 프레임 추출
│   │       ├── scene_analyzer.py     # GPT-4o Vision → PII 전략 수립 (판단 1)
│   │       ├── sam3_segmentor.py     # SAM3 텍스트 프롬프트 탐지 + 세그멘테이션
│   │       ├── sam3_tracker.py       # SAM3 전체 프레임 추적 + 마스크 전파
│   │       ├── confidence_checker.py # 탐지 결과 검증 → 재탐지 판단 (판단 2)
│   │       ├── masking_engine.py     # SAM3 마스크 위에 blur/blackbox/pixelate 적용 (판단 3)
│   │       ├── video_composer.py     # ffmpeg 최종 합성
│   │       └── report_generator.py  # PII 요약 리포트 생성 (판단 4)
│   ├── models/
│   │   └── sam3_loader.py       # SAM3 모델 싱글톤 로드 (startup에 1회만)
│   ├── tasks.py                 # Celery 비동기 작업 정의
│   ├── schemas.py               # Pydantic 요청/응답 스키마
│   └── config.py                # 환경변수, 경로 설정
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── UploadZone.jsx        # 영상 업로드 드래그앤드롭
│   │   │   ├── AgentLog.jsx          # 에이전트 판단 로그 스트리밍
│   │   │   ├── VideoPreview.jsx      # 마스킹 미리보기 + 세그멘테이션 마스크 오버레이
│   │   │   ├── ReportPanel.jsx       # PII 탐지 리포트 테이블
│   │   │   └── DownloadButton.jsx    # 최종 영상 다운로드
│   │   ├── hooks/
│   │   │   ├── useJobStatus.js       # SSE로 처리 상태 구독
│   │   │   └── useAgentLog.js        # 에이전트 로그 스트림 수신
│   │   └── api/
│   │       └── client.js             # axios 인스턴스, API 호출 함수
│   └── package.json
├── checkpoints/
│   └── sam3.pt                  # SAM3 모델 가중치 (HuggingFace에서 별도 다운로드)
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── CLAUDE.md
```

---

## 에이전트 판단 루프 (핵심)

### AgentState

```python
# backend/agent/state.py
from typing import TypedDict, List, Literal

class PIIObject(TypedDict):
    type: Literal["face", "document", "screen", "nameplate", "id_card"]
    mask: List[List[int]]        # SAM3 픽셀 마스크 (polygon points)
    confidence: float
    track_id: int                # SAM3 다중 객체 추적 ID
    frame_range: List[int]       # [start_frame, end_frame]
    mask_strategy: str           # "blur" | "blackbox" | "pixelate"

class AgentState(TypedDict):
    job_id: str
    video_path: str
    frames_dir: str
    scene_type: str                       # "meeting" | "lecture" | "interview" | ...
    estimated_pii_types: List[str]        # 판단 1: GPT-4o가 결정한 탐지 대상
    sam3_text_prompts: List[str]          # SAM3에 넘길 텍스트 프롬프트 목록
    detected_objects: List[PIIObject]     # 판단 2: SAM3 탐지 결과
    retry_count: int
    masked_frames_dir: str
    output_video_path: str
    report: dict
    logs: List[dict]                      # SSE로 프론트에 스트리밍할 판단 로그
```

### LangGraph 그래프 정의

```python
# backend/agent/graph.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from .tools import (
    frame_extractor, scene_analyzer,
    sam3_segmentor, sam3_tracker,
    confidence_checker, masking_engine,
    video_composer, report_generator,
)

def should_retry(state: AgentState) -> str:
    """판단 2: SAM3 탐지 결과가 충분한지 검증"""
    low_conf = [o for o in state["detected_objects"] if o["confidence"] < 0.65]
    if low_conf and state["retry_count"] < 2:
        return "retry"
    return "proceed"

graph = StateGraph(AgentState)

graph.add_node("extract_frames",    frame_extractor.run)
graph.add_node("analyze_scene",     scene_analyzer.run)      # 판단 1
graph.add_node("segment_pii",       sam3_segmentor.run)      # SAM3 탐지
graph.add_node("check_confidence",  confidence_checker.run)  # 판단 2
graph.add_node("track_pii",         sam3_tracker.run)        # SAM3 추적
graph.add_node("mask_frames",       masking_engine.run)      # 판단 3
graph.add_node("compose_video",     video_composer.run)
graph.add_node("generate_report",   report_generator.run)   # 판단 4

graph.set_entry_point("extract_frames")
graph.add_edge("extract_frames",   "analyze_scene")
graph.add_edge("analyze_scene",    "segment_pii")
graph.add_edge("segment_pii",      "check_confidence")
graph.add_conditional_edges("check_confidence", should_retry, {
    "retry":   "segment_pii",   # 프롬프트 보완 후 재탐지 루프
    "proceed": "track_pii"
})
graph.add_edge("track_pii",        "mask_frames")
graph.add_edge("mask_frames",      "compose_video")
graph.add_edge("compose_video",    "generate_report")
graph.add_edge("generate_report",  END)

app_graph = graph.compile()
```

---

## SAM3 모델 로드 (싱글톤)

FastAPI startup 시 1회만 로드해 모든 요청에서 재사용한다.
매 요청마다 로드하면 GPU 메모리 초과 및 수십 초 지연 발생.

```python
# backend/models/sam3_loader.py
from ultralytics import SAM

_sam3_model = None

def get_sam3() -> SAM:
    global _sam3_model
    if _sam3_model is None:
        _sam3_model = SAM("checkpoints/sam3.pt")
        _sam3_model.to("cuda")
    return _sam3_model
```

```python
# backend/main.py — startup 이벤트
@app.on_event("startup")
async def startup_event():
    get_sam3()  # 서버 시작 시 GPU에 SAM3 올려두기
    print("SAM3 loaded on GPU")
```

---

## 툴 구현 가이드

### Tool 1: frame_extractor

```python
# backend/agent/tools/frame_extractor.py
import subprocess, os
from ..state import AgentState

def run(state: AgentState) -> AgentState:
    out_dir = f"uploads/{state['job_id']}/frames"
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", state["video_path"],
        "-vf", "fps=1",
        f"{out_dir}/%04d.jpg", "-y"
    ], check=True)
    state["frames_dir"] = out_dir
    state["logs"].append({
        "step": 1, "action": "frame_extractor",
        "result": f"추출 완료: {out_dir}"
    })
    return state
```

### Tool 2: scene_analyzer (판단 1)

GPT-4o Vision으로 씬을 분석하고 SAM3에 넘길 텍스트 프롬프트를 결정한다.

```python
# backend/agent/tools/scene_analyzer.py
import openai, base64, json, os
from ..state import AgentState

PII_PROMPT_MAP = {
    "face":      "human face",
    "document":  "paper document",
    "screen":    "computer screen",
    "nameplate": "name badge",
    "id_card":   "id card",
}

SYSTEM_PROMPT = """
You are a privacy analysis agent.
Given a video frame, identify scene type and privacy-sensitive objects likely present.
Respond ONLY in JSON:
{
  "scene_type": "meeting" | "lecture" | "interview" | "public" | "other",
  "expected_pii": ["face", "document", "screen", "nameplate", "id_card"],
  "reasoning": "..."
}
"""

def run(state: AgentState) -> AgentState:
    frames = sorted(os.listdir(state["frames_dir"]))
    sample = frames[len(frames) // 10]
    with open(f"{state['frames_dir']}/{sample}", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": SYSTEM_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}],
        response_format={"type": "json_object"}
    )
    result = json.loads(resp.choices[0].message.content)
    state["scene_type"]          = result["scene_type"]
    state["estimated_pii_types"] = result["expected_pii"]
    state["sam3_text_prompts"]   = [PII_PROMPT_MAP[p] for p in result["expected_pii"]
                                     if p in PII_PROMPT_MAP]
    state["logs"].append({
        "step": 1, "action": "scene_analyzer",
        "thinking": f"씬: {result['scene_type']} → SAM3 프롬프트: {state['sam3_text_prompts']}",
        "tool_call": f"gpt4o_vision(frame={sample})",
        "result": result
    })
    return state
```

### Tool 3: sam3_segmentor (SAM3 탐지 + 픽셀 마스크)

SAM3의 텍스트 기반 Promptable Concept Segmentation(PCS)을 사용한다.

```python
# backend/agent/tools/sam3_segmentor.py
import os
from ..state import AgentState, PIIObject
from models.sam3_loader import get_sam3

MASK_STRATEGY_MAP = {
    "face":      "blur",
    "document":  "blackbox",
    "screen":    "pixelate",
    "nameplate": "blackbox",
    "id_card":   "blackbox",
}

def run(state: AgentState) -> AgentState:
    sam3 = get_sam3()
    frames = sorted(os.listdir(state["frames_dir"]))
    # 대표 프레임 3장 샘플링 (10%, 30%, 50% 지점)
    samples = [f"{state['frames_dir']}/{frames[i]}"
               for i in [len(frames)//10, len(frames)//3, len(frames)//2]]

    detected: list[PIIObject] = []
    for idx, text_prompt in enumerate(state["sam3_text_prompts"]):
        results = sam3.predict(
            source=samples,
            texts=text_prompt,   # SAM3 텍스트 프롬프트 (PCS 모드)
            task="segment",
            conf=0.3,
        )
        for result in results:
            if result.masks is None:
                continue
            for mask, conf in zip(result.masks.xy, result.boxes.conf.tolist()):
                pii_type = state["estimated_pii_types"][idx]
                detected.append(PIIObject(
                    type=pii_type,
                    mask=mask.tolist(),
                    confidence=float(conf),
                    track_id=-1,
                    frame_range=[-1, -1],
                    mask_strategy=MASK_STRATEGY_MAP.get(pii_type, "blur"),
                ))

    state["detected_objects"] = detected
    state["retry_count"]      = state.get("retry_count", 0)
    state["logs"].append({
        "step": 2, "action": "sam3_segmentor",
        "thinking": f"SAM3 텍스트 프롬프트로 {len(detected)}개 객체 탐지",
        "tool_call": f"sam3.predict(texts={state['sam3_text_prompts']})",
        "result": {"detected_count": len(detected)}
    })
    return state
```

### Tool 4: confidence_checker (판단 2)

SAM3 탐지 결과의 신뢰도를 검증하고 재탐지 시 프롬프트를 보완한다.

```python
# backend/agent/tools/confidence_checker.py
from ..state import AgentState

THRESHOLD = 0.65
REFINED_PROMPTS = {
    "face":      "close-up human face portrait",
    "document":  "printed paper sheet with text",
    "screen":    "laptop or monitor screen display",
    "nameplate": "name tag badge on clothing",
    "id_card":   "identity card with photo",
}

def run(state: AgentState) -> AgentState:
    low_conf = [o for o in state["detected_objects"] if o["confidence"] < THRESHOLD]
    if low_conf:
        state["retry_count"] += 1
        failed = list({o["type"] for o in low_conf})
        state["sam3_text_prompts"] = [
            REFINED_PROMPTS.get(t, p)
            for t, p in zip(state["estimated_pii_types"], state["sam3_text_prompts"])
            if t in failed
        ]
        state["logs"].append({
            "step": 2, "action": "confidence_checker",
            "thinking": f"confidence 낮음({[round(o['confidence'],2) for o in low_conf]}) → 프롬프트 보완 후 재탐지",
            "tool_call": f"confidence_checker(threshold={THRESHOLD})",
            "result": {"retry": True, "refined_prompts": state["sam3_text_prompts"]}
        })
    else:
        state["logs"].append({
            "step": 2, "action": "confidence_checker",
            "thinking": "confidence 충분 → 추적 단계 진행",
            "tool_call": f"confidence_checker(threshold={THRESHOLD})",
            "result": {"retry": False, "objects": len(state["detected_objects"])}
        })
    return state
```

### Tool 5: sam3_tracker (SAM3 전체 프레임 추적)

SAM3 track 모드로 메모리뱅크 기반 다중 객체 추적을 수행한다.
별도 OpenCV CSRT tracker 불필요.

```python
# backend/agent/tools/sam3_tracker.py
from ..state import AgentState
from models.sam3_loader import get_sam3

def run(state: AgentState) -> AgentState:
    sam3 = get_sam3()

    # SAM3 track 모드: 메모리뱅크로 프레임 간 객체 ID 유지
    results = sam3.track(
        source=state["video_path"],
        texts=state["sam3_text_prompts"],
        persist=True,    # 프레임 간 메모리뱅크 유지
        conf=0.4,
        iou=0.5,
    )

    track_map: dict[int, dict] = {}
    for frame_idx, result in enumerate(results):
        if result.masks is None or result.boxes.id is None:
            continue
        for mask, track_id, conf in zip(
            result.masks.xy,
            result.boxes.id.int().tolist(),
            result.boxes.conf.tolist()
        ):
            if track_id not in track_map:
                track_map[track_id] = {"mask": mask.tolist(), "start": frame_idx, "end": frame_idx}
            else:
                track_map[track_id]["end"]  = frame_idx
                track_map[track_id]["mask"] = mask.tolist()

    for obj in state["detected_objects"]:
        for tid, info in track_map.items():
            obj["track_id"]    = tid
            obj["frame_range"] = [info["start"], info["end"]]

    state["logs"].append({
        "step": 2, "action": "sam3_tracker",
        "thinking": f"SAM3 메모리뱅크로 {len(track_map)}개 객체 전체 영상 추적 완료",
        "tool_call": "sam3.track(source=video, persist=True)",
        "result": {"tracked_objects": len(track_map)}
    })
    return state
```

### Tool 6: masking_engine (판단 3)

SAM3가 생성한 픽셀 단위 마스크 위에 PII 타입별 마스킹을 적용한다.

```python
# backend/agent/tools/masking_engine.py
import cv2, numpy as np, os
from ..state import AgentState

def run(state: AgentState) -> AgentState:
    out_dir = f"uploads/{state['job_id']}/masked_frames"
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(os.listdir(state["frames_dir"]))

    for frame_name in frames:
        img = cv2.imread(f"{state['frames_dir']}/{frame_name}")
        if img is None:
            continue
        h, w = img.shape[:2]
        for obj in state["detected_objects"]:
            pts = np.array(obj["mask"], dtype=np.int32)
            binary = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(binary, [pts], 255)
            img = _apply_mask(img, binary, obj["mask_strategy"])
        cv2.imwrite(f"{out_dir}/{frame_name}", img)

    state["masked_frames_dir"] = out_dir
    state["logs"].append({
        "step": 3, "action": "masking_engine",
        "thinking": "face→blur / document→blackbox / screen→pixelate 적용",
        "tool_call": f"masking_engine(frames={len(frames)})",
        "result": {"masked_frames": len(frames)}
    })
    return state

def _apply_mask(img: np.ndarray, mask: np.ndarray, strategy: str) -> np.ndarray:
    result = img.copy()
    roi = mask == 255
    if strategy == "blur":
        blurred = cv2.GaussianBlur(img, (51, 51), 15)
        result[roi] = blurred[roi]
    elif strategy == "blackbox":
        result[roi] = 0
    elif strategy == "pixelate":
        ys, xs = np.where(roi)
        if len(ys) == 0:
            return result
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        patch = img[y1:y2, x1:x2]
        if patch.size > 0:
            block = 12
            small = cv2.resize(patch, (max(1,(x2-x1)//block), max(1,(y2-y1)//block)))
            pix   = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            tmp = result.copy()
            tmp[y1:y2, x1:x2] = pix
            result[roi] = tmp[roi]
    return result
```

---

## FastAPI 엔드포인트

```python
# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from pathlib import Path
from models.sam3_loader import get_sam3
import uuid, asyncio, os, json

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    get_sam3()

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    path = f"uploads/{job_id}/input{Path(file.filename).suffix}"
    os.makedirs(f"uploads/{job_id}", exist_ok=True)
    with open(path, "wb") as f:
        f.write(await file.read())
    run_agent.delay(job_id, path)
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}/stream")
async def stream_logs(job_id: str):
    async def event_generator():
        last_idx = 0
        while True:
            logs = redis_client.lrange(f"logs:{job_id}", last_idx, -1)
            for log in logs:
                yield f"data: {log.decode()}\n\n"
                last_idx += 1
            status = redis_client.get(f"status:{job_id}")
            if status and status.decode() == "done":
                yield 'data: {"event":"done"}\n\n'
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/jobs/{job_id}/report")
async def get_report(job_id: str):
    return json.loads(redis_client.get(f"report:{job_id}"))

@app.get("/api/jobs/{job_id}/download")
async def download_video(job_id: str):
    return FileResponse(f"outputs/{job_id}/masked.mp4",
                        media_type="video/mp4", filename="masked.mp4")
```

---

## Frontend — AgentLog 컴포넌트

```jsx
// frontend/src/components/AgentLog.jsx
import { useEffect, useState } from "react"

const STEP_LABELS = {
  scene_analyzer:     "판단 1 — PII 전략 수립",
  sam3_segmentor:     "판단 2 — SAM3 탐지 + 세그멘테이션",
  confidence_checker: "판단 2 — 탐지 충분성 확인",
  sam3_tracker:       "판단 2 — SAM3 전체 프레임 추적",
  masking_engine:     "판단 3 — 마스킹 방식 결정",
  report_generator:   "판단 4 — 리포트 생성",
}

export default function AgentLog({ jobId }) {
  const [logs, setLogs] = useState([])

  useEffect(() => {
    if (!jobId) return
    const es = new EventSource(`/api/jobs/${jobId}/stream`)
    es.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.event === "done") { es.close(); return }
      setLogs(prev => [...prev, data])
    }
    return () => es.close()
  }, [jobId])

  return (
    <div className="flex flex-col gap-2 p-4">
      {logs.map((log, i) => (
        <div key={i}>
          {STEP_LABELS[log.action] && (
            <div className="text-xs text-tertiary border rounded px-2 py-0.5 w-fit mb-1">
              {STEP_LABELS[log.action]}
            </div>
          )}
          {log.thinking  && <p className="italic text-sm bg-secondary px-3 py-2 rounded-lg">"{log.thinking}"</p>}
          {log.tool_call && <p className="font-mono text-xs text-info bg-info/10 px-3 py-2 rounded">→ {log.tool_call}</p>}
          {log.result    && <p className="text-xs text-success bg-success/10 px-3 py-2 rounded">✓ {JSON.stringify(log.result)}</p>}
        </div>
      ))}
    </div>
  )
}
```

---

## requirements.txt

```txt
fastapi
uvicorn[standard]
celery
redis
python-multipart
openai
ultralytics>=8.3.237   # SAM3 통합 버전
torch>=2.0.0
torchvision
opencv-python-headless
numpy
pydantic
langgraph
```

> SAM3 모델 가중치 별도 설치:
> 1. https://huggingface.co/ultralytics/assets 에서 access request
> 2. `sam3.pt` 다운로드 → `checkpoints/sam3.pt` 위치에 배치
> 3. clip 패키지 충돌 해결:
>    `pip uninstall clip -y && pip install git+https://github.com/ultralytics/CLIP.git`

---

## 환경변수 (.env)

```bash
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379/0
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
SAM3_CHECKPOINT=checkpoints/sam3.pt
MAX_VIDEO_DURATION=300     # 5분 제한 (MVP)
MAX_VIDEO_SIZE_MB=200
SAMPLE_FPS=1
CONFIDENCE_THRESHOLD=0.65
MAX_RETRY_COUNT=2
```

---

## Docker Compose

```yaml
services:
  api:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./checkpoints:/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on: [redis]

  worker:
    build: ./backend
    command: celery -A tasks worker --loglevel=info --concurrency=1
    env_file: .env
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./checkpoints:/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on: [redis]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    environment:
      - VITE_API_URL=http://localhost:8000
```

> `--concurrency=1` 필수: SAM3가 GPU 전체를 점유하므로 병렬 영상 처리 불가.

---

## 구현 순서 (MVP)

```
Phase 1 (1~2주)
  □ FastAPI 기본 구조 + 파일 업로드 엔드포인트
  □ SAM3 싱글톤 로드 (startup 이벤트)
  □ frame_extractor 툴 (ffmpeg)
  □ scene_analyzer 툴 (GPT-4o → sam3_text_prompts 생성)

Phase 2 (3~4주)
  □ sam3_segmentor 툴 (텍스트 프롬프트 → 픽셀 마스크)
  □ confidence_checker 툴 + 재탐지 루프
  □ sam3_tracker 툴 (전체 영상 추적)
  □ LangGraph 그래프 완성 + 조건부 엣지

Phase 3 (5~6주)
  □ masking_engine 툴 (SAM3 마스크 위에 blur/blackbox/pixelate)
  □ video_composer 툴 (ffmpeg 합성)
  □ report_generator 툴
  □ Celery + Redis 비동기 작업 큐

Phase 4 (7~8주)
  □ SSE 로그 스트리밍 엔드포인트
  □ React 프론트 (UploadZone, AgentLog, VideoPreview)
  □ 전체 통합 테스트 (480p, 5분 이내 영상)
  □ Docker Compose GPU 배포
  □ 발표 데모 영상 준비
```

---

## 주요 제약 및 주의사항

- **SAM3 가중치**: HuggingFace access request 필요. `ultralytics>=8.3.237` 설치 후 `sam3.pt` 수동 배치.
- **clip 패키지 충돌**: SAM3 사용 시 반드시 `pip uninstall clip -y && pip install git+https://github.com/ultralytics/CLIP.git` 실행.
- **GPU 메모리**: SAM3는 약 8~12GB VRAM 사용. 부족 시 입력 해상도를 640p로 제한.
- **GPT-4o 비용**: scene_analyzer는 대표 프레임 1장만 사용. SAM3가 추적을 담당하므로 프레임별 Vision API 호출 불필요.
- **Celery concurrency=1**: SAM3가 GPU를 점유하므로 병렬 영상 처리 불가. 큐에 순차 처리.
- **ffmpeg**: Docker 이미지에 `RUN apt-get install -y ffmpeg` 포함 필수.
- **MVP 범위**: 200MB 초과 업로드 거부, 5분 초과 영상은 앞 5분만 처리.
