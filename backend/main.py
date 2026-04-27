import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .agent.log_emitter import _logs_path, _status_path, read_status
from .agent.runner import run_agent_job
from .config import settings
from .models.sam3_loader import get_load_error, is_available, load_sam3
from .schemas import JobCreateResponse, JobStatusResponse, ReportResponse

app = FastAPI(title="Privacy Guard Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_models() -> None:
    load_sam3()


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip() or 0.0)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "upload_dir": str(settings.upload_path),
        "output_dir": str(settings.output_path),
        "sam3_loaded": is_available(),
        "sam3_error": get_load_error(),
    }


@app.post("/api/jobs", response_model=JobCreateResponse)
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> JobCreateResponse:
    if not file.filename:
        raise HTTPException(422, "filename missing")

    suffix = Path(file.filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise HTTPException(422, f"unsupported extension: {suffix}")

    job_id = str(uuid.uuid4())
    job_dir = settings.upload_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / f"input{suffix}"

    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
                f.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"file too large (>{settings.MAX_VIDEO_SIZE_MB} MB)")
            f.write(chunk)

    try:
        duration = _probe_duration(dest)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(422, f"could not probe video duration: {e}")

    if duration > settings.MAX_VIDEO_DURATION:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            413, f"video too long ({duration:.1f}s > {settings.MAX_VIDEO_DURATION}s)"
        )

    background_tasks.add_task(run_agent_job, job_id, str(dest))
    return JobCreateResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_status(job_id: str) -> JobStatusResponse:
    status, error = read_status(job_id)
    return JobStatusResponse(job_id=job_id, status=status, error=error)


@app.get("/api/jobs/{job_id}/stream")
async def stream_logs(job_id: str) -> StreamingResponse:
    logs_path = _logs_path(job_id)
    status_path = _status_path(job_id)

    async def event_generator():
        sent = 0
        idle_ticks = 0
        while True:
            if logs_path.exists():
                lines = logs_path.read_text(encoding="utf-8").splitlines()
                for line in lines[sent:]:
                    yield f"data: {line}\n\n"
                sent = len(lines)
            status = "pending"
            if status_path.exists():
                status = status_path.read_text(encoding="utf-8").splitlines()[0].strip()
            if status in ("done", "failed"):
                yield f'data: {json.dumps({"event": status})}\n\n'
                break
            idle_ticks += 1
            if idle_ticks > 600:
                yield 'data: {"event": "timeout"}\n\n'
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/report", response_model=ReportResponse)
def get_report(job_id: str) -> ReportResponse:
    path = settings.output_path / job_id / "report.json"
    if not path.exists():
        raise HTTPException(404, "report not ready")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ReportResponse(
        job_id=data.get("job_id", job_id),
        scene_type=data.get("scene_type"),
        total_objects=data.get("total_objects", 0),
        by_type=data.get("by_type", {}),
        detected_objects=data.get("detected_objects", []),
    )


@app.get("/api/jobs/{job_id}/download")
def download_video(job_id: str):
    path = settings.output_path / job_id / "output.mp4"
    if not path.exists():
        raise HTTPException(404, "output video not ready")
    return FileResponse(str(path), media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/api/jobs/{job_id}/report.pdf")
def download_report_pdf(job_id: str):
    path = settings.output_path / job_id / "report.pdf"
    if not path.exists():
        raise HTTPException(404, "report.pdf not ready")
    return FileResponse(
        str(path),
        media_type="application/pdf",
        filename=f"{job_id}-report.pdf",
    )
