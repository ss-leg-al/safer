from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class JobStore:
    job_id: str
    video_path: str | None = None
    frames_dir: str | None = None
    sample_frame: str | None = None
    scene_type: str | None = None
    expected_pii: list[str] = field(default_factory=list)
    detected_objects: list[dict] = field(default_factory=list)
    per_frame_bboxes: dict[str, list[dict]] = field(default_factory=dict)
    masked_frames_dir: str | None = None
    output_video_path: str | None = None
    report: dict | None = None
    detect_attempts: int = 0


_stores: dict[str, JobStore] = {}
_lock = threading.Lock()


def get_store(job_id: str) -> JobStore:
    with _lock:
        if job_id not in _stores:
            _stores[job_id] = JobStore(job_id=job_id)
        return _stores[job_id]


def reset_store(job_id: str) -> None:
    with _lock:
        _stores.pop(job_id, None)
