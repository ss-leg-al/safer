import json
from pathlib import Path
from typing import Any

from ..config import settings


def _logs_path(job_id: str) -> Path:
    return settings.upload_path / job_id / "logs.jsonl"


def _status_path(job_id: str) -> Path:
    return settings.upload_path / job_id / "status.txt"


def emit_log(job_id: str, payload: dict) -> None:
    path = _logs_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_default) + "\n")


def write_status(job_id: str, status: str, error: str | None = None) -> None:
    path = _status_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = status if not error else f"{status}\n{error}"
    path.write_text(payload, encoding="utf-8")


def read_status(job_id: str) -> tuple[str, str | None]:
    path = _status_path(job_id)
    if not path.exists():
        return ("pending", None)
    raw = path.read_text(encoding="utf-8").strip().split("\n", 1)
    status = raw[0]
    error = raw[1] if len(raw) > 1 else None
    return (status, error)


def _default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
