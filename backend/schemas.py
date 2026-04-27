from typing import List, Literal, Optional
from pydantic import BaseModel


class PIIObjectSchema(BaseModel):
    type: Literal["face", "document", "screen", "nameplate", "id_card"]
    bbox: List[int]
    confidence: float
    mask_strategy: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    error: Optional[str] = None


class ReportResponse(BaseModel):
    job_id: str
    scene_type: Optional[str] = None
    total_objects: int
    by_type: dict
    detected_objects: List[PIIObjectSchema]
