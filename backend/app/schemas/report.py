from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List, Optional
from app.core.enum import ReportStatus

class ReportId(BaseModel):
    id_report: str

class ReportCreate(BaseModel):
    id_report: str
    user_id: int
    template_id: int
    input_file_path: dict
    measurements: dict
    meta: dict

class ReportUpdate(BaseModel):
    user_id: Optional[int] = None
    template_id: Optional[int] = None

    status: Optional[ReportStatus] = None
    error_message: Optional[str] = None

    input_files: Optional[dict] = None
    measurements: Optional[dict] = None
    meta: Optional[dict] = None

    llm_response: Optional[dict] = None

    html_object_key: Optional[str] = None
    pdf_object_key: Optional[str] = None

    review_score: Optional[int] = None
    review_text: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    reviewer_user_id: Optional[int] = None

    generation_started_at: Optional[datetime] = None
    generation_completed_at: Optional[datetime] = None

class ReportOut(BaseModel):
    id_report: str
    user_id: int
    template_id: int
    status: ReportStatus
    input_files: dict
    measurements: dict
    meta: dict
    llm_response: dict
    html_object_key: Optional[str] = None
    pdf_object_key: Optional[str] = None
    review_score: Optional[int] = None
    review_text: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ReportStorageUpdate(BaseModel):
    input_files: Optional[dict] = None
    html_object_key: Optional[str] = None
    pdf_object_key: Optional[str] = None

class ReportsList(BaseModel):
    reports: List[ReportOut]