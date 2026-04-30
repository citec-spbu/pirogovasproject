from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List, Optional

class ReportIdResponse(BaseModel):
    id_report: str

class ReportOut(BaseModel):
    id_report: str
    user_id: int
    path_to_photo: str
    measurements: dict
    meta: dict
    llm_response: dict
    trace_data: Optional[dict] = None
    review_score: Optional[int] = None
    review_text: Optional[str] = None
    html_path: Optional[str] = None
    pdf_path: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ReportsListResponse(BaseModel):
    reports: List[ReportOut]

class ReviewCreate(BaseModel):
    id_report: str
    review_score: int
    review_text: Optional[str] = None