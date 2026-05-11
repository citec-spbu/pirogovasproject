from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional
from app.core.enum import ClinicalProtocolStatus

class ClinicalProtocolCreate(BaseModel):
    title: str
    version: Optional[str] = None
    content: Optional[str] = None
    uploaded_by_user_id: int
    file_object_key: Optional[str] = None
    uploaded_by_user_id: int

class ClinicalProtocolOut(ClinicalProtocolCreate):
    id: int
    status: ClinicalProtocolStatus
    error_message: Optional[str] = None
    uploaded_by_user_id: int
    uploaded_at: datetime
    updated_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
