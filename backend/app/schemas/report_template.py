from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class ReportTemplateCreate(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    content: str
    is_active: bool = False
    created_by_user_id: int

class ReportTemplateOut(ReportTemplateCreate):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
