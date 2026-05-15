from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class ReportTemplateCreate(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    content: str
    is_active: bool = False

class ReportTemplateOut(BaseModel):
    id: int
    name: str
    version: str
    description: Optional[str] = None
    is_active: bool
    created_by_user_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

