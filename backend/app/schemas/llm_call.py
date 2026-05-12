from pydantic import BaseModel, ConfigDict
from app.core.enum.call_type import CallType, CallStatus
from typing import Optional
from datetime import datetime

class LLMCallCreate(BaseModel):
    report_id: int
    call_type: CallType
    provider: str
    model: str
    prompt: str
    template_id: Optional[int] = None
    input_json: Optional[dict] = None

class LLMCallFinish(BaseModel):
    output_json: Optional[dict] = None
    trace_json: Optional[dict] = None

class LLMCallFail(BaseModel):
    error_message: str
    trace_json: Optional[dict] = None

class LLMCallOut(LLMCallCreate):
    status: CallStatus
    output_json: Optional[dict] = None
    trace_json: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


