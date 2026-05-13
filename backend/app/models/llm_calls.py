from sqlalchemy import Column, DateTime, ForeignKey, Integer, Boolean, String,Enum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from backend.app.core.database import Base
from backend.app.core.enum.call_type import CallType, CallStatus

class LLMCall(Base):
    __tablename__ = "llm_calls"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    status = Column(Enum(CallStatus), nullable=False, default=CallStatus.QUEUED)
    call_type = Column(Enum(CallType), nullable=False)

    is_judge = Column(Boolean, nullable=False, default=False, server_default="false")

    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    prompt = Column(String, nullable=False)
    template_id = Column(Integer, ForeignKey("report_templates.id"), nullable=True)

    input_json = Column(JSONB, nullable=True)
    error_message = Column(String, nullable=True)
    output_json = Column(JSONB, nullable=True)
    trace_json = Column(JSONB, nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


