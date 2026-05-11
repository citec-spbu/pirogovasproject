from sqlalchemy import Column, Integer, String,ForeignKey,DateTime, Enum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.core.database import Base
from app.core.enum.report_status import ReportStatus

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer,primary_key=True, index=True)
    id_report = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer,ForeignKey("users.id"), nullable=False)

    status = Column(Enum(ReportStatus), nullable=False, default=ReportStatus.PROCESSING)
    template_id = Column(Integer, ForeignKey("report_templates.id"), nullable=True)
    error_message = Column(String, nullable=True)

    input_files = Column(JSONB, nullable=False)
    measurements = Column(JSONB, nullable=False)
    meta = Column(JSONB, nullable=False)

    llm_response = Column(JSONB, nullable=True)

    html_object_key = Column(String, nullable=True)
    pdf_object_key = Column(String, nullable=True)

    review_score = Column(Integer, nullable=True)
    review_text = Column(String, nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewer_user_id = Column(Integer,ForeignKey("users.id"), nullable=True)

    generation_started_at = Column(DateTime(timezone=True), nullable=True)
    generation_completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())