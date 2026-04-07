from sqlalchemy import Column, Integer, String,JSON,ForeignKey,DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer,primary_key=True, index=True)
    id_report = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer,ForeignKey("users.id"), nullable=False)
    path_to_photo = Column(String, nullable=False)
    measurements = Column(JSON, nullable=False)
    meta = Column(JSON, nullable=False)
    llm_response = Column(JSON, nullable=False)
    trace_data = Column(JSON, nullable=True)
    review_score = Column(Integer, nullable=True)
    review_text = Column(String, nullable=True)
    html_path = Column(String, nullable=True)
    pdf_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())