from sqlalchemy import Column, Integer,Text, String, DateTime, Boolean, ForeignKey
from sqlalchemy import Column, Integer, Text, String, DateTime, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from app.core.database import Base

class ReportTemplate(Base):
    __tablename__ = "report_templates"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_report_template_name_version"),)

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    description = Column(String, nullable=True)

    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)

    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True),server_default=func.now(), onupdate=func.now(), nullable=False)