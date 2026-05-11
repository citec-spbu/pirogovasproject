from sqlalchemy import Column, Integer, String, Text, DateTime, Enum, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base
from app.core.enum.clinical_protocol_status import ClinicalProtocolStatus

class ClinicalProtocol(Base):
    __tablename__ = "clinical_protocols"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    version = Column(String, nullable=True)

    content = Column(Text, nullable=True)
    file_object_key = Column(String, nullable=True)

    status = Column(Enum(ClinicalProtocolStatus), nullable=False, default=ClinicalProtocolStatus.UPLOADED)
    error_message = Column(String, nullable=True)

    uploaded_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    indexed_at = Column(DateTime(timezone=True), nullable=True)