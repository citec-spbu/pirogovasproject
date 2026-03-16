from sqlalchemy import AsyncSession
from app.models.report import Report
import uuid
from app.utils.file_handler import save_photo_to_disk

async def save_diagnosis_report(db: AsyncSession, measurements,photo_path,metadata, llm_response, trace_data):
    id_report = str(uuid.uuid4())
    report = Report(
        id=id_report,
        user_id=None,
        photo_path=photo_path,
        measurements=measurements,
        metadata=metadata,
        llm_response=llm_response,
        trace_data=trace_data
    )

    db.add(report)
    await db.commit()
    await db.refresh(report)
    return id_report

