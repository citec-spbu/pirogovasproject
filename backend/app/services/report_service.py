from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List, Optional

from app.models.report import Report
from app.models.user import User
from app.models.llm_calls import LLMCall
from app.core.enum.report_status import ReportStatus
from app.schemas.report import ReportReviewUpdate
from app.core.enum.call_type import CallType, CallStatus
from app.core.config import get_settings
from app.services import storage_service

from app.utils.pdf_generator import generate_pdf_from_html
from app.utils.html_report_generator import generate_html_report

async def create_queued_report(
    db: AsyncSession,
    measurements: dict,
    input_files: dict,
    meta: dict,
    user_id: int,
    template_id: Optional[int] = None,
) -> tuple[Report, LLMCall]:
    settings = get_settings()
    
    id_report = str(uuid.uuid4())
    report = Report(
        id_report=id_report,
        user_id=user_id,
        status =ReportStatus.PROCESSING,
        input_files=input_files,
        measurements=measurements,
        meta=meta,
        llm_response=None,
    )
    db.add(report)
    await db.flush()

    llm_call = LLMCall(
        report_id=report.id,
        user_id=user_id,
        status=CallStatus.QUEUED,
        call_type=CallType.REPORT_GENERATION,
        provider="vllm",
        model=settings.VLLM_MODEL,
        prompt=meta.get("anamnesis", ""),
        template_id=template_id,
        input_json={
            "measurements": measurements,
            "meta": meta,
        },
    )
    db.add(llm_call)

    await db.commit()
    await db.refresh(report)
    await db.refresh(llm_call)

    return report, llm_call

async def get_report_by_id(db:AsyncSession, id_report:str) -> Report:
    result = await db.execute(select(Report).where(Report.id_report == id_report))
    report = result.scalar_one_or_none()

    if not report:
        raise ValueError("Report not found")

    return report

async def get_reports_by_login(db: AsyncSession, login: str):
    result = await db.execute(select(User).where(User.login == login))
    user = result.scalar_one_or_none()
    if not user:
        raise ValueError("User not found")
    
    result = await db.execute(select(Report).where(Report.user_id == user.id))
    reports = result.scalars().all()
    return reports

async def get_reports_by_user_id(
    db: AsyncSession,
    user_id: int,
) -> List[Report]:
    result = await db.execute(select(Report).where(Report.user_id == user_id))
    result = result.scalars().all()
    return result

async def render_and_store_report_files(
    db: AsyncSession,
    report: Report,
) -> tuple[str, str]:

    html_content = generate_html_report(report)
    html_object_key = await storage_service.upload_text(
        text=html_content,
        prefix=f"reports/{report.id_report}/result",
        filename="report.html",
        content_type="text/html; charset=utf-8",
    )

    pdf_content = generate_pdf_from_html(html_content)
    pdf_object_key = await storage_service.upload_bytes_file(
        data=pdf_content,
        prefix=f"reports/{report.id_report}/result",
        filename="report.pdf",
    )

    report.html_object_key = html_object_key
    report.pdf_object_key = pdf_object_key

    await db.flush()

    return html_object_key, pdf_object_key
    
async def add_review(db: AsyncSession, review: ReportReviewUpdate, id_report: str):
    result = await db.execute(select(Report).where(Report.id_report == id_report))
    report = result.scalar_one_or_none()
    if not report:
        raise ValueError("Report not found")
    report.review_score = review.review_score
    report.review_text = review.review_text
    await db.commit()
