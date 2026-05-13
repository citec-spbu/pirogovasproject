from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List

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

async def save_report(db: AsyncSession, measurements,input_files,meta, llm_response,trace_data, user_id: int, judge_enabled: bool = False,):
    
    settings = get_settings()
    
    id_report = str(uuid.uuid4())
    report = Report(
        id_report=id_report,
        user_id=user_id,
        llm_response=llm_response,
        status =ReportStatus.PROCESSING,
        input_files=input_files,
        measurements=measurements,
        meta=meta,
        judge_enabled=judge_enabled,
        judge_status="queued" if judge_enabled else None,
    )

    db.add(report)
    await db.flush()


    has_errors = bool((trace_data or {}).get("errors"))
    llm_call = LLMCall(
        report_id=report.id,
        user_id=user_id,
        status=CallStatus.FAILED if has_errors else CallStatus.COMPLETED,
        call_type=CallType.REPORT_GENERATION,
        provider="vllm",
        model=settings.VLLM_MODEL,
        prompt=meta.get("anamnesis", ""),
        input_json={
            "measurements": measurements,
            "meta": meta,
        },
        output_json=llm_response,
        trace_json=trace_data,
    )
    db.add(llm_call)

    await db.commit()
    await db.refresh(report)
    await db.refresh(llm_call)

    return id_report

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

async def generate_report(db: AsyncSession, id_report: str, output_dir: str = "reports") -> str:
    result = await db.execute(select(Report).where(Report.id_report == id_report))
    report = result.scalar_one_or_none()

    if not report:
        raise ValueError("Report not found")

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
    report.status = ReportStatus.COMPLETED

    await db.commit()
    await db.refresh(report)

    return html_object_key, pdf_object_key
    
async def add_review(db: AsyncSession, review: ReportReviewUpdate, id_report: str):
    result = await db.execute(select(Report).where(Report.id_report == id_report))
    report = result.scalar_one_or_none()
    if not report:
        raise ValueError("Report not found")
    report.review_score = review.review_score
    report.review_text = review.review_text
    await db.commit()
