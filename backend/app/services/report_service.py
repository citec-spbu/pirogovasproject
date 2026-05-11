from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.report import Report
from app.models.user import User
from app.models.llm_calls import LLMCall
from app.schemas.report import ReviewCreate

from app.utils.pdf_generator import generate_pdf_from_html
from app.utils.html_report_generator import generate_html_report
from app.utils.file_handler import save_photo_to_disk
from app.core.enum.report_status import ReportStatus

import uuid

async def save_report(db: AsyncSession, measurements,input_files,meta, llm_response,llm_call_id, trace_data):
    id_report = str(uuid.uuid4())
    report = Report(
        id_report=id_report,
        user_id=None,
        llm_response=llm_response,
        status =ReportStatus.PROCESSING,
        input_files=input_files,
        measurements=measurements,
        meta=meta,
    )
    db.add(report)

    llm_call = await db.execute(select(LLMCall).where(LLMCall.id == llm_call_id))
    llm_call = llm_call.scalar_one_or_none()
    llm_call.trace_json["report_id"] = id_report
    db.add(llm_call)

    await db.commit()
    await db.refresh(report)
    return id_report

async def get_reports_by_login(db: AsyncSession, login: str):
    result = await db.execute(select(User).where(User.login == login))
    user = result.scalar_one_or_none()
    if not user:
        raise ValueError("User not found")
    
    result = await db.execute(select(Report).where(Report.user_id == user.id))
    reports = result.scalars().all()
    return reports

async def generate_report(db: AsyncSession, id_report: str, output_dir: str) -> str:
    result = await db.execute(select(Report).where(Report.id == id_report))
    report = result.scalar_one_or_none()
    if not report:
        raise ValueError("Report not found")
    html_object_key = generate_html_report(report, output_dir)
    pdf_object_key = generate_pdf_from_html(html_object_key, output_dir)
    report.html_object_key = html_object_key
    report.pdf_object_key = pdf_object_key
    await db.commit()
    return html_object_key, pdf_object_key
    
async def add_review(db: AsyncSession, review: ReviewCreate):
    result = await db.execute(select(Report).where(Report.id == review.id_report))
    report = result.scalar_one_or_none()
    if not report:
        raise ValueError("Report not found")
    report.review_score = review.review_score
    report.review_text = review.review_text
    await db.commit()
