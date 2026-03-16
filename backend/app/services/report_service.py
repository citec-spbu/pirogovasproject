from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.report import Report
from app.models.user import User
from app.schemas.report import ReviewCreate
from app.utils.pdf_generator import generate_pdf_from_html
from app.utils.html_report_generator import generate_html_report
import uuid
from app.utils.file_handler import save_photo_to_disk

async def save_report(db: AsyncSession, measurements,path_to_photo,metadata, llm_response, trace_data):
    id_report = str(uuid.uuid4())
    report = Report(
        id_report=id_report,
        user_id=None,
        path_to_photo=path_to_photo,
        measurements=measurements,
        metadata=metadata,
        llm_response=llm_response,
        trace_data=trace_data
    )

    db.add(report)
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
    html_path = generate_html_report(report, output_dir)
    pdf_path = generate_pdf_from_html(html_path, output_dir)
    report.html_path = html_path
    report.pdf_path = pdf_path
    await db.commit()
    return html_path, pdf_path
    
async def save_review(db: AsyncSession, review: ReviewCreate):
    result = await db.execute(select(Report).where(Report.id == review.id_report))
    report = result.scalar_one_or_none()
    if not report:
        raise ValueError("Report not found")
    report.review_score = review.review_score
    report.review_comment = review.review_comment
    await db.commit()
