from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.report import ReportsListResponse, ReviewCreate
from app.services import report_service
from app.core.database import get_db

router = APIRouter(prefix="/reports", tags=["reports"])

@router.get("get_id_reports", response_model=ReportsListResponse)
async def get_id_reports(login: str = Query(...), db: AsyncSession = Depends(get_db)):
    try:
        reports = await report_service.get_reports_by_login(db, login)
        return ReportsListResponse(reports=reports)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail = str(e))
    
@router.get("/make_report")
async def make_report(id_report: str = Query(...), db: AsyncSession = Depends(get_db)):
    try:
        html_path, pdf_path = await report_service.generate_report(db, id_report)
        return {"html_path": html_path, "pdf_path": pdf_path}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
@router.post("/add_review")
async def add_review(review: ReviewCreate, db: AsyncSession = Depends(get_db)):
    try:
        await report_service.add_review(db, review)
        return {"message": "Review added successfully"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))