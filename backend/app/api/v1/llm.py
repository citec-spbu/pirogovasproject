from fastapi import APIRouter, File, UploadFile, Depends, HTTPException,status, Form
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.report import ReportIdResponse
from app.services import llm_service, report_service
from app.core.database import get_db
from app.utils import file_handler
from datetime import date
from typing import Optional

router = APIRouter(prefix="llm", tags=["llm"])

@router.post("/create_report", response_model=ReportIdResponse)
async def create_report(
    # metadata
    patient_name: str = Form(..., description="Patient full name / Code"),
    patient_sex: str = Form(..., description="Sex (Male/Female)"),
    birth_date: date = Form(..., description="Date of birth"),
    ct_date: date = Form(..., description="CT study date"),
    anamnesis: Optional[str] = Form("", description="Medical history"),
    # files
    ct_images: UploadFile = File(..., description="ZIP archive with CT images"),
    measurements_file: UploadFile = File(..., description="Measurements file (CSV/JSON)"),
    db: AsyncSession = Depends(get_db)
):
    try:
        metadata = {
                    "name": patient_name,
                    "sex": patient_sex,
                    "birth_date": birth_date.isoformat(),
                    "ct_date": ct_date.isoformat(),
                    "anamnesis": anamnesis
                }
        

        ct_zip_bytes = await ct_images.read()
        measurements_bytes = await measurements_file.read()


        measurements_dict = file_handler.parse_measurements_file(measurements_bytes, measurements_file.filename)


        photo_path = file_handler.extract_images_from_zip(ct_zip_bytes, ct_images.filename)


        llm_response, trace_data = await llm_service.process_llm_request(measurements_dict)
        

        id_report = await report_service.save_report(
            db=db,
            measurements=measurements_dict,
            photo_path=photo_path,
            metadata=metadata,
            llm_response=llm_response,
            trace_data=trace_data
        )
        return ReportIdResponse(id=id_report)
    except Exception as e:
        raise HTTPException (status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))