from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status, Form
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import date

from app.schemas.report import ReportCreateResponse
from app.core.enum.report_status import ReportStatus
from app.services import llm_service, report_service
from app.core.database import get_db
from app.utils import file_handler

from app.api.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/llm", tags=["llm"])

@router.post("/create_report", response_model=ReportCreateResponse)
async def create_report(
        # metadata
        patient_name: str = Form(..., description="Patient full name"),
        patient_sex: str = Form(..., description="Sex (Male/Female)"),
        birth_date: date = Form(..., description="Date of birth"),
        ct_date: date = Form(..., description="CT study date"),
        medical_text: str = Form("", description="Medical history + symptoms"),
        # files
        ct_images: UploadFile = File(..., description="ZIP archive with CT images"),
        measurements_file: UploadFile = File(..., description="Measurements file (CSV/JSON)"),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    try:
        meta = {
            "name": patient_name,
            "sex": patient_sex,
            "birth_date": birth_date.isoformat(),
            "ct_date": ct_date.isoformat(),
            "anamnesis": medical_text
        }

        ct_zip_bytes = await ct_images.read()
        measurements_bytes = await measurements_file.read()

        measurements_dict = file_handler.parse_measurements_file(measurements_bytes, measurements_file.filename)

        input_files = {
        "ct_archive_filename": ct_images.filename,
        "measurements_filename": measurements_file.filename,
        }

        llm_response, trace_data = await llm_service.process_llm_request(patient_data=measurements_dict, medical_text=medical_text)

        # Capture warnings/errors from llm_response
        trace_data.update({
            "warnings": llm_response.get("warnings", []),
            "errors": llm_response.get("errors", [])
        })

        id_report = await report_service.save_report(
        db=db,
        measurements=measurements_dict,
        input_files=input_files,
        meta=meta,
        llm_response=llm_response.get("report"),
        trace_data=trace_data,
        user_id=current_user.id,
        )

        return ReportCreateResponse(
            id_report=id_report,
            status=ReportStatus.PROCESSING
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e