from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status, Form
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import date
from fastapi import BackgroundTasks
from app.services.llm_judge_runner import run_llm_judge_for_report

from app.schemas.report import ReportCreateResponse
from app.core.enum.report_status import ReportStatus
from app.services import llm_service, report_service, storage_service
from app.core.database import get_db
from app.utils import file_handler

from app.api.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/llm", tags=["llm"])

@router.post("/create_report", response_model=ReportCreateResponse)
async def create_report(
        # metadata
        background_tasks: BackgroundTasks,
        patient_name: str = Form(..., description="Patient full name"),
        patient_sex: str = Form(..., description="Sex (Male/Female)"),
        birth_date: date = Form(..., description="Date of birth"),
        ct_date: date = Form(..., description="CT study date"),
        medical_text: str = Form("", description="Medical history + symptoms"),
        enable_llm_judge: bool = Form(False, description="Run LLM-as-judge after report generation"),
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

        ct_images_object_key = await storage_service.upload_file(
            file=ct_images,
            prefix=f"reports/{current_user.id}/ct_images",
        )

        measurements_bytes = await measurements_file.read()
        measurements_object_key = storage_service.build_object_key(
            prefix=f"reports/{current_user.id}/measurements",
            filename=measurements_file.filename,
        )
        await storage_service.upload_bytes(
            data=measurements_bytes,
            object_key=measurements_object_key,
            content_type=measurements_file.content_type or "application/octet-stream",
        )

        input_files  = {
            "ct_images": {
                "filename": ct_images.filename,
                "object_key": ct_images_object_key,
                "content_type": ct_images.content_type or "application/zip",
            },
            "measurements": {
                "filename": measurements_file.filename,
                "object_key": measurements_object_key,
                "content_type": measurements_file.content_type or "application/octet-stream",
            }
        }

        measurements_dict = file_handler.parse_measurements_file(measurements_bytes, measurements_file.filename)

        id_report = await report_service.create_report_entry(db=db, measurements=measurements_dict, input_files=input_files,
                                meta=meta, user_id=current_user.id, judge_enabled=enable_llm_judge)

        llm_response, trace_data = await llm_service.process_llm_request(patient_data=measurements_dict, medical_text=medical_text)
        trace_data.update({
            "warnings": llm_response.get("warnings", []),
            "errors": llm_response.get("errors", [])
        })

        await report_service.save_report_after_answer(db=db, id_report=id_report, llm_response=llm_response.get("report"), trace_data=trace_data)

        if enable_llm_judge:
            background_tasks.add_task(run_llm_judge_for_report, id_report, current_user.id)

        return ReportCreateResponse(
            id_report=id_report,
            status=ReportStatus.PROCESSING
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
