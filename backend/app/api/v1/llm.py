from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status, Form
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import date
import logging
from typing import List

from app.schemas.report import ReportCreateResponse
from app.core.enum.report_status import ReportStatus
from app.services import storage_service, report_service
from app.tasks.report_tasks import generate_report_task
from app.core.database import get_db
from app.utils import file_handler

from app.api.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/llm", tags=["llm"])

logger = logging.getLogger(__name__)

async def cleanup_uploaded_objects(object_keys: list[str]) -> None:
    for object_key in object_keys:
        try:
            await storage_service.delete_object(object_key)
        except Exception:
            logger.exception("Failed to delete uploaded CT object during rollback: %s", object_key)

@router.post("/create_report", response_model=ReportCreateResponse)
async def create_report(
        # metadata
        patient_name: str = Form(..., description="Patient full name"),
        patient_sex: str = Form(..., description="Sex (Male/Female)"),
        birth_date: date = Form(..., description="Date of birth"),
        ct_date: date = Form(..., description="CT study date"),
        medical_text: str = Form("", description="Medical history + symptoms"),
        enable_llm_judge: bool = Form(False, description="Run LLM-as-judge after report generation"),
        # files
        ct_images: List[UploadFile] = File(..., description="One ZIP archive or multiple CT images PNG/JPEG"),
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

        allowed_image_extensions = {".png", ".jpg", ".jpeg"}
        allowed_archive_extensions = {".zip"}

        if not ct_images:
            raise ValueError("At least one CT image file or ZIP archive is required")

        ct_files_to_upload = []

        for file in ct_images:
            filename = file.filename or ""
            lower_filename = filename.lower()

            is_image = any(lower_filename.endswith(ext) for ext in allowed_image_extensions)
            is_archive = any(lower_filename.endswith(ext) for ext in allowed_archive_extensions)

            if not is_image and not is_archive:
                raise ValueError("CT files must be PNG, JPG, JPEG or ZIP")

            ct_files_to_upload.append(
                {
                    "file": file,
                    "filename": filename,
                    "kind": "archive" if is_archive else "image",
                }
            )

        archive_count = sum(1 for item in ct_files_to_upload if item["kind"] == "archive")
        image_count = sum(1 for item in ct_files_to_upload if item["kind"] == "image")

        if archive_count > 1:
            raise ValueError("Only one ZIP archive is allowed")

        if archive_count == 1 and image_count > 0:
            raise ValueError("Upload either one ZIP archive or multiple CT images, not both")

        uploaded_object_keys = []
        uploaded_ct_files = []

        for item in ct_files_to_upload:
            object_key = await storage_service.upload_file(
                file=item["file"],
                prefix=f"reports/{current_user.id}/ct_images",
            )
            uploaded_object_keys.append(object_key)

            uploaded_ct_files.append(
                {
                    "filename": item["filename"],
                    "object_key": object_key,
                    "content_type": item["file"].content_type or "application/octet-stream",
                    "kind": item["kind"],
                }
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
            "ct_images": uploaded_ct_files,
            "measurements": {
                "filename": measurements_file.filename,
                "object_key": measurements_object_key,
                "content_type": measurements_file.content_type or "application/octet-stream",
            }
        }

        measurements_dict = file_handler.parse_measurements_file(measurements_bytes, measurements_file.filename)
        
        report, llm_call = await report_service.create_queued_report(
            db=db,
            measurements=measurements_dict,
            input_files = input_files,
            meta = meta,
            user_id = current_user.id,
            judge_enabled = enable_llm_judge,
        )

        generate_report_task.delay(report.id, llm_call.id, enable_llm_judge)

        return ReportCreateResponse(
            id_report=report.id_report,
            status=report.status,
        )
    except ValueError as e:
        await cleanup_uploaded_objects(locals().get("uploaded_object_keys", []))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    except Exception as e:
        await cleanup_uploaded_objects(locals().get("uploaded_object_keys", []))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e