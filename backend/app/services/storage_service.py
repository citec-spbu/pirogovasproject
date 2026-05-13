from datetime import timedelta
from io import BytesIO
from uuid import uuid4
import logging

from fastapi import UploadFile
from minio import Minio
from minio.error import S3Error
from starlette.concurrency import run_in_threadpool

from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

def get_minio_client() -> Minio:
    settings = get_settings()
    return Minio(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

def ensure_bucket_exists_sync() -> None:
    settings = get_settings()
    client = get_minio_client()

    if not client.bucket_exists(settings.MINIO_BUCKET_NAME):
        client.make_bucket(settings.MINIO_BUCKET_NAME)

async def ensure_bucket_exists() -> None:
    await run_in_threadpool(ensure_bucket_exists_sync)


def build_object_key(prefix: str, filename: Optional[str] = None) -> str:
    safe_filename = filename or "file"
    return f"{prefix.rstrip('/')}/{uuid4()}_{safe_filename}"

def upload_bytes_sync(
        data: bytes,
        object_key: str,
        content_type: Optional[str] = "application/octet-stream"
) -> str:
    settings = get_settings()
    client = get_minio_client()


    client.put_object(
        bucket_name = settings.MINIO_BUCKET_NAME,
        object_name = object_key,
        data= BytesIO(data),
        length=len(data),
        content_type=content_type,
    )

    return object_key

async def upload_bytes(
        data: bytes,
        object_key: str,
        content_type: Optional[str] = "application/octet-stream"
) -> str:
    return await run_in_threadpool(
        upload_bytes_sync,
        data,
        object_key,
        content_type,
    )

async def upload_file(
        file: UploadFile,
        prefix: str,
) -> str:
    data = await file.read()
    object_key = build_object_key(prefix=prefix, filename=file.filename)

    return await upload_bytes(
        data=data,
        object_key=object_key,
        content_type=file.content_type
    )

async def upload_text(
    text: str,
    prefix: str,
    filename: str,
    content_type: str = "text/plain; charset=utf-8",
) -> str:
    data = text.encode("utf-8")
    object_key = build_object_key(prefix=prefix, filename=filename)

    return await upload_bytes(
        data=data,
        object_key=object_key,
        content_type=content_type
    )

async def upload_bytes_file(
    data: bytes,
    prefix: str,
    filename: str,
    content_type: str = "application/octet-stream"
) -> str:
    object_key = build_object_key(prefix=prefix, filename=filename)

    return await upload_bytes(
        data=data,
        object_key=object_key,
        content_type=content_type
)

def get_object_bytes_sync(object_key: str) -> bytes:
    settings = get_settings()
    client = get_minio_client()

    response = client.get_object(
        bucket_name=settings.MINIO_BUCKET_NAME,
        object_name=object_key,
    )

    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()

async def get_object_bytes(object_key: str) -> bytes:
    return await run_in_threadpool(
        get_object_bytes_sync,
        object_key,
    )

def delete_object_sync(object_key: str) -> None:
    settings = get_settings()
    client = get_minio_client()

    try:
        client.remove_object(
            bucket_name=settings.MINIO_BUCKET_NAME,
            object_name=object_key,
        )
    except S3Error as e:
        logger.error(f"Error deleting object {object_key} from MinIO: {e}", exc_info=True)
        raise RuntimeError(f"Failed to delete object {object_key} from storage") from e

async def delete_object(object_key: str) -> None:
    return await run_in_threadpool(
        delete_object_sync,
        object_key,
    )

def get_presigned_url_sync(object_key: str) -> str:
    settings = get_settings()
    client = get_minio_client()

    return client.presigned_get_object(
        bucket_name=settings.MINIO_BUCKET_NAME,
        object_name=object_key,
        expires=timedelta(seconds=settings.MINIO_PRESIGNED_EXPIRES_SECONDS),
    )

async def get_presigned_url(object_key: str) -> str:
    return await run_in_threadpool(
        get_presigned_url_sync,
        object_key,
    )