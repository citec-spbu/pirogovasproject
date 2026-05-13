from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional
from functools import lru_cache
from datetime import date


class Settings(BaseSettings):
    SECRET_KEY: str

    FIRST_ADMIN_LOGIN: str
    FIRST_ADMIN_PASSWORD: str
    FIRST_ADMIN_NAME: str
    FIRST_ADMIN_SURNAME: str
    FIRST_ADMIN_PATRONYMIC: str
    FIRST_ORGANIZATION_NAME: str
    FIRST_ADMIN_DATE_OF_BIRTH: date

    DATABASE_URL: str = "postgresql://postgres:password@localhost/db"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str
    MINIO_SECURE: bool
    MINIO_PRESIGNED_EXPIRES_SECONDS: int

    GUIDELINE_PATHS: Optional[str] = None

    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100
    CHUNK_SEPARATORS: str = "\n\n,\n,. ,! ,? , , "

    EMBEDDING_MODEL_NAME: str = "DmitryPogrebnoy/MedRuBertTiny2"
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    EMBEDDING_BATCH_SIZE: int = 8
    EMBEDDING_NORMALIZE: bool = True

    TOP_K_RETRIEVAL: int = 4
    TOP_K_CANDIDATES: int = 20
    MIN_RELEVANCE_SCORE: float = 0.0
    RRF_K: float = 60.0
    CROSS_ENCODER_WEIGHT: float = 0.7

    GRAPH_MAX_HOPS: int = 2
    GRAPH_MAX_CHUNKS: int = 8
    # Пути к кешу
    KB_CACHE_ROOT: str = ".kb_cache"
    KB_CACHE_GRAPH_ROOT: str = ".kb_cache_graph"
    KB_CACHE_BM25_ROOT: str = ".kb_cache_bm25"

    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_API_KEY: Optional[str] = None
    VLLM_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"

    model_config = SettingsConfigDict(env_file=".env")

@lru_cache
def get_settings() -> Settings:
    return Settings()
