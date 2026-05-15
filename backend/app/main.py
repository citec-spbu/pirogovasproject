from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.v1 import users, admin, reports, llm

from app.core.database import engine
from app.services.bootstrap_service import bootstrap
from app.services.storage_service import ensure_bucket_exists

@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_bucket_exists()
    await bootstrap()
    yield
    await engine.dispose()

app = FastAPI(title="CT AI Analysis API", lifespan=lifespan)

app.include_router(users.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(llm.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the CT AI Analysis API!"}