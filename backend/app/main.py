from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(users.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(llm.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the CT AI Analysis API!"}