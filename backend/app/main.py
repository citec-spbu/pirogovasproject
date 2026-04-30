from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.v1 import users
from app.api.v1 import reports
from app.api.v1 import llm
from app.core.database import engine
from app.services.bootstrap_service import bootstrap

@asynccontextmanager
async def lifespan(app: FastAPI):
    await bootstrap()
    yield
    await engine.dispose()

app = FastAPI(title="CT AI Analysis API", lifespan=lifespan)

app.include_router(users.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(llm.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the CT AI Analysis API!"}