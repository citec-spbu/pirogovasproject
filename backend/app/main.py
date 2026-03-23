from fastapi import FastAPI
from app.api.v1 import users
from app.api.v1 import reports
from app.api.v1 import llm
from contextlib import asynccontextmanager
from app.core.database import engine, Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="CT AI Analysis API")

app.include_router(users.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(llm.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the CT AI Analysis API!"}