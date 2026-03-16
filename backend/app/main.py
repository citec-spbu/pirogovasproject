from fastapi import FastAPI
from app.api.v1 import users
from app.api.v1 import reports
from app.api.v1 import llm

app = FastAPI(title="CT AI Analysis API")

app.include_router(users.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(llm.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the CT AI Analysis API!"}