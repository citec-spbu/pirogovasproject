from fastapi import FastAPI
from app.api.v1 import users

app = FastAPI(title="CT AI Analysis API")

app.include_router(users.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the CT AI Analysis API!"}