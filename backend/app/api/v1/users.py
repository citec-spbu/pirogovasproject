from fastapi import APIRouter, Depends, HTTPException,status
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.user import UserCreate, UserLogin
from app.services import user_service
from app.core.database import get_db

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user_data:UserCreate,db: AsyncSession = Depends(get_db)):
    try:
        await user_service.register_user(db,user_data)
        return {"message": "User registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail = str(e))
    
@router.post("/login", status_code=status.HTTP_200_OK)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    try:
        await user_service.login_user(db, user_data)
        return {"message": "User logged in successfully"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))