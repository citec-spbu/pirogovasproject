from fastapi import APIRouter, Depends, HTTPException,status
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.user import UserLogin
from app.services import user_service
from app.core.database import get_db
from app.core.security import create_access_token

router = APIRouter(prefix="/users", tags=["users"])
   
@router.post("/login", status_code=status.HTTP_200_OK)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    try:
        await user_service.login_user(db, user_data)
        data_for_token = {"sub": user_data.login, "role": user_data.role, "organization_id": user_data.organization_id}
        jwt_token = create_access_token(data_for_token)
        return { "access_token": jwt_token, "token_type": "bearer" }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e), headers={"WWW-Authenticate": "Bearer"})