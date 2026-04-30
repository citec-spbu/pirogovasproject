from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.admin import AdminCreateUser, AdminUserOut, AdminUpdateUser
from app.services import admin_service
from app.api.dependencies import require_admin
from app.models.user import User

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/create_user", response_model=AdminUserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: AdminCreateUser,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    try:
        user = await admin_service.create_user(db, user_data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return user

@router.patch("/update_user/{user_id}", response_model=AdminUserOut, status_code=status.HTTP_200_OK)
async def update_user(
    user_id: int,
    user_data: AdminUpdateUser,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
    ):
    try:
        user = await admin_service.update_user(db, user_id, user_data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return user