from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.admin import AdminCreateUser, AdminUserOut, AdminUpdateUser
from app.services import admin_service

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/create_user", response_model=AdminUserOut, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: AdminCreateUser):
    try:
        user = await admin_service.create_user(user_data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    user_out = AdminUserOut(user)
    return user_out

@router.patch("/update_user/{user_id}", response_model=AdminUserOut, status_code=status.HTTP_200_OK)
async def update_user(user_id: int, user_data: AdminUpdateUser):
    try:
        user = await admin_service.update_user(user_id, user_data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    user_out = AdminUserOut(user)
    return user_out