from fastapi import HTTPException, status

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User, Organization
from app.schemas.admin import AdminCreateUser, AdminUserOut, AdminUpdateUser
from app.core.security import get_password_hash, verify_password
from app.core.enum.role import UserRole

async def create_user(db: AsyncSession, user_data: AdminCreateUser) -> User:
    result = await db.execute(select(User).where(User.login == user_data.login))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Login is already registered"
        )
    
    organization_result = await db.execute( select(Organization).where(Organization.id == user_data.organization_id))
    organization = organization_result.scalar_one_or_none()
    if not organization:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization not found"
        )
    
    hashed_password = get_password_hash(user_data.password)

    new_user = User(
        login=user_data.login,
        hashed_password=hashed_password,
        role = user_data.role,
        organization_id=user_data.organization_id,
        name=user_data.name,
        surname=user_data.surname,
        patronymic=user_data.patronymic,
        date_of_birth=user_data.date_of_birth
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

async def update_user(db: AsyncSession, user_id: int, user_data: AdminUpdateUser):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user_data.organization_id:
        organization_result = await db.execute( select(Organization).where(Organization.id == user_data.organization_id))
        organization = organization_result.scalar_one_or_none()
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization not found"
            )
    
    if user_data.role == UserRole.USER and user.role == UserRole.ADMIN or user_data.is_active == False and user.role == UserRole.ADMIN:
        admins = await db.execute(select(User).where(User.role == UserRole.ADMIN))
        admins = admins.scalars().all()
        if len(admins) == 1:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can assign admin role"
            )
        
    update_dict = user_data.model_dump(exclude_unset=True)

    for key, value in update_dict.items():
        setattr(user, key, value)
    
    await db.commit()
    await db.refresh(user)

    return user
    
    