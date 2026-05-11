from fastapi import Depends, HTTPException,status
from fastapi.security import OAuth2PasswordBearer

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.enum.role import UserRole
from app.core.security import decode_access_token
from app.models.user import User

oauth_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login")


async def get_current_user(
        token: str = Depends(oauth_scheme),
        db: AsyncSession = Depends(get_db)
) -> User:
    payload = decode_access_token(token)
    login = payload.get("sub")

    if login is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail= "Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
            )
    
    result = await db.execute(select(User).where(User.login == login))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not active",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return current_user
    

def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user
