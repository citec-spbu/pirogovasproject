from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.schemas.user import UserCreate
from app.core.security import get_password_hash

async def register_user(db: AsyncSession, user_data: UserCreate) -> User:
    # Check if the login is already registered
    result = await db.execute(select(User).where(User.login == user_data.login))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise ValueError("Email is already registered")

    # Hash the password
    hashed_password = get_password_hash(user_data.password)

    # Create a new user instance
    new_user = User(
        login=user_data.login,
        hashed_password=hashed_password,
        name=user_data.name,
        surname=user_data.surname,
        patronymic=user_data.patronymic,
        date_of_birth=user_data.date_of_birth
    )

    # Add the new user to the database
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user