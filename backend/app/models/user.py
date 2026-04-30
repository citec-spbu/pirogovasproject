from sqlalchemy import Column, Integer, String, Date, Boolean,ForeignKey, Enum
from app.core.database import Base
from enum import Enum
from backend.app.core.role import UserRole

class Organization(Base):
    __tablename__ = "organizations"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(Date, nullable=False)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    role = Column(Enum(UserRole, name= "user_role"), nullable=False, default=UserRole.USER)

    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    login = Column(String, unique=True,index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    surname = Column(String, nullable=False)
    patronymic = Column(String, nullable=True)
    date_of_birth = Column(Date, nullable=False)
    is_active = Column(Boolean, default=True)

