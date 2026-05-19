from pydantic import BaseModel, ConfigDict
from datetime import date
from typing import Optional

from app.core.enum.role import UserRole

class UserLogin(BaseModel):
    login: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserFioRoleOut(BaseModel):
    fio: str
    role: UserRole

class UserShortOut(BaseModel):
    fio: str
    login: str
    organization_name: str

class UserFullOut(BaseModel):
    id: int
    login: str
    role: UserRole
    organization_name: str
    name: str
    surname: str
    patronymic: Optional[str] = None
    date_of_birth: date
    is_active: bool

    model_config = ConfigDict(from_attributes=True)

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str