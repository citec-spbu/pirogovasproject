from pydantic import BaseModel,ConfigDict
from datetime import date
from backend.app.core.role import UserRole

class AdminCreateUser(BaseModel):
    login: str
    password: str
    role: UserRole
    organization_id: int
    name: str
    surname: str
    patronymic: str
    date_of_birth: date

class AdminUpdateUser(BaseModel):
    role: UserRole | None = None
    name: str | None = None
    surname: str | None = None
    organization_id: int | None = None
    patronymic: str | None = None
    date_of_birth: date | None = None
    is_active: bool | None = None


class AdminUserOut(BaseModel):
    id: int
    login: str
    role: UserRole
    organization_id: int
    name: str
    surname: str
    patronymic: str
    date_of_birth: date
    is_active: bool

    model_config = ConfigDict(from_attributes=True)