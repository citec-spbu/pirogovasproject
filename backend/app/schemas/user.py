from pydantic import BaseModel,ConfigDict
from datetime import date
from backend.app.core.role import UserRole

class UserCreate(BaseModel):
    login: str
    password: str
    role: UserRole
    name: str
    surname: str
    organization_id: int
    patronymic: str
    date_of_birth: date

class UserOut(BaseModel):
    id: int
    login: str
    role: UserRole
    name: str
    surname: str
    organization_id: int
    patronymic: str
    date_of_birth: date

    model_config = ConfigDict(from_attributes=True)

class UserLogin(BaseModel):
    login: str
    password: str