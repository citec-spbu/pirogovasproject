from pydantic import BaseModel,ConfigDict
from datetime import date
from app.core.enum.role import UserRole
from typing import Optional

class AdminCreateUser(BaseModel):
    login: str
    password: str
    role: UserRole
    organization_name: str
    name: str
    surname: str
    patronymic: str
    date_of_birth: date

class AdminUpdateUser(BaseModel):
    role: Optional[UserRole] = None
    name: Optional[str] = None
    surname: Optional[str] = None
    organization_name: Optional[str] = None
    patronymic: Optional[str] = None
    date_of_birth: Optional[date] = None
    is_active: Optional[bool] = None


class AdminUserOut(BaseModel):
    id: int
    login: str
    role: UserRole
    organization_name: str
    name: str
    surname: str
    patronymic: str
    date_of_birth: date
    is_active: bool

    model_config = ConfigDict(from_attributes=True)

class AdminMetricsOut(BaseModel):
    llm_calls_total: int
    llm_calls_failed: int
    llm_error_percent: float
    reviewed_reports_total: int
    average_review_score: Optional[float] = None