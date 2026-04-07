from pydantic import BaseModel,ConfigDict
from datetime import date

class UserCreate(BaseModel):
    login: str
    password: str
    name: str
    surname: str
    patronymic: str
    date_of_birth: date

class UserLogin(BaseModel):
    login: str
    password: str

class UserOut(BaseModel):
    id: int
    login: str
    name: str
    surname: str
    patronymic: str
    date_of_birth: date

    model_config = ConfigDict(from_attributes=True)