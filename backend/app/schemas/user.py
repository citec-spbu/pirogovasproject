from pydantic import BaseModel,ConfigDict
from datatime import date

class UserCreate(BaseModel):
    login: str
    password: str
    name: str
    surname: str
    patronymic: str
    date_of_birth: date

class UserOut(BaseModel):
    id: int
    login: str
    name: str
    surname: str
    patronymic: str
    date_of_birth: date

    model_config = ConfigDict(from_attributes=True)