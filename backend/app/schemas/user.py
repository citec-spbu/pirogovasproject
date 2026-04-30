from pydantic import BaseModel

class UserLogin(BaseModel):
    login: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"