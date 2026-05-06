from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from datetime import date

class Settings(BaseSettings):
    SECRET_KEY: str
    
    FIRST_ADMIN_LOGIN: str
    FIRST_ADMIN_PASSWORD: str
    FIRST_ADMIN_NAME: str
    FIRST_ADMIN_SURNAME: str
    FIRST_ADMIN_PATRONYMIC: str
    FIRST_ORGANIZATION_NAME: str
    FIRST_ADMIN_DATE_OF_BIRTH: date

    DATABASE_URL: str = "postgresql://postgres:password@localhost/db"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    model_config = SettingsConfigDict(env_file=".env")
    

@lru_cache
def get_settings() -> Settings:
    return Settings()