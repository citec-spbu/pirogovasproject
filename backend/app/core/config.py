from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:password@localhost/db"
    SECRET_KEY: str = "who-knows-what-this-is-for"
    ALGORITHM: str = "HS256"

    class Config:
        env_file = ".env"
    
settings = Settings()