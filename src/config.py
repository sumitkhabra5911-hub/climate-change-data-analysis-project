"""
Enterprise Configuration for Climate Intelligence Platform.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "TerraPulse Climate Intelligence API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8008"))
    CORS_ORIGINS: list[str] = ["*"]
    
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_PATH: str = os.path.join(BASE_DIR, "data", "climate_data_cleaned.csv")
    MODELS_DIR: str = os.path.join(BASE_DIR, "outputs", "models")
    
    model_config = SettingsConfigDict(case_sensitive=True)

settings = Settings()
