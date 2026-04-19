from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="config/.env", env_file_encoding="utf-8", extra="ignore"
    )
    app_name: str = "Unpaper the Patient"
    log_level: str = "INFO"
    max_upload_mb: int = 20
    gemini_api_key: str = ""


settings = Settings()
