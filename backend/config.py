from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENAI_API_KEY: str = ""
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    MAX_VIDEO_DURATION: int = 300
    MAX_VIDEO_SIZE_MB: int = 200
    SAMPLE_FPS: int = 1
    CONFIDENCE_THRESHOLD: float = 0.7
    MAX_RETRY_COUNT: int = 2

    @property
    def upload_path(self) -> Path:
        return Path(self.UPLOAD_DIR).resolve()

    @property
    def output_path(self) -> Path:
        return Path(self.OUTPUT_DIR).resolve()


settings = Settings()
settings.upload_path.mkdir(parents=True, exist_ok=True)
settings.output_path.mkdir(parents=True, exist_ok=True)
