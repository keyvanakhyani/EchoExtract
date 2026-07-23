"""Central configuration for EchoExtract using pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Values can be overridden via environment variables (prefixed with
    ECHO_) or a .env file. For example, ECHO_MODEL_SIZE=small.
    """

    model_config = SettingsConfigDict(
        env_prefix="ECHO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Whisper model settings
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 5

    # Where HuggingFace stores downloaded models
    hf_home: Path = Path(r"D:\whisper_models\huggingface")

    # Default output formats when none are specified
    default_formats: list[str] = ["txt", "srt", "vtt", "json"]

    # Default output directory (None means "next to the input video")
    default_output_dir: Path | None = None


# A single shared instance imported across the app.
settings = Settings()