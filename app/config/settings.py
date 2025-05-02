import os
from typing import Optional, List, Dict, Any

# Import BaseSettings from pydantic-settings package as required in Pydantic v2
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "New Energy Intelligent Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API settings
    API_PREFIX: str = "/api"

    # File paths
    UPLOAD_DIR: str = "app/static/uploads"
    EXPORT_DIR: str = "app/static/exports"
    TEMPLATE_DIR: str = "app/static/templates"

    # Vector database settings
    VECTOR_DB_PATH: str = "app/data/vector_db"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # LLM settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    # RAG settings
    DEFAULT_RETRIEVAL_STRATEGY: str = "hybrid"
    DEFAULT_VECTOR_SIMILARITY_THRESHOLD: float = 0.7
    DEFAULT_MAX_CHUNKS_PER_STRATEGY: int = 5
    DEFAULT_RERANKING_ENABLED: bool = True

    # Content generation settings
    DEFAULT_MAX_TOKENS: int = 1000
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_INCLUDE_SOURCES: bool = True

    # Domain-specific settings
    NEW_ENERGY_GLOSSARY_PATH: str = "app/data/new_energy_glossary.json"

    # Security settings
    CORS_ORIGINS: List[str] = ["*"]

    # Use SettingsConfigDict instead of inner Config class
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


settings = Settings()