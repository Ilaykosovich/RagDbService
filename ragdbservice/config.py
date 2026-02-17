from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    pg_url: str
    text2sql_db_url: str
    chroma_persist_dir: str = "./chroma_db"
    chroma_history_collection: str = "chroma_history_collection"
    chroma_collection: str = "pg_schema"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    statement_timeout_seconds: int = 30

    class Config:
        env_file = ".env"
        env_prefix = ""


settings = AppSettings()
