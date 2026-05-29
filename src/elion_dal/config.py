"""Конфигурация сервиса (pydantic-settings, читается из окружения/.env)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # gRPC
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "elion_chunks"

    # Postgres (source-of-truth)
    pg_dsn: str = "postgresql+psycopg://elion:elion@localhost:5432/elion"

    # Эмбеддинги
    # fastembed -> ONNX multilingual-e5 (быстро на CPU); flag -> настоящий BGE-M3 (вариант A).
    embedding_backend: str = "fastembed"  # fastembed | flag
    # Пусто => дефолтная модель бэкенда (fastembed: multilingual-e5-large; flag: BAAI/bge-m3).
    embedding_model: str = ""
    embedding_dim: int = 1024

    # Чанкинг
    chunk_tokens: int = 400
    chunk_overlap: int = 64
    # Токенайзер для подсчёта длины чанков. Намеренно НЕ привязан к embedding_model:
    # bge-m3-токенайзер даёт стабильное сегментирование независимо от бэкенда эмбеддингов.
    chunk_tokenizer_model: str = "BAAI/bge-m3"

    # Поиск
    search_top_k: int = 3
    search_prefetch: int = 20
    # Во сколько раз больше детей тянуть, чтобы схлопнуть в top_k уникальных родителей.
    search_parent_fanout: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
