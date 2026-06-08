"""Конфигурация сервиса (pydantic-settings, читается из окружения/.env)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # gRPC / сервер
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    grpc_max_workers: int = 8
    grpc_max_message_mb: int = 32
    # Ретраи подключения к Qdrant/Postgres на старте (backoff).
    startup_retries: int = 10
    startup_retry_delay_s: float = 3.0

    # Логирование
    log_level: str = "INFO"

    # Веб-админка (FastAPI в том же процессе, что и gRPC)
    admin_enabled: bool = True
    admin_host: str = "0.0.0.0"
    admin_port: int = 8080
    # Basic-auth админки: логин/пароль из env. Пустой пароль => auth выключен (dev).
    admin_user: str = "admin"
    admin_password: str = ""

    # Фиксированный токен доступа к gRPC API (ручкам). Пусто => проверка выключена.
    # Может переопределяться в админке (app_settings.api_token), env — бутстрап/фолбэк.
    api_token: str = ""

    # Подключение локальной админки к удалённому REST-серверу.
    api_base_url: str = ""        # https://elion-dal.vibenest.net (без хвостового /)
    # (gRPC-параметры сохранены — могут пригодиться, когда платформа научится
    # проксировать gRPC; см. ADR-006.)
    grpc_target: str = ""
    grpc_insecure: bool = False

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "elion_chunks"
    # Runtime-устойчивость к сбоям Qdrant (транзиентные IO/сетевые ошибки).
    qdrant_timeout_s: float = 10.0          # глобальный таймаут http-клиента (сек)
    qdrant_retry_attempts: int = 3          # всего попыток на сетевой вызов (1 = без ретраев)
    qdrant_retry_base_delay_s: float = 0.5  # база экспоненциального backoff (сек)
    # Размер окна embed+upsert при индексации; большие документы пишутся батчами.
    upsert_batch_size: int = 256

    # Postgres (source-of-truth)
    pg_dsn: str = "postgresql+psycopg://elion:elion@localhost:5432/elion"
    # Создавать схему БД при старте (create_all) — чтобы деплой работал из коробки.
    # Идемпотентно. Для окружений с alembic выставить false и катать миграции отдельно.
    auto_migrate: bool = True

    # Эмбеддинги
    # st-bm25 -> рекомендуемый прод-конфиг: dense USER-bge-m3 (sentence-transformers) + BM25 sparse
    #            (по исследованию leaderboard: hybrid USER-bge-m3+BM25 -> Recall@5 0.98).
    # fastembed -> ONNX multilingual-e5 + BM25 (легче/быстрее на CPU, R@5 0.92);
    # flag      -> настоящий BGE-M3 (dense + learned sparse).
    embedding_backend: str = "st-bm25"  # st-bm25 | fastembed | flag
    # Пусто => дефолтная модель бэкенда. Для st-bm25 dense — deepvk/USER-bge-m3.
    embedding_model: str = "deepvk/USER-bge-m3"
    embedding_dim: int = 1024
    # int8-квантизация эмбеддинг-модели. ВНИМАНИЕ: для flag/BGE-M3 это torch dynamic
    # int8, и по замерам RSS он НЕ снижается (а ~удваивается из-за fp32-копии на пике) —
    # см. ADR-004. Поэтому default OFF. Реальное снижение RAM даёт int8 ONNX-экспорт.
    embedding_quantize: bool = False

    # Чанкинг (рекомендуемый конфиг по исследованию: recursive 1024/102, structured-сепараторы).
    chunk_tokens: int = 1024
    chunk_overlap: int = 102  # ~10% от 1024
    # Мин. размер чанка в токенах. 0 = выкл. >0 => чанки короче дропаются при индексации
    # (фильтр мусора: заголовки-сироты, хвосты-обрезки). Меняет состав индекса — см. Chunker.
    chunk_min_tokens: int = 0
    # Стратегия сепараторов нарезки: structured (абзацы→предложения→слова, по умолчанию)
    # или token (жёстко: абзац/строка/слово/символ, без учёта границ предложений).
    chunk_separator_mode: str = "structured"
    # Токенайзер для подсчёта длины чанков. Намеренно НЕ привязан к embedding_model:
    # bge-m3-токенайзер даёт стабильное сегментирование независимо от бэкенда эмбеддингов.
    chunk_tokenizer_model: str = "BAAI/bge-m3"

    # Поиск
    search_top_k: int = 3
    search_prefetch: int = 20
    # Во сколько раз больше детей тянуть, чтобы схлопнуть в top_k уникальных родителей.
    search_parent_fanout: int = 5

    # Приоритет свежести: множитель к скору по дате. 0 = выключено (поведение по умолчанию).
    recency_weight: float = 0.0
    recency_halflife_days: float = 365.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
