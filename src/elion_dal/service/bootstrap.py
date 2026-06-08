"""Сборка IndexService из конфига (без зависимости от gRPC).

Используется и сервером, и сид-утилитой, и CLI — чтобы не дублировать проводку.
Эффективный конфиг = .env, поверх которого ложатся override из БД (app_settings):
restart-настройки (backend/model/quantize) применяются здесь, на старте;
live-настройки читаются IndexService на каждом запросе.
"""

from __future__ import annotations

from ..chunking.chunker import Chunker
from ..config import Settings, get_settings
from ..embedding.factory import build_provider
from ..store.pg_repo import PgRepo
from ..store.qdrant_repo import QdrantRepo
from ..store.settings_store import SettingsStore
from .sync import IndexService


def build_index_service(settings: Settings | None = None, ensure: bool = True) -> IndexService:
    settings = settings or get_settings()
    pg = PgRepo(settings.pg_dsn)
    store = SettingsStore(pg.engine)
    store.load()

    # restart-настройки: override из БД поверх .env (применяются на старте).
    backend = store.get("embedding_backend") or settings.embedding_backend
    model_ovr = store.get("embedding_model")
    quant_ovr = store.get("embedding_quantize")
    eff = settings.model_copy(
        update={
            "embedding_backend": backend,
            "embedding_model": model_ovr if model_ovr is not None else settings.embedding_model,
            "embedding_quantize": settings.embedding_quantize if quant_ovr is None else quant_ovr,
        }
    )

    provider = build_provider(eff)
    qdrant = QdrantRepo(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=provider.dim,  # размерность из реальной модели, не из конфига
        sparse_uses_idf=provider.sparse_uses_idf,
        prefetch=settings.search_prefetch,
        timeout_s=settings.qdrant_timeout_s,
        retry_attempts=settings.qdrant_retry_attempts,
        retry_base_delay_s=settings.qdrant_retry_base_delay_s,
    )
    # chunk_tokenizer_model — restart-настройка: override из БД поверх .env (как embedding_*).
    tokenizer_model = store.get("chunk_tokenizer_model") or settings.chunk_tokenizer_model
    chunker = Chunker(
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
        model_name=tokenizer_model,
        min_tokens=settings.chunk_min_tokens,
        separator_mode=settings.chunk_separator_mode,
    )

    if ensure:
        qdrant.ensure_collection()
    return IndexService(
        pg,
        qdrant,
        provider,
        chunker,
        parent_fanout=settings.search_parent_fanout,
        recency_weight=settings.recency_weight,
        recency_halflife_days=settings.recency_halflife_days,
        settings_store=store,
        base_settings=settings,
        upsert_batch_size=settings.upsert_batch_size,
    )
