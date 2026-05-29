"""Сборка IndexService из конфига (без зависимости от gRPC).

Используется и сервером, и сид-утилитой, и CLI — чтобы не дублировать проводку.
ВНИМАНИЕ: build_index_service грузит эмбеддинг-модель (на первом запуске
скачивает BGE-M3) и создаёт коллекцию Qdrant при отсутствии.
"""

from __future__ import annotations

from ..chunking.chunker import Chunker
from ..config import Settings, get_settings
from ..embedding.factory import build_provider
from ..store.pg_repo import PgRepo
from ..store.qdrant_repo import QdrantRepo
from .sync import IndexService


def build_index_service(settings: Settings | None = None, ensure: bool = True) -> IndexService:
    settings = settings or get_settings()
    provider = build_provider(settings)
    pg = PgRepo(settings.pg_dsn)
    qdrant = QdrantRepo(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
        sparse_uses_idf=provider.sparse_uses_idf,
        prefetch=settings.search_prefetch,
    )
    chunker = Chunker(
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
        model_name=settings.chunk_tokenizer_model,
    )
    if ensure:
        qdrant.ensure_collection()
    return IndexService(pg, qdrant, provider, chunker, parent_fanout=settings.search_parent_fanout)
