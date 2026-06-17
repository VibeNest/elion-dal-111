"""Доступ к Qdrant: коллекция с named-векторами dense+sparse и гибридный поиск (RRF)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from ..embedding.base import Embedding
from ..util.retry import call_with_retry
from .models import point_id

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"


@dataclass(slots=True)
class PointInput:
    parent_id: str
    chunk_index: int
    embedding: Embedding
    payload: dict


@dataclass(slots=True)
class SearchHit:
    """Дочернее попадание. sync схлопывает их в уникальных родителей."""

    chunk_id: str
    parent_id: str
    doc_id: str
    source_id: str
    text: str
    score: float


def _make_client(location: str, timeout_s: float | None = None) -> QdrantClient:
    """Создать клиент Qdrant из строки конфига.

    Поддерживаются три режима (выбор по значению QDRANT_URL), без изменения кода:
      * "http://..." / "https://..."  -> внешний сервер (прод/Docker);
      * ":memory:"                    -> embedded in-memory (тесты);
      * любой другой путь             -> embedded on-disk (локальная разработка без Docker).
    Embedded-режим — штатная возможность qdrant-client, поддерживает sparse и RRF.

    `timeout_s` применяется только к http-режиму — чтобы зависший Qdrant-сервер не
    вешал запрос навечно (embedded работает в процессе, таймаут не нужен).
    """
    # check_compatibility=False: версия клиента пиннится через requirements.lock,
    # а проверка дёргает сервер на старте (лишний запрос + warning в логах).
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, timeout=timeout_s, check_compatibility=False)
    if location == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=location)


class QdrantRepo:
    def __init__(
        self,
        url: str,
        collection: str,
        dim: int,
        sparse_uses_idf: bool,
        prefetch: int = 20,
        timeout_s: float | None = None,
        retry_attempts: int = 1,
        retry_base_delay_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = _make_client(url, timeout_s)
        self.collection = collection
        self.dim = dim
        self.sparse_uses_idf = sparse_uses_idf
        self.prefetch = prefetch
        self._retry_attempts = max(1, retry_attempts)
        self._retry_base_delay_s = retry_base_delay_s
        self._sleep = sleep

    def _retry(self, fn: Callable[[], object], op_name: str) -> object:
        """Вызвать сетевую операцию Qdrant с ретраями на транзиентных сбоях."""
        return call_with_retry(
            fn,
            attempts=self._retry_attempts,
            base_delay_s=self._retry_base_delay_s,
            sleep=self._sleep,
            op_name=op_name,
        )

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self._warn_on_dim_mismatch()
            return
        sparse_params = models.SparseVectorParams(
            modifier=models.Modifier.IDF if self.sparse_uses_idf else None
        )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: sparse_params},
        )
        self.client.create_payload_index(
            self.collection, field_name="source_id", field_schema=models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            self.collection, field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            self.collection,
            field_name="published_ts",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    def recreate_collection(self) -> None:
        """Снести коллекцию целиком и создать заново (чистое восстановление при reindex
        после повреждения storage). Удаление идемпотентно (нет коллекции — не ошибка)."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.ensure_collection()

    def _warn_on_dim_mismatch(self) -> None:
        """Если коллекция уже есть с другой размерностью dense — предупредить
        (например, сменили модель эмбеддингов без пересоздания коллекции)."""
        try:
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors[DENSE].size
            if existing != self.dim:
                logger.warning(
                    "Qdrant collection '%s' имеет dense-размерность %d, а модель даёт %d. "
                    "Пересоздайте коллекцию или смените модель.",
                    self.collection,
                    existing,
                    self.dim,
                )
        except Exception:  # noqa: BLE001 — диагностика не должна ронять старт
            pass

    def upsert_chunks(self, points: Sequence[PointInput]) -> int:
        structs = [
            models.PointStruct(
                id=point_id(p.parent_id, p.chunk_index),
                vector={
                    DENSE: p.embedding.dense,
                    SPARSE: models.SparseVector(
                        indices=p.embedding.sparse.indices, values=p.embedding.sparse.values
                    ),
                },
                payload=p.payload,
            )
            for p in points
        ]
        if not structs:
            return 0
        self._retry(
            lambda: self.client.upsert(
                collection_name=self.collection, points=structs, wait=True
            ),
            "upsert_chunks",
        )
        return len(structs)

    def delete_by_doc(self, doc_id: str) -> None:
        self._retry(
            lambda: self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id", match=models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                ),
                wait=True,
            ),
            "delete_by_doc",
        )

    def delete_by_source(self, source_id: str) -> None:
        self._retry(
            lambda: self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_id", match=models.MatchValue(value=source_id)
                            )
                        ]
                    )
                ),
                wait=True,
            ),
            "delete_by_source",
        )

    def _filter(self, source_ids: Sequence[str], min_published_ts: int, academic_year: int | None = None,
    is_active: bool | None = None,) -> models.Filter | None:
        must: list = []
        if source_ids:
            must.append(
                models.FieldCondition(key="source_id", match=models.MatchAny(any=list(source_ids)))
            )
        if min_published_ts > 0:
            must.append(
                models.FieldCondition(key="published_ts", range=models.Range(gte=min_published_ts))
            )
        if academic_year is not None:
            must.append(
            models.FieldCondition(key="academic_year", match=models.MatchValue(value=academic_year))
        )
        if is_active is not None:
            must.append(
                models.FieldCondition(key="is_active", match=models.MatchValue(value=is_active))
            )
        return models.Filter(must=must) if must else None

    def search(
        self,
        query: Embedding,
        limit: int,
        source_ids: Sequence[str] = (),
        min_published_ts: int = 0,
        prefetch_limit: int | None = None,
        academic_year: int | None = None,
        is_active: bool | None = None,
    ) -> list[SearchHit]:
        """Гибридный поиск по детям. `limit` — сколько дочерних попаданий вернуть
        (sync схлопнёт их в уникальных родителей). `prefetch_limit` переопределяет
        число кандидатов на ветку (живая настройка)."""
        qfilter = self._filter(source_ids, min_published_ts, academic_year=academic_year,
    is_active=is_active,)
        pf = prefetch_limit or self.prefetch
        result = self._retry(
            lambda: self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(query=query.dense, using=DENSE, limit=pf, filter=qfilter),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=query.sparse.indices, values=query.sparse.values
                        ),
                        using=SPARSE,
                        limit=pf,
                        filter=qfilter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            ),
            "search",
        )
        hits: list[SearchHit] = []
        for p in result.points:
            payload = p.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(payload.get("chunk_id", "")),
                    parent_id=str(payload.get("parent_id", "")),
                    doc_id=str(payload.get("doc_id", "")),
                    source_id=str(payload.get("source_id", "")),
                    text=str(payload.get("text", "")),
                    score=float(p.score),
                )
            )
        return hits

    def dense_scores(
        self,
        query: Embedding,
        limit: int,
        source_ids: Sequence[str] = (),
        min_published_ts: int = 0,
    ) -> dict[str, float]:
        """Только-dense поиск -> {chunk_id: cosine}. Нужен для confidence-сигнала
        (RRF-fusion не отдаёт сырые косинусы)."""
        qfilter = self._filter(source_ids, min_published_ts)
        result = self._retry(
            lambda: self.client.query_points(
                collection_name=self.collection,
                query=query.dense,
                using=DENSE,
                limit=limit,
                with_payload=["chunk_id"],
                query_filter=qfilter,
            ),
            "dense_scores",
        )
        out: dict[str, float] = {}
        for p in result.points:
            cid = (p.payload or {}).get("chunk_id")
            if cid:
                out[str(cid)] = float(p.score)
        return out
