# ruff: noqa: B008
"""REST API сервера (FastAPI). Заменил gRPC как публичный контракт.

Все ручки (кроме `/healthz`) требуют Bearer-токен из env/админки (`API_TOKEN`):
    Authorization: Bearer <token>

Эндпоинты:
- GET    /healthz                    — health (открыт)
- POST   /api/v1/search               — гибридный поиск, top-k родителей
- POST   /api/v1/documents            — upsert документа (для админки upload)
- DELETE /api/v1/sources/{source_id}  — удалить источник
- DELETE /api/v1/documents/{doc_id}   — удалить документ
- GET    /api/v1/sources              — список источников + объёмы
- GET    /api/v1/stats                — суммарная статистика
- GET    /api/v1/settings             — текущие настройки (live + restart)
- POST   /api/v1/settings             — обновить настройки (items: dict)
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ..config import Settings
from ..service.sync import IndexService, UpsertCounts
from ..store.pg_repo import DocInput, SectionInput

logger = logging.getLogger(__name__)


# ----------- pydantic-схемы -----------


class SectionIn(BaseModel):
    section_id: str = "0"
    heading_path: list[str] = Field(default_factory=list)
    url: str = ""
    text: str
    published_ts: int = 0
    content_hash: str = ""


class DocumentIn(BaseModel):
    doc_id: str
    source_id: str
    url: str = ""
    title: str = ""
    lang: str = "ru"
    published_ts: int = 0
    content_hash: str = ""
    index_in_rag: bool = True
    sections: list[SectionIn] = Field(default_factory=list)
    text: str = ""  # fallback: если sections пусто, весь текст = одна секция


class SearchIn(BaseModel):
    query: str
    top_k: int = 0
    source_ids: list[str] = Field(default_factory=list)
    min_published_ts: int = 0


class SettingsUpdateIn(BaseModel):
    items: dict[str, str]


# ----------- auth -----------


def _effective_token(index: IndexService, settings: Settings) -> str:
    store = getattr(index, "settings_store", None)
    if store is not None:
        tok = store.get("api_token")
        if tok:
            return str(tok)
    return getattr(settings, "api_token", "") or ""


def _make_auth_dep(index: IndexService, settings: Settings):
    """Зависимость: Bearer-токен; HealthCheck не использует её (он открыт)."""
    bearer = HTTPBearer(auto_error=False)

    def check(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        expected = _effective_token(index, settings)
        if not expected:
            return  # токен не сконфигурирован -> ручки открыты
        given = creds.credentials if creds and creds.scheme.lower() == "bearer" else ""
        if not secrets.compare_digest(given, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing API token")

    return check


# ----------- app factory -----------


def create_api(index: IndexService, settings: Settings) -> FastAPI:
    app = FastAPI(title="Элион — DAL REST API", version="0.2")
    auth = _make_auth_dep(index, settings)

    @app.get("/healthz")
    def healthz() -> dict:
        # Открыт — для health-проб платформы.
        return {"status": "ok"}

    # --- поиск ---
    @app.post("/api/v1/search", dependencies=[Depends(auth)])
    def search(req: SearchIn, request: Request) -> dict:
        top_k = req.top_k or settings.search_top_k
        t0 = time.perf_counter()
        try:
            hits = index.search(
                query=req.query,
                top_k=top_k,
                source_ids=req.source_ids,
                min_published_ts=req.min_published_ts,
            )
        except Exception as e:  # noqa: BLE001 — деградируем мягко, не голым 500
            # Полный traceback — в логи; клиенту — 503 (бэкенд поиска недоступен).
            # TODO(diag): временно отдаём тип/сообщение в detail для диагностики 500
            #             на проде; после фикса заменить на generic-текст.
            logger.exception("search failed query=%r", req.query)
            raise HTTPException(
                status_code=503,
                detail=f"search backend error: {type(e).__name__}: {str(e)[:300]}",
            ) from e
        dt_ms = (time.perf_counter() - t0) * 1000
        if hits:
            logger.info(
                "search hits=%d top_score=%.4f dense=%.4f %.0fms query=%r",
                len(hits),
                hits[0].score,
                hits[0].dense_score,
                dt_ms,
                req.query,
            )
        else:
            logger.info("search NO-HIT %.0fms query=%r", dt_ms, req.query)
        return {
            "hits": [
                {
                    "parent_id": h.parent_id,
                    "doc_id": h.doc_id,
                    "source_id": h.source_id,
                    "url": h.url,
                    "title": h.title,
                    "heading_path": list(h.heading_path),
                    "text": h.text,
                    "matched_child": h.matched_child,
                    "score": h.score,
                    "dense_score": h.dense_score,
                }
                for h in hits
            ]
        }

    # --- индексация документа (приходит из админки upload) ---
    @app.post("/api/v1/documents", dependencies=[Depends(auth)])
    def upsert_document(payload: DocumentIn) -> dict:
        sections = [
            SectionInput(
                section_id=s.section_id,
                heading_path=list(s.heading_path),
                url=s.url,
                text=s.text,
                published_ts=s.published_ts,
                content_hash=s.content_hash,
            )
            for s in payload.sections
        ]
        if not sections and payload.text:
            sections = [
                SectionInput(section_id="0", heading_path=[], url=payload.url, text=payload.text)
            ]
        doc = DocInput(
            doc_id=payload.doc_id,
            source_id=payload.source_id or "unknown",
            url=payload.url,
            title=payload.title,
            lang=payload.lang or "ru",
            published_ts=payload.published_ts,
            content_hash=payload.content_hash,
            index_in_rag=payload.index_in_rag,
            sections=sections,
        )
        counts = UpsertCounts()
        try:
            index.process_document(doc, counts)
        except Exception:  # noqa: BLE001
            counts.failed += 1
            logger.exception("Не удалось обработать документ doc_id=%s", doc.doc_id)
        return {
            "received": counts.received,
            "indexed": counts.indexed,
            "skipped": counts.skipped,
            "blank": counts.blank,
            "failed": counts.failed,
            "parents_upserted": counts.parents_upserted,
            "chunks_upserted": counts.chunks_upserted,
        }

    # --- удаление ---
    @app.delete("/api/v1/sources/{source_id}", dependencies=[Depends(auth)])
    def delete_source(source_id: str) -> dict:
        docs, chunks = index.delete_source(source_id)
        return {"documents_deleted": docs, "chunks_deleted": chunks}

    @app.delete("/api/v1/documents/{doc_id}", dependencies=[Depends(auth)])
    def delete_doc(doc_id: str) -> dict:
        docs, chunks = index.delete_doc(doc_id)
        return {"documents_deleted": docs, "chunks_deleted": chunks}

    # --- статистика ---
    def _source_to_dict(s) -> dict:
        return {
            "source_id": s.source_id,
            "name": s.name,
            "last_indexed_ts": s.last_indexed_ts,
            "document_count": s.document_count,
            "parent_count": s.parent_count,
            "chunk_count": s.chunk_count,
        }

    @app.get("/api/v1/sources", dependencies=[Depends(auth)])
    def list_sources() -> dict:
        return {"sources": [_source_to_dict(s) for s in index.list_sources()]}

    @app.get("/api/v1/stats", dependencies=[Depends(auth)])
    def get_stats() -> dict:
        st = index.get_stats()
        return {
            "total_documents": st.total_documents,
            "total_parents": st.total_parents,
            "total_chunks": st.total_chunks,
            "sources": [_source_to_dict(s) for s in st.sources],
        }

    # --- настройки ---
    @app.get("/api/v1/settings", dependencies=[Depends(auth)])
    def get_settings_endpoint() -> dict:
        return {
            "fields": [
                {
                    "key": v.key,
                    "label": v.label,
                    "tier": v.tier,
                    "type": v.type,
                    "value": "" if v.value is None else str(v.value),
                    "is_override": bool(v.is_override),
                }
                for v in index.settings_view()
            ]
        }

    @app.post("/api/v1/settings", dependencies=[Depends(auth)])
    def update_settings(req: SettingsUpdateIn) -> dict:
        index.update_settings(dict(req.items))
        return get_settings_endpoint()

    # Защитный обработчик: всё остальное — 404 без подробностей (минимум информации).
    @app.exception_handler(404)
    async def _not_found(_req: Request, _exc: Any) -> JSONResponse:
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    return app
