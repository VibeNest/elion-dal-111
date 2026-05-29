"""Доступ к Postgres (source-of-truth): документы, родители (секции), дети.

Дедуп по content_hash. Родители возвращаются на поиске для генерации.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from ..chunking.chunker import Chunk as TextChunk
from .models import Base, Chunk, Document, Parent, Source, chunk_id


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class SectionInput:
    section_id: str
    heading_path: list[str]
    url: str
    text: str
    published_ts: int = 0
    content_hash: str = ""


@dataclass(slots=True)
class DocInput:
    doc_id: str
    source_id: str
    url: str
    title: str
    lang: str
    published_ts: int
    content_hash: str
    index_in_rag: bool
    sections: list[SectionInput] = field(default_factory=list)


@dataclass(slots=True)
class ParentBuild:
    """Готовый к записи родитель с его дочерними чанками."""

    parent_id: str
    section_id: str
    heading_path: list[str]
    url: str
    text: str
    token_count: int
    ordinal: int
    children: list[TextChunk]


@dataclass(slots=True)
class ParentRecord:
    """Данные родителя для ответа на поиске (join с документом)."""

    parent_id: str
    doc_id: str
    source_id: str
    title: str
    url: str
    heading_path: list[str]
    text: str


class PgRepo:
    def __init__(self, dsn: str) -> None:
        self.engine = create_engine(dsn, pool_pre_ping=True, future=True)
        # SQLite (локальный режим без Postgres) не включает внешние ключи по
        # умолчанию — без этого не сработает ON DELETE CASCADE.
        if dsn.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(self.engine, "connect")
            def _fk_on(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        self._sm: sessionmaker[Session] = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        """Создать схему напрямую (для тестов; в проде — alembic)."""
        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def ensure_source(self, source_id: str, name: str | None = None) -> None:
        with self._sm.begin() as s:
            if s.get(Source, source_id) is None:
                s.add(Source(source_id=source_id, name=name or source_id))

    def get_content_hash(self, doc_id: str) -> str | None:
        with self._sm() as s:
            return s.execute(
                select(Document.content_hash).where(Document.doc_id == doc_id)
            ).scalar_one_or_none()

    def upsert_document(self, doc: DocInput, raw_text: str) -> None:
        with self._sm.begin() as s:
            existing = s.get(Document, doc.doc_id)
            if existing is None:
                s.add(
                    Document(
                        doc_id=doc.doc_id,
                        source_id=doc.source_id,
                        url=doc.url,
                        title=doc.title,
                        lang=doc.lang,
                        published_ts=doc.published_ts,
                        content_hash=doc.content_hash,
                        raw_text=raw_text,
                        index_in_rag=doc.index_in_rag,
                    )
                )
            else:
                existing.source_id = doc.source_id
                existing.url = doc.url
                existing.title = doc.title
                existing.lang = doc.lang
                existing.published_ts = doc.published_ts
                existing.content_hash = doc.content_hash
                existing.raw_text = raw_text
                existing.index_in_rag = doc.index_in_rag

    def replace_parents_and_chunks(self, doc_id: str, parents: Sequence[ParentBuild]) -> None:
        with self._sm.begin() as s:
            # Удаляем старых родителей документа — каскад снесёт и детей.
            s.execute(delete(Parent).where(Parent.doc_id == doc_id))
            for p in parents:
                s.add(
                    Parent(
                        parent_id=p.parent_id,
                        doc_id=doc_id,
                        section_id=p.section_id,
                        heading_path=p.heading_path,
                        url=p.url,
                        text=p.text,
                        token_count=p.token_count,
                        ordinal=p.ordinal,
                    )
                )
                for c in p.children:
                    s.add(
                        Chunk(
                            chunk_id=chunk_id(p.parent_id, c.index),
                            parent_id=p.parent_id,
                            doc_id=doc_id,
                            chunk_index=c.index,
                            text=c.text,
                            token_count=c.token_count,
                            content_hash=sha256(c.text),
                        )
                    )

    def get_parents(self, parent_ids: Sequence[str]) -> dict[str, ParentRecord]:
        if not parent_ids:
            return {}
        with self._sm() as s:
            rows = s.execute(
                select(Parent, Document.source_id, Document.title)
                .join(Document, Document.doc_id == Parent.doc_id)
                .where(Parent.parent_id.in_(list(parent_ids)))
            ).all()
        result: dict[str, ParentRecord] = {}
        for parent, source_id, title in rows:
            result[parent.parent_id] = ParentRecord(
                parent_id=parent.parent_id,
                doc_id=parent.doc_id,
                source_id=source_id,
                title=title,
                url=parent.url,
                heading_path=list(parent.heading_path or []),
                text=parent.text,
            )
        return result

    def touch_source_indexed(self, source_id: str) -> None:
        with self._sm.begin() as s:
            src = s.get(Source, source_id)
            if src is not None:
                src.last_indexed_at = func.now()

    def delete_by_source(self, source_id: str) -> tuple[int, int]:
        """Удалить документы (с родителями и детьми) источника. Возвращает (docs, chunks)."""
        with self._sm.begin() as s:
            doc_ids = list(
                s.execute(select(Document.doc_id).where(Document.source_id == source_id)).scalars()
            )
            chunks_n = (
                s.execute(
                    select(func.count()).select_from(Chunk).where(Chunk.doc_id.in_(doc_ids))
                ).scalar_one()
                if doc_ids
                else 0
            )
            # ON DELETE CASCADE снесёт родителей и детей вместе с документами.
            s.execute(delete(Document).where(Document.source_id == source_id))
            return len(doc_ids), int(chunks_n)
