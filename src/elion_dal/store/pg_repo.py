"""Доступ к Postgres (source-of-truth): документы, родители (секции), дети.

Дедуп по content_hash. Родители возвращаются на поиске для генерации.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
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
    academic_year: int | None = None
    is_active: bool | None = None

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
    published_ts: int = 0


@dataclass(slots=True)
class SourceStats:
    source_id: str
    name: str
    last_indexed_ts: int
    document_count: int
    parent_count: int
    chunk_count: int


@dataclass(slots=True)
class StoreStats:
    total_documents: int
    total_parents: int
    total_chunks: int
    sources: list[SourceStats]


@dataclass(slots=True)
class ChunkRow:
    """Готовый дочерний чанк из PG — для переэмбеддинга при reindex."""

    parent_id: str
    chunk_index: int
    text: str


@dataclass(slots=True)
class ParentReindex:
    """Поля родителя, нужные для payload точки Qdrant при reindex."""

    url: str
    heading_path: list[str]


@dataclass(slots=True)
class DocReindexRow:
    """Документ с готовыми parents+chunks для пересборки индекса из PG (SoT)."""

    doc_id: str
    source_id: str
    title: str
    lang: str
    published_ts: int
    parents: dict[str, ParentReindex]  # parent_id -> (url, heading_path)
    chunks: list[ChunkRow]  # все дети документа в порядке (parent, index)


@dataclass(slots=True)
class DocSummary:
    """Строка списка документов для браузера чанков в админке."""

    doc_id: str
    source_id: str
    title: str
    lang: str
    published_ts: int
    index_in_rag: bool
    indexed: bool  # content_hash != "" — закоммичен в индекс
    parent_count: int
    chunk_count: int


@dataclass(slots=True)
class ChunkDetail:
    """Дочерний чанк для просмотра (с длиной в токенах и chunk_id)."""

    chunk_id: str
    chunk_index: int
    text: str
    token_count: int


@dataclass(slots=True)
class ParentDetail:
    """Секция-родитель с её детьми — для просмотра в админке."""

    parent_id: str
    section_id: str
    heading_path: list[str]
    ordinal: int
    token_count: int
    text: str
    chunks: list[ChunkDetail]


@dataclass(slots=True)
class DocDetail:
    """Документ с секциями(parents) и чанками — для просмотра в админке."""

    doc_id: str
    source_id: str
    title: str
    url: str
    lang: str
    published_ts: int
    index_in_rag: bool
    indexed: bool
    parents: list[ParentDetail]


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
        """Записать мету документа. content_hash НЕ трогаем здесь — он
        фиксируется отдельно через set_content_hash() только после успешной
        индексации в Qdrant (см. sync), чтобы сбой Qdrant не приводил к дрейфу
        (документ считался бы «проиндексированным» без точек)."""
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
                        content_hash="",  # pending — выставится после успеха
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
                existing.raw_text = raw_text
                existing.index_in_rag = doc.index_in_rag
                # content_hash намеренно не обновляем здесь.

    def set_content_hash(self, doc_id: str, content_hash: str) -> None:
        """Зафиксировать content_hash после успешной индексации (commit point)."""
        with self._sm.begin() as s:
            d = s.get(Document, doc_id)
            if d is not None:
                d.content_hash = content_hash

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
                select(Parent, Document.source_id, Document.title, Document.published_ts)
                .join(Document, Document.doc_id == Parent.doc_id)
                .where(Parent.parent_id.in_(list(parent_ids)))
            ).all()
        result: dict[str, ParentRecord] = {}
        for parent, source_id, title, published_ts in rows:
            result[parent.parent_id] = ParentRecord(
                parent_id=parent.parent_id,
                doc_id=parent.doc_id,
                source_id=source_id,
                title=title,
                url=parent.url,
                heading_path=list(parent.heading_path or []),
                text=parent.text,
                published_ts=int(published_ts or 0),
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

    def delete_by_doc(self, doc_id: str) -> tuple[int, int]:
        """Удалить один документ (с родителями и детьми). Возвращает (docs, chunks)."""
        with self._sm.begin() as s:
            exists = s.get(Document, doc_id) is not None
            chunks_n = s.execute(
                select(func.count()).select_from(Chunk).where(Chunk.doc_id == doc_id)
            ).scalar_one()
            s.execute(delete(Document).where(Document.doc_id == doc_id))
            return (1 if exists else 0), int(chunks_n)

    def list_sources(self) -> list[SourceStats]:
        with self._sm() as s:
            sources = list(s.execute(select(Source)).scalars())
            doc_counts = dict(
                s.execute(
                    select(Document.source_id, func.count()).group_by(Document.source_id)
                ).all()
            )
            parent_counts = dict(
                s.execute(
                    select(Document.source_id, func.count())
                    .select_from(Parent)
                    .join(Document, Document.doc_id == Parent.doc_id)
                    .group_by(Document.source_id)
                ).all()
            )
            chunk_counts = dict(
                s.execute(
                    select(Document.source_id, func.count())
                    .select_from(Chunk)
                    .join(Document, Document.doc_id == Chunk.doc_id)
                    .group_by(Document.source_id)
                ).all()
            )
        result: list[SourceStats] = []
        for src in sources:
            ts = int(src.last_indexed_at.timestamp()) if src.last_indexed_at else 0
            result.append(
                SourceStats(
                    source_id=src.source_id,
                    name=src.name or src.source_id,
                    last_indexed_ts=ts,
                    document_count=int(doc_counts.get(src.source_id, 0)),
                    parent_count=int(parent_counts.get(src.source_id, 0)),
                    chunk_count=int(chunk_counts.get(src.source_id, 0)),
                )
            )
        return result

    def get_stats(self) -> StoreStats:
        sources = self.list_sources()
        with self._sm() as s:
            td = s.execute(select(func.count()).select_from(Document)).scalar_one()
            tp = s.execute(select(func.count()).select_from(Parent)).scalar_one()
            tc = s.execute(select(func.count()).select_from(Chunk)).scalar_one()
        return StoreStats(int(td), int(tp), int(tc), sources)

    def list_documents(self, source_id: str | None = None) -> list[DocSummary]:
        """Список документов с объёмами (parents/chunks) для браузера чанков.

        В отличие от iter_documents_for_reindex здесь НЕ фильтруем по content_hash —
        показываем и pending/незакоммиченные (флаг `indexed` это отражает)."""
        with self._sm() as s:
            q = select(Document)
            if source_id:
                q = q.where(Document.source_id == source_id)
            docs = list(s.execute(q.order_by(Document.source_id, Document.title)).scalars())
            parent_counts = dict(
                s.execute(
                    select(Parent.doc_id, func.count()).group_by(Parent.doc_id)
                ).all()
            )
            chunk_counts = dict(
                s.execute(
                    select(Chunk.doc_id, func.count()).group_by(Chunk.doc_id)
                ).all()
            )
        return [
            DocSummary(
                doc_id=d.doc_id,
                source_id=d.source_id,
                title=d.title,
                lang=d.lang,
                published_ts=int(d.published_ts or 0),
                index_in_rag=bool(d.index_in_rag),
                indexed=bool(d.content_hash),
                parent_count=int(parent_counts.get(d.doc_id, 0)),
                chunk_count=int(chunk_counts.get(d.doc_id, 0)),
            )
            for d in docs
        ]

    def get_document_detail(self, doc_id: str) -> DocDetail | None:
        """Документ + его секции(parents, по ordinal) + чанки (по chunk_index)."""
        with self._sm() as s:
            d = s.get(Document, doc_id)
            if d is None:
                return None
            parents = list(
                s.execute(
                    select(Parent).where(Parent.doc_id == doc_id).order_by(Parent.ordinal)
                ).scalars()
            )
            chunks = list(
                s.execute(
                    select(Chunk)
                    .where(Chunk.doc_id == doc_id)
                    .order_by(Chunk.parent_id, Chunk.chunk_index)
                ).scalars()
            )
            doc = DocDetail(
                doc_id=d.doc_id,
                source_id=d.source_id,
                title=d.title,
                url=d.url,
                lang=d.lang,
                published_ts=int(d.published_ts or 0),
                index_in_rag=bool(d.index_in_rag),
                indexed=bool(d.content_hash),
                parents=[],
            )
        chunks_by_parent: dict[str, list[ChunkDetail]] = {}
        for c in chunks:
            chunks_by_parent.setdefault(c.parent_id, []).append(
                ChunkDetail(
                    chunk_id=c.chunk_id,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    token_count=int(c.token_count or 0),
                )
            )
        for p in parents:
            doc.parents.append(
                ParentDetail(
                    parent_id=p.parent_id,
                    section_id=p.section_id,
                    heading_path=list(p.heading_path or []),
                    ordinal=int(p.ordinal or 0),
                    token_count=int(p.token_count or 0),
                    text=p.text,
                    chunks=chunks_by_parent.get(p.parent_id, []),
                )
            )
        return doc

    def iter_documents_for_reindex(
        self, source_id: str | None = None, batch: int = 200
    ) -> Iterator[DocReindexRow]:
        """Стримит документы (готовые к индексации) с их parents и chunks — для reindex.

        Отдаются только закоммиченные документы: `index_in_rag=True` И `content_hash != ""`
        (pending/полузаписанные пропускаем). Пагинация по `doc_id`, чтобы не держать весь
        корпус в памяти. Чанки берутся ГОТОВЫМИ из PG (не перенарезаются) — их
        `chunk_id = parent_id#index` даёт те же детерминированные point_id, т.е. точное
        восстановление коллекции.
        """
        with self._sm() as s:
            q = select(Document.doc_id).where(
                Document.index_in_rag.is_(True), Document.content_hash != ""
            )
            if source_id is not None:
                q = q.where(Document.source_id == source_id)
            doc_ids = list(s.execute(q.order_by(Document.doc_id)).scalars())

        for i in range(0, len(doc_ids), batch):
            window = doc_ids[i : i + batch]
            with self._sm() as s:
                docs = {
                    d.doc_id: d
                    for d in s.execute(
                        select(Document).where(Document.doc_id.in_(window))
                    ).scalars()
                }
                parents = list(
                    s.execute(select(Parent).where(Parent.doc_id.in_(window))).scalars()
                )
                chunks = list(
                    s.execute(
                        select(Chunk).where(Chunk.doc_id.in_(window)).order_by(
                            Chunk.parent_id, Chunk.chunk_index
                        )
                    ).scalars()
                )
            parents_by_doc: dict[str, dict[str, ParentReindex]] = {}
            for p in parents:
                parents_by_doc.setdefault(p.doc_id, {})[p.parent_id] = ParentReindex(
                    url=p.url, heading_path=list(p.heading_path or [])
                )
            chunks_by_doc: dict[str, list[ChunkRow]] = {}
            for c in chunks:
                chunks_by_doc.setdefault(c.doc_id, []).append(
                    ChunkRow(parent_id=c.parent_id, chunk_index=c.chunk_index, text=c.text)
                )
            for doc_id in window:
                d = docs.get(doc_id)
                if d is None:
                    continue
                yield DocReindexRow(
                    doc_id=d.doc_id,
                    source_id=d.source_id,
                    title=d.title,
                    lang=d.lang,
                    published_ts=int(d.published_ts or 0),
                    parents=parents_by_doc.get(doc_id, {}),
                    chunks=chunks_by_doc.get(doc_id, []),
                )
