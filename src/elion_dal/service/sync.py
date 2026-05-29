"""Оркестрация parent-child: документ -> секции(родители) -> дети -> PG + Qdrant.

Индексируем детей, ищем по детям, но возвращаем РОДИТЕЛЕЙ (схлопывая дублирующие
попадания) — точный матч + богатый контекст для генерации. Идемпотентно по хешу.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..chunking.chunker import Chunker
from ..embedding.base import EmbeddingProvider
from ..store.models import chunk_id, parent_pk
from ..store.pg_repo import DocInput, ParentBuild, PgRepo, sha256
from ..store.qdrant_repo import PointInput, QdrantRepo


@dataclass(slots=True)
class UpsertCounts:
    received: int = 0
    indexed: int = 0
    skipped: int = 0
    blank: int = 0
    parents_upserted: int = 0
    chunks_upserted: int = 0


@dataclass(slots=True)
class ParentHit:
    parent_id: str
    doc_id: str
    source_id: str
    title: str
    url: str
    heading_path: list[str] = field(default_factory=list)
    text: str = ""
    matched_child: str = ""
    score: float = 0.0


class IndexService:
    def __init__(
        self,
        pg: PgRepo,
        qdrant: QdrantRepo,
        provider: EmbeddingProvider,
        chunker: Chunker,
        parent_fanout: int = 5,
    ) -> None:
        self.pg = pg
        self.qdrant = qdrant
        self.provider = provider
        self.chunker = chunker
        # Во сколько раз больше детей тянуть, чтобы схлопнуть в top_k уникальных родителей.
        self.parent_fanout = max(1, parent_fanout)

    def process_document(self, doc: DocInput, counts: UpsertCounts) -> None:
        counts.received += 1
        self.pg.ensure_source(doc.source_id)

        raw_text = "\n\n".join(s.text for s in doc.sections)
        if not doc.content_hash:
            doc.content_hash = sha256(raw_text)

        # Бланк-на-скачивание: храним в SoT, но не индексируем; чистим старые точки.
        if not doc.index_in_rag:
            self.pg.upsert_document(doc, raw_text)
            self.pg.replace_parents_and_chunks(doc.doc_id, [])
            self.qdrant.delete_by_doc(doc.doc_id)
            counts.blank += 1
            return

        prev_hash = self.pg.get_content_hash(doc.doc_id)
        if prev_hash is not None and prev_hash == doc.content_hash:
            counts.skipped += 1
            return

        self.pg.upsert_document(doc, raw_text)

        # Секция -> родитель, текст секции -> дети.
        parents: list[ParentBuild] = []
        for ordinal, section in enumerate(doc.sections):
            section_id = section.section_id or str(ordinal)
            pid = parent_pk(doc.doc_id, section_id)
            children = self.chunker.split(section.text)
            parents.append(
                ParentBuild(
                    parent_id=pid,
                    section_id=section_id,
                    heading_path=section.heading_path,
                    url=section.url or doc.url,
                    text=section.text,
                    token_count=sum(c.token_count for c in children),
                    ordinal=ordinal,
                    children=children,
                )
            )

        self.pg.replace_parents_and_chunks(doc.doc_id, parents)
        self.qdrant.delete_by_doc(doc.doc_id)

        # Эмбеддим всех детей одним батчем.
        texts = [c.text for p in parents for c in p.children]
        if texts:
            embeddings = self.provider.embed_documents(texts)
            points: list[PointInput] = []
            i = 0
            for p in parents:
                for c in p.children:
                    points.append(
                        PointInput(
                            parent_id=p.parent_id,
                            chunk_index=c.index,
                            embedding=embeddings[i],
                            payload={
                                "chunk_id": chunk_id(p.parent_id, c.index),
                                "parent_id": p.parent_id,
                                "doc_id": doc.doc_id,
                                "source_id": doc.source_id,
                                "url": p.url,
                                "title": doc.title,
                                "heading_path": p.heading_path,
                                "text": c.text,
                                "published_ts": doc.published_ts,
                                "lang": doc.lang,
                            },
                        )
                    )
                    i += 1
            counts.chunks_upserted += self.qdrant.upsert_chunks(points)

        counts.parents_upserted += len(parents)
        self.pg.touch_source_indexed(doc.source_id)
        counts.indexed += 1

    def search(
        self, query: str, top_k: int, source_ids: list[str], min_published_ts: int
    ) -> list[ParentHit]:
        embedding = self.provider.embed_query(query)
        child_hits = self.qdrant.search(
            embedding,
            limit=top_k * self.parent_fanout,
            source_ids=source_ids,
            min_published_ts=min_published_ts,
        )

        # Схлопываем детей в уникальных родителей, сохраняя порядок (RRF уже отсортировал).
        ordered_parent_ids: list[str] = []
        best: dict[str, tuple[float, str]] = {}  # parent_id -> (score, matched_child_text)
        for h in child_hits:
            if h.parent_id not in best:
                ordered_parent_ids.append(h.parent_id)
                best[h.parent_id] = (h.score, h.text)
            if len(ordered_parent_ids) >= top_k:
                break

        records = self.pg.get_parents(ordered_parent_ids)
        results: list[ParentHit] = []
        for pid in ordered_parent_ids:
            rec = records.get(pid)
            if rec is None:
                continue
            score, matched_child = best[pid]
            results.append(
                ParentHit(
                    parent_id=rec.parent_id,
                    doc_id=rec.doc_id,
                    source_id=rec.source_id,
                    title=rec.title,
                    url=rec.url,
                    heading_path=rec.heading_path,
                    text=rec.text,
                    matched_child=matched_child,
                    score=score,
                )
            )
        return results

    def delete_source(self, source_id: str) -> tuple[int, int]:
        docs, chunks = self.pg.delete_by_source(source_id)
        self.qdrant.delete_by_source(source_id)
        return docs, chunks

    def health(self) -> dict:
        qok = self.qdrant.ping()
        pok = self.pg.ping()
        return {
            "ok": qok and pok,
            "qdrant_ok": qok,
            "postgres_ok": pok,
            "embedding_backend": self.provider.name,
        }
