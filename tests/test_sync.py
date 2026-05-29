"""Тесты оркестрации parent-child на фейках (без Qdrant/Postgres/моделей).

Проверяем дедуп по хешу, бланки, переиндексацию, разбиение на родителей и
схлопывание детей в уникальных родителей на поиске.
"""

from __future__ import annotations

import pytest

from elion_dal.chunking.chunker import Chunk
from elion_dal.embedding.base import Embedding, SparseVector
from elion_dal.service.sync import IndexService, UpsertCounts
from elion_dal.store.pg_repo import DocInput, ParentRecord, SectionInput
from elion_dal.store.qdrant_repo import SearchHit


class FakePg:
    def __init__(self):
        self.hashes: dict[str, str] = {}
        self.docs: dict[str, DocInput] = {}
        self.parents: dict[str, list] = {}
        self.sources: set[str] = set()
        self.touched: list[str] = []

    def ensure_source(self, source_id, name=None):
        self.sources.add(source_id)

    def get_content_hash(self, doc_id):
        return self.hashes.get(doc_id)

    def upsert_document(self, doc, raw_text):
        # Хеш здесь НЕ трогаем (как в реальном PgRepo) — только set_content_hash.
        self.docs[doc.doc_id] = doc

    def set_content_hash(self, doc_id, content_hash):
        self.hashes[doc_id] = content_hash

    def replace_parents_and_chunks(self, doc_id, parents):
        self.parents[doc_id] = list(parents)

    def get_parents(self, parent_ids):
        out: dict[str, ParentRecord] = {}
        for doc_id, plist in self.parents.items():
            doc = self.docs[doc_id]
            for p in plist:
                if p.parent_id in parent_ids:
                    out[p.parent_id] = ParentRecord(
                        parent_id=p.parent_id,
                        doc_id=doc_id,
                        source_id=doc.source_id,
                        title=doc.title,
                        url=p.url,
                        heading_path=p.heading_path,
                        text=p.text,
                    )
        return out

    def touch_source_indexed(self, source_id):
        self.touched.append(source_id)

    def delete_by_source(self, source_id):
        return 0, 0


class FakeQdrant:
    def __init__(self):
        self.points: dict[str, list] = {}
        self.deleted_docs: list[str] = []
        self.search_hits: list[SearchHit] = []
        self.fail_upserts: int = 0  # сколько ближайших upsert_chunks уронить

    def delete_by_doc(self, doc_id):
        self.deleted_docs.append(doc_id)
        self.points.pop(doc_id, None)

    def upsert_chunks(self, points):
        if self.fail_upserts > 0:
            self.fail_upserts -= 1
            raise RuntimeError("qdrant upsert failed")
        for p in points:
            self.points.setdefault(p.payload["doc_id"], []).append(p)
        return len(points)

    def delete_by_source(self, source_id):
        pass

    def search(self, embedding, limit, source_ids=(), min_published_ts=0):
        return self.search_hits[:limit]

    def ping(self):
        return True


class FakeProvider:
    name = "fake"
    dim = 4
    sparse_uses_idf = False

    def embed_documents(self, texts):
        return [Embedding(dense=[0.0] * 4, sparse=SparseVector([1], [1.0])) for _ in texts]

    def embed_query(self, text):
        return Embedding(dense=[0.0] * 4, sparse=SparseVector([1], [1.0]))


class FakeChunker:
    def split(self, text):
        parts = [p for p in text.split("|") if p]
        return [Chunk(index=i, text=p, token_count=len(p)) for i, p in enumerate(parts)]


def make_service():
    return IndexService(FakePg(), FakeQdrant(), FakeProvider(), FakeChunker(), parent_fanout=5)


def section(text, sid="0", url="u"):
    return SectionInput(section_id=sid, heading_path=[], url=url, text=text)


def doc(text="a|b|c", h="h1", index=True, doc_id="d1", sections=None):
    if sections is None:
        sections = [section(text)]
    return DocInput(
        doc_id=doc_id,
        source_id="s1",
        url="u",
        title="t",
        lang="ru",
        published_ts=0,
        content_hash=h,
        index_in_rag=index,
        sections=sections,
    )


def test_new_document_indexed():
    svc = make_service()
    counts = UpsertCounts()
    svc.process_document(doc(), counts)
    assert counts.indexed == 1
    assert counts.parents_upserted == 1
    assert counts.chunks_upserted == 3
    assert svc.qdrant.points["d1"]


def test_unchanged_document_skipped():
    svc = make_service()
    counts = UpsertCounts()
    svc.process_document(doc(h="same"), counts)
    svc.process_document(doc(h="same"), counts)
    assert counts.indexed == 1
    assert counts.skipped == 1


def test_changed_document_reindexed():
    svc = make_service()
    counts = UpsertCounts()
    svc.process_document(doc(text="a|b", h="v1"), counts)
    svc.process_document(doc(text="a|b|c|d", h="v2"), counts)
    assert counts.indexed == 2
    assert "d1" in svc.qdrant.deleted_docs
    assert len(svc.qdrant.points["d1"]) == 4


def test_blank_not_indexed():
    svc = make_service()
    counts = UpsertCounts()
    svc.process_document(doc(index=False), counts)
    assert counts.blank == 1
    assert counts.chunks_upserted == 0
    assert svc.qdrant.points.get("d1") is None


def test_content_hash_autocomputed():
    svc = make_service()
    counts = UpsertCounts()
    svc.process_document(doc(h=""), counts)
    assert svc.pg.hashes["d1"]


def test_hash_not_committed_until_qdrant_success():
    """A1: при сбое Qdrant хеш не фиксируется -> следующий прогон переиндексирует."""
    svc = make_service()
    svc.qdrant.fail_upserts = 1  # первый upsert падает
    counts = UpsertCounts()
    with pytest.raises(RuntimeError):
        svc.process_document(doc(h="v1"), counts)
    assert svc.pg.get_content_hash("d1") is None  # хеш НЕ зафиксирован

    # Qdrant ожил: повторный прогон индексирует, а не пропускает по хешу.
    counts2 = UpsertCounts()
    svc.process_document(doc(h="v1"), counts2)
    assert counts2.indexed == 1
    assert counts2.skipped == 0
    assert svc.pg.get_content_hash("d1") == "v1"
    assert svc.qdrant.points["d1"]


def test_multisection_creates_two_parents():
    svc = make_service()
    counts = UpsertCounts()
    sections = [section("a|b", sid="1"), section("c|d|e", sid="2")]
    svc.process_document(doc(sections=sections), counts)
    assert counts.parents_upserted == 2
    assert counts.chunks_upserted == 5  # 2 + 3 детей


def test_search_collapses_children_to_parents():
    svc = make_service()
    counts = UpsertCounts()
    sections = [section("a|b", sid="1"), section("c|d", sid="2")]
    svc.process_document(doc(sections=sections), counts)

    pa, pb = "d1::1", "d1::2"
    # Дети двух родителей вперемешку; RRF-порядок уже задан списком.
    svc.qdrant.search_hits = [
        SearchHit(
            chunk_id=f"{pb}#0", parent_id=pb, doc_id="d1", source_id="s1", text="c", score=0.9
        ),
        SearchHit(
            chunk_id=f"{pa}#0", parent_id=pa, doc_id="d1", source_id="s1", text="a", score=0.7
        ),
        SearchHit(
            chunk_id=f"{pb}#1", parent_id=pb, doc_id="d1", source_id="s1", text="d", score=0.5
        ),
    ]
    hits = svc.search("q", top_k=2, source_ids=[], min_published_ts=0)
    assert [h.parent_id for h in hits] == [pb, pa]  # уникальные родители в порядке RRF
    assert hits[0].score == 0.9
    assert hits[0].matched_child == "c"  # сниппет ребёнка-победителя
    assert hits[0].text == "c|d"  # текст родителя (вся секция)
