"""Тесты оркестрации parent-child на фейках (без Qdrant/Postgres/моделей).

Проверяем дедуп по хешу, бланки, переиндексацию, разбиение на родителей и
схлопывание детей в уникальных родителей на поиске.
"""

from __future__ import annotations

import time

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
                        published_ts=doc.published_ts,
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
        self.prefetch: int = 20
        self.last_limit: int | None = None
        self.last_prefetch: int | None = None

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

    def search(self, embedding, limit, source_ids=(), min_published_ts=0, prefetch_limit=None):
        self.last_limit = limit
        self.last_prefetch = prefetch_limit
        return self.search_hits[:limit]

    def dense_scores(self, embedding, limit, source_ids=(), min_published_ts=0):
        return getattr(self, "dense_map", {})

    def ping(self):
        return True


class FakeStore:
    """Минимальный SettingsStore для тестов: get() отдаёт уже типизированные значения."""

    def __init__(self, values: dict | None = None):
        self._v = values or {}

    def get(self, key):
        return self._v.get(key)

    def view(self, base):
        return []


class FakeReranker:
    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text

    def rerank(self, query, docs):
        return [self.scores_by_text[d] for d in docs]


class FakeProvider:
    name = "fake"
    dim = 4
    sparse_uses_idf = False

    def embed_documents(self, texts):
        return [Embedding(dense=[0.0] * 4, sparse=SparseVector([1], [1.0])) for _ in texts]

    def embed_query(self, text):
        return Embedding(dense=[0.0] * 4, sparse=SparseVector([1], [1.0]))


class FakeChunker:
    def __init__(self):
        self.chunk_tokens = 10
        self.chunk_overlap = 2

    def split(self, text):
        parts = [p for p in text.split("|") if p]
        return [Chunk(index=i, text=p, token_count=len(p)) for i, p in enumerate(parts)]

    def count_tokens(self, text):
        return len(text)


def make_service(reranker=None, recency_weight=0.0):
    return IndexService(
        FakePg(),
        FakeQdrant(),
        FakeProvider(),
        FakeChunker(),
        parent_fanout=5,
        reranker=reranker,
        recency_weight=recency_weight,
    )


def section(text, sid="0", url="u"):
    return SectionInput(section_id=sid, heading_path=[], url=url, text=text)


def doc(text="a|b|c", h="h1", index=True, doc_id="d1", sections=None, published_ts=0):
    if sections is None:
        sections = [section(text)]
    return DocInput(
        doc_id=doc_id,
        source_id="s1",
        url="u",
        title="t",
        lang="ru",
        published_ts=published_ts,
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


def test_dense_score_populated():
    """B4: dense_score берётся из dense-only запроса по chunk_id matched-ребёнка."""
    svc = make_service()
    counts = UpsertCounts()
    svc.process_document(doc(sections=[section("a|b", sid="1")], doc_id="d1"), counts)
    svc.qdrant.dense_map = {"d1::1#0": 0.83}
    svc.qdrant.search_hits = [
        SearchHit(
            chunk_id="d1::1#0", parent_id="d1::1", doc_id="d1", source_id="s1", text="a", score=0.5
        )
    ]
    hits = svc.search("q", top_k=1, source_ids=[], min_published_ts=0)
    assert abs(hits[0].dense_score - 0.83) < 1e-6


def test_reranker_reorders_parents():
    """B5: реранкер переупорядочивает схлопнутых родителей."""
    rr = FakeReranker({"a|b": 0.9, "c|d": 0.1})
    svc = make_service(reranker=rr)
    counts = UpsertCounts()
    svc.process_document(
        doc(sections=[section("a|b", sid="1"), section("c|d", sid="2")], doc_id="d1"), counts
    )
    # RRF-порядок: сначала d1::2 ("c|d"), потом d1::1 ("a|b").
    svc.qdrant.search_hits = [
        SearchHit(
            chunk_id="d1::2#0", parent_id="d1::2", doc_id="d1", source_id="s1", text="c", score=0.9
        ),
        SearchHit(
            chunk_id="d1::1#0", parent_id="d1::1", doc_id="d1", source_id="s1", text="a", score=0.7
        ),
    ]
    hits = svc.search("q", top_k=2, source_ids=[], min_published_ts=0)
    # Реранкер выше оценил "a|b" (d1::1) -> он должен выйти первым.
    assert [h.parent_id for h in hits] == ["d1::1", "d1::2"]


def test_live_settings_override_fanout_and_prefetch():
    """DB-настройки читаются на лету: parent_fanout и prefetch влияют на запрос к Qdrant."""
    svc = IndexService(
        FakePg(),
        FakeQdrant(),
        FakeProvider(),
        FakeChunker(),
        settings_store=FakeStore({"search_parent_fanout": 3, "search_prefetch": 42}),
    )
    svc.search("q", top_k=4, source_ids=[], min_published_ts=0)
    assert svc.qdrant.last_limit == 12  # top_k(4) * fanout(3)
    assert svc.qdrant.last_prefetch == 42


def test_live_rerank_toggle_uses_lazy_factory():
    """rerank_enabled=true в БД -> реранкер лениво грузится фабрикой и переупорядочивает."""
    built = []

    def factory():
        rr = FakeReranker({"a|b": 0.9, "c|d": 0.1})
        built.append(rr)
        return rr

    svc = IndexService(
        FakePg(),
        FakeQdrant(),
        FakeProvider(),
        FakeChunker(),
        settings_store=FakeStore({"rerank_enabled": True}),
        reranker_factory=factory,
    )
    counts = UpsertCounts()
    svc.process_document(
        doc(sections=[section("a|b", sid="1"), section("c|d", sid="2")], doc_id="d1"), counts
    )
    svc.qdrant.search_hits = [
        SearchHit(
            chunk_id="d1::2#0", parent_id="d1::2", doc_id="d1", source_id="s1", text="c", score=0.9
        ),
        SearchHit(
            chunk_id="d1::1#0", parent_id="d1::1", doc_id="d1", source_id="s1", text="a", score=0.7
        ),
    ]
    hits = svc.search("q", top_k=2, source_ids=[], min_published_ts=0)
    assert len(built) == 1  # фабрика вызвана один раз (ленивая загрузка)
    assert [h.parent_id for h in hits] == ["d1::1", "d1::2"]  # реранкер переупорядочил


def test_recency_boost_reorders():
    """B6: свежий документ обгоняет старый при recency_weight>0."""
    svc = make_service(recency_weight=1.0)
    now = int(time.time())
    counts = UpsertCounts()
    svc.process_document(
        doc(text="a", doc_id="dA", h="hA", published_ts=now - 5 * 365 * 86400), counts
    )
    svc.process_document(doc(text="b", doc_id="dB", h="hB", published_ts=now), counts)
    # RRF: старый dA (0.6) выше свежего dB (0.5).
    svc.qdrant.search_hits = [
        SearchHit(
            chunk_id="dA::0#0", parent_id="dA::0", doc_id="dA", source_id="s1", text="a", score=0.6
        ),
        SearchHit(
            chunk_id="dB::0#0", parent_id="dB::0", doc_id="dB", source_id="s1", text="b", score=0.5
        ),
    ]
    hits = svc.search("q", top_k=2, source_ids=[], min_published_ts=0)
    # Свежесть домножает скор -> dB обгоняет dA.
    assert [h.parent_id for h in hits] == ["dB::0", "dA::0"]
