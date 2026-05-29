"""Unit-тесты gRPC-servicer: маппинг proto<->DTO, в т.ч. fallback «text -> один родитель»."""

from __future__ import annotations

from elion_dal.config import Settings
from elion_dal.grpc_gen import vectorstore_pb2 as pb
from elion_dal.service.servicer import VectorStoreServicer
from elion_dal.service.sync import ParentHit


class FakeIndex:
    def __init__(self):
        self.docs = []
        self._hits = []

    def process_document(self, doc, counts):
        self.docs.append(doc)
        counts.received += 1
        counts.indexed += 1
        counts.parents_upserted += len(doc.sections)
        counts.chunks_upserted += sum(1 for _ in doc.sections)

    def set_hits(self, hits):
        self._hits = hits

    def search(self, query, top_k, source_ids, min_published_ts):
        return self._hits


def make_servicer():
    return VectorStoreServicer(FakeIndex(), Settings())


def test_upsert_maps_sections():
    svc = make_servicer()
    d = pb.Document(doc_id="d1", source_id="s1", index_in_rag=True)
    sec = d.sections.add()
    sec.section_id = "4.9"
    sec.text = "текст секции"
    sec.heading_path.append("4. Регистрация")

    result = svc.UpsertDocuments(iter([d]), None)
    assert result.documents_received == 1
    mapped = svc.index.docs[0]
    assert len(mapped.sections) == 1
    assert mapped.sections[0].section_id == "4.9"
    assert mapped.sections[0].heading_path == ["4. Регистрация"]


def test_upsert_fallback_text_becomes_single_parent():
    svc = make_servicer()
    d = pb.Document(doc_id="d2", source_id="s1", text="плоский текст", index_in_rag=True)

    svc.UpsertDocuments(iter([d]), None)
    mapped = svc.index.docs[0]
    assert len(mapped.sections) == 1
    assert mapped.sections[0].section_id == "0"
    assert mapped.sections[0].text == "плоский текст"


def test_search_maps_parent_hits():
    svc = make_servicer()
    svc.index.set_hits(
        [
            ParentHit(
                parent_id="d1::0", doc_id="d1", source_id="s1", title="t", url="u",
                heading_path=["A"], text="родитель", matched_child="ребёнок", score=0.42,
            )
        ]
    )
    resp = svc.Search(pb.SearchRequest(query="q", top_k=1), None)
    assert len(resp.hits) == 1
    h = resp.hits[0]
    assert h.parent_id == "d1::0"
    assert list(h.heading_path) == ["A"]
    assert h.text == "родитель"
    assert h.matched_child == "ребёнок"
    assert abs(h.score - 0.42) < 1e-6


def test_search_uses_config_top_k_when_zero():
    svc = make_servicer()
    captured = {}
    orig = svc.index.search

    def spy(query, top_k, source_ids, min_published_ts):
        captured["top_k"] = top_k
        return orig(query, top_k, source_ids, min_published_ts)

    svc.index.search = spy
    svc.Search(pb.SearchRequest(query="q", top_k=0), None)
    assert captured["top_k"] == Settings().search_top_k
