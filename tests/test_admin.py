"""Тесты веб-админки (FastAPI TestClient) на фейковом IndexService — без модели/инфры."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from elion_dal.admin.web import create_app
from elion_dal.service.sync import ParentHit
from elion_dal.store.pg_repo import SourceStats, StoreStats
from elion_dal.store.settings_store import SettingView


class FakeIndex:
    def __init__(self):
        self.deleted_sources = []
        self.deleted_docs = []
        self.uploaded = []
        self.updated_settings = None

    def get_stats(self):
        return StoreStats(2, 3, 9, [SourceStats("s1", "Источник 1", 1700000000, 2, 3, 9)])

    def list_sources(self):
        return self.get_stats().sources

    def search(self, query, top_k, source_ids, min_published_ts):
        return [
            ParentHit(
                parent_id="d1::0",
                doc_id="d1",
                source_id="s1",
                title="Заголовок",
                url="u",
                heading_path=["A"],
                text="текст родителя",
                matched_child="ребёнок",
                score=0.5,
                dense_score=0.8,
            )
        ]

    def delete_source(self, source_id):
        self.deleted_sources.append(source_id)
        return 1, 3

    def delete_doc(self, doc_id):
        self.deleted_docs.append(doc_id)
        return 1, 3

    def process_document(self, doc, counts):
        self.uploaded.append(doc)

    def settings_view(self):
        return [
            SettingView("search_parent_fanout", "Fan-out", "live", "int", 5, False),
            SettingView("rerank_enabled", "Реранкер", "live", "bool", False, False),
            SettingView("embedding_backend", "Бэкенд", "restart", "str", "fastembed", False),
        ]

    def update_settings(self, items):
        self.updated_settings = items


def client():
    return TestClient(create_app(FakeIndex()))


def test_dashboard_renders():
    r = client().get("/")
    assert r.status_code == 200
    assert "Элион — DAL Admin" in r.text
    assert "Источник 1" in r.text  # строка таблицы источников
    assert "Настройки" in r.text  # секция редактирования настроек
    assert "после рестарта" in r.text  # пометка у restart-настройки


def test_settings_post_updates_index():
    idx = FakeIndex()
    c = TestClient(create_app(idx))
    r = c.post(
        "/settings",
        data={"search_parent_fanout": "7", "rerank_enabled": "on"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert idx.updated_settings["search_parent_fanout"] == "7"
    assert idx.updated_settings["rerank_enabled"] == "true"  # чекбокс отмечен
    assert idx.updated_settings["embedding_quantize"] == "false"  # bool не отмечен -> false


def test_api_stats():
    r = client().get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_documents"] == 2
    assert data["total_chunks"] == 9
    assert data["sources"][0]["source_id"] == "s1"


def test_api_search_returns_dense_score():
    r = client().post("/api/search", data={"query": "вопрос", "top_k": 3})
    assert r.status_code == 200
    hits = r.json()
    assert hits[0]["parent_id"] == "d1::0"
    assert hits[0]["dense_score"] == 0.8


def test_delete_source_and_doc():
    app = create_app(FakeIndex())
    c = TestClient(app)
    # follow_redirects=False: эндпоинт отвечает 303 на "/".
    r1 = c.post("/sources/s1/delete", follow_redirects=False)
    assert r1.status_code == 303
    r2 = c.post("/docs/d1/delete", follow_redirects=False)
    assert r2.status_code == 303


def test_upload_indexes_docx():
    import docx

    app_index = FakeIndex()
    c = TestClient(create_app(app_index))
    # Сформируем настоящий .docx в памяти.
    buf = io.BytesIO()
    d = docx.Document()
    d.add_paragraph("Правила приёма 2026: баллы и сроки.")
    d.save(buf)
    buf.seek(0)
    r = c.post(
        "/upload",
        files={
            "file": (
                "rules.docx",
                buf,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        data={"source_id": "knowledge_base"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert len(app_index.uploaded) == 1
    doc = app_index.uploaded[0]
    assert doc.source_id == "knowledge_base"
    assert "Правила приёма" in doc.sections[0].text
