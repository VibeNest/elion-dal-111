"""Тесты веб-админки (FastAPI TestClient) на фейковом IndexService — без модели/инфры."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from elion_dal.admin.web import create_app
from elion_dal.config import Settings
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

    def list_documents(self, source_id=""):
        return [
            {
                "doc_id": "d1", "source_id": "s1", "title": "Док", "lang": "ru",
                "published_ts": 0, "index_in_rag": True, "indexed": True,
                "parent_count": 1, "chunk_count": 2,
            }
        ]

    def get_document_detail(self, doc_id):
        return {
            "doc_id": doc_id, "source_id": "s1", "title": "Док", "url": "u", "lang": "ru",
            "published_ts": 0, "index_in_rag": True, "indexed": True,
            "parents": [
                {
                    "parent_id": "d1::0", "section_id": "0", "heading_path": ["A"],
                    "ordinal": 0, "token_count": 2, "text": "p",
                    "chunks": [
                        {"chunk_id": "d1::0#0", "chunk_index": 0, "text": "c0", "token_count": 1}
                    ],
                }
            ],
        }

    def preview_chunking(self, text, chunk_tokens=None, chunk_overlap=None,
                         min_tokens=None, separator_mode=None):
        return {
            "chunks": [{"index": 0, "text": text, "token_count": 2}],
            "summary": {
                "count": 1, "total_tokens": 2, "avg_tokens": 2, "dropped": 0,
                "chunk_tokens": chunk_tokens or 400, "chunk_overlap": chunk_overlap or 64,
                "min_tokens": min_tokens or 0, "separator_mode": separator_mode or "structured",
            },
        }

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
        data={"search_parent_fanout": "7", "embedding_quantize": "on"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert idx.updated_settings["search_parent_fanout"] == "7"
    assert idx.updated_settings["embedding_quantize"] == "true"  # чекбокс отмечен


def test_dashboard_has_chunk_sections():
    t = client().get("/").text
    assert "Превью нарезки" in t
    assert "Документы и чанки" in t
    # JS-функции просмотра/превью присутствуют
    assert "doPreview" in t
    assert "loadDocs" in t
    assert "showChunks" in t


def test_api_documents_proxy():
    r = client().get("/api/documents")
    assert r.status_code == 200
    data = r.json()
    assert data[0]["doc_id"] == "d1"
    assert data[0]["chunk_count"] == 2


def test_api_document_detail_proxy():
    r = client().get("/api/documents/d1/detail")
    assert r.status_code == 200
    body = r.json()
    assert body["parents"][0]["chunks"][0]["chunk_id"] == "d1::0#0"


def test_api_chunk_preview_proxy():
    r = client().post("/api/chunk-preview", data={"text": "привет мир", "chunk_tokens": "50"})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["count"] == 1
    assert body["summary"]["chunk_tokens"] == 50


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


def test_admin_basic_auth_enforced_when_password_set():
    app = create_app(FakeIndex(), Settings(admin_user="admin", admin_password="secret"))
    c = TestClient(app)
    assert c.get("/").status_code == 401  # без креды
    assert c.get("/", auth=("admin", "wrong")).status_code == 401  # неверный пароль
    assert c.get("/", auth=("admin", "secret")).status_code == 200  # верные креды


def test_admin_open_when_no_password():
    # Пустой ADMIN_PASSWORD -> auth выключен (dev).
    c = TestClient(create_app(FakeIndex(), Settings(admin_password="")))
    assert c.get("/").status_code == 200


def test_healthz_open_even_with_auth():
    # /healthz доступен без креды даже при включённом Basic-auth (для проб платформы).
    c = TestClient(create_app(FakeIndex(), Settings(admin_password="secret")))
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
