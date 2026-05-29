"""Unit-тесты PgRepo на временном SQLite (offline, без модели и без Postgres-сервера).

Проверяем дедуп по хешу, запись родителей/детей, get_parents с join документа и
каскадное удаление по источнику.
"""

from __future__ import annotations

from elion_dal.chunking.chunker import Chunk
from elion_dal.store.pg_repo import DocInput, ParentBuild, PgRepo, SectionInput


def make_repo(tmp_path):
    repo = PgRepo(f"sqlite:///{(tmp_path / 'elion_test.db').as_posix()}")
    repo.create_all()
    return repo


def make_doc(doc_id="d1", content_hash="h1"):
    return DocInput(
        doc_id=doc_id, source_id="s1", url="u", title="Заголовок", lang="ru",
        published_ts=0, content_hash=content_hash, index_in_rag=True,
        sections=[SectionInput(section_id="0", heading_path=["A"], url="u", text="секция")],
    )


def make_parent(parent_id="d1::0"):
    return ParentBuild(
        parent_id=parent_id, section_id="0", heading_path=["A", "A.1"], url="u",
        text="текст родителя", token_count=2, ordinal=0,
        children=[Chunk(0, "ребёнок1", 1), Chunk(1, "ребёнок2", 1)],
    )


def test_upsert_and_get_content_hash(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    assert repo.get_content_hash("d1") is None
    repo.upsert_document(make_doc(content_hash="abc"), raw_text="секция")
    assert repo.get_content_hash("d1") == "abc"


def test_parents_and_children_with_join(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])

    recs = repo.get_parents(["d1::0"])
    assert "d1::0" in recs
    rec = recs["d1::0"]
    assert rec.text == "текст родителя"
    assert rec.source_id == "s1"          # join с documents
    assert rec.title == "Заголовок"
    assert rec.heading_path == ["A", "A.1"]


def test_replace_parents_is_idempotent(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])
    # Повторная запись (меньше детей) не плодит дубликаты.
    p = make_parent()
    p.children = [Chunk(0, "только один", 1)]
    repo.replace_parents_and_chunks("d1", [p])
    docs, chunks = repo.delete_by_source("s1")
    assert docs == 1
    assert chunks == 1


def test_delete_by_source_cascades(tmp_path):
    repo = make_repo(tmp_path)
    repo.ensure_source("s1")
    repo.upsert_document(make_doc(), raw_text="секция")
    repo.replace_parents_and_chunks("d1", [make_parent()])

    docs, chunks = repo.delete_by_source("s1")
    assert docs == 1
    assert chunks == 2
    assert repo.get_parents(["d1::0"]) == {}  # каскад снёс родителей и детей
