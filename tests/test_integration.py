"""Integration: реальный round-trip upsert -> search на поднятых Qdrant + Postgres.

Требует:
    docker compose up -d qdrant postgres
    alembic upgrade head   (или PgRepo.create_all)
    EMBEDDING_BACKEND=fastembed  (модель скачается на первом запуске)

Запуск только этих тестов:  pytest -m integration
По умолчанию (без -m) тоже выполнятся, но упадут без инфраструктуры — поэтому
помечены skip при недоступности зависимостей.
"""

from __future__ import annotations

import pytest

from elion_dal.config import get_settings
from elion_dal.service.bootstrap import build_index_service
from elion_dal.service.sync import UpsertCounts
from elion_dal.store.pg_repo import DocInput, SectionInput, sha256

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def index():
    settings = get_settings()
    try:
        svc = build_index_service(settings)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"инфраструктура/модель недоступна: {e}")
    if not svc.pg.ping() or not svc.qdrant.ping():
        pytest.skip("Qdrant/Postgres не подняты")
    svc.pg.create_all()  # на случай, если миграции не прогнаны
    return svc


def test_roundtrip(index):
    text = (
        "Олимпиада Физтех по биологии проводится в два этапа. "
        "Отборочный этап проходит онлайн осенью, заключительный — очно весной. "
        "Победители и призёры получают льготы при поступлении на биомед."
    )
    doc = DocInput(
        doc_id="it-bio-1",
        source_id="it_source",
        url="https://bio-olymp.mipt.ru/",
        title="Олимпиада по биологии",
        lang="ru",
        published_ts=0,
        content_hash=sha256(text),
        index_in_rag=True,
        sections=[
            SectionInput(
                section_id="0", heading_path=[], url="https://bio-olymp.mipt.ru/", text=text
            )
        ],
    )
    counts = UpsertCounts()
    index.process_document(doc, counts)
    assert counts.indexed == 1
    assert counts.parents_upserted == 1
    assert counts.chunks_upserted >= 1

    hits = index.search(
        "когда проходит олимпиада по биологии",
        top_k=3,
        source_ids=["it_source"],
        min_published_ts=0,
    )
    assert hits, "поиск ничего не вернул"
    assert hits[0].source_id == "it_source"
    assert hits[0].parent_id == "it-bio-1::0"
    assert "олимпиад" in hits[0].text.lower()  # текст родителя

    # очистка
    index.delete_source("it_source")
