"""Unit-тесты SettingsStore на SQLite (offline)."""

from __future__ import annotations

from elion_dal.config import Settings
from elion_dal.store.pg_repo import PgRepo
from elion_dal.store.settings_store import SettingsStore


def make_store(tmp_path):
    repo = PgRepo(f"sqlite:///{(tmp_path / 'set.db').as_posix()}")
    repo.create_all()  # создаёт в т.ч. app_settings
    store = SettingsStore(repo.engine)
    store.load()
    return store


def test_get_none_when_not_set(tmp_path):
    store = make_store(tmp_path)
    assert store.get("search_parent_fanout") is None
    assert store.get("embedding_quantize") is None


def test_set_and_typed_get(tmp_path):
    store = make_store(tmp_path)
    store.set_many(
        {"search_parent_fanout": "7", "recency_weight": "0.5", "embedding_quantize": "true"}
    )
    assert store.get("search_parent_fanout") == 7  # int
    assert store.get("recency_weight") == 0.5  # float
    assert store.get("embedding_quantize") is True  # bool

    # Перечитать из БД (новый стор) — значения сохранились.
    store2 = SettingsStore(store.engine)
    store2.load()
    assert store2.get("search_parent_fanout") == 7


def test_unknown_keys_ignored(tmp_path):
    store = make_store(tmp_path)
    store.set_many({"not_a_setting": "x", "search_prefetch": "30"})
    assert store.get("search_prefetch") == 30
    assert store.get("not_a_setting") is None


def test_view_marks_overrides(tmp_path):
    store = make_store(tmp_path)
    store.set_many({"search_parent_fanout": "9"})
    view = {v.key: v for v in store.view(Settings())}
    assert view["search_parent_fanout"].value == 9
    assert view["search_parent_fanout"].is_override is True
    # не переопределённое -> значение из .env-дефолтов, override=False
    assert view["search_prefetch"].is_override is False
    assert view["search_prefetch"].value == Settings().search_prefetch
    # restart-тир помечен
    assert view["embedding_backend"].tier == "restart"


def test_new_chunk_fields_present_and_typed(tmp_path):
    store = make_store(tmp_path)
    store.set_many(
        {"chunk_min_tokens": "20", "chunk_separator_mode": "token", "search_top_k": "7"}
    )
    assert store.get("chunk_min_tokens") == 20  # int
    assert store.get("chunk_separator_mode") == "token"  # str
    assert store.get("search_top_k") == 7  # int
    view = {v.key: v for v in store.view(Settings())}
    # токенайзер длины — restart-тир (применяется на старте)
    assert view["chunk_tokenizer_model"].tier == "restart"
    assert view["chunk_tokenizer_model"].value == Settings().chunk_tokenizer_model
    # фильтр мусора и стратегия — live
    assert view["chunk_min_tokens"].tier == "live"
    assert view["chunk_separator_mode"].tier == "live"
    assert view["search_top_k"].tier == "live"


def test_load_without_table_is_safe(tmp_path):
    # Стор поверх БД без миграций (нет таблицы) -> пустые overrides, без падения.
    repo = PgRepo(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    store = SettingsStore(repo.engine)
    store.load()
    assert store.get("search_parent_fanout") is None
