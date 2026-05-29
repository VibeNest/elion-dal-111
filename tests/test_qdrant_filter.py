"""Unit-тесты построения Qdrant-фильтра (offline: embedded :memory:-клиент, без сервера)."""

from __future__ import annotations

from elion_dal.store.qdrant_repo import QdrantRepo


def make_repo():
    # ":memory:" -> embedded-клиент в процессе, без сетевого сервера.
    return QdrantRepo(url=":memory:", collection="t", dim=4, sparse_uses_idf=False)


def test_no_filters_returns_none():
    assert make_repo()._filter([], 0) is None


def test_source_filter_only():
    f = make_repo()._filter(["a", "b"], 0)
    assert f is not None
    assert len(f.must) == 1


def test_date_filter_only():
    f = make_repo()._filter([], 1700000000)
    assert f is not None
    assert len(f.must) == 1


def test_both_filters_combined():
    f = make_repo()._filter(["a"], 1700000000)
    assert f is not None
    assert len(f.must) == 2
