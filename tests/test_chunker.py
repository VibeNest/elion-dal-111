"""Unit-тесты чанкера (offline: длина считается через length_fn, без токенайзера)."""

from __future__ import annotations

import pytest

from elion_dal.chunking.chunker import Chunker

# Длина в "словах" — детерминированно и без загрузки модели.
WORDS = lambda s: len(s.split())  # noqa: E731


def make_chunker(tokens=10, overlap=2):
    return Chunker(chunk_tokens=tokens, chunk_overlap=overlap, length_fn=WORDS)


def test_empty_text_returns_no_chunks():
    assert make_chunker().split("") == []
    assert make_chunker().split("   \n  ") == []


def test_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunk_tokens=10, chunk_overlap=10, length_fn=WORDS)


def test_splits_long_text_into_sequential_chunks():
    text = " ".join(f"слово{i}" for i in range(45))  # 45 "токенов" при tokens=10
    chunks = make_chunker(tokens=10, overlap=2).split(text)
    assert len(chunks) >= 4
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Каждый чанк не превышает лимит (текст дробится по пробелам).
    assert all(c.token_count <= 10 for c in chunks)
    assert all(c.text for c in chunks)


def test_short_text_is_single_chunk():
    chunks = make_chunker(tokens=10).split("одно два три")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].token_count == 3
