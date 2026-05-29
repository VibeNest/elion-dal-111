"""FastEmbed-провайдер: лёгкий ONNX-бэкенд на CPU (dense + BM25 sparse).

ВАЖНО: fastembed НЕ содержит BGE-M3. Настоящий BGE-M3 (dense + learned sparse,
вариант A по ТЗ) даёт только FlagEmbedding (см. flag_provider.py). Этот провайдер —
быстрый ONNX-вариант для сравнения по латентности на CPU: dense = multilingual-e5
(1024, хорошо тянет русский), sparse = BM25 (Qdrant/bm25, IDF-модификатор Qdrant).

Выбор между этим бэкендом и FlagEmbedding делается по bench/benchmark_embeddings.py.
Для e5/bge-моделей fastembed сам подставляет query/passage-префиксы через
query_embed()/passage_embed().
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import Embedding, EmbeddingProvider, SparseVector


class FastEmbedProvider(EmbeddingProvider):
    name = "fastembed"
    sparse_uses_idf = True

    def __init__(
        self,
        dense_model: str = "intfloat/multilingual-e5-large",
        dim: int = 1024,
        sparse_model: str = "Qdrant/bm25",
    ) -> None:
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._dense = TextEmbedding(model_name=dense_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)
        # Размерность берём по факту из модели, а не из конфига (снимает footgun
        # с рассинхроном EMBEDDING_DIM и реальной моделью).
        probe = next(iter(self._dense.embed(["x"])))
        self.dim = len(probe.tolist()) if hasattr(probe, "tolist") else len(probe)

    @staticmethod
    def _to_sparse(s) -> SparseVector:
        return SparseVector(
            indices=[int(i) for i in s.indices.tolist()],
            values=[float(v) for v in s.values.tolist()],
        )

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        texts = list(texts)
        dense = list(self._dense.passage_embed(texts))
        sparse = list(self._sparse.embed(texts))
        return [
            Embedding(dense=[float(x) for x in d.tolist()], sparse=self._to_sparse(s))
            for d, s in zip(dense, sparse, strict=True)
        ]

    def embed_query(self, text: str) -> Embedding:
        dense = next(iter(self._dense.query_embed([text])))
        sparse = next(iter(self._sparse.query_embed([text])))
        return Embedding(dense=[float(x) for x in dense.tolist()], sparse=self._to_sparse(sparse))
