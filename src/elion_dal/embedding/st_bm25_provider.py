"""Прод-провайдер рекомендуемой конфигурации (по исследованию leaderboard).

dense  = deepvk/USER-bge-m3 (sentence-transformers, L2-норма, cosine)
sparse = BM25 (fastembed Qdrant/bm25; IDF применяет Qdrant -> sparse_uses_idf=True)

Слияние dense+sparse делает Qdrant (FusionQuery RRF) — см. store/qdrant_repo.py.
Выбор обоснован бенчем: hybrid USER-bge-m3 + BM25 -> Recall@5 0.98 (лучший конфиг).
USER-bge-m3 — dense-only (нет sparse-головы BGE-M3), поэтому лексическую ветку даёт
отдельная BM25-модель fastembed, а не learned-веса.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import Embedding, EmbeddingProvider, SparseVector


class StBm25Provider(EmbeddingProvider):
    name = "st-bm25"
    sparse_uses_idf = True  # BM25 -> Qdrant применяет IDF-модификатор коллекции

    def __init__(
        self,
        dense_model: str = "deepvk/USER-bge-m3",
        sparse_model: str = "Qdrant/bm25",
        dim: int = 1024,
        quantize: bool = False,
    ) -> None:
        from fastembed import SparseTextEmbedding
        from sentence_transformers import SentenceTransformer

        self._dense = SentenceTransformer(dense_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)
        # ST-dense работает в fp32; параметр quantize принимаем для единообразия фабрики.
        self.quantized = False
        # Размерность берём по факту из модели (снимает рассинхрон с EMBEDDING_DIM).
        probe = self._dense.encode(["x"], normalize_embeddings=True, convert_to_numpy=True)
        self.dim = int(len(probe[0]))

    @staticmethod
    def _to_sparse(s) -> SparseVector:
        return SparseVector(
            indices=[int(i) for i in s.indices.tolist()],
            values=[float(v) for v in s.values.tolist()],
        )

    def _dense_encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._dense.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [[float(x) for x in v.tolist()] for v in vecs]

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        texts = list(texts)
        dense = self._dense_encode(texts)
        sparse = list(self._sparse.embed(texts))
        return [
            Embedding(dense=d, sparse=self._to_sparse(s))
            for d, s in zip(dense, sparse, strict=True)
        ]

    def embed_query(self, text: str) -> Embedding:
        dense = self._dense_encode([text])[0]
        sparse = next(iter(self._sparse.query_embed([text])))
        return Embedding(dense=dense, sparse=self._to_sparse(sparse))
