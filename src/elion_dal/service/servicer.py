"""Реализация gRPC-сервиса VectorStore поверх IndexService."""

from __future__ import annotations

from ..config import Settings
from ..grpc_gen import vectorstore_pb2 as pb
from ..grpc_gen import vectorstore_pb2_grpc as pb_grpc
from ..store.pg_repo import DocInput, SectionInput
from .sync import IndexService, UpsertCounts


class VectorStoreServicer(pb_grpc.VectorStoreServicer):
    def __init__(self, index: IndexService, settings: Settings) -> None:
        self.index = index
        self.settings = settings

    def UpsertDocuments(self, request_iterator, context) -> pb.UpsertResult:
        counts = UpsertCounts()
        for d in request_iterator:
            sections = [
                SectionInput(
                    section_id=s.section_id,
                    heading_path=list(s.heading_path),
                    url=s.url,
                    text=s.text,
                    published_ts=s.published_ts,
                    content_hash=s.content_hash,
                )
                for s in d.sections
            ]
            # Fallback: документ без секций -> весь текст как один родитель.
            if not sections and d.text:
                sections = [SectionInput(section_id="0", heading_path=[], url=d.url, text=d.text)]
            doc = DocInput(
                doc_id=d.doc_id,
                source_id=d.source_id or "unknown",
                url=d.url,
                title=d.title,
                lang=d.lang or "ru",
                published_ts=d.published_ts,
                content_hash=d.content_hash,
                index_in_rag=d.index_in_rag,
                sections=sections,
            )
            self.index.process_document(doc, counts)
        return pb.UpsertResult(
            documents_received=counts.received,
            documents_indexed=counts.indexed,
            documents_skipped=counts.skipped,
            documents_blank=counts.blank,
            chunks_upserted=counts.chunks_upserted,
            parents_upserted=counts.parents_upserted,
        )

    def Search(self, request, context) -> pb.SearchResponse:
        top_k = request.top_k or self.settings.search_top_k
        hits = self.index.search(
            query=request.query,
            top_k=top_k,
            source_ids=list(request.source_ids),
            min_published_ts=request.min_published_ts,
        )
        return pb.SearchResponse(
            hits=[
                pb.Hit(
                    parent_id=h.parent_id,
                    doc_id=h.doc_id,
                    source_id=h.source_id,
                    url=h.url,
                    title=h.title,
                    heading_path=h.heading_path,
                    text=h.text,
                    matched_child=h.matched_child,
                    score=h.score,
                )
                for h in hits
            ]
        )

    def DeleteBySource(self, request, context) -> pb.DeleteResult:
        docs, chunks = self.index.delete_source(request.source_id)
        return pb.DeleteResult(documents_deleted=docs, chunks_deleted=chunks)

    def HealthCheck(self, request, context) -> pb.HealthStatus:
        h = self.index.health()
        return pb.HealthStatus(
            ok=h["ok"],
            qdrant_ok=h["qdrant_ok"],
            postgres_ok=h["postgres_ok"],
            embedding_backend=h["embedding_backend"],
            detail="",
        )
