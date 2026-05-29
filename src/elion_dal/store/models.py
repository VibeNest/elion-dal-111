"""SQLAlchemy-модели source-of-truth (Postgres) для parent-child retrieval.

Иерархия: Document -> Parent (секция) -> Chunk (ребёнок).
В Qdrant индексируются только дети; родители живут здесь и отдаются на поиске
для генерации. Postgres — истина: можно пересобрать индекс без перекраулинга,
дедуп по content_hash.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

QDRANT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def parent_pk(doc_id: str, section_id: str) -> str:
    """Стабильный ID родителя (секции) в пределах документа."""
    return f"{doc_id}::{section_id}"


def chunk_id(parent_id: str, index: int) -> str:
    """Стабильный ID ребёнка в пределах родителя (он же основа point_id Qdrant)."""
    return f"{parent_id}#{index}"


def point_id(parent_id: str, index: int) -> str:
    """Детерминированный UUID точки Qdrant — для идемпотентного upsert."""
    return str(uuid.uuid5(QDRANT_NAMESPACE, chunk_id(parent_id, index)))


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    source_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    source_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("sources.source_id", ondelete="CASCADE"), index=True
    )
    url: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str] = mapped_column(Text, default="")
    lang: Mapped[str] = mapped_column(String(16), default="ru")
    published_ts: Mapped[int] = mapped_column(BigInteger, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    index_in_rag: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    parents: Mapped[list[Parent]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Parent(Base):
    """Родитель = секция документа. Возвращается на поиске для генерации."""

    __tablename__ = "parents"

    parent_id: Mapped[str] = mapped_column(String(384), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(256), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    section_id: Mapped[str] = mapped_column(String(128), default="0")
    heading_path: Mapped[list] = mapped_column(JSON, default=list)
    url: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="parents")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="parent", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    """Ребёнок — то, что эмбеддится и индексируется в Qdrant."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(400), primary_key=True)
    parent_id: Mapped[str] = mapped_column(
        String(384), ForeignKey("parents.parent_id", ondelete="CASCADE"), index=True
    )
    doc_id: Mapped[str] = mapped_column(String(256), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    parent: Mapped[Parent] = relationship(back_populates="chunks")
