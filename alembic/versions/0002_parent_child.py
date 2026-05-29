"""parent-child: таблица parents, chunks.parent_id

Revision ID: 0002_parent_child
Revises: 0001_init
Create Date: 2026-05-29
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_parent_child"
down_revision: str | None = "0001_init"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Данных ещё нет — пересоздаём chunks под новую схему (parent-child).
    op.drop_index("ix_chunks_doc_id", table_name="chunks")
    op.drop_table("chunks")

    op.create_table(
        "parents",
        sa.Column("parent_id", sa.String(384), primary_key=True),
        sa.Column(
            "doc_id",
            sa.String(256),
            sa.ForeignKey("documents.doc_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("section_id", sa.String(128), nullable=False, server_default="0"),
        sa.Column("heading_path", sa.JSON(), nullable=True),
        sa.Column("url", sa.Text(), nullable=False, server_default=""),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_parents_doc_id", "parents", ["doc_id"])

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(400), primary_key=True),
        sa.Column(
            "parent_id",
            sa.String(384),
            sa.ForeignKey("parents.parent_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("doc_id", sa.String(256), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content_hash", sa.String(64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chunks_parent_id", "chunks", ["parent_id"])
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_doc_id", table_name="chunks")
    op.drop_index("ix_chunks_parent_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_parents_doc_id", table_name="parents")
    op.drop_table("parents")
    # Восстановить chunks из 0001_init.
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(280), primary_key=True),
        sa.Column(
            "doc_id",
            sa.String(256),
            sa.ForeignKey("documents.doc_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content_hash", sa.String(64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
