"""Testing filtering by academic_year and is_active."""
import os

# Set environment variables for embedded mode
os.environ["PG_DSN"] = "sqlite:///./elion_dev.db"
os.environ["QDRANT_URL"] = "./qdrant_local"

from elion_dal.service.bootstrap import build_index_service
from elion_dal.service.sync import UpsertCounts
from elion_dal.store.pg_repo import DocInput, PgRepo, SectionInput


def ensure_db():
    """Create database tables if they don't exist."""
    repo = PgRepo(os.environ["PG_DSN"])
    repo.create_all()
    return repo


def test_academic_year_filter():
    """Test filtering by academic year."""
    print("\n=== Test: filtering by academic_year ===")
    ensure_db()
    index = build_index_service(ensure=True)
    doc_id = "test_filter_2026"

    try:
        # Create test document with explicit year
        doc = DocInput(
            doc_id=doc_id,
            source_id="test",
            url="https://example.com",
            title="Test document 2026",
            lang="ru",
            published_ts=0,
            content_hash="",
            index_in_rag=True,
            academic_year=2026,
            is_active=True,
            sections=[
                SectionInput(
                    section_id="0",
                    heading_path=[],
                    url="https://example.com",
                    text="This is a test document for filtering by year.",
                )
            ],
        )
        counts = UpsertCounts()
        index.process_document(doc, counts)
        print(f"Document loaded: {counts.indexed}")

        # Test filtering
        hits_all = index.search(
            query="test",
            top_k=5,
            source_ids=[],
            min_published_ts=0
        )
        hits_2026 = index.search(
            query="test",
            top_k=5,
            source_ids=[],
            min_published_ts=0,
            academic_year=2026,
        )
        hits_2025 = index.search(
            query="test",
            top_k=5,
            source_ids=[],
            min_published_ts=0,
            academic_year=2025
        )

        print(f"No filter: {len(hits_all)}")
        print(f"Filter 2026: {len(hits_2026)}")
        print(f"Filter 2025: {len(hits_2025)}")

        assert len(hits_2026) > 0, "Document 2026 should be found"
        assert len(hits_2025) == 0, "Documents 2025 should not be found"
        print("Test by year passed")

    finally:
        # Clean up test document
        try:
            index.delete_doc(doc_id)
            print(f"Test document {doc_id} deleted")
        except Exception as e:
            print(f"Could not delete {doc_id}: {e}")


def test_is_active_filter():
    """Test filtering by active status."""
    print("\n=== Test: filtering by is_active ===")
    ensure_db()
    index = build_index_service(ensure=True)
    doc_id = "test_filter_inactive"

    try:
        # Create inactive document
        doc = DocInput(
            doc_id=doc_id,
            source_id="test",
            url="https://example.com",
            title="Inactive document",
            lang="ru",
            published_ts=0,
            content_hash="",
            index_in_rag=True,
            academic_year=2026,
            is_active=False,
            sections=[
                SectionInput(
                    section_id="0",
                    heading_path=[],
                    url="https://example.com",
                    text="This is an inactive test document.",
                )
            ],
        )
        counts = UpsertCounts()
        index.process_document(doc, counts)
        print(f"Document loaded: {counts.indexed}")

        # Test filtering
        hits_all = index.search(
            query="inactive",
            top_k=5,
            source_ids=[],
            min_published_ts=0
        )
        hits_active = index.search(
            query="inactive",
            top_k=5,
            source_ids=[],
            min_published_ts=0,
            is_active=True
        )
        hits_inactive = index.search(
            query="inactive",
            top_k=5,
            source_ids=[],
            min_published_ts=0,
            is_active=False
        )

        print(f"No filter: {len(hits_all)}")
        print(f"Only active: {len(hits_active)}")
        print(f"Only inactive: {len(hits_inactive)}")

        assert len(hits_inactive) > 0, "Inactive document should be found"
        assert len(hits_active) == 0, "Active documents should not be found"
        print("Test by active status passed")

    finally:
        # Clean up test document
        try:
            index.delete_doc(doc_id)
            print(f"Test document {doc_id} deleted")
        except Exception as e:
            print(f"Could not delete {doc_id}: {e}")


def test_none_filter_behavior():
    """Test that None does not apply any filter."""
    print("\n=== Test: None = no filter ===")
    ensure_db()
    index = build_index_service(ensure=True)
    doc_id = "test_filter_none"

    try:
        # Create a document
        doc = DocInput(
            doc_id=doc_id,
            source_id="test",
            url="https://example.com",
            title="Document for None test",
            lang="ru",
            published_ts=0,
            content_hash="",
            index_in_rag=True,
            academic_year=2026,
            is_active=True,
            sections=[
                SectionInput(
                    section_id="0",
                    heading_path=[],
                    url="https://example.com",
                    text="Document for testing filter with None.",
                )
            ],
        )
        counts = UpsertCounts()
        index.process_document(doc, counts)

        query = "document for testing"
        hits_none = index.search(
            query=query,
            top_k=5,
            source_ids=[],
            min_published_ts=0,
            is_active=None
        )
        hits_all = index.search(
            query=query,
            top_k=5,
            source_ids=[],
            min_published_ts=0
        )

        print(f"With is_active=None: {len(hits_none)}")
        print(f"No filter: {len(hits_all)}")

        assert len(hits_none) == len(hits_all), "None should behave like 'no filter'"
        print("Test for None passed")

    finally:
        # Clean up test document
        try:
            index.delete_doc(doc_id)
            print(f"Test document {doc_id} deleted")
        except Exception as e:
            print(f"Could not delete {doc_id}: {e}")


if __name__ == "__main__":
    test_academic_year_filter()
    test_is_active_filter()
    test_none_filter_behavior()
    print("\nAll tests passed!")