"""Сидинг локальной «Базы знаний» (PDF/DOCX) в индекс — для демо без ETL.

Содержательные документы (положения, регламенты, правила приёма, программы)
индексируются. Бланки-на-скачивание (заявления) помечаются index_in_rag=False
и в RAG не попадают, но сохраняются в SoT.

Запуск:
    python -m elion_dal.ingestion.seed_knowledge_base [путь_к_папке]
По умолчанию ищет «База знаний» в корне ML_prj (рядом с elion-dal).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from ..service.bootstrap import build_index_service
from ..service.sync import UpsertCounts
from ..store.pg_repo import DocInput, SectionInput, sha256
from .loaders import load_document

SOURCE_ID = "knowledge_base"
# Маркеры файлов-бланков (не индексируем содержимое, только отдаём на скачивание).
BLANK_MARKERS = ("заявление",)


def default_kb_path() -> Path:
    # .../elion-dal/src/elion_dal/ingestion/seed_knowledge_base.py -> parents[4] = ML_prj
    ml_prj = Path(__file__).resolve().parents[4]
    return ml_prj / "База знаний"


def is_blank(filename: str) -> bool:
    low = filename.lower()
    return any(m in low for m in BLANK_MARKERS)


def doc_id_for(path: Path) -> str:
    h = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]
    return f"kb-{h}"


def main(argv: list[str]) -> int:
    kb_path = Path(argv[1]) if len(argv) > 1 else default_kb_path()
    if not kb_path.exists():
        print(f"Папка не найдена: {kb_path}")
        return 1

    files = sorted(p for p in kb_path.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".doc"})
    print(f"Найдено файлов: {len(files)} в {kb_path}")

    index = build_index_service()
    counts = UpsertCounts()

    for path in files:
        try:
            text = load_document(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {path.name}: {e}")
            continue
        blank = is_blank(path.name)
        url = f"file://{path.name}"
        # У плоских PDF/DOCX нет структуры -> весь документ = один родитель (секция).
        # Когда ETL начнёт присылать секции из Markdown, здесь появятся реальные parents.
        doc = DocInput(
            doc_id=doc_id_for(path),
            source_id=SOURCE_ID,
            url=url,
            title=path.stem,
            lang="ru",
            published_ts=0,
            content_hash=sha256(text),
            index_in_rag=not blank,
            sections=[SectionInput(section_id="0", heading_path=[], url=url, text=text)],
        )
        index.process_document(doc, counts)
        tag = "BLANK" if blank else f"{len(text)} симв."
        print(f"  [ok] {path.name} ({tag})")

    print(
        f"\nИтого: получено={counts.received} проиндексировано={counts.indexed} "
        f"пропущено(не изменились)={counts.skipped} бланков={counts.blank} "
        f"чанков={counts.chunks_upserted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
