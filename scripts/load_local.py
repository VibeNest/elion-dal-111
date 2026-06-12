"""Залить корпус (JSONL) в локально запущенный DAL — для экспериментов.

    python scripts/load_local.py <путь_к.jsonl> [source_id] [limit]

Поля строки JSONL: обязательны doc_id и text; опц. title, url, lang, published_ts.
Сервер должен быть запущен (python -m elion_dal.service.server, порт 8080).
"""

import json
import sys
import urllib.request

BASE = "http://localhost:8080"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/load_local.py <path.jsonl> [source_id] [limit]")
        return 1
    path = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else "kb-local"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

    ok = fail = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("text", "").strip():
                continue
            payload = {
                "doc_id": d["doc_id"],
                "source_id": source,
                "url": d.get("url", ""),
                "title": d.get("title", ""),
                "lang": d.get("lang", "ru"),
                "published_ts": d.get("published_ts", 0) or 0,
                "content_hash": d.get("content_hash", ""),
                "index_in_rag": True,
                "text": d["text"],
            }
            req = urllib.request.Request(
                BASE + "/api/v1/documents",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=180).read())
                ok += r.get("indexed", 0)
                fail += r.get("failed", 0)
                print(f"[{i}] {d.get('title', '')[:48]:48} chunks={r.get('chunks_upserted', 0)}")
            except Exception as e:  # noqa: BLE001 — простой скрипт, печатаем и едем дальше
                fail += 1
                print(f"[{i}] FAIL {e}")

    print(f"\nГотово: source={source!r} indexed={ok} failed={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
