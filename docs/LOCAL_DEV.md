# Локальный запуск без Docker — для экспериментов

Поднять DAL на своей машине **без Docker** и потестить гипотезы: своя нарезка,
своя модель, свои данные. Postgres → **SQLite-файл**, Qdrant → **embedded on-disk**,
эмбеддинг-модель → из HF-кэша. Код сервиса тот же, что в проде — меняется только конфиг.

> Проверено вживую 2026-06-12: 15 реальных док залились (0 ошибок), гибридный поиск
> отдал релевантную выдачу. Так что флоу рабочий, не теоретический.

---

## 1. Установка (один раз)

```powershell
cd elion-dal
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.lock

# Тяжёлый стек рекомендуемого конфига st-bm25 (dense USER-bge-m3 через sentence-transformers).
# torch берём CPU-сборкой, иначе приедет CUDA-вариант (~5 ГБ):
.\.venv\Scripts\python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install sentence-transformers==5.5.1

# Кодоген gRPC (часть модулей его импортирует) + сам пакет:
.\.venv\Scripts\python scripts\gen_proto.py
.\.venv\Scripts\python -m pip install --no-deps -e .
```

**Лёгкая альтернатива для быстрых экспериментов:** вместо тяжёлого стека выше можно
взять `EMBEDDING_BACKEND=fastembed` (ONNX multilingual-e5 + BM25, **без torch**) — в
~6× быстрее индексирует и легче по RAM (R@5 0.92 против 0.98). Тогда два `pip install`
с torch/sentence-transformers не нужны.

---

## 2. Запуск

Создай `.env` в корне `elion-dal/` (он в `.gitignore`, в репо не попадёт):

```ini
# Без Docker: SQLite + embedded Qdrant (файлы лягут в ./.local_run/)
PG_DSN=sqlite:///.local_run/elion.db
QDRANT_URL=./.local_run/qdrant

# Рекомендуемый конфиг (как в проде). Для лёгкого варианта: EMBEDDING_BACKEND=fastembed
EMBEDDING_BACKEND=st-bm25
EMBEDDING_MODEL=deepvk/USER-bge-m3

# Куда кэшировать модель (чтобы не качать заново каждый раз)
HF_HOME=D:/hf_home

ADMIN_PORT=8080
AUTO_MIGRATE=true
# ADMIN_PASSWORD и API_TOKEN пустые => админка и ручки открыты (локально удобно)
```

```powershell
.\.venv\Scripts\python -m elion_dal.service.server
```

- **Первый запуск качает модель** (~2 ГБ для USER-bge-m3, ~1 ГБ для e5-base) + крошечный
  BM25 (стоп-слова, <1 МБ). Дальше всё из `HF_HOME`, офлайн. Под троттлингом сети учитывай время.
- Поднялось, когда в логе `Uvicorn running on http://0.0.0.0:8080`.
- Проверка: открой <http://localhost:8080/readyz> → `{"ok":true,...}`.
- Swagger со всеми ручками: <http://localhost:8080/docs>. Админка: <http://localhost:8080/admin/>.

> Embedded-Qdrant — **однопроцессный** (файловый лок): один запущенный сервер на хранилище.
> Warning `Payload indexes have no effect in the local Qdrant` — это норма, поиск работает.
> `QDRANT_URL=:memory:` — эфемерный режим (данные не сохраняются между запусками).

---

## 3. Залить данные

**Вариант А — через админку (файлы PDF/DOCX):** <http://localhost:8080/admin/> →
«Загрузить документ» → выбрать файл и `source_id`.

**Вариант Б — один документ через API:**
```powershell
curl.exe -X POST http://localhost:8080/api/v1/documents `
  -H "Content-Type: application/json" `
  -d '{\"doc_id\":\"test-1\",\"source_id\":\"my-exp\",\"title\":\"Проба\",\"text\":\"Текст документа для индексации.\"}'
```

**Вариант В — пачкой из JSONL** (поля `doc_id, title, url, text`; под свой корпус):
```python
# load_jsonl.py — python load_jsonl.py путь_к.jsonl my-exp 50
import json, sys, urllib.request
path, source, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= n: break
        d = json.loads(line)
        if not d.get("text", "").strip(): continue
        payload = {"doc_id": d["doc_id"], "source_id": source, "url": d.get("url", ""),
                   "title": d.get("title", ""), "text": d["text"], "index_in_rag": True}
        req = urllib.request.Request("http://localhost:8080/api/v1/documents",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        print(f"[{i}] {d.get('title','')[:50]:50} chunks={r.get('chunks_upserted',0)} failed={r.get('failed',0)}")
```
```powershell
.\.venv\Scripts\python load_jsonl.py "C:\путь\kb_final.jsonl" my-exp 50
```

Индексация = чанкинг + эмбеддинг на CPU, ~2–4 док/с для USER-bge-m3.

---

## 4. Дёрнуть поиск

**Через API** (hybrid RRF dense+BM25, возвращает родителей):
```powershell
curl.exe -X POST http://localhost:8080/api/v1/search `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"твой вопрос\",\"top_k\":5,\"source_ids\":[\"my-exp\"]}'
```
В ответе у каждого hit: `title`, `text` (родитель-секция), `score` (RRF), `dense_score`
(сырой cosine лучшего ребёнка — сигнал уверенности), `matched_child`.

**Через админку:** блок «Поиск» на <http://localhost:8080/admin/> — там же `dense_score` виден.

**Готовые рецепты для своего кода:** `handoff/reader.py` (поиск, breadcrumbs, no-hit,
склейка контекста) и `examples/dal_client.py` (тонкий клиент).

---

## 5. Что крутить в экспериментах

- **Конфиг нарезки/модели** — через `.env` (рестарт) или **на лету в админке** (раздел
  «Настройки»): `chunk_tokens`, `chunk_overlap`, `chunk_min_tokens`, `chunk_separator_mode`,
  `search_top_k`. `embedding_backend`/`embedding_model` — только рестартом + переиндексация.
- **Превью нарезки (dry-run)** в админке — вставить текст, увидеть как режется, **не трогая индекс**.
- **Сравнение методов/моделей вне сервиса** — отдельный харнесс `leaderboard/` (свой venv,
  считает Recall@5/MRR по golden-set). Для гипотез «какая нарезка/модель лучше» — туда.

---

## 6. Известные мелкие пункты (не блокеры)

Наша часть проекта в основном сдана; ниже — что осознанно отложено/опционально:

1. **Durability Qdrant** — у embedded и у временного прода защиты от порчи storage нет
   (ловили RocksDB-порчу на проде; чинится `reindex --recreate` из PG). Серьёзный
   хардненинг (память/graceful shutdown/бэкапы) планируется **позже, на серверах заказчика**.
2. **Косметика по запросу смежников** — `published_ts` в ответе поиска (если нужна дата),
   warm-up первого запроса (убрать холодный старт), счётчик точек Qdrant в `/stats`. Часы работы, делаем если попросят.
3. **semantic-чанкинг** — остался заглушкой (нужен llama-index); recursive по бенчу
   победил, доделывать смысла нет.

---

## TL;DR

```powershell
# 1. .env с SQLite + embedded Qdrant (см. §2)
# 2. запуск
.\.venv\Scripts\python -m elion_dal.service.server
# 3. залить
.\.venv\Scripts\python load_jsonl.py corpus.jsonl my-exp 50
# 4. спросить
curl.exe -X POST http://localhost:8080/api/v1/search -H "Content-Type: application/json" -d '{\"query\":\"...\",\"source_ids\":[\"my-exp\"]}'
```
