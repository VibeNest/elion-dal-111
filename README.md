# elion-dal — Векторизация и Хранение (Data Access Layer «Элиона»)

Микросервис Этапа 2 ТЗ: принимает документы (секции + метаданные), внутри режет
секции на **дочерние чанки**, **эмбеддит** (BGE-M3: dense + sparse), хранит в
**Postgres** (source-of-truth) и **Qdrant** (производный индекс), отдаёт результаты
**гибридного поиска** (dense + sparse, fusion = RRF) по **gRPC**.

**Parent-child retrieval:** поиск идёт по детям (точный матч), но возвращаются
**родители** — секции документа — для богатого контекста генерации. Сервис не
вызывает LLM и не решает про fallback; Confidence/роутинг/карточки — на стороне RAG-ядра.

## Архитектура

```
ETL / сидер ──UpsertDocuments──► [секция→родитель; текст→дети; embed(dense+sparse)]
                                       ├──► Postgres (SoT): documents/parents/chunks
                                       └──► Qdrant (index): только дети + parent_id
RAG-ядро ──Search──► embed(query) → Qdrant hybrid (RRF) по детям
                     → схлопывание в уникальных родителей → top-k родителей
```

- **Эмбеддинги за интерфейсом** `EmbeddingProvider` (`src/elion_dal/embedding/`):
  - `flag` — настоящий BGE-M3 dense + learned sparse (вариант A; `pip install -e ".[flag]"`);
  - `fastembed` — лёгкий ONNX на CPU: `multilingual-e5-large` dense + BM25 sparse (IDF).
  Выбор — по итогам `bench/benchmark_embeddings.py` (всё на CPU, GPU не нужен).
  `EMBEDDING_QUANTIZE` (default **false**) — переключатель int8. ВНИМАНИЕ: для `flag`
  (BGE-M3) это torch dynamic int8, и по замерам RSS он НЕ снижается, а удваивается
  (~4 ГБ против ~2 ГБ: fp32-копия удерживается на пике). Реальное снижение RAM даёт
  int8 ONNX-экспорт (на будущее). Подробности — `docs/adr.md`, ADR-004.
- **Qdrant**: коллекция `elion_chunks`, named-векторы `dense`(1024, Cosine) + `sparse`,
  payload-индексы `source_id` / `doc_id` / `published_ts`. Индексируются **только дети**
  с `parent_id` в payload. `point_id` детерминирован → идемпотентный upsert.
- **Postgres**: `sources` / `documents` / `parents` / `chunks`; дедуп по `content_hash`,
  полная пересборка индекса без перекраулинга. Родители (секции) отдаются на поиске.
- **Контракт с ETL**: документ несёт `sections[]` (родители) — желательно из Markdown со
  структурой; документ без секций трактуется как один родитель (fallback `text`).

## Быстрый старт (локально)

```bash
# 1. Окружение
python -m venv .venv && . .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Сгенерировать gRPC-код из proto
python scripts/gen_proto.py

# 3. Поднять инфраструктуру и накатить схему
docker compose up -d qdrant postgres
copy .env.example .env          # при необходимости поправить
alembic upgrade head

# 4. (опц.) Сравнить эмбеддинг-провайдеры на CPU и выбрать EMBEDDING_BACKEND
python -m bench.benchmark_embeddings

# 5. Засидить локальную «Базу знаний» (PDF/DOCX из ../База знаний)
python -m elion_dal.ingestion.seed_knowledge_base

# 6. Проверить поиск из консоли
python -m elion_dal.cli.query "как получить справку для налогового вычета"
python -m elion_dal.cli.query "когда олимпиада Физтех по биологии"

# 7. Поднять gRPC-сервер
python -m elion_dal.service.server
```

### Полностью в Docker

```bash
docker compose --profile full up --build
```

### Локальный запуск без Docker (embedded-бэкенды)

Тот же код умеет работать без серверов — через embedded-режим Qdrant и SQLite.
Бэкенды выбираются конфигом, код сервиса не меняется:

```bash
# Qdrant — встроенный on-disk режим, Postgres -> SQLite-файл
set QDRANT_URL=./qdrant_local        # или ":memory:" для эфемерного
set PG_DSN=sqlite:///./elion_dev.db

python -m elion_dal.ingestion.seed_knowledge_base
python -m elion_dal.cli.query "когда олимпиада Физтех по биологии"
```

`QDRANT_URL` интерпретируется так: `http(s)://…` — внешний сервер; `:memory:` —
эфемерный embedded; иначе — путь к локальному on-disk-хранилищу. Прод остаётся на
Qdrant-сервере + Postgres, embedded-режим — для dev/CI без Docker.

## gRPC API (`proto/vectorstore.proto`)

| RPC | Назначение |
|---|---|
| `UpsertDocuments(stream Document)` | индексация (чанкинг+эмбеддинг внутри, идемпотентно по хешу; изоляция ошибок по документу) |
| `Search(SearchRequest)` | гибридный поиск (RRF) по детям → Топ-k родителей + `matched_child` + `dense_score` (confidence) |
| `DeleteBySource(SourceRef)` | удалить источник из PG и Qdrant (переиндексация) |
| `DeleteByDoc(DocRef)` | удалить один документ (страница исчезла из sitemap) |
| `ListSources` / `GetStats` | админ-статистика: источники с датами синхронизации и объёмами |
| `HealthCheck` | живость + доступность Qdrant/Postgres |

**Ранжирование:** hybrid (RRF) → опциональный реранкер (`RERANK_ENABLED`, cross-encoder, default off)
→ опциональный recency-boost (`RECENCY_WEIGHT`, приоритет свежих дат) → Топ-k. `dense_score` —
сырой cosine лучшего ребёнка, сигнал уверенности для fallback на стороне RAG-ядра.

Проверка через `grpcurl` (включена reflection):
```bash
grpcurl -plaintext localhost:50051 list
grpcurl -plaintext -d '{"query":"налоговый вычет","top_k":3}' \
  localhost:50051 elion.vectorstore.v1.VectorStore/Search
```

## CI и воспроизводимость

- **CI** (`.github/workflows/ci.yml`): на push/PR — установка, кодоген proto, `ruff`,
  offline unit-тесты (`-m "not integration"`).
- **Lock** (`requirements.lock`, через `pip-compile`): пиннинг рантайм-зависимостей;
  Docker ставит по локу (`pip install -r requirements.lock` + `pip install --no-deps .`).
  Windows-only пакеты помечены маркером `sys_platform == "win32"`. Для прод-деплоя лок
  желательно регенерировать на целевой платформе (Linux) или через `uv`.

## Веб-админка

Поднимается в **том же процессе**, что и gRPC (общий `IndexService`, одна загруженная
модель — без отдельных контейнеров). Старт сервера → `http://localhost:8080`:
- дашборд: объёмы (документы/родители/чанки) и таблица источников с датой синхронизации;
- поиск с показом `dense_score` и matched-сниппета;
- удаление источника/документа;
- загрузка PDF/DOCX → парсинг и индексация в один клик;
- **редактирование настроек** (хранятся в БД `app_settings`, override поверх `.env`):
  - *live* (применяются сразу): `search_prefetch`, `search_parent_fanout`, `recency_weight`,
    `recency_halflife_days`, `rerank_enabled`, `chunk_tokens`, `chunk_overlap`;
  - *restart* (подхватываются после перезапуска, помечены в UI): `embedding_backend`,
    `embedding_model`, `embedding_quantize`.
  Связь с инфраструктурой (`PG_DSN`/`QDRANT_URL`/порты) остаётся в `.env` (нужна для
  bootstrap) и не редактируется из админки.

Отключается `ADMIN_ENABLED=false`.

## Тесты

```bash
pytest                 # unit (быстрые, без инфраструктуры)
pytest -m integration  # round-trip на поднятых Qdrant+Postgres (+скачивание модели)
```

## Конфигурация (`.env`)

См. `.env.example`: `GRPC_*`, `QDRANT_URL`, `PG_DSN`, `EMBEDDING_BACKEND`
(`fastembed`|`flag`), `CHUNK_TOKENS`/`CHUNK_OVERLAP`, `SEARCH_TOP_K`/`SEARCH_PREFETCH`.

## За рамками сервиса

Веб-краулинг/ETL, RAG-ядро и LLM, роутинг интентов, виджет, админ-панель — отдельные
части системы. Документы сюда приходят уже очищенными (от ETL) либо из сид-утилиты.
