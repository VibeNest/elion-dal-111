# Локальный запуск без Docker — для экспериментов

DAL на своей машине без Docker: SQLite вместо Postgres, embedded Qdrant вместо сервера.
Код тот же, что в проде — меняется только конфиг.

## Быстрый старт

1. **Установка** (один раз):
   ```powershell
   cd elion-dal
   py -3.12 -m venv .venv
   .\.venv\Scripts\python -m pip install -r requirements.lock
   .\.venv\Scripts\python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
   .\.venv\Scripts\python -m pip install sentence-transformers==5.5.1
   .\.venv\Scripts\python scripts\gen_proto.py
   .\.venv\Scripts\python -m pip install --no-deps -e .
   ```

2. **Конфиг** — скопировать готовый env:
   ```powershell
   copy .env.local .env
   ```

3. **Данные** — скачать корпус, запомнить локальный путь:
   <https://drive.google.com/file/d/1l2khwVZJJLT2PMIm8E4lJlsP03m_y58x/view>

4. **Запуск**:
   ```powershell
   .\.venv\Scripts\python -m elion_dal.service.server
   ```
   Готово, когда в логе `Uvicorn running on http://0.0.0.0:8080`.
   Первый запуск качает модель (~2 ГБ для USER-bge-m3), дальше из кэша.

5. **Залить данные** (путь к скачанному файлу из шага 3):
   ```powershell
   .\.venv\Scripts\python scripts\load_local.py "C:\путь\kb_final.jsonl"
   ```

6. **Спросить**:
   ```powershell
   curl.exe -X POST http://localhost:8080/api/v1/search -H "Content-Type: application/json" -d '{\"query\":\"твой вопрос\",\"source_ids\":[\"kb-local\"]}'
   ```
   Или поиск в админке: <http://localhost:8080/admin/>

---

## Что и зачем — что менять

| Хочешь | Меняй |
|---|---|
| Другую модель эмбеддингов | `EMBEDDING_BACKEND` / `EMBEDDING_MODEL` в `.env` → рестарт + перезалить данные |
| Лёгкий быстрый прогон (без torch, ~6× быстрее) | `EMBEDDING_BACKEND=fastembed` (e5 ONNX + BM25) |
| Размер / перекрытие чанков | `CHUNK_TOKENS` / `CHUNK_OVERLAP` в `.env` (или на лету в админке) |
| Сбросить всё и залить заново | удалить `elion_dev.db` и папку `qdrant_local/` |
| Свой набор данных в поиске | `source_id` при заливке: `load_local.py <файл> мой-id` и тот же id в запросе |
| Кэш модели в другом месте | переменная окружения ОС `HF_HOME=...` (не через `.env`) |

**Что внутри:**
- `elion_dev.db` (SQLite) = source-of-truth, как Postgres в проде. `qdrant_local/` (embedded Qdrant) = индекс.
- Поиск — гибрид RRF (dense USER-bge-m3 + BM25), возвращает родителей-секции. У каждого hit: `score` (RRF), `dense_score` (уверенность по лучшему ребёнку).
- Готовые рецепты под свой код — `handoff/reader.py`; тонкий клиент — `examples/dal_client.py`; все ручки — Swagger `/docs`.
- embedded Qdrant — однопроцессный (файловый лок); `QDRANT_URL=:memory:` — эфемерный режим без сохранения.

---

## Отложено (не блокеры)
- **Durability Qdrant** (защита storage от порчи) — позже, на серверах заказчика.
- **Косметика по запросу смежников** — `published_ts` в выдаче, warm-up первого запроса, счётчик точек в `/stats`.
- **semantic-чанкинг** — заглушка; по бенчу победил recursive, доделывать смысла нет.
