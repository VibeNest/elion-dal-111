"""Лёгкая веб-админка (FastAPI) в том же процессе, что и gRPC.

Работает с IndexService напрямую (in-process) — одна загруженная модель на оба
интерфейса, без отдельных контейнеров. Возможности: дашборд (объёмы + источники),
поиск с dense_score, удаление источника/документа, загрузка файла (PDF/DOCX → индекс).
"""

from __future__ import annotations

# Длинные строки — это HTML/JS-шаблоны; File()/Form() в дефолтах — идиома FastAPI.
# ruff: noqa: E501, B008
import hashlib
import html
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from ..ingestion.loaders import load_document
from ..service.sync import IndexService, UpsertCounts
from ..store.pg_repo import DocInput, SectionInput, sha256
from ..store.settings_store import FIELDS

_HEAD = """<!doctype html><html lang=ru><head><meta charset=utf-8>
<title>Элион — DAL Admin</title>
<style>
 body{font-family:system-ui,Arial;margin:24px;max-width:1000px;color:#222}
 h1{font-size:20px} h2{font-size:16px;margin-top:28px}
 table{border-collapse:collapse;width:100%} td,th{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:14px}
 .cards{display:flex;gap:16px;margin:12px 0}
 .card{border:1px solid #ddd;border-radius:8px;padding:12px 16px;min-width:120px}
 .card b{font-size:22px;display:block}
 input,button{font-size:14px;padding:6px 8px} button{cursor:pointer}
 .hit{border:1px solid #eee;border-radius:8px;padding:10px;margin:8px 0}
 .muted{color:#888;font-size:12px}
</style></head><body>
<h1>Элион — DAL Admin</h1>"""

_SCRIPT = """
<script>
async function doSearch(e){
  e.preventDefault();
  const q=document.getElementById('q').value, k=document.getElementById('k').value;
  const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body:`query=${encodeURIComponent(q)}&top_k=${k}`});
  const data=await r.json(); const box=document.getElementById('results'); box.innerHTML='';
  if(!data.length){box.innerHTML='<p class=muted>Ничего не найдено (no-hit).</p>';return;}
  for(const h of data){
    const el=document.createElement('div'); el.className='hit';
    el.innerHTML=`<div><b>${h.title||h.parent_id}</b> <span class=muted>score=${h.score.toFixed(4)} dense=${h.dense_score.toFixed(4)} · ${h.source_id}</span></div>`+
      (h.heading_path?`<div class=muted>${h.heading_path.join(' › ')}</div>`:'')+
      `<div class=muted>нашли по: ${h.matched_child.slice(0,160)}</div>`+
      `<div>${h.text.slice(0,400)}</div><div class=muted><a href='${h.url}'>${h.url}</a></div>`;
    box.appendChild(el);
  }
}
</script></body></html>"""


def _fmt_ts(ts: int) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M")


def _doc_id(filename: str) -> str:
    return "kb-" + hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]


def _settings_form(views) -> str:
    rows = ""
    for v in views:
        badge = " <span class=muted>(после рестарта)</span>" if v.tier == "restart" else ""
        ovr = " <span class=muted>· override</span>" if v.is_override else ""
        if v.type == "bool":
            checked = "checked" if v.value else ""
            field = f"<input type=checkbox name='{v.key}' {checked}>"
        else:
            val = "" if v.value is None else html.escape(str(v.value))
            typ = "number" if v.type in ("int", "float") else "text"
            step = " step=any" if v.type == "float" else ""
            field = f"<input type={typ}{step} name='{v.key}' value='{val}'>"
        rows += f"<tr><td>{html.escape(v.label)}{badge}{ovr}</td><td>{field}</td></tr>"
    if not rows:
        return ""
    return (
        "<h2>Настройки</h2><form method=post action='/settings'>"
        "<table>" + rows + "</table>"
        "<button>Сохранить</button> "
        "<span class=muted>live применяются сразу; restart — после перезапуска сервиса</span>"
        "</form>"
    )


def create_app(index: IndexService) -> FastAPI:
    app = FastAPI(title="Элион — DAL Admin")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        st = index.get_stats()
        rows = ""
        for s in st.sources:
            rows += (
                f"<tr><td>{html.escape(s.source_id)}</td><td>{html.escape(s.name)}</td>"
                f"<td>{s.document_count}</td><td>{s.parent_count}</td><td>{s.chunk_count}</td>"
                f"<td>{_fmt_ts(s.last_indexed_ts)}</td>"
                f"<td><form method=post action='/sources/{html.escape(s.source_id)}/delete' "
                f"onsubmit=\"return confirm('Удалить источник {html.escape(s.source_id)}?')\">"
                f"<button>Удалить</button></form></td></tr>"
            )
        body = f"""
        <div class=cards>
          <div class=card>документы<b>{st.total_documents}</b></div>
          <div class=card>родители<b>{st.total_parents}</b></div>
          <div class=card>чанки<b>{st.total_chunks}</b></div>
        </div>
        <h2>Источники</h2>
        <table><tr><th>source_id</th><th>имя</th><th>док.</th><th>род.</th><th>чанки</th>
          <th>синхронизация</th><th></th></tr>{rows or "<tr><td colspan=7 class=muted>пусто</td></tr>"}</table>
        <h2>Загрузить документ</h2>
        <form method=post action='/upload' enctype='multipart/form-data'>
          <input type=file name=file required>
          <input type=text name=source_id value='knowledge_base' title='source_id'>
          <button>Загрузить и проиндексировать</button>
        </form>
        <h2>Поиск</h2>
        <form onsubmit='doSearch(event)'>
          <input id=q size=60 placeholder='запрос...' required>
          <input id=k type=number value=5 min=1 max=20 style='width:60px'>
          <button>Искать</button>
        </form>
        <div id=results></div>
        {_settings_form(index.settings_view())}
        """
        return _HEAD + body + _SCRIPT

    @app.post("/settings")
    async def update_settings(request: Request) -> RedirectResponse:
        form = await request.form()
        items: dict[str, str] = {}
        for f in FIELDS:
            if f.type == "bool":
                # снятый чекбокс не приходит в форме -> false
                items[f.key] = "true" if form.get(f.key) is not None else "false"
            else:
                val = form.get(f.key)
                if val is not None and str(val) != "":
                    items[f.key] = str(val)
        index.update_settings(items)
        return RedirectResponse("/", status_code=303)

    @app.get("/api/stats")
    def api_stats() -> dict:
        st = index.get_stats()
        return {
            "total_documents": st.total_documents,
            "total_parents": st.total_parents,
            "total_chunks": st.total_chunks,
            "sources": [asdict(s) for s in st.sources],
        }

    @app.post("/api/search")
    def api_search(query: str = Form(...), top_k: int = Form(5)) -> list[dict]:
        hits = index.search(query=query, top_k=top_k, source_ids=[], min_published_ts=0)
        return [
            {
                "parent_id": h.parent_id,
                "doc_id": h.doc_id,
                "source_id": h.source_id,
                "title": h.title,
                "url": h.url,
                "heading_path": h.heading_path,
                "text": h.text,
                "matched_child": h.matched_child,
                "score": h.score,
                "dense_score": h.dense_score,
            }
            for h in hits
        ]

    @app.post("/sources/{source_id}/delete")
    def delete_source(source_id: str) -> RedirectResponse:
        index.delete_source(source_id)
        return RedirectResponse("/", status_code=303)

    @app.post("/docs/{doc_id}/delete")
    def delete_doc(doc_id: str) -> RedirectResponse:
        index.delete_doc(doc_id)
        return RedirectResponse("/", status_code=303)

    @app.post("/upload")
    def upload(file: UploadFile = File(...), source_id: str = Form("knowledge_base")):
        data = file.file.read()
        suffix = Path(file.filename or "f").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            text = load_document(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        name = file.filename or "upload"
        url = f"file://{name}"
        doc = DocInput(
            doc_id=_doc_id(name),
            source_id=source_id,
            url=url,
            title=Path(name).stem,
            lang="ru",
            published_ts=0,
            content_hash=sha256(text),
            index_in_rag=True,
            sections=[SectionInput(section_id="0", heading_path=[], url=url, text=text)],
        )
        index.process_document(doc, UpsertCounts())
        return RedirectResponse("/", status_code=303)

    return app
