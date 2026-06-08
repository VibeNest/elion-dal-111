"""Редактируемые настройки в БД (override поверх .env).

Тир `live` — читается на каждом запросе (применяется мгновенно).
Тир `restart` — применяется на старте сервиса (bootstrap), в UI помечается.
Связь с инфраструктурой (PG_DSN/QDRANT_URL/порты) тут НЕ управляется — она нужна,
чтобы вообще подключиться к БД, и остаётся в .env.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Field:
    key: str
    type: str  # int | float | bool | str
    tier: str  # live | restart
    label: str


# Управляемые из админки настройки. Порядок = порядок отображения.
FIELDS: list[Field] = [
    Field("search_prefetch", "int", "live", "Кандидатов на ветку prefetch"),
    Field("search_parent_fanout", "int", "live", "Fan-out детей на родителя"),
    Field("recency_weight", "float", "live", "Вес свежести (0 = выкл)"),
    Field("recency_halflife_days", "float", "live", "Полупериод свежести, дни"),
    Field("search_top_k", "int", "live", "Top-K результатов поиска"),
    Field("chunk_tokens", "int", "live", "Размер чанка (токены)"),
    Field("chunk_overlap", "int", "live", "Перекрытие чанков (токены)"),
    Field("chunk_min_tokens", "int", "live", "Мин. размер чанка, токены (0 = выкл)"),
    Field("chunk_separator_mode", "str", "live", "Стратегия нарезки (structured|token)"),
    Field("chunk_tokenizer_model", "str", "restart", "Токенайзер длины чанков"),
    Field("embedding_backend", "str", "restart", "Бэкенд эмбеддингов (st-bm25|fastembed|flag)"),
    Field("embedding_model", "str", "restart", "Модель эмбеддингов (пусто = дефолт)"),
    Field("embedding_quantize", "bool", "restart", "int8-квантизация модели"),
    Field("api_token", "str", "live", "Токен доступа к gRPC API (ручкам)"),
]
FIELD_BY_KEY: dict[str, Field] = {f.key: f for f in FIELDS}


def _convert(field: Field, raw: str):
    if field.type == "int":
        return int(raw)
    if field.type == "float":
        return float(raw)
    if field.type == "bool":
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    return raw


@dataclass(slots=True)
class SettingView:
    key: str
    label: str
    tier: str
    type: str
    value: object
    is_override: bool


class SettingsStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._raw: dict[str, str] = {}

    def load(self) -> None:
        """Прочитать overrides из БД. Если таблицы ещё нет (не накатили миграции) —
        работаем на дефолтах из .env."""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text("SELECT key, value FROM app_settings")).all()
            self._raw = {k: v for k, v in rows}
        except Exception as e:  # noqa: BLE001
            logger.warning("app_settings недоступны (%s) — используем дефолты .env", e)
            self._raw = {}

    def get(self, key: str):
        """Типизированное значение override или None, если не задано."""
        field = FIELD_BY_KEY.get(key)
        if field is None or key not in self._raw:
            return None
        try:
            return _convert(field, self._raw[key])
        except (ValueError, TypeError):
            return None

    def set_many(self, items: dict[str, str]) -> None:
        upsert = text(
            "INSERT INTO app_settings(key, value, updated_at) "
            "VALUES (:k, :v, CURRENT_TIMESTAMP) "
            "ON CONFLICT (key) DO UPDATE SET value = :v, updated_at = CURRENT_TIMESTAMP"
        )
        with self.engine.begin() as conn:
            for k, v in items.items():
                if k not in FIELD_BY_KEY:
                    continue
                conn.execute(upsert, {"k": k, "v": str(v)})
                self._raw[k] = str(v)

    def view(self, base) -> list[SettingView]:
        """Список настроек для админки: эффективное значение + признак override."""
        out: list[SettingView] = []
        for f in FIELDS:
            override = self.get(f.key)
            if override is not None:
                value, is_override = override, True
            else:
                value, is_override = getattr(base, f.key, None), False
            out.append(SettingView(f.key, f.label, f.tier, f.type, value, is_override))
        return out
