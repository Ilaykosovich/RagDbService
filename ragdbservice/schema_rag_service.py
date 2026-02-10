from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.types import Collection

from service.config import settings

# твои функции
from DB.pg_chroma_export import build_chroma_from_pg_url, build_schema_context_from_db


def _pick_query_text(analysis: Dict[str, Any]) -> str:
    # 1) самый точный — search_queries[0]
    sq = analysis.get("search_queries") or []
    if sq:
        return str(sq[0])

    # 2) fallback: intent + keywords + entities
    parts: List[str] = []
    if analysis.get("intent"):
        parts.append(str(analysis["intent"]))

    kw = analysis.get("keywords") or []
    if kw:
        parts.append(" ".join(map(str, kw)))

    ents = analysis.get("entities") or []
    if ents:
        ent_str = " ".join(
            f'{e.get("type","")}:{e.get("value","")}' for e in ents
        )
        parts.append(ent_str)

    return " | ".join([p for p in parts if p]).strip()


def _table_fq_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    # Мы в чанках сохраняем schema_name + table_name
    s = meta.get("schema_name")
    t = meta.get("table_name")
    if s and t:
        return f"{s}.{t}"

    # для fk чанка можно достать из from_schema/from_table
    fs = meta.get("from_schema")
    ft = meta.get("from_table")
    if fs and ft:
        return f"{fs}.{ft}"

    return None


@dataclass
class QueryResult:
    tables: Dict[str, Any]
    foreign_keys: List[Dict[str, Any]]
    matched_tables: List[str]
    query_text: str


class SchemaRagService:
    """
    Singleton-like service:
    - build chroma on startup
    - query method -> returns relevant schema JSON
    - update method -> rebuilds chroma
    """

    def __init__(self) -> None:
        self._collection: Optional[Collection] = None

    def start(self) -> None:
        # создаём коллекцию при старте (если уже есть — просто откроется)
        self._collection = build_chroma_from_pg_url(
            pg_url=settings.pg_url,
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
            embedding_model=settings.embedding_model,
            statement_timeout_seconds=settings.statement_timeout_seconds,
            reset_collection=False,
        )

    def update(self) -> None:
        # полная пересборка (без дублей)
        self._collection = build_chroma_from_pg_url(
            pg_url=settings.pg_url,
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
            embedding_model=settings.embedding_model,
            statement_timeout_seconds=settings.statement_timeout_seconds,
            reset_collection=True,
        )

    def query_schema(
        self,
        analysis: Dict[str, Any],
        *,
        top_k: int = 30,
    ) -> QueryResult:
        if not self._collection:
            raise RuntimeError("Service not started: collection is None")

        query_text = _pick_query_text(analysis)
        if not query_text:
            return QueryResult(tables={}, foreign_keys=[], matched_tables=[], query_text="")

        # 1) поиск релевантных чанков
        res = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        # 2) собрать список таблиц из top-k чанков
        matched: Set[str] = set()
        for meta in (res.get("metadatas") or [[]])[0]:
            if not meta:
                continue
            fq = _table_fq_from_meta(meta)
            if fq:
                matched.add(fq)

        matched_tables = sorted(matched)

        if not matched_tables:
            return QueryResult(tables={}, foreign_keys=[], matched_tables=[], query_text=query_text)

        # 3) взять «истину» по схеме напрямую из БД (все таблицы/колонки/комменты/fk),
        # и отфильтровать только найденные таблицы
        full = build_schema_context_from_db(
            settings.pg_url,
            statement_timeout_seconds=settings.statement_timeout_seconds,
        )

        filtered_tables = {fq: full["tables"][fq] for fq in matched_tables if fq in full["tables"]}

        # FK фильтруем по from/to
        filtered_fks = [
            fk for fk in (full.get("foreign_keys") or [])
            if fk.get("from") in filtered_tables or fk.get("to") in filtered_tables
        ]

        return QueryResult(
            tables=filtered_tables,
            foreign_keys=filtered_fks,
            matched_tables=matched_tables,
            query_text=query_text,
        )


# Один глобальный экземпляр (как get_chroma() у тебя было)
schema_rag_service = SchemaRagService()
