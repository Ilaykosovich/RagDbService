from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from chromadb.types import Collection
from ragdbservice.config import settings
from DB.build_vector_store import build_chroma_from_pg_url, build_schema_context_from_db
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional

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
        self._schema_cache: Optional[Dict[str, Any]] = None
        self._schema_cache_ts: float = 0.0
        self._schema_cache_ttl_s: int = 10 * 60  # 10 минут

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

    def _get_full_schema_cached(self) -> Dict[str, Any]:
        now = time.time()
        if self._schema_cache and (now - self._schema_cache_ts) < self._schema_cache_ttl_s:
            return self._schema_cache

        full = build_schema_context_from_db(
            settings.pg_url,
            statement_timeout_seconds=settings.statement_timeout_seconds,
        )
        self._schema_cache = full
        self._schema_cache_ts = now
        return full

    def _expand_tables_one_hop(
            self,
            base_tables: List[str],
            full: Dict[str, Any],
            *,
            max_extra: int = 5,
    ) -> List[str]:
        """
        Добавляет таблицы-соседи на 1 шаг по FK, чтобы LLM видел мосты для JOIN.
        max_extra ограничивает рост.
        """
        base_set: Set[str] = set(base_tables)
        extras: List[str] = []

        for fk in (full.get("foreign_keys") or []):
            frm = fk.get("from")
            to = fk.get("to")
            if not frm or not to:
                continue

            if frm in base_set and to not in base_set:
                extras.append(to)
            elif to in base_set and frm not in base_set:
                extras.append(frm)

        # дедуп + ограничение
        out: List[str] = list(base_tables)
        for t in extras:
            if t not in base_set:
                out.append(t)
                base_set.add(t)
                if len(out) - len(base_tables) >= max_extra:
                    break
        return out



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
            max_tables: int = 8,  # сколько таблиц отдаём “в базовом наборе”
            fk_expand_extra: int = 5,  # сколько максимум соседей по FK добавить
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

        metas: List[Dict[str, Any]] = (res.get("metadatas") or [[]])[0] or []
        if not metas:
            return QueryResult(tables={}, foreign_keys=[], matched_tables=[], query_text=query_text)

        # 2) скоринг таблиц по рангу + типу чанка
        # rank-based scoring устойчивее, чем distances (cosine/l2/ip могут отличаться)
        table_score: Dict[str, float] = defaultdict(float)
        table_best_rank: Dict[str, int] = {}
        table_hit_count: Dict[str, int] = defaultdict(int)

        # Небольшие бонусы по типу чанка (можно тюнить)
        type_bonus = {
            "table_summary": 3.0,
            "table_comment": 2.0,
            "column": 1.0,
            "fk": 1.5,
            "column_comment": 1.2,
        }

        for rank, meta in enumerate(metas, start=1):
            if not meta:
                continue
            fq = _table_fq_from_meta(meta)
            if not fq:
                continue

            chunk_type = (meta.get("chunk_type") or "").strip()
            base = (top_k - rank + 1)  # чем выше в выдаче — тем больше
            bonus = type_bonus.get(chunk_type, 0.5)

            table_score[fq] += base + bonus
            table_hit_count[fq] += 1
            if fq not in table_best_rank or rank < table_best_rank[fq]:
                table_best_rank[fq] = rank

        if not table_score:
            return QueryResult(tables={}, foreign_keys=[], matched_tables=[], query_text=query_text)

        # 3) выбрать top max_tables таблиц
        # tie-breaker: лучшее место + количество хитов
        ranked_tables = sorted(
            table_score.keys(),
            key=lambda fq: (
                -table_score[fq],
                table_best_rank.get(fq, 10 ** 9),
                -table_hit_count.get(fq, 0),
            ),
        )
        matched_tables = ranked_tables[:max_tables]

        # 4) “истина” по схеме из БД (но с кешем)
        full = self._get_full_schema_cached()

        # 5) 1-hop расширение по FK (добавим соседей)
        matched_tables = self._expand_tables_one_hop(
            matched_tables,
            full,
            max_extra=fk_expand_extra,
        )

        # 6) фильтрация таблиц
        filtered_tables = {
            fq: full["tables"][fq]
            for fq in matched_tables
            if fq in (full.get("tables") or {})
        }

        # 7) фильтрация FK по from/to
        filtered_fks = [
            fk for fk in (full.get("foreign_keys") or [])
            if fk.get("from") in filtered_tables or fk.get("to") in filtered_tables
        ]

        return QueryResult(
            tables=filtered_tables,
            foreign_keys=filtered_fks,
            matched_tables=list(filtered_tables.keys()),  # только реально найденные в full
            query_text=query_text,
        )


# Один глобальный экземпляр (как get_chroma() у тебя было)
schema_rag_service = SchemaRagService()
