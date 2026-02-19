from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.types import Collection
from sentence_transformers import SentenceTransformer

from ragdbservice.config import settings
from ragdbservice.history_repo_orm import HistoryRepoORM
from ragdbservice.schema_rag_service import schema_rag_service
from DB.models import RagQueryHistory


def _doc_text(user_query: str) -> str:
    return (user_query or "").strip()


def _tables_overlap(a: Optional[List[str]], b: Optional[List[str]]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa))


@dataclass
class HistoryMatch:
    score: float
    history_id: str
    user_query: str
    sql: str
    tables_used: Optional[List[str]]
    created_at: str
    hit_count: int
    duration_ms: Optional[int]
    rows_count: Optional[int]


class HistoryRagService:
    def __init__(self) -> None:
        self._repo: Optional[HistoryRepoORM] = None
        self._collection: Optional[Collection] = None
        self._embedder: Optional[SentenceTransformer] = None

    def start(self) -> None:
        self._repo = HistoryRepoORM()

        self._collection = build_history_chroma_from_db(
            repo=self._repo,
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_history_collection,
            embedding_model=settings.embedding_model,
        )

        self._embedder = SentenceTransformer(settings.embedding_model)



    def ingest_success(
        self,
        *,
        db_fingerprint: str,
        user_query: str,
        sql: str,
        tables_used: Optional[List[str]] = None,
        duration_ms: Optional[int] = None,
        rows_count: Optional[int] = None,
    ) -> str:
        if not self._repo or not self._collection or not self._embedder:
            raise RuntimeError("HistoryRagService not started")

        row: RagQueryHistory = self._repo.upsert_success(
            db_fingerprint=db_fingerprint,
            user_query=user_query,
            sql=sql,
            tables_used=tables_used,
            duration_ms=duration_ms,
            rows_count=rows_count,
        )

        doc = _doc_text(row.user_query)
        emb = self._embedder.encode([doc], normalize_embeddings=True).tolist()[0]

        self._collection.upsert(
            ids=[str(row.id)],
            documents=[doc],
            metadatas=[{"db_fingerprint": row.db_fingerprint}],
            embeddings=[emb],
        )

        return str(row.id)

    def search(
        self,
        *,
        db_fingerprint: str,
        user_query: str,
        top_k: int = 5,
        prefetch_k: int = 30,
        schema_top_k: int = 30,
        table_boost: float = 0.15,
    ) -> List[HistoryMatch]:
        if not self._repo or not self._collection or not self._embedder:
            raise RuntimeError("HistoryRagService not started")

        # 1) ✅ сначала достаём релевантные таблицы из SchemaRAG
        analysis = {"search_queries": [user_query]}
        schema_res = schema_rag_service.query_schema(analysis, top_k=schema_top_k)
        tables_filter = schema_res.matched_tables  # List[str]

        qdoc = _doc_text(user_query)
        res = self._collection.query(
            query_texts=[qdoc],
            n_results=max(prefetch_k, top_k),
            include=["distances"],
            where={"db_fingerprint": db_fingerprint},
        )

        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        if not ids:
            return []

        rows_map = self._repo.get_by_ids(ids)

        scored: List[Tuple[float, RagQueryHistory]] = []
        for hid, dist in zip(ids, dists):
            row = rows_map.get(hid)
            if not row:
                continue
            score = 1.0 - float(dist)
            if tables_filter:
                score += table_boost * _tables_overlap(row.tables_used, tables_filter)

            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[HistoryMatch] = []
        for score, row in scored[:top_k]:
            out.append(HistoryMatch(
                score=round(score, 4),
                history_id=str(row.id),
                user_query=row.user_query,
                sql=row.sql,
                tables_used=row.tables_used,
                created_at=row.created_at.isoformat() if row.created_at else "",
                hit_count=row.hit_count or 1,
                duration_ms=row.duration_ms,
                rows_count=row.rows_count,
            ))
        return out


def build_history_chroma_from_db(
    repo: HistoryRepoORM,
    *,
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    reset_collection: bool = False,
):
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)

    model = SentenceTransformer(embedding_model)

    rows = repo.get_rows_from_query_history()

    texts = []
    metadatas = []
    ids = []

    for row in rows:
        text = (row.user_query or "").strip()

        texts.append(text)
        ids.append(str(row.id))

        metadatas.append({
            "db_fingerprint": row.db_fingerprint,
            "sql": row.sql,
            "created_at": row.created_at.isoformat() if row.created_at else ""
        })

    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection



history_rag_service = HistoryRagService()
