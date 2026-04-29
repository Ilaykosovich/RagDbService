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
        self._embedder = SentenceTransformer(settings.embedding_model)

        backfill_minilm_embeddings(
            repo=self._repo,
            model=self._embedder,
        )

        self._collection = build_history_chroma_from_db(
            repo=self._repo,
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_history_collection,
            model=self._embedder,
        )



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

        doc = _doc_text(user_query)
        emb = self._embedder.encode([doc], normalize_embeddings=True).tolist()[0]

        row: RagQueryHistory = self._repo.upsert_success(
            db_fingerprint=db_fingerprint,
            user_query=user_query,
            sql=sql,
            tables_used=tables_used,
            duration_ms=duration_ms,
            rows_count=rows_count,
            embedings_allMiniLML6v2=emb,
        )

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
        query_embedding = self._embedder.encode([qdoc], normalize_embeddings=True).tolist()[0]
        res = self._collection.query(
            query_embeddings=[query_embedding],
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
    model: SentenceTransformer,
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

    rows = repo.get_rows_from_query_history()

    texts = []
    metadatas = []
    ids = []
    embeddings = []

    for row in rows:
        text = (row.user_query or "").strip()
        texts.append(text)
        ids.append(str(row.id))
        embeddings.append(row.embedings_allMiniLML6v2)

        metadatas.append({
            "db_fingerprint": row.db_fingerprint,
            "sql": row.sql,
            "created_at": row.created_at.isoformat() if row.created_at else ""
        })

    if not ids:
        return collection

    missing_indexes = [i for i, emb in enumerate(embeddings) if emb is None]
    if missing_indexes:
        missing_texts = [texts[i] for i in missing_indexes]
        generated = model.encode(missing_texts, normalize_embeddings=True).tolist()
        for i, emb in zip(missing_indexes, generated):
            embeddings[i] = emb

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection


def backfill_minilm_embeddings(
    repo: HistoryRepoORM,
    model: SentenceTransformer,
    *,
    batch_size: int = 128,
) -> int:
    updated = 0

    while True:
        rows = repo.get_rows_missing_minilm_embeddings(limit=batch_size)
        if not rows:
            return updated

        texts = [_doc_text(row.user_query) for row in rows]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        repo.update_minilm_embeddings({
            row.id: embedding
            for row, embedding in zip(rows, embeddings)
        })
        updated += len(rows)



history_rag_service = HistoryRagService()
