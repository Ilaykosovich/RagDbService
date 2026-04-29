"""RagDBService — FastAPI application exposing RAG-based schema and query-history endpoints.

Endpoints
---------
POST /schema
    Given a structured analysis object, returns the minimal relevant DB schema
    (tables + foreign keys) retrieved via vector similarity search.

POST /update
    Triggers a full rebuild of the schema vector store from the live database.

POST /rag/history/ingest
    Persists a successful SQL execution record and indexes it in the history
    vector store for future similarity retrieval.

POST /rag/history/search
    Searches past successful queries semantically similar to the current user
    query, optionally boosted by schema-table overlap.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from ragdbservice.history_rag_service import history_rag_service

from ragdbservice.schema_rag_service import schema_rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Initialises both RAG services (schema + history) on startup,
    building or loading their respective Chroma vector stores.
    """
    schema_rag_service.start()
    history_rag_service.start()
    yield


app = FastAPI(
    title="RagDBService",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/schema")
def get_schema(analysis: Dict[str, Any]):
    """Return the minimal relevant DB schema for a given query analysis.

    Performs a vector similarity search over the schema Chroma collection,
    ranks candidate tables, expands by one FK hop, and returns the pruned
    schema context.

    Args:
        analysis: Structured query analysis produced by the LLM. Expected keys
            include ``search_queries``, ``intent``, ``keywords``, and
            ``entities`` (all optional — at least one should be present).

    Returns:
        JSON with ``ok=True`` and ``schema`` (tables + foreign_keys) on success,
        or ``ok=False`` with an ``error`` message when no relevant tables are found.
    """
    result = schema_rag_service.query_schema(analysis)

    if not result.tables:
        return JSONResponse({
            "mode": "db_query_chain",
            "ok": False,
            "error": "No relevant tables found in schema RAG for this request.",
            "analysis": analysis,
        })

    return JSONResponse({
        "mode": "db_query_chain",
        "ok": True,
        "analysis": analysis,
        "schema": {
            "tables": result.tables,
            "foreign_keys": result.foreign_keys,
        },
    })


@app.post("/update")
def update_schema():
    """Rebuild the schema vector store from the live database.

    Drops the existing Chroma collection and re-indexes all table/column
    chunks from the configured PostgreSQL database. Use this after schema
    migrations or significant DDL changes.

    Returns:
        ``{"ok": True}`` on success.
    """
    schema_rag_service.update()
    return {"ok": True}


@app.post("/rag/history/ingest")
def ingest_history(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Persist a successful SQL execution and index it for future retrieval.

    Upserts the record in the relational store (deduplicating by
    ``db_fingerprint`` + ``user_query``) and updates the Chroma embedding.

    Args:
        payload: JSON body with the following fields:

            - ``db_fingerprint`` *(str, required)* — unique identifier of the
              target database (e.g. a hash of the connection string or schema).
            - ``user_query`` *(str, required)* — original natural-language query.
            - ``sql`` *(str, required)* — the SQL that was successfully executed.
            - ``tables_used`` *(list[str], optional)* — fully-qualified table
              names referenced by the query.
            - ``duration_ms`` *(int, optional)* — query execution time in ms.
            - ``rows_count`` *(int, optional)* — number of rows returned.

    Returns:
        ``{"ok": True, "history_id": "<uuid>"}``
    """
    db_fingerprint = payload["db_fingerprint"]
    user_query = payload["user_query"]
    sql = payload["sql"]

    tables_used = payload.get("tables_used")
    duration_ms = payload.get("duration_ms")
    rows_count = payload.get("rows_count")

    hid = history_rag_service.ingest_success(
        db_fingerprint=db_fingerprint,
        user_query=user_query,
        sql=sql,
        tables_used=tables_used,
        duration_ms=duration_ms,
        rows_count=rows_count,
    )

    return JSONResponse({"ok": True, "history_id": hid})


@app.post("/rag/history/search")
def search_history(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Search past successful queries semantically similar to the current request.

    Retrieves candidate history entries from Chroma filtered by
    ``db_fingerprint``, then re-ranks them by combining cosine similarity with
    a schema-table overlap boost.

    Args:
        payload: JSON body with the following fields:

            - ``db_fingerprint`` *(str, required)* — database identifier used
              to scope the search to the correct database.
            - ``user_query`` *(str, required)* — current natural-language query.
            - ``top_k`` *(int, optional, default 5)* — number of results to
              return.

    Returns:
        ``{"ok": True, "matches": [...]}`` where each match contains
        ``score``, ``history_id``, ``user_query``, ``sql``, ``tables_used``,
        ``created_at``, ``hit_count``, ``duration_ms``, and ``rows_count``.
    """
    db_fingerprint = payload["db_fingerprint"]
    user_query = payload["user_query"]
    top_k = int(payload.get("top_k", 5))

    matches = history_rag_service.search(
        db_fingerprint=db_fingerprint,
        user_query=user_query,
        top_k=top_k,
        prefetch_k=max(30, top_k * 6),
    )
    return JSONResponse({"ok": True, "matches": [m.__dict__ for m in matches]})
