from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from ragdbservice.history_rag_service import history_rag_service

from ragdbservice.schema_rag_service import schema_rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    schema_rag_service.start()
    history_rag_service.start()
    yield
    # ---- SHUTDOWN ----
    # если у сервисов есть stop()/close() — вызови тут
    # history_rag_service.stop()
    # schema_rag_service.stop()

app = FastAPI(
    title="RagDBService",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/schema")
def get_schema(analysis: Dict[str, Any]):
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
    schema_rag_service.update()
    return {"ok": True}


@app.post("/rag/history/ingest")
def ingest_history(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    # обязательные
    db_fingerprint = payload["db_fingerprint"]
    user_query = payload["user_query"]
    sql = payload["sql"]

    # опциональные
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
