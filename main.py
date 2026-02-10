from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

from service.chroma_service import schema_rag_service


app = FastAPI(title="Schema RAG Service", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    schema_rag_service.start()


@app.post("/schema")
def get_schema(analysis: Dict[str, Any] = Body(...)) -> JSONResponse:
    result = schema_rag_service.query_schema(analysis, top_k=30)

    if not result.tables:
        return JSONResponse({
            "mode": "db_query_chain",
            "ok": False,
            "error": "No relevant tables found in schema RAG for this request.",
            "analysis": analysis,
            "query_text_used": result.query_text,
            "matched_tables": result.matched_tables,
            "schema": {"tables": {}, "foreign_keys": []},
        })

    return JSONResponse({
        "mode": "db_query_chain",
        "ok": True,
        "analysis": analysis,
        "query_text_used": result.query_text,
        "matched_tables": result.matched_tables,
        "schema": {
            "tables": result.tables,
            "foreign_keys": result.foreign_keys,
        },
    })


@app.post("/update")
def update_schema_index() -> JSONResponse:
    schema_rag_service.update()
    return JSONResponse({
        "ok": True,
        "message": "Chroma schema index rebuilt",
    })
