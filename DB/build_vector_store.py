from __future__ import annotations
import psycopg
from chromadb.types import Collection
from __future__ import annotations
import re
import uuid
from typing import Any,Dict, List, Tuple, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ---------- Parsing helpers ----------

SECTION_TITLE_RE = re.compile(r"^[A-Z0-9_]+$")  # e.g. TABLES, COLUMNS, TABLE_COMMENTS


# Reuse your existing QUERIES dict exactly as before
QUERIES = {
    "tables": """
        select table_schema, table_name
        from information_schema.tables
        where table_type = 'BASE TABLE'
          and table_schema not in ('pg_catalog', 'information_schema')
        order by table_schema, table_name;
    """,
    "columns": """
        select
          table_schema,
          table_name,
          ordinal_position,
          column_name,
          data_type,
          is_nullable,
          column_default
        from information_schema.columns
        where table_schema not in ('pg_catalog', 'information_schema')
        order by table_schema, table_name, ordinal_position;
    """,
    "table_comments": """
        select
          n.nspname as schema_name,
          c.relname as table_name,
          obj_description(c.oid, 'pg_class') as table_description
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where c.relkind = 'r'
          and n.nspname not in ('pg_catalog', 'information_schema')
        order by schema_name, table_name;
    """,
    "column_comments": """
        select
          n.nspname as schema_name,
          c.relname as table_name,
          a.attname as column_name,
          col_description(c.oid, a.attnum) as column_description
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        join pg_attribute a on a.attrelid = c.oid
        where c.relkind = 'r'
          and n.nspname not in ('pg_catalog', 'information_schema')
          and a.attnum > 0
          and not a.attisdropped
        order by schema_name, table_name, a.attnum;
    """,
    "foreign_keys": """
        select
          n1.nspname as from_schema,
          c1.relname as from_table,
          a1.attname as from_column,
          n2.nspname as to_schema,
          c2.relname as to_table,
          a2.attname as to_column,
          con.conname as constraint_name
        from pg_constraint con
        join pg_class c1 on c1.oid = con.conrelid
        join pg_namespace n1 on n1.oid = c1.relnamespace
        join pg_class c2 on c2.oid = con.confrelid
        join pg_namespace n2 on n2.oid = c2.relnamespace
        join lateral unnest(con.conkey) with ordinality as k1(attnum, ord) on true
        join lateral unnest(con.confkey) with ordinality as k2(attnum, ord) on k1.ord = k2.ord
        join pg_attribute a1 on a1.attrelid = c1.oid and a1.attnum = k1.attnum
        join pg_attribute a2 on a2.attrelid = c2.oid and a2.attnum = k2.attnum
        where con.contype = 'f'
          and n1.nspname not in ('pg_catalog', 'information_schema')
        order by from_schema, from_table, constraint_name;
    """,
}


def _fetch_section(conn: psycopg.Connection, section_name: str, sql: str) -> Section:
    with conn.cursor() as cur:
        cur.execute(sql)
        colnames = [d.name for d in cur.description]
        rows = cur.fetchall()

    # build_chunks expects rows as List[List[str]]
    str_rows = [["" if v is None else str(v) for v in row] for row in rows]
    return Section(name=section_name, columns=colnames, rows=str_rows)


def _fetch_all(conn: psycopg.Connection, sql: str) -> List[Tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()

def build_schema_context_from_db(
    pg_url: str,
    *,
    statement_timeout_seconds: int = 30,
) -> Dict[str, Any]:
    with psycopg.connect(pg_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"set statement_timeout = '{statement_timeout_seconds}s';")

        tables = _fetch_all(conn, QUERIES["tables"])
        columns = _fetch_all(conn, QUERIES["columns"])
        table_comments = _fetch_all(conn, QUERIES["table_comments"])
        column_comments = _fetch_all(conn, QUERIES["column_comments"])
        fks = _fetch_all(conn, QUERIES["foreign_keys"])

    # индексы комментариев для быстрого маппинга
    tbl_desc: Dict[Tuple[str, str], Optional[str]] = {
        (s, t): d for (s, t, d) in table_comments
    }
    col_desc: Dict[Tuple[str, str, str], Optional[str]] = {
        (s, t, c): d for (s, t, c, d) in column_comments
    }

    # соберём таблицы
    tables_map: Dict[str, Any] = {}
    for (schema, table) in tables:
        fq = f"{schema}.{table}"
        tables_map[fq] = {
            "schema": schema,
            "name": table,
            "description": tbl_desc.get((schema, table)),
            "columns": [],
        }

    # добавим колонки
    for (schema, table, pos, col, dtype, is_nullable, default) in columns:
        fq = f"{schema}.{table}"
        if fq not in tables_map:
            continue
        tables_map[fq]["columns"].append({
            "ordinal_position": int(pos),
            "name": col,
            "type": dtype,
            "nullable": (is_nullable == "YES"),
            "default": default,
            "description": col_desc.get((schema, table, col)),
        })

    foreign_keys: List[Dict[str, str]] = []
    for (fs, ft, fc, ts, tt, tc, cn) in fks:
        foreign_keys.append({
            "from": f"{fs}.{ft}",
            "from_column": fc,
            "to": f"{ts}.{tt}",
            "to_column": tc,
            "constraint": cn,
        })

    return {
        "tables": tables_map,
        "foreign_keys": foreign_keys,
    }







def build_chroma_from_pg_url(
    pg_url: str,
    *,
    persist_dir: str = "./chroma_db",
    collection_name: str = "pg_schema",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    statement_timeout_seconds: int = 30,
    reset_collection: bool = False,
) -> Collection:
    """
    Connects to Postgres using a URL/DSN string, exports schema metadata in-memory,
    builds chunks, and saves them into a persistent ChromaDB collection.
    """

    # 1) Connect to Postgres using the URL string
    with psycopg.connect(pg_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"set statement_timeout = '{statement_timeout_seconds}s';")

        # 2) Build sections directly (no TXT)
        sections: Dict[str, Section] = {}
        for name, sql in QUERIES.items():
            sections[name.lower()] = _fetch_section(conn, name.lower(), sql)

    # 3) Chunk + embed + save to Chroma
    chunks = build_chunks(sections)

    # Optional: reset collection to avoid duplicate embeddings on repeated runs
    if reset_collection:
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # collection may not exist yet

    save_to_chroma(
        chunks=chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    # Optional: quick sanity check
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(name=collection_name)
def build_chunks(sections: Dict[str, Section]) -> List[Tuple[str, Dict[str, str]]]:
    """
    Returns list of (text, metadata) tuples.
    Creates:
      - table_summary chunks
      - column chunks (per column)
      - table_comment chunks (per table)
      - column_comment chunks (per column)
      - fk chunks (per relationship)
    """
    chunks: List[Tuple[str, Dict[str, str]]] = []

    # Index some sections for easier lookup
    tables = sections.get("tables")
    columns = sections.get("columns")
    table_comments = sections.get("table_comments")
    column_comments = sections.get("column_comments")
    foreign_keys = sections.get("foreign_keys")

    # Build lookup maps
    table_desc: Dict[Tuple[str, str], str] = {}
    if table_comments:
        # expected columns: schema_name, table_name, table_description
        colmap = {name: idx for idx, name in enumerate(table_comments.columns)}
        for r in table_comments.rows:
            schema = safe_get(r, colmap.get("schema_name", 0))
            table = safe_get(r, colmap.get("table_name", 1))
            desc = safe_get(r, colmap.get("table_description", 2))
            table_desc[(schema, table)] = desc

    col_desc: Dict[Tuple[str, str, str], str] = {}
    if column_comments:
        colmap = {name: idx for idx, name in enumerate(column_comments.columns)}
        for r in column_comments.rows:
            schema = safe_get(r, colmap.get("schema_name", 0))
            table = safe_get(r, colmap.get("table_name", 1))
            col = safe_get(r, colmap.get("column_name", 2))
            desc = safe_get(r, colmap.get("column_description", 3))
            col_desc[(schema, table, col)] = desc

    cols_by_table: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    if columns:
        colmap = {name: idx for idx, name in enumerate(columns.columns)}
        for r in columns.rows:
            schema = safe_get(r, colmap.get("table_schema", 0))
            table = safe_get(r, colmap.get("table_name", 1))
            col = safe_get(r, colmap.get("column_name", 3))
            dtype = safe_get(r, colmap.get("data_type", 4))
            nullable = safe_get(r, colmap.get("is_nullable", 5))
            default = safe_get(r, colmap.get("column_default", 6))
            desc = col_desc.get((schema, table, col), "")
            cols_by_table.setdefault((schema, table), []).append({
                "column_name": col,
                "data_type": dtype,
                "is_nullable": nullable,
                "column_default": default,
                "column_description": desc,
            })

    fks_by_table: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    if foreign_keys:
        colmap = {name: idx for idx, name in enumerate(foreign_keys.columns)}
        for r in foreign_keys.rows:
            from_schema = safe_get(r, colmap.get("from_schema", 0))
            from_table = safe_get(r, colmap.get("from_table", 1))
            from_col = safe_get(r, colmap.get("from_column", 2))
            to_schema = safe_get(r, colmap.get("to_schema", 3))
            to_table = safe_get(r, colmap.get("to_table", 4))
            to_col = safe_get(r, colmap.get("to_column", 5))
            cname = safe_get(r, colmap.get("constraint_name", 6))
            fks_by_table.setdefault((from_schema, from_table), []).append({
                "from_column": from_col,
                "to_schema": to_schema,
                "to_table": to_table,
                "to_column": to_col,
                "constraint_name": cname,
            })

            # Also useful to have incoming refs (optional)
            fks_by_table.setdefault((to_schema, to_table), [])

            # FK chunk itself (row-level)
            fk_text = (
                f"Foreign key {cname}: "
                f"{from_schema}.{from_table}.{from_col} -> {to_schema}.{to_table}.{to_col}"
            )
            chunks.append((fk_text, {
                "chunk_type": "fk",
                "from_schema": from_schema,
                "from_table": from_table,
                "from_column": from_col,
                "to_schema": to_schema,
                "to_table": to_table,
                "to_column": to_col,
                "constraint_name": cname,
            }))

    # Determine all tables from TABLES section if present, else from columns map
    table_list: List[Tuple[str, str]] = []
    if tables:
        colmap = {name: idx for idx, name in enumerate(tables.columns)}
        for r in tables.rows:
            schema = safe_get(r, colmap.get("table_schema", 0))
            table = safe_get(r, colmap.get("table_name", 1))
            table_list.append((schema, table))
    else:
        table_list = sorted(cols_by_table.keys())

    # Build per-table chunks
    for (schema, table) in table_list:
        desc = table_desc.get((schema, table), "")

        # TABLE_COMMENT chunk
        if desc:
            chunks.append((f"Table {schema}.{table} description: {desc}", {
                "chunk_type": "table_comment",
                "schema_name": schema,
                "table_name": table,
            }))

        # COLUMN chunks (row-level)
        for c in cols_by_table.get((schema, table), []):
            col_name = c["column_name"]
            col_text = (
                f"Column {schema}.{table}.{col_name} "
                f"type={c['data_type']} nullable={c['is_nullable']} default={c['column_default'] or 'NULL'}"
            )
            if c["column_description"]:
                col_text += f" description={c['column_description']}"
            chunks.append((col_text, {
                "chunk_type": "column",
                "schema_name": schema,
                "table_name": table,
                "column_name": col_name,
            }))

        # TABLE_SUMMARY chunk (most important for retrieval)
        # Keep it compact: top N columns + all FKs
        cols = cols_by_table.get((schema, table), [])
        fk_list = fks_by_table.get((schema, table), [])

        top_cols = cols[:25]  # limit for token efficiency
        col_lines = []
        for c in top_cols:
            line = f"- {c['column_name']} ({c['data_type']})"
            if c["column_description"]:
                line += f": {c['column_description']}"
            col_lines.append(line)

        fk_lines = []
        for fk in fk_list[:30]:
            fk_lines.append(
                f"- {fk['constraint_name']}: {schema}.{table}.{fk['from_column']} -> "
                f"{fk['to_schema']}.{fk['to_table']}.{fk['to_column']}"
            )

        summary_parts = [
            f"TABLE {schema}.{table}",
            f"Description: {desc}" if desc else "Description: (none)",
            "Columns:",
            *(col_lines if col_lines else ["(no columns found)"]),
            "Foreign keys:",
            *(fk_lines if fk_lines else ["(no foreign keys found)"]),
        ]

        summary_text = "\n".join(summary_parts)

        chunks.append((summary_text, {
            "chunk_type": "table_summary",
            "schema_name": schema,
            "table_name": table,
        }))

    return chunks


# ---------- Vector store (Chroma) ----------

def save_to_chroma(
    chunks: List[Tuple[str, Dict[str, str]]],
    persist_dir: str = "./chroma_db",
    collection_name: str = "pg_schema",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:


    """
    Saves chunks into a persistent local Chroma store.
    """
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=collection_name)

    model = SentenceTransformer(embedding_model)

    texts = [t for (t, _) in chunks]
    metadatas = [m for (_, m) in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    # Upsert in batches to avoid large memory spikes
    batch_size = 50
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )