# RagDbService

A FastAPI microservice that provides two RAG (Retrieval-Augmented Generation) capabilities for a Text-to-SQL pipeline:

- **Schema RAG** — dynamically retrieves the minimal relevant database schema for a given natural-language query
- **History RAG** — stores and retrieves previously successful SQL executions to assist future query generation

---

## How it works

### Schema RAG

On startup the service connects to your PostgreSQL database, introspects the full schema (tables, columns, comments, foreign keys), and indexes it into a local [ChromaDB](https://www.trychroma.com/) vector store using `sentence-transformers/all-MiniLM-L6-v2` embeddings.

When a query analysis arrives at `POST /schema`, the service:
1. Converts the analysis into a search string
2. Runs a vector similarity search over the schema chunks
3. Ranks candidate tables by a score combining retrieval rank and chunk type
4. Expands the result by one FK hop to include JOIN bridges
5. Returns only the tables and foreign keys relevant to the query

### History RAG

Successful SQL executions are persisted in a PostgreSQL `query_history` table and simultaneously indexed in a separate Chroma collection.

When `POST /rag/history/search` is called, the service retrieves semantically similar past queries filtered by `db_fingerprint`, then re-ranks them with a schema-table overlap boost.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/schema` | Return minimal relevant schema for a query analysis |
| `POST` | `/update` | Rebuild the schema vector store from the live database |
| `POST` | `/rag/history/ingest` | Persist a successful SQL execution and index it |
| `POST` | `/rag/history/search` | Search past queries similar to the current request |

### `POST /schema`

Request body — structured query analysis:
```json
{
  "search_queries": ["total revenue by region last quarter"],
  "intent": "aggregation",
  "keywords": ["revenue", "region"],
  "entities": [{"type": "time", "value": "last quarter"}]
}
```

Response:
```json
{
  "mode": "db_query_chain",
  "ok": true,
  "analysis": { "...": "..." },
  "schema": {
    "tables": { "public.orders": { "...": "..." } },
    "foreign_keys": [{ "from": "public.order_items", "to": "public.orders", "..." : "..." }]
  }
}
```

### `POST /rag/history/ingest`

```json
{
  "db_fingerprint": "abc123",
  "user_query": "total revenue by region",
  "sql": "SELECT region, SUM(amount) FROM orders GROUP BY region",
  "tables_used": ["public.orders"],
  "duration_ms": 42,
  "rows_count": 5
}
```

### `POST /rag/history/search`

```json
{
  "db_fingerprint": "abc123",
  "user_query": "revenue per region",
  "top_k": 5
}
```

---

## Configuration

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `pg_url` | ✅ | — | PostgreSQL URL of the **business database** (schema is introspected from here) |
| `text2sql_db_url` | ✅ | — | PostgreSQL URL of the **history database** (stores `query_history` table) |
| `chroma_persist_dir` | | `./chroma_db` | Local directory for ChromaDB persistence |
| `chroma_collection` | | `pg_schema` | Chroma collection name for schema embeddings |
| `chroma_history_collection` | | `chroma_history_collection` | Chroma collection name for history embeddings |
| `embedding_model` | | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `statement_timeout_seconds` | | `30` | PostgreSQL statement timeout for schema introspection queries |

---

## Running with Docker

```bash
# 1. Configure environment
cp .env.example .env

# 2. Build the image
docker build -t ragdbservice .

# 3. Run the container
docker run -p 8000:8000 --env-file .env ragdbservice
```

Service will be available at `http://localhost:8000`.  
Interactive API docs at `http://localhost:8000/docs`.

---

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Database migrations

The `query_history` table is managed with Alembic:

```bash
alembic upgrade head
```

---

## Project structure

```
RagDbService/
├── main.py                        # FastAPI app & endpoints
├── ragdbservice/
│   ├── config.py                  # Settings (pydantic-settings)
│   ├── schema_rag_service.py      # Schema vector search logic
│   ├── history_rag_service.py     # History ingest & search logic
│   └── history_repo_orm.py        # SQLAlchemy repository for query_history
├── DB/
│   ├── build_vector_store.py      # Schema introspection & Chroma indexing
│   ├── models.py                  # SQLAlchemy ORM models
│   └── session.py                 # DB session factory
├── alembic/                       # Database migrations
├── chroma_db/                     # Persisted vector store (gitignored)
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Tech stack

| Component | Library |
|-----------|---------|
| API framework | FastAPI + Uvicorn |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| ORM / migrations | SQLAlchemy 2.0 + Alembic |
| Database driver | psycopg 3 |
| Settings | pydantic-settings |
