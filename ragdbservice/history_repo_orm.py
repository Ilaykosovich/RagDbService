from __future__ import annotations

import hashlib
import re
import uuid
from typing import List, Optional, Dict
from sqlalchemy import select, create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker
from ragdbservice.config import settings
from DB.models import RagQueryHistory
from datetime import datetime, timedelta, timezone


_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    return _WS_RE.sub(" ", s)


def compute_query_hash(db_fingerprint: str, user_query: str, sql: str) -> str:
    base = f"{_norm(db_fingerprint)}|{_norm(user_query)}|{_norm(sql)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def to_sqlalchemy_url(url: str) -> str:
    # SQLAlchemy needs the driver prefix; psycopg.connect does NOT.
    return url.replace("postgresql://", "postgresql+psycopg://", 1)



_history_engine = create_engine(
    to_sqlalchemy_url(settings.text2sql_db_url),
    pool_pre_ping=True,
    future=True,
)


HistorySessionLocal = sessionmaker(
    bind=_history_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


class HistoryRepoORM:
    def upsert_success(
        self,
        *,
        db_fingerprint: str,
        user_query: str,
        sql: str,
        tables_used: Optional[List[str]] = None,
        duration_ms: Optional[int] = None,
        rows_count: Optional[int] = None,
    ) -> RagQueryHistory:
        qh = compute_query_hash(db_fingerprint, user_query, sql)


        with HistorySessionLocal() as session:
            stmt = (
                insert(RagQueryHistory)
                .values(
                    id=uuid.uuid4(),
                    db_fingerprint=db_fingerprint,
                    user_query=user_query,
                    sql=sql,
                    tables_used=tables_used,
                    duration_ms=duration_ms,
                    rows_count=rows_count,
                    query_hash=qh,
                )
                .on_conflict_do_update(
                    index_elements=[RagQueryHistory.query_hash],
                    set_={
                        "last_seen_at": insert(RagQueryHistory).excluded.last_seen_at,
                        "hit_count": RagQueryHistory.hit_count + 1,
                        "duration_ms": insert(RagQueryHistory).excluded.duration_ms,
                        "rows_count": insert(RagQueryHistory).excluded.rows_count,
                        "tables_used": insert(RagQueryHistory).excluded.tables_used,
                    },
                )
                .returning(RagQueryHistory)
            )

            row = session.execute(stmt).scalar_one()
            session.commit()
            return row

    def get_by_ids(self, ids: List[str]) -> Dict[str, RagQueryHistory]:
        if not ids:
            return {}

        uuids = [uuid.UUID(x) for x in ids]
        with HistorySessionLocal() as session:
            rows = (
                session.execute(select(RagQueryHistory).where(RagQueryHistory.id.in_(uuids)))
                .scalars()
                .all()
            )

        return {str(r.id): r for r in rows}

    def get_rows_from_query_history(
            self,
            *,
            db_fingerprint: Optional[str] = None,
            days_back: Optional[int] = None,
            limit: Optional[int] = None,
            offset: int = 0,
    ) -> List[RagQueryHistory]:
        """
        Returns rows from query_history for building/rebuilding Chroma history collection.

        - db_fingerprint: if provided, only rows for this DB
        - days_back: if provided, only rows newer than now - days_back
        - limit/offset: optional batching
        """
        stmt = select(RagQueryHistory)

        if db_fingerprint:
            stmt = stmt.where(RagQueryHistory.db_fingerprint == db_fingerprint)

        if days_back is not None:
            # created_at is timezone=True in your model, so use aware datetime
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            stmt = stmt.where(RagQueryHistory.created_at >= cutoff)

        stmt = stmt.order_by(RagQueryHistory.created_at.desc())

        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)

        with HistorySessionLocal() as session:
            rows = session.execute(stmt).scalars().all()

        return rows
