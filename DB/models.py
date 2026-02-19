from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import DateTime, Integer, String, Text, func, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RagQueryHistory(Base):
    __tablename__ = "query_history"
    __table_args__ = (
        Index("ix_query_history_db_fingerprint", "db_fingerprint"),
        Index("ix_query_history_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    db_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)

    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    sql: Mapped[str] = mapped_column(Text, nullable=False)

    tables_used: Mapped[Optional[List[str]]] = mapped_column(JSONB, nullable=True)

    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rows_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
