"""Resumable chunk-state store.

Every chunk of a document moves through PENDING -> PROCESSING -> DONE | FAILED. The state
is persisted in a small SQLite database (WAL) keyed by a content+config hash, so an
interrupted run resumes exactly where it stopped.

The connection is shared by the main (extract/edit) thread and the TTS worker. WAL protects
the file, but a Python ``sqlite3`` connection is not safe for concurrent use, so every access
is serialized through one lock (se-brain concurrency-patterns: double-write prevention).
"""

from __future__ import annotations

import sqlite3
import threading
from enum import StrEnum
from pathlib import Path
from types import TracebackType

from pdf2audio.errors import DatabaseError


class ChunkStatus(StrEnum):
    """Lifecycle of a single chunk. Stored as its string value in SQLite."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class ChunkStateStore:
    """Thread-safe repository for per-chunk processing state, backed by SQLite (WAL).

    Use as a context manager so the connection is always closed::

        with ChunkStateStore(db_path) as store:
            store.reset_stale(doc_hash)
            ...
    """

    def __init__(self, db_path: Path) -> None:
        try:
            # check_same_thread=False: the connection is shared across threads and we
            # serialize access ourselves with _lock (below).
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            self._conn.execute("PRAGMA synchronous=NORMAL")  # balance durability vs. speed
            self._lock = threading.Lock()
            self._execute(
                "CREATE TABLE IF NOT EXISTS chunks "
                "(pdf_hash TEXT, chunk_idx INTEGER, status TEXT, "
                "PRIMARY KEY(pdf_hash, chunk_idx))"
            )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not open state database {db_path}: {exc}") from exc

    def __enter__(self) -> ChunkStateStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _execute(self, sql: str, params: tuple = ()) -> None:
        try:
            with self._lock:
                self._conn.execute(sql, params)
                self._conn.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"State write failed: {exc}") from exc

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        try:
            with self._lock:
                return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"State read failed: {exc}") from exc

    def reset_stale(self, doc_hash: str) -> None:
        """On restart, revert any chunk left PROCESSING (from a crash) back to PENDING."""
        self._execute(
            "UPDATE chunks SET status=? WHERE pdf_hash=? AND status=?",
            (ChunkStatus.PENDING.value, doc_hash, ChunkStatus.PROCESSING.value),
        )

    def status(self, doc_hash: str, chunk_idx: int) -> ChunkStatus | None:
        """Return the stored status of a chunk, or None if it has never been recorded."""
        rows = self._query(
            "SELECT status FROM chunks WHERE pdf_hash=? AND chunk_idx=?",
            (doc_hash, chunk_idx),
        )
        return ChunkStatus(rows[0][0]) if rows else None

    def mark(self, doc_hash: str, chunk_idx: int, status: ChunkStatus) -> None:
        """Upsert a chunk's status (idempotent — safe to call from either thread)."""
        self._execute(
            "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
            (doc_hash, chunk_idx, status.value),
        )

    def pending_count(self, doc_hash: str) -> int:
        """How many chunks are not yet DONE (the merge gate — 0 means ready to merge)."""
        rows = self._query(
            "SELECT COUNT(*) FROM chunks WHERE pdf_hash=? AND status!=?",
            (doc_hash, ChunkStatus.DONE.value),
        )
        return int(rows[0][0])

    def done_indices(self, doc_hash: str) -> list[int]:
        """The chunk indices that finished, in order — the exact merge input."""
        rows = self._query(
            "SELECT chunk_idx FROM chunks WHERE pdf_hash=? AND status=? ORDER BY chunk_idx",
            (doc_hash, ChunkStatus.DONE.value),
        )
        return [int(r[0]) for r in rows]

    def close(self) -> None:
        self._conn.close()
