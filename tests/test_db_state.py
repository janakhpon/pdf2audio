"""The resumable chunk-state contract.

The state machine lives inline in src/__main__.process_single_document (threaded, hard to
unit-test in isolation). These tests replicate the exact ``chunks`` table schema and the
key SQL transitions to document and lock in the contract:

    PENDING -> PROCESSING -> DONE
                          -> FAILED
    on restart: PROCESSING -> PENDING (reset in-flight work)
    "ready to merge" when COUNT(status != 'DONE') == 0
"""

from __future__ import annotations

import sqlite3

import pytest

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS chunks "
    "(pdf_hash TEXT, chunk_idx INTEGER, status TEXT, PRIMARY KEY(pdf_hash, chunk_idx))"
)
_HASH = "deadbeef"


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute(_SCHEMA)
    yield c
    c.close()


def _status(conn, idx):
    row = conn.execute(
        "SELECT status FROM chunks WHERE pdf_hash=? AND chunk_idx=?", (_HASH, idx)
    ).fetchone()
    return row[0] if row else None


def _pending_count(conn):
    return conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE pdf_hash=? AND status!='DONE'", (_HASH,)
    ).fetchone()[0]


def test_composite_primary_key_enforced(conn):
    conn.execute(
        "INSERT INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
        (_HASH, 1, "PENDING"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
            (_HASH, 1, "DONE"),
        )


def test_insert_or_replace_marks_processing(conn):
    conn.execute(
        "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
        (_HASH, 1, "PROCESSING"),
    )
    assert _status(conn, 1) == "PROCESSING"
    # Re-processing the same idx replaces rather than duplicating.
    conn.execute(
        "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
        (_HASH, 1, "PROCESSING"),
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE pdf_hash=? AND chunk_idx=?", (_HASH, 1)
        ).fetchone()[0]
        == 1
    )


def test_transition_processing_to_done(conn):
    conn.execute(
        "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
        (_HASH, 1, "PROCESSING"),
    )
    conn.execute("UPDATE chunks SET status='DONE' WHERE pdf_hash=? AND chunk_idx=?", (_HASH, 1))
    assert _status(conn, 1) == "DONE"


def test_transition_processing_to_failed(conn):
    conn.execute(
        "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
        (_HASH, 2, "PROCESSING"),
    )
    conn.execute("UPDATE chunks SET status='FAILED' WHERE pdf_hash=? AND chunk_idx=?", (_HASH, 2))
    assert _status(conn, 2) == "FAILED"


def test_restart_resets_processing_to_pending(conn):
    # Simulate a crash mid-flight: two PROCESSING, one DONE, one FAILED.
    for idx, status in [(1, "PROCESSING"), (2, "DONE"), (3, "PROCESSING"), (4, "FAILED")]:
        conn.execute(
            "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
            (_HASH, idx, status),
        )

    # The startup reset that process_single_document runs.
    conn.execute(
        "UPDATE chunks SET status='PENDING' WHERE pdf_hash=? AND status='PROCESSING'", (_HASH,)
    )

    assert _status(conn, 1) == "PENDING"
    assert _status(conn, 3) == "PENDING"
    # DONE and FAILED are untouched.
    assert _status(conn, 2) == "DONE"
    assert _status(conn, 4) == "FAILED"


def test_pending_count_zero_means_ready_to_merge(conn):
    for idx in (1, 2, 3):
        conn.execute(
            "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
            (_HASH, idx, "DONE"),
        )
    assert _pending_count(conn) == 0


def test_pending_count_nonzero_blocks_merge(conn):
    conn.execute(
        "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
        (_HASH, 1, "DONE"),
    )
    conn.execute(
        "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
        (_HASH, 2, "FAILED"),
    )
    # FAILED counts as not-DONE -> merge is blocked.
    assert _pending_count(conn) == 1


def test_hash_scoping_isolates_documents(conn):
    conn.execute(
        "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
        (_HASH, 1, "PENDING"),
    )
    conn.execute(
        "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
        ("other_doc", 1, "DONE"),
    )
    # The reset for _HASH must not touch the other document's rows.
    conn.execute(
        "UPDATE chunks SET status='PENDING' WHERE pdf_hash=? AND status='PROCESSING'", (_HASH,)
    )
    assert _pending_count(conn) == 1  # only the _HASH PENDING row
    other = conn.execute(
        "SELECT status FROM chunks WHERE pdf_hash='other_doc' AND chunk_idx=1"
    ).fetchone()[0]
    assert other == "DONE"


def test_done_ordering_query_for_merge_paths(conn):
    for idx in (3, 1, 2):
        conn.execute(
            "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
            (_HASH, idx, "DONE"),
        )
    rows = conn.execute(
        "SELECT chunk_idx FROM chunks WHERE pdf_hash=? AND status='DONE' ORDER BY chunk_idx",
        (_HASH,),
    ).fetchall()
    assert [r[0] for r in rows] == [1, 2, 3]
