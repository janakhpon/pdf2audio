"""ChunkStateStore — the real resumable chunk-state repository (pdf2audio/state.py).

Exercises the actual class against a temp SQLite file (not a replicated schema), covering:

    PENDING -> PROCESSING -> DONE
                          -> FAILED
    on restart: PROCESSING -> PENDING (reset in-flight work)
    "ready to merge" when pending_count() == 0
"""

from __future__ import annotations

import pytest
from pdf2audio.errors import DatabaseError
from pdf2audio.state import ChunkStateStore, ChunkStatus

_HASH = "deadbeef"


@pytest.fixture
def store(tmp_path):
    s = ChunkStateStore(tmp_path / "state.db")
    yield s
    s.close()


def test_unknown_chunk_status_is_none(store):
    assert store.status(_HASH, 1) is None


def test_mark_is_idempotent_upsert(store):
    store.mark(_HASH, 1, ChunkStatus.PROCESSING)
    store.mark(_HASH, 1, ChunkStatus.PROCESSING)  # re-mark must replace, not duplicate
    assert store.status(_HASH, 1) == ChunkStatus.PROCESSING
    assert store.done_indices(_HASH) == []


def test_transition_processing_to_done(store):
    store.mark(_HASH, 1, ChunkStatus.PROCESSING)
    store.mark(_HASH, 1, ChunkStatus.DONE)
    assert store.status(_HASH, 1) == ChunkStatus.DONE


def test_transition_processing_to_failed(store):
    store.mark(_HASH, 2, ChunkStatus.PROCESSING)
    store.mark(_HASH, 2, ChunkStatus.FAILED)
    assert store.status(_HASH, 2) == ChunkStatus.FAILED


def test_reset_stale_reverts_only_processing(store):
    for idx, status in [
        (1, ChunkStatus.PROCESSING),
        (2, ChunkStatus.DONE),
        (3, ChunkStatus.PROCESSING),
        (4, ChunkStatus.FAILED),
    ]:
        store.mark(_HASH, idx, status)

    store.reset_stale(_HASH)

    assert store.status(_HASH, 1) == ChunkStatus.PENDING
    assert store.status(_HASH, 3) == ChunkStatus.PENDING
    # DONE and FAILED are untouched.
    assert store.status(_HASH, 2) == ChunkStatus.DONE
    assert store.status(_HASH, 4) == ChunkStatus.FAILED


def test_pending_count_zero_means_ready_to_merge(store):
    for idx in (1, 2, 3):
        store.mark(_HASH, idx, ChunkStatus.DONE)
    assert store.pending_count(_HASH) == 0


def test_failed_counts_as_pending_and_blocks_merge(store):
    store.mark(_HASH, 1, ChunkStatus.DONE)
    store.mark(_HASH, 2, ChunkStatus.FAILED)
    assert store.pending_count(_HASH) == 1


def test_hash_scoping_isolates_documents(store):
    store.mark(_HASH, 1, ChunkStatus.PENDING)
    store.mark("other_doc", 1, ChunkStatus.DONE)

    store.reset_stale(_HASH)  # must not touch the other document's rows

    assert store.pending_count(_HASH) == 1  # only the _HASH PENDING row
    assert store.status("other_doc", 1) == ChunkStatus.DONE


def test_done_indices_are_ordered(store):
    for idx in (3, 1, 2):
        store.mark(_HASH, idx, ChunkStatus.DONE)
    assert store.done_indices(_HASH) == [1, 2, 3]


def test_state_survives_reopen(tmp_path):
    db = tmp_path / "state.db"
    with ChunkStateStore(db) as store:
        store.mark(_HASH, 1, ChunkStatus.DONE)
        store.mark(_HASH, 2, ChunkStatus.PROCESSING)
    # A fresh process resuming the same document sees the persisted state.
    with ChunkStateStore(db) as reopened:
        assert reopened.status(_HASH, 1) == ChunkStatus.DONE
        reopened.reset_stale(_HASH)
        assert reopened.status(_HASH, 2) == ChunkStatus.PENDING


def test_operations_after_close_raise_database_error(tmp_path):
    store = ChunkStateStore(tmp_path / "state.db")
    store.close()
    with pytest.raises(DatabaseError):
        store.mark(_HASH, 1, ChunkStatus.DONE)
