"""The typed exception hierarchy: every domain error is a PDF2AudioError is an Exception."""

from __future__ import annotations

import pytest

from src.errors import (
    AudioError,
    ConfigError,
    DatabaseError,
    EditorError,
    ExtractionError,
    MergeError,
    PDF2AudioError,
)

_SUBCLASSES = [
    ConfigError,
    ExtractionError,
    EditorError,
    AudioError,
    MergeError,
    DatabaseError,
]


def test_base_is_exception():
    assert issubclass(PDF2AudioError, Exception)


@pytest.mark.parametrize("err_cls", _SUBCLASSES)
def test_subclass_hierarchy(err_cls):
    assert issubclass(err_cls, PDF2AudioError)
    assert issubclass(err_cls, Exception)
    instance = err_cls("boom")
    assert isinstance(instance, PDF2AudioError)
    assert isinstance(instance, Exception)
    assert str(instance) == "boom"


def test_family_can_be_caught_as_base():
    with pytest.raises(PDF2AudioError):
        raise ExtractionError("caught as family")
