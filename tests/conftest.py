"""Shared test setup."""

from __future__ import annotations

import pytest

from chiprl import cache


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path_factory, monkeypatch):
    """Never read or write the developer's real result cache from a test."""
    monkeypatch.setattr(cache, "DB", tmp_path_factory.mktemp("cache") / "results.sqlite")
