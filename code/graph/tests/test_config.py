"""
Tests for config.py.

Mainly guards against the import-time bug this package actually had: the
NEO4J_PASSWORD check used to run at module import, which meant importing
load_subset.py for its pure logic (see test_load_subset.py) would crash
without Docker even running. That check now lives in get_driver() instead.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def test_module_imports_without_neo4j_password_set(monkeypatch):
    monkeypatch.delenv('NEO4J_PASSWORD', raising=False)
    import importlib
    importlib.reload(config)  # re-run module body with the var unset
    # Getting here at all is the test -- import must not raise.


def test_get_driver_raises_clearly_when_password_missing(monkeypatch):
    monkeypatch.delenv('NEO4J_PASSWORD', raising=False)
    import importlib
    importlib.reload(config)
    with pytest.raises(RuntimeError, match='NEO4J_PASSWORD'):
        config.get_driver()


def test_paths_point_at_repo_data_directory():
    # code/graph/config.py -> code/ -> repo root -> data/
    assert config.PATH_COMP.name == '2025_Completeness_783.csv'
    assert config.PATH_CONN.name == '2025_Connectivity_783.parquet'
    assert config.PATH_COMP.parent.name == 'data'
    assert config.PATH_COMP.parent == config.PATH_CONN.parent
