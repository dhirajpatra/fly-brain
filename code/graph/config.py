"""
Shared Neo4j connection config for the code/graph/ package.

Reads connection details from environment variables so credentials never
land in source control:

    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=...      # same value used to start docker-compose

Matches the paths already defined in benchmark.py so this package stays in
sync with the existing repo layout.
"""

import os
from pathlib import Path

NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

if NEO4J_PASSWORD is None:
    raise RuntimeError(
        'NEO4J_PASSWORD is not set. Export it (same value used for '
        'docker-compose) before running anything in code/graph/.'
    )

# Mirrors benchmark.py's path layout: code/graph/ -> code/ -> repo root -> data/
_current_dir = Path(__file__).resolve().parent
_code_dir = _current_dir.parent
_repo_dir = _code_dir.parent

PATH_COMP = (_repo_dir / 'data/2025_Completeness_783.csv').resolve()
PATH_CONN = (_repo_dir / 'data/2025_Connectivity_783.parquet').resolve()
PATH_WT = (_repo_dir / 'data').resolve()


def get_driver():
    """Return a configured Neo4j driver. Caller is responsible for closing it."""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
