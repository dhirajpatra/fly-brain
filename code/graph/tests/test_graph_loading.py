"""
Tests for load_subset.load_nodes / load_edges.

Uses a fake session that just records what Cypher + params it was called
with, so these run without Docker or a real Neo4j instance. What they
actually check:

  - load_nodes: `idx` on each node is the position in the FULL completeness
    file, not a position recomputed over the filtered subset. This is the
    exact bug class flagged in load_subset.py's own comments -- silently
    renumbering neurons would let a simulation run with plausible-looking
    but wrong connectivity.
  - load_edges: only edges where BOTH endpoints are in the subset are
    loaded (no dangling half-edges to neurons outside the subset), and the
    'Excitatory x Connectivity' column is renamed to 'weight' without its
    value being altered.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load_subset import load_edges, load_nodes  # noqa: E402


class FakeSession:
    """Records the params passed to run() instead of executing anything."""
    def __init__(self):
        self.calls = []

    def run(self, query, **kwargs):
        self.calls.append({'query': query, **kwargs})
        return None


def test_load_nodes_preserves_full_file_idx_not_subset_position(comp_df):
    # Subset skips id 200 (idx=1) and 300 (idx=2) entirely. If idx were
    # recomputed over just this subset, 400 would wrongly get idx=1 instead
    # of its true idx=3.
    session = FakeSession()
    load_nodes(session, comp_df, neuron_ids={100, 400, 500})

    rows_by_id = {r['flywire_id']: r for r in session.calls[0]['rows']}
    assert rows_by_id[100]['idx'] == 0
    assert rows_by_id[400]['idx'] == 3   # NOT 1
    assert rows_by_id[500]['idx'] == 4   # NOT 2


def test_load_nodes_carries_completed_flag(comp_df):
    session = FakeSession()
    load_nodes(session, comp_df, neuron_ids={300})
    row = session.calls[0]['rows'][0]
    assert row['flywire_id'] == 300
    assert row['completed'] is False


def test_load_nodes_returns_count(comp_df):
    session = FakeSession()
    n = load_nodes(session, comp_df, neuron_ids={100, 200, 300})
    assert n == 3


def test_load_edges_excludes_edges_leaving_the_subset(conn_df):
    # Full conn_df has 100->200, 200->300, 300->400, 500->100.
    # Subset {100, 200, 500}: only 100->200 and 500->100 have both
    # endpoints inside; 200->300 and 300->400 must be dropped.
    session = FakeSession()
    load_edges(session, conn_df, neuron_ids={100, 200, 500})

    rows = session.calls[0]['rows']
    pairs = {(r['Presynaptic_ID'], r['Postsynaptic_ID']) for r in rows}
    assert pairs == {(100, 200), (500, 100)}


def test_load_edges_renames_weight_without_changing_value(conn_df):
    session = FakeSession()
    load_edges(session, conn_df, neuron_ids={100, 200})

    rows = session.calls[0]['rows']
    assert len(rows) == 1
    row = rows[0]
    assert 'weight' in row
    assert 'Excitatory x Connectivity' not in row
    assert row['weight'] == 5   # matches conn_df fixture: 100->200 weight=5


def test_load_edges_returns_count(conn_df):
    session = FakeSession()
    n = load_edges(session, conn_df, neuron_ids={100, 200, 300, 400, 500})
    assert n == 4   # all four edges have both endpoints in the full set
