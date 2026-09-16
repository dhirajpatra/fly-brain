"""
Integration tests against a REAL Neo4j instance.

These auto-skip if Neo4j isn't reachable, so `pytest` runs clean with no
setup, but `docker compose up -d && pytest` exercises the actual
schema/load/verify path end to end. Deliberately uses its own throwaway
neurons (IDs 900001+) so it never collides with a real subset load and can
clean up after itself.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def _neo4j_available():
    if config.NEO4J_PASSWORD is None:
        return False
    try:
        driver = config.get_driver()
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


requires_neo4j = pytest.mark.skipif(
    not _neo4j_available(),
    reason='Neo4j not reachable -- start it with `docker compose up -d` '
           'and export NEO4J_PASSWORD to run this test.',
)


@pytest.fixture
def clean_test_neurons():
    """Delete any leftover test nodes before AND after the test."""
    def _cleanup():
        driver = config.get_driver()
        try:
            with driver.session() as session:
                session.run(
                    'MATCH (n:Neuron) WHERE n.flywire_id >= 900001 '
                    'AND n.flywire_id <= 900010 DETACH DELETE n'
                )
        finally:
            driver.close()
    _cleanup()
    yield
    _cleanup()


@requires_neo4j
def test_schema_apply_is_idempotent():
    from schema import apply_schema
    apply_schema()
    apply_schema()  # must not raise on a second run


@requires_neo4j
def test_load_and_verify_roundtrip(clean_test_neurons):
    from load_subset import load_edges, load_nodes

    comp_df = pd.DataFrame(
        {'Completed': [True, True]},
        index=[900001, 900002],
    )
    conn_df = pd.DataFrame({
        'Presynaptic_ID': [900001],
        'Postsynaptic_ID': [900002],
        'Presynaptic_Index': [0],
        'Postsynaptic_Index': [1],
        'Connectivity': [4],
        'Excitatory': [1],
        'Excitatory x Connectivity': [4],
    })

    driver = config.get_driver()
    try:
        with driver.session() as session:
            n_nodes = load_nodes(session, comp_df, {900001, 900002})
            n_edges = load_edges(session, conn_df, {900001, 900002})
            assert n_nodes == 2
            assert n_edges == 1

            result = session.run(
                'MATCH (pre:Neuron {flywire_id: 900001})'
                '-[r:SYNAPSES_TO]->(post:Neuron {flywire_id: 900002}) '
                'RETURN r.weight AS weight, pre.idx AS pre_idx, '
                'post.idx AS post_idx'
            )
            record = result.single()
            assert record is not None, (
                'Edge not found -- check orientation in load_edges().'
            )
            assert record['weight'] == 4
            assert record['pre_idx'] == 0
            assert record['post_idx'] == 1
    finally:
        driver.close()
