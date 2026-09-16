"""
Load an N-hop subset of the connectome into Neo4j, seeded from an
experiment's stimulated neurons (e.g. benchmark.EXPERIMENTS['sugar']).

Filters by raw FlyWire ID directly against 2025_Connectivity_783.parquet
(Presynaptic_ID / Postsynaptic_ID columns), so no index remapping happens
anywhere in this path -- every node keeps the exact `idx` value it has in
2025_Completeness_783.csv, unchanged. That's what keeps this subset
compatible with the pickled weight tensors if it's ever used for anything
beyond querying (see verify_weights.py).

Usage:
    python load_subset.py --experiment sugar --hops 2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import PATH_COMP, PATH_CONN, get_driver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/
from benchmark import get_experiment  # noqa: E402


def expand_hops(seed_ids, conn_df, hops):
    """Return the set of FlyWire IDs reachable within `hops` steps of seed_ids.

    Undirected expansion (follows edges in either direction) since the goal
    is a queryable neighbourhood, not a directional traversal.
    """
    frontier = set(seed_ids)
    visited = set(seed_ids)
    for _ in range(hops):
        touched = conn_df[
            conn_df['Presynaptic_ID'].isin(frontier)
            | conn_df['Postsynaptic_ID'].isin(frontier)
        ]
        next_frontier = (
            set(touched['Presynaptic_ID']) | set(touched['Postsynaptic_ID'])
        ) - visited
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier
    return visited


def load_nodes(session, comp_df, neuron_ids):
    # idx must be the position in the FULL completeness file (matches
    # Presynaptic_Index/Postsynaptic_Index and the pickled weight tensors),
    # never a position recomputed over a filtered subset.
    full_positions = {fid: i for i, fid in enumerate(comp_df.index)}
    records = [
        {'flywire_id': int(fid), 'idx': int(full_positions[fid]),
         'completed': bool(comp_df.loc[fid, 'Completed'])}
        for fid in neuron_ids
    ]
    session.run(
        """
        UNWIND $rows AS row
        MERGE (n:Neuron {flywire_id: row.flywire_id})
        SET n.idx = row.idx, n.completed = row.completed
        """,
        rows=records,
    )
    return len(records)


def load_edges(session, conn_df, neuron_ids):
    subset = conn_df[
        conn_df['Presynaptic_ID'].isin(neuron_ids)
        & conn_df['Postsynaptic_ID'].isin(neuron_ids)
    ]
    records = subset[[
        'Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity',
        'Excitatory', 'Excitatory x Connectivity',
    ]].rename(columns={'Excitatory x Connectivity': 'weight'}).to_dict('records')

    session.run(
        """
        UNWIND $rows AS row
        MATCH (pre:Neuron {flywire_id: row.Presynaptic_ID})
        MATCH (post:Neuron {flywire_id: row.Postsynaptic_ID})
        MERGE (pre)-[r:SYNAPSES_TO]->(post)
        SET r.weight = row.weight,
            r.connectivity = row.Connectivity,
            r.excitatory = row.Excitatory
        """,
        rows=records,
    )
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='sugar')
    parser.add_argument('--hops', type=int, default=2)
    args = parser.parse_args()

    experiment = get_experiment(args.experiment)
    seed_ids = set(experiment['neu_exc']) | set(experiment.get('neu_exc2', []))
    print(f"Experiment '{experiment['key']}': {len(seed_ids)} seed neuron(s)")

    print('Reading connectivity parquet...')
    conn_df = pd.read_parquet(PATH_CONN)
    print('Reading completeness CSV...')
    comp_df = pd.read_csv(PATH_COMP, index_col=0)

    print(f'Expanding {args.hops} hop(s) from seed set...')
    neuron_ids = expand_hops(seed_ids, conn_df, args.hops)
    print(f'Subset: {len(neuron_ids)} neurons')

    driver = get_driver()
    try:
        with driver.session() as session:
            n_nodes = load_nodes(session, comp_df, neuron_ids)
            n_edges = load_edges(session, conn_df, neuron_ids)
        print(f'Loaded {n_nodes} nodes, {n_edges} edges.')
    finally:
        driver.close()


if __name__ == '__main__':
    main()
