"""
Schema setup for the connectome graph.

Node label:  Neuron
  flywire_id : int   (raw FlyWire ID, e.g. 720575940624963786 — UNIQUE)
  idx        : int   (positional index from 2025_Completeness_783.csv —
                       this is what Presynaptic_Index / Postsynaptic_Index
                       and the pickled weight tensors are keyed on; NEVER
                       recompute this for a subset, always carry the
                       original value through)
  completed  : bool

Relationship: (:Neuron)-[:SYNAPSES_TO {weight, connectivity, excitatory}]->(:Neuron)
  Direction is pre -> post (the natural/intuitive direction). Note this is
  the OPPOSITE of how run_pytorch.get_weights() builds its COO tensor,
  which indexes [Postsynaptic_Index, Presynaptic_Index] (post, pre). Any
  code that reconstructs a weight tensor FROM this graph must swap back to
  that (post, pre) order -- see verify_weights.py, which checks this
  automatically. Getting this backwards means the reconstructed network
  runs signals in reverse and everything downstream is silently wrong.

  weight       = 'Excitatory x Connectivity' from the parquet (signed int)
  connectivity = raw 'Connectivity' count (unsigned int)
  excitatory   = 'Excitatory' flag as loaded (int, 1 or -1 per source data)

Run this once against a fresh database before loading any data.
"""

from config import get_driver

CONSTRAINTS_AND_INDEXES = [
    "CREATE CONSTRAINT neuron_flywire_id IF NOT EXISTS "
    "FOR (n:Neuron) REQUIRE n.flywire_id IS UNIQUE",

    "CREATE INDEX neuron_idx IF NOT EXISTS "
    "FOR (n:Neuron) ON (n.idx)",

    # Provenance graph (Phase 4) -- created now so later scripts don't need
    # a second schema pass.
    "CREATE CONSTRAINT run_key IF NOT EXISTS "
    "FOR (r:Run) REQUIRE (r.run_label, r.round, r.backend_key) IS UNIQUE",

    "CREATE CONSTRAINT spike_export_path IF NOT EXISTS "
    "FOR (s:SpikeExport) REQUIRE s.spike_path IS UNIQUE",
]


def apply_schema():
    driver = get_driver()
    try:
        with driver.session() as session:
            for stmt in CONSTRAINTS_AND_INDEXES:
                session.run(stmt)
                print(f'Applied: {stmt.splitlines()[0]}...')
    finally:
        driver.close()


if __name__ == '__main__':
    apply_schema()
