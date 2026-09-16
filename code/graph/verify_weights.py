"""
Round-trip check for the graph -> weight tensor path.

Rebuilds a sparse COO tensor from what's currently in Neo4j and compares it
exactly against the existing data/weight_coo.pkl produced by
run_pytorch.get_weights(). This only proves anything for neuron pairs that
are actually present in the graph (e.g. the small first-pass subset) --
run load_subset.py first, and only expect a match on the rows/cols that
were loaded.

Run this before trusting ANY Neo4j-derived weight tensor for a simulation.
get_weights() indexes as [Postsynaptic_Index, Presynaptic_Index] (row=post,
col=pre) -- this script reproduces that exact orientation and will fail
loudly if a future edit to load_subset.py flips it.
"""

import pickle

import torch

from config import PATH_WT, get_driver


def rebuild_coo_from_graph(num_neurons):
    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (pre:Neuron)-[r:SYNAPSES_TO]->(post:Neuron)
                RETURN post.idx AS post_idx, pre.idx AS pre_idx, r.weight AS weight
                """
            )
            rows, cols, vals = [], [], []
            for record in result:
                rows.append(record['post_idx'])   # row = post  (matches get_weights)
                cols.append(record['pre_idx'])    # col = pre
                vals.append(record['weight'])
    finally:
        driver.close()

    if not rows:
        raise RuntimeError(
            'No SYNAPSES_TO edges found in Neo4j. Run load_subset.py first.'
        )

    return torch.sparse_coo_tensor(
        [rows, cols], vals, (num_neurons, num_neurons)
    ).to(torch.float32).coalesce()


def main():
    coo_path = PATH_WT / 'weight_coo.pkl'
    if not coo_path.exists():
        raise RuntimeError(
            f'{coo_path} not found. Run the PyTorch benchmark once first '
            'so the reference pickle exists.'
        )

    with open(coo_path, 'rb') as f:
        reference = pickle.load(f).coalesce()

    num_neurons = reference.shape[0]
    print(f'Reference weight_coo.pkl: {reference.shape}, '
          f'{reference._nnz()} non-zero entries')

    rebuilt = rebuild_coo_from_graph(num_neurons)
    print(f'Rebuilt from Neo4j:       {rebuilt.shape}, '
          f'{rebuilt._nnz()} non-zero entries')

    # Only compare entries the graph actually has -- a subset load will
    # legitimately have far fewer non-zeros than the full reference.
    ref_indices = reference.indices()
    ref_values = reference.values()
    rebuilt_dense_lookup = {
        (int(r), int(c)): float(v)
        for r, c, v in zip(*rebuilt.indices(), rebuilt.values())
    }

    mismatches = []
    checked = 0
    for i in range(ref_indices.shape[1]):
        r, c = int(ref_indices[0, i]), int(ref_indices[1, i])
        if (r, c) not in rebuilt_dense_lookup:
            continue
        checked += 1
        ref_val = float(ref_values[i])
        graph_val = rebuilt_dense_lookup[(r, c)]
        if ref_val != graph_val:
            mismatches.append((r, c, ref_val, graph_val))

    print(f'Checked {checked} overlapping entries.')
    if mismatches:
        print(f'MISMATCH: {len(mismatches)} entries differ. First few:')
        for r, c, ref_val, graph_val in mismatches[:10]:
            print(f'  (post={r}, pre={c}): pickle={ref_val} graph={graph_val}')
        raise SystemExit(1)

    if checked == 0:
        print('WARNING: zero overlapping entries checked -- verified nothing. '
              'Is the loaded subset disjoint from expectations?')
        raise SystemExit(1)

    print('OK: all overlapping entries match exactly.')


if __name__ == '__main__':
    main()
