"""
Tests for verify_weights.compare_coo.

This is the check that's supposed to catch a post/pre orientation flip
before anyone trusts a Neo4j-derived weight tensor for a simulation, so it
gets tested against exactly that failure mode: a tensor that's correct
except transposed.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verify_weights import compare_coo  # noqa: E402


def make_coo(entries, size=5):
    """entries: list of (row, col, value)."""
    if not entries:
        rows, cols, vals = [], [], []
    else:
        rows, cols, vals = zip(*entries)
    return torch.sparse_coo_tensor(
        [list(rows), list(cols)], list(vals), (size, size)
    ).to(torch.float32).coalesce()


def test_identical_tensors_have_no_mismatches():
    ref = make_coo([(1, 0, 5.0), (2, 1, -3.0)])
    checked, mismatches = compare_coo(ref, ref)
    assert checked == 2
    assert mismatches == []


def test_subset_with_fewer_entries_is_not_a_mismatch():
    # Reference has 3 entries, rebuilt (subset) has only 1 -- that 1 should
    # match cleanly and the other 2 are simply not checked, not flagged.
    ref = make_coo([(1, 0, 5.0), (2, 1, -3.0), (3, 2, 7.0)])
    rebuilt = make_coo([(1, 0, 5.0)], size=5)
    checked, mismatches = compare_coo(ref, rebuilt)
    assert checked == 1
    assert mismatches == []


def test_transposed_tensor_is_caught_as_mismatch():
    # This is the exact bug class flagged throughout the codebase: rebuilt
    # with (pre, post) instead of (post, pre). Same values, wrong
    # positions -- should NOT silently pass.
    ref = make_coo([(1, 0, 5.0), (2, 1, -3.0)])
    transposed = make_coo([(0, 1, 5.0), (1, 2, -3.0)])  # rows/cols swapped
    checked, mismatches = compare_coo(ref, transposed)
    # None of the transposed positions overlap the reference positions, so
    # nothing gets checked at all -- which itself must be treated as a
    # failure by the caller (see verify_weights.main()'s checked==0 guard),
    # not silently reported as "0 mismatches, all good".
    assert checked == 0


def test_wrong_value_at_correct_position_is_caught():
    ref = make_coo([(1, 0, 5.0)])
    wrong_value = make_coo([(1, 0, 999.0)])
    checked, mismatches = compare_coo(ref, wrong_value)
    assert checked == 1
    assert mismatches == [(1, 0, 5.0, 999.0)]
