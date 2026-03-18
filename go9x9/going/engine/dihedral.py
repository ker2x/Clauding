"""
Dihedral symmetry transforms for 9x9 Go.

The square board has 8 symmetries (D4 group): identity, 3 rotations, 2 reflections,
2 diagonal reflections. Precomputed permutation tables map flat board indices.
"""

import numpy as np

BOARD_SIZE = 9


def _build_permutation(transform_fn):
    """Build a flat-index permutation array from a (row, col) -> (row, col) function."""
    perm = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            nr, nc = transform_fn(r, c)
            perm[r * BOARD_SIZE + c] = nr * BOARD_SIZE + nc
    return perm


# The 8 dihedral transforms: (row, col) -> (new_row, new_col)
_TRANSFORMS = [
    lambda r, c: (r, c),                           # identity
    lambda r, c: (c, BOARD_SIZE - 1 - r),           # rot90 CW
    lambda r, c: (BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c),  # rot180
    lambda r, c: (BOARD_SIZE - 1 - c, r),           # rot270 CW
    lambda r, c: (r, BOARD_SIZE - 1 - c),            # flip horizontal
    lambda r, c: (BOARD_SIZE - 1 - r, c),            # flip vertical
    lambda r, c: (c, r),                             # transpose
    lambda r, c: (BOARD_SIZE - 1 - c, BOARD_SIZE - 1 - r),  # anti-transpose
]

# Precomputed permutation arrays for flat board indices (0-80)
PERMUTATIONS = [_build_permutation(fn) for fn in _TRANSFORMS]


def transform_state(state: np.ndarray, perm_idx: int) -> np.ndarray:
    """Transform a (C, 9, 9) state tensor using dihedral symmetry."""
    flat = state.reshape(state.shape[0], -1)  # (C, 81)
    perm = PERMUTATIONS[perm_idx]
    transformed = flat[:, perm].reshape(state.shape)
    return transformed


def transform_policy(policy: np.ndarray, perm_idx: int) -> np.ndarray:
    """Transform an 82-element policy vector (81 positions + pass)."""
    perm = PERMUTATIONS[perm_idx]
    result = np.zeros_like(policy)
    result[perm] = policy[:81]  # remap board positions
    result[81] = policy[81]     # pass unchanged
    return result


def transform_ownership(ownership: np.ndarray, perm_idx: int) -> np.ndarray:
    """Transform an 81-element ownership vector."""
    perm = PERMUTATIONS[perm_idx]
    result = np.zeros_like(ownership)
    result[perm] = ownership
    return result


def augment_sample(state, policy, value, ownership, surprise):
    """Yield all 8 dihedral transforms of a single training sample."""
    for i in range(8):
        yield (
            transform_state(state, i),
            transform_policy(policy, i),
            value,
            transform_ownership(ownership, i),
            surprise,
        )
