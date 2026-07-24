"""Required sparse Krenn witnesses."""

from __future__ import annotations

from experiments.krenn_quantum_graph.witness import SparseWitness


def fixture_n4_d3() -> SparseWitness:
    """The six-entry exact witness requested for ``n=4,d=3``."""

    return SparseWitness.from_coordinates(
        4,
        3,
        {
            (0, 1, 0, 0): 1,
            (2, 3, 0, 0): 1,
            (0, 2, 1, 1): 1,
            (1, 3, 1, 1): 1,
            (0, 3, 2, 2): 1,
            (1, 2, 2, 2): 1,
        },
    )


def fixture_n6_d2() -> SparseWitness:
    """The six-entry exact witness requested for ``n=6,d=2``."""

    return SparseWitness.from_coordinates(
        6,
        2,
        {
            (0, 1, 0, 0): 1,
            (2, 3, 0, 0): 1,
            (4, 5, 0, 0): 1,
            (0, 5, 1, 1): 1,
            (1, 2, 1, 1): 1,
            (3, 4, 1, 1): 1,
        },
    )

