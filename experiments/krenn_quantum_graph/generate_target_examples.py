"""Generate small exact examples for the target-generic Krenn stack.

The collection is deliberately separate from the original canonical-GHZ
milestone bundles. One transparent ``n=2`` problem carries an exact rational
image witness; the remaining named targets are structure-only examples and
make no membership or nonmembership claim.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.artifacts import DEFAULT_RESULTS
from experiments.krenn_quantum_graph.target_artifacts import (
    verify_target_artifact_bundle,
    write_target_artifact_bundle,
)
from experiments.krenn_quantum_graph.targets import (
    ColoringTarget,
    heralded_ghz_target,
    unnormalized_dicke_target,
    unnormalized_graph_state_target,
    unnormalized_w_target,
)
from experiments.krenn_quantum_graph.tensor_map import MatchingTensorMap
from experiments.krenn_quantum_graph.witness import SparseWitness


DEFAULT_TARGET_EXAMPLE_RESULTS = DEFAULT_RESULTS / "target_examples"

BUNDLE_NAMES = (
    "n2_d2_arbitrary_rational_exact_witness",
    "n4_d3_heralded_ghz_k2_structure_only",
    "n4_d2_w_target_structure_only",
    "n4_d2_dicke2_target_structure_only",
    "n4_d2_one_edge_graph_state_structure_only",
)


class KrennTargetExampleError(RuntimeError):
    """The released target-example collection changed unexpectedly."""


def _arbitrary_rational_example(
) -> tuple[ColoringTarget, SparseWitness]:
    coefficients = {
        (0, 0): Fraction(1, 2),
        (0, 1): Fraction(-2, 3),
        (1, 0): Fraction(5, 7),
        (1, 1): Fraction(11, 5),
    }
    target = ColoringTarget.from_sparse(2, 2, coefficients)
    witness = SparseWitness.from_coordinates(
        2,
        2,
        {
            (0, 1, a, b): value
            for (a, b), value in coefficients.items()
        },
    )
    return target, witness


def _bundle_specs(
) -> tuple[
    tuple[str, MatchingTensorMap, ColoringTarget, SparseWitness | None],
    ...,
]:
    arbitrary_target, arbitrary_witness = _arbitrary_rational_example()
    return (
        (
            BUNDLE_NAMES[0],
            MatchingTensorMap(2, 2),
            arbitrary_target,
            arbitrary_witness,
        ),
        (
            BUNDLE_NAMES[1],
            MatchingTensorMap(4, 3),
            heralded_ghz_target(4, 3, 2, trigger_color=0),
            None,
        ),
        (
            BUNDLE_NAMES[2],
            MatchingTensorMap(4, 2),
            unnormalized_w_target(4),
            None,
        ),
        (
            BUNDLE_NAMES[3],
            MatchingTensorMap(4, 2),
            unnormalized_dicke_target(4, 2),
            None,
        ),
        (
            BUNDLE_NAMES[4],
            MatchingTensorMap(4, 2),
            unnormalized_graph_state_target(4, ((0, 1),)),
            None,
        ),
    )


def _resolved_collection_root(
    output_root: Path | str,
    *,
    create: bool,
) -> Path:
    raw_root = Path(output_root)
    if raw_root.is_symlink():
        raise KrennTargetExampleError(
            "target-example root must not be a symbolic link"
        )
    root = raw_root.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise KrennTargetExampleError(
            "target-example root must be a directory"
        )
    return root


def _check_root_inventory(
    output_root: Path,
    *,
    complete: bool,
) -> None:
    expected = set(BUNDLE_NAMES)
    actual = {entry.name for entry in output_root.iterdir()}
    allowed = actual == expected if complete else actual <= expected
    if not allowed:
        qualifier = "complete " if complete else ""
        raise KrennTargetExampleError(
            f"{qualifier}target-example root inventory changed: "
            f"expected {sorted(expected)}, found {sorted(actual)}"
        )
    for entry in output_root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            raise KrennTargetExampleError(
                "target-example entries must be regular directories"
            )


def generate_all(
    output_root: Path | str = DEFAULT_TARGET_EXAMPLE_RESULTS,
) -> Mapping[str, Mapping]:
    """Write all examples and immediately replay every certificate."""

    root = _resolved_collection_root(output_root, create=True)
    _check_root_inventory(root, complete=False)
    for name, tensor_map, target, witness in _bundle_specs():
        write_target_artifact_bundle(
            tensor_map,
            target,
            root / name,
            witness=witness,
        )
    return verify_all(root)


def verify_all(
    output_root: Path | str = DEFAULT_TARGET_EXAMPLE_RESULTS,
) -> Mapping[str, Mapping]:
    """Replay the fixed collection and reject missing or extra bundles."""

    root = _resolved_collection_root(output_root, create=False)
    _check_root_inventory(root, complete=True)
    summary = {}
    for name in BUNDLE_NAMES:
        bundle = verify_target_artifact_bundle(root / name)
        claims = bundle.certificate["claims"]
        summary[name] = {
            "exact_artifact_replay": bundle.exact,
            "status": bundle.certificate["status"],
            "n": bundle.tensor_map.n,
            "d": bundle.tensor_map.d,
            "variables": bundle.tensor_map.variable_count,
            "coefficients": bundle.tensor_map.coefficient_count,
            "degree": bundle.tensor_map.degree,
            "target_support": bundle.target.support_size,
            "witness_present": bundle.witness is not None,
            "affine_image_certified": claims[
                "exact_rational_affine_image_membership_certified"
            ],
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify target-generic Krenn examples."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_TARGET_EXAMPLE_RESULTS,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify existing bundles instead of regenerating them",
    )
    arguments = parser.parse_args(argv)
    summary = (
        verify_all(arguments.output_root)
        if arguments.verify
        else generate_all(arguments.output_root)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
