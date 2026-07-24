"""Generate or verify the first complete Krenn application milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.krenn_quantum_graph.artifacts import (
    DEFAULT_RESULTS,
    verify_artifact_bundle,
    write_artifact_bundle,
)
from experiments.krenn_quantum_graph.deformation import (
    certify_n4_d3_deformation,
)
from experiments.krenn_quantum_graph.fixtures import (
    fixture_n4_d3,
    fixture_n6_d2,
)
from experiments.krenn_quantum_graph.independent_verifier import (
    canonical_claim_boundary,
)
from experiments.krenn_quantum_graph.system import generate_sparse_system


BUNDLE_NAMES = (
    "n4_d3_fixture",
    "n6_d2_fixture",
    "n4_d4_negative_benchmark",
    "n6_d4_production_system",
)


def negative_benchmark_boundary() -> Mapping:
    return canonical_claim_boundary(4, 4)


def production_search_boundary() -> Mapping:
    return canonical_claim_boundary(6, 4)


def generate_all(
    output_root: Path | str = DEFAULT_RESULTS,
) -> Mapping[str, Mapping]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_inputs = (Path(__file__).resolve(),)

    system_43 = generate_sparse_system(4, 3)
    witness_43 = fixture_n4_d3()
    deformation = certify_n4_d3_deformation(system_43, witness_43)
    write_artifact_bundle(
        system_43,
        output_root / BUNDLE_NAMES[0],
        witness=witness_43,
        deformation=deformation,
        input_paths=source_inputs,
    )

    system_62 = generate_sparse_system(6, 2)
    write_artifact_bundle(
        system_62,
        output_root / BUNDLE_NAMES[1],
        witness=fixture_n6_d2(),
        input_paths=source_inputs,
    )

    write_artifact_bundle(
        generate_sparse_system(4, 4),
        output_root / BUNDLE_NAMES[2],
        boundary=negative_benchmark_boundary(),
        input_paths=source_inputs,
    )

    write_artifact_bundle(
        generate_sparse_system(6, 4),
        output_root / BUNDLE_NAMES[3],
        boundary=production_search_boundary(),
        input_paths=source_inputs,
    )
    return verify_all(output_root)


def verify_all(
    output_root: Path | str = DEFAULT_RESULTS,
) -> Mapping[str, Mapping]:
    output_root = Path(output_root).resolve()
    summary = {}
    for name in BUNDLE_NAMES:
        bundle = verify_artifact_bundle(output_root / name)
        summary[name] = {
            "exact": bundle.exact,
            "status": bundle.certificate["status"],
            "variables": bundle.system.variable_count,
            "equations": bundle.system.equation_count,
            "degree": bundle.system.degree,
            "matchings_per_equation": bundle.system.matching_count,
            "monomials": bundle.system.monomial_count,
            "witness_present": bundle.witness is not None,
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify standalone Krenn artifacts."
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_RESULTS
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
