# Target-generic perfect-matching tensor stack

This directory is a standalone application layer for the polynomial tensor
map

\[
\Phi_{n,d}(W)_c =
\sum_{M\in\operatorname{PM}(K_n)}
\prod_{\{i,j\}\in M} W[i,j,c_i,c_j].
\]

The application problem is to solve `Phi(W) = T` for a prescribed target
tensor `T`. The canonical GHZ tensor is one target, not a built-in restriction
of the map.

Krenn code and released artifacts live only under
`experiments/krenn_quantum_graph`, `tests/test_krenn_*.py`, and
`results/krenn_quantum_graph`. The native tensor, target, artifact,
deformation, invariant, local-action, and search layers are self-contained.
The private compatibility bridge used during development is deliberately not
part of this repository.

## Exact conventions

- An edge variable is `W[i,j,a,b]` with canonical endpoints `i < j`.
- Edges are lexicographic; variables are edge-major, then `a`, then `b`.
- Colorings use product/C order, with the last vertex changing fastest.
- Matchings are deterministic: pair the least unused vertex with possible
  partners in ascending order.
- Under a vertex permutation, an edge that reverses orientation is
  recanonicalized and its two endpoint-color slots are swapped.
- Symmetry transports coordinates and indexes orbits. It never imposes equal
  weights.

All exact target and witness arithmetic currently uses integers or rational
numbers. Exact cyclotomic and general complex coefficient adapters are not
implemented.

## Layers

`targets.py` defines a canonical sparse exact-rational `ColoringTarget` and
constructors for:

- canonical and heralded GHZ targets;
- unnormalized qubit W and Dicke targets;
- unnormalized qudit Dicke occupation tensors;
- unnormalized qubit CZ graph-state tensors; and
- arbitrary rational target coefficients.

The named states are intentionally unnormalized so their coefficients remain
exact. `tensor_map.py` implements the target-independent `MatchingTensorMap`,
exact output and residual evaluation, target transport, and exact squared
normalized overlap for nonzero real-rational tensors. Fidelity is a metric,
not an image-membership certificate.

`system.py` attaches any exact rational target to the compact polynomial
system. Omitting `target=` retains the original canonical-GHZ behavior,
including the original artifact encoding.

`transport.py` implements exact `S_n x S_d` coordinate transport.
`local_actions.py` implements rational maps

\[
A_i:\mathbb Q^d\longrightarrow\mathbb Q^e
\]

on every vertex and verifies

\[
\Phi_{n,e}(A.W)=(A_0\otimes\cdots\otimes A_{n-1})\Phi_{n,d}(W)
\]

coefficientwise. This transports finite solutions forward. It does not, by
itself, propagate a no-go statement backward.

`tensor_invariants.py` provides exact, native diagnostics:

- universal scalar-gauge dimension bounds for the image closure;
- support-character ranks and local-diagonal stabilizer dimensions;
- rational flattening ranks; and
- exact-support necessary conditions.

These are one-sided invariants. A support or flattening test that passes is
not an image certificate, and the current invariants do not exclude a named
target from the whole image.

## Artifact formats

There are two deliberately separate schemas.

The original canonical-GHZ system bundles store:

- `equation_offsets`;
- `monomial_variable_indices`;
- `rhs_values`; and
- an optional sparse witness and deformation certificate.

The target-generic bundles store the target-independent map only:

- `equation_offsets`;
- `monomial_variable_indices`;
- exact rational `target.json`; and
- optional exact rational `witness.json`.

The target-generic NPZ contains no RHS. Both schemas use round-trippable
certificates, SHA-256 manifests, canonical source ledgers, safe path and ZIP
validation, primary reconstruction, and an independent matching
reconstruction. The independent verifier does not import or reuse the primary
perfect-matching enumerator. Corruption tests cover both stale hashes and
semantically corrupted files with refreshed hashes.

## Reproduce

Python 3.10 or newer and NumPy are required. A minimal setup is:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the complete 101-test standalone suite from the repository root:

```bash
python -m unittest discover -v -s tests -p "test_krenn_*.py"
```

Generate or verify the four original canonical-GHZ milestone bundles:

```bash
python -m experiments.krenn_quantum_graph.generate_results
python -m experiments.krenn_quantum_graph.generate_results --verify
```

Generate or verify five target-generic examples:

```bash
python -m experiments.krenn_quantum_graph.generate_target_examples
python -m experiments.krenn_quantum_graph.generate_target_examples --verify
```

Generate or verify the bounded `n=6,d=3` ternary milestone:

```bash
python -m experiments.krenn_quantum_graph.ternary_milestone_artifact \
  results/krenn_quantum_graph/n6_d3_ternary_milestone

python -m experiments.krenn_quantum_graph.ternary_milestone_artifact \
  results/krenn_quantum_graph/n6_d3_ternary_milestone --verify
```

## Original exact milestones

| Bundle | Variables | Equations | Degree | Terms/equation | Status |
|---|---:|---:|---:|---:|---|
| `n4_d3_fixture` | 54 | 81 | 2 | 3 | exact rational witness and deformation certificate |
| `n6_d2_fixture` | 60 | 64 | 3 | 15 | exact rational witness |
| `n4_d4_negative_benchmark` | 96 | 256 | 2 | 3 | structure only; nonexistence not reproved |
| `n6_d4_production_system` | 240 | 4,096 | 3 | 15 | structure and bounded support diagnostics only |

The `n=6,d=4` archive has 61,440 cubic monomials. No uncontrolled complex
weight search is started by its generator.

## Target-generic examples

| Bundle | Target | Status |
|---|---|---|
| `n2_d2_arbitrary_rational_exact_witness` | four unrelated rational coefficients | exact affine image witness over Q, hence over C |
| `n4_d3_heralded_ghz_k2_structure_only` | heralded GHZ | target and map structure only |
| `n4_d2_w_target_structure_only` | unnormalized W | target and map structure only |
| `n4_d2_dicke2_target_structure_only` | unnormalized weight-two Dicke | target and map structure only |
| `n4_d2_one_edge_graph_state_structure_only` | one-CZ-edge graph state | target and map structure only |

The four structure-only bundles assert no affine, projective,
local-diagonal-orbit, border-image, or nonexistence result.

## Bounded `n=6,d=3` ternary milestone

The fixed-target ternary search is over
`W[i,j,a,b] in {-1,0,1}` for canonical GHZ:

- 135 variables;
- 729 cubic equations;
- 15 matching monomials per equation;
- 10,935 monomials total.

The deterministic nine-weight seed satisfies 728 equations and has one exact
residual `+1`, at coloring `002121`. The seed census independently enumerates
all `15^3 = 3,375` ordered choices with one perfect matching assigned to each
color. They form 680 color-unordered multisets and exactly eight
`S_6 x S_3` orbits, with ordered orbit sizes summing back to 3,375.

The released bounded plan has:

- node cap 32;
- time cap 30 seconds;
- support cap 12; and
- frontier cap 400.

Its deterministic termination is `node-cap-reached`; the best replayed
candidate still has one nonzero residual. This is a replayable bounded search
record, not an exhaustive tree certificate. The eight seed orbits cover only
the one-perfect-matching-per-color seed family, not all ternary assignments.
The search is fixed-target: it requires each constant GHZ coefficient to be
exactly one, so it is narrower than projective or local-diagonal GHZ-orbit
search.

## Sharp deformation regression

At the `n=4,d=3` fixture, the exact `81 x 54` Jacobian has rank 51 and
nullity 3 over `Q`. Its kernel is precisely the reciprocal rescaling of the
three complementary edge pairs:

```text
(01)/(23), (02)/(13), (03)/(12).
```

The native code constructs an explicit split quotient of `Q^54` by these
three gauge directions. The induced 51-dimensional exact-Q Jacobian has rank
51 and nullity zero, so the quotient removes exactly the gauge kernel.

An arithmetic regression repeats the ranks over `F_31`. This finite-field
computation is not used as a proof over `C`.

## Claim boundary and next research work

- Exact integer or rational witnesses certify affine image membership over
  `Q` and therefore over `C`.
- An `F_31` result alone is not a proof over `C`.
- No independently checkable nonexistence certificate for `n=4,d=4` is
  emitted, so this application makes no new proof claim for that benchmark.
- The bounded `n=6,d=3` ternary run proves only what happened at its replayed
  nodes.
- Projective, local-diagonal-orbit, and border-image modes are named
  explicitly but do not yet have solvers or proof certificates.
- Forward local-projection equivariance is exact. Backward no-go propagation
  still requires an independently certified nonimage result.

The next high-value mathematical outputs are genuine polynomial invariants
vanishing on the image closure, independently replayable Nullstellensatz
nonimage certificates, exact-image versus border-image separation, certified
backward local-projection laws, and eventually a two-vertex contraction or
forbidden matching-minor theorem connecting different vertex counts.
