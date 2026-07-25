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

Run the complete standalone suite from the repository root:

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

Generate or verify the exact near-miss forensic bundle:

```bash
python -m experiments.krenn_quantum_graph.defect_mining
python -m experiments.krenn_quantum_graph.defect_mining --verify
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

## What the `728/729` defect actually proves

The lone failed equation of the natural seed is equation 70:

```text
coloring 002121, occupation (2,2,2), target coefficient 0.
```

Its residual is the full 15-term hafnian coefficient. On the natural
support, exactly one term survives:

\[
W_{01}^{00}W_{24}^{22}W_{35}^{11}.
\]

The eight one-perfect-matching-per-color support orbits have respectively

```text
24, 12, 6, 6, 5, 2, 1, 3
```

mixed defects. The same literal equation does not fail on every orbit.
Instead there is a support dichotomy:

- if two selected color matchings share an edge, they force a `4+2` mixed
  coloring;
- if all three are edge-disjoint, their cubic union has a rainbow perfect
  matching and forces a `2+2+2` coloring.

Every forced coefficient has exactly one active matching monomial. Therefore
no choice of nonzero signs, phases, or rescalings on those same nine slots can
cancel it. This proves a finite no-go theorem for the complete
one-diagonal-perfect-matching-per-color support family, covering all 3,375
ordered seeds.

It is not a no-go theorem for the full 135-variable map. The fixed-support
inconsistency has the small exact receipt

\[
1=DQ-\bigl((P_0-1)P_1P_2+(P_1-1)P_2+(P_2-1)\bigr),
\]

where `P_a` is the selected constant-color matching product, `D` is a forced
mixed monomial, and `P_0 P_1 P_2 = DQ`.

The forensic bundle also records:

- all 15 terms of the exact victim polynomial;
- every residual on all eight orbit representatives;
- the identity
  \[
  \sum_c\Phi(W)_c\prod_i g_i(c_i)
  =\operatorname{haf}\bigl(g_i^T W_{ij}g_j\bigr);
  \]
- the all-ones contraction `haf(s_ij)=4` versus GHZ target sum `3`;
- negative audits for linear matching dependence, the `S_6` sign character,
  invariant linear output directions, and an `n=4` contraction
  obstruction; and
- deterministic Singular inputs over `F_31` for the equal-`g` `d=3` and
  `d=4` target fibers.

The equal-`g` systems have 90 variables and 28 equations for `d=3`, and 150
variables and 84 equations for `d=4`. Concrete integer Jacobian minors reduce
to nonzero determinants modulo 31 and give full ranks 28 and 84. Hence both
equal-`g` maps are dominant in characteristic zero: their image closures fill
the occupation-coefficient spaces. No nonzero universal polynomial invariant
can live solely in this symmetric shadow.

For `d=3`, the conclusion is stronger. Every perfect matching contains
exactly one edge incident to vertex `0`. After fixing the 60 non-star shadow
variables, all 28 equations are jointly linear in the 30 star variables. A
deterministic `{-1,+1}` choice for the non-star variables gives a `28 x 28`
minor with exact determinant

\[
2^{54}.
\]

Thus this fixed rational slice surjects onto every rational occupation
target. In particular, an exact rational GHZ shadow solution exists, with
denominators at most 32. A second sparse certificate over
\(\mathbf Q(\omega)\), where \(\omega^2+\omega+1=0\), realizes

\[
\frac{A^3+B^3+C^3+6ABC}{9}=X^3+Y^3+Z^3
\]

on nine prism edges. These are shadow solutions, not full tensor solutions:
the sparse lift still has nonzero `002121` amplitude
\(\frac23\omega^2\). The antisymmetric and distinct-`g_i` equations
therefore remain the relevant exact-image arena. The special `d=4` shadow
fiber is still undecided.

## Exact border-image certificate at `n=6,d=3`

The natural near-miss also contains a full-tensor Laurent degeneration. For
nonzero `t`, replace

```text
W[0,1,0,0] = t
W[2,3,0,0] = t^-1
```

and leave the other seven natural seed weights equal to one. Exact symbolic
evaluation of all 10,935 matching monomials gives

\[
\Phi(W(t))=\operatorname{GHZ}_{6,3}
            +t\,e_{002121}.
\]

Although the input has a pole at `t=0`, its output extends polynomially and
specializes there to GHZ. Taking real \(t\to0\) through nonzero values proves
membership in the ordinary Euclidean closure over \(\mathbf C\), and hence in
the Zariski closure of the image. This is an exact border-image membership
certificate. On its own, a Laurent degeneration does not decide whether GHZ
also lies in the affine image. The external nonimage theorem recorded below
supplies that missing half.

## External `n=6,d=3` affine-nonimage theorem

The external repository
[`algal/krenn-gu-6x3-certificate`](https://github.com/algal/krenn-gu-6x3-certificate)
proves at proof-content commit
`105ffbc50b0443433fc53c248272617cc022f3e2`:

```text
not exists W : MonochromaticQuantumGraph.WeightsN 6 3 C,
  MonochromaticQuantumGraph.EqSystemN 6 3 W
```

Its published verification record reports a clean 8,421-job Lean build, all
50 artifact hashes passing, no `sorryAx`, and the axiom closure
`[propext, Classical.choice, Lean.ofReduceBool, Lean.trustCompiler,
Quot.sound]`. The use of `native_decide` makes this a compiler-trusting Lean
proof rather than a kernel-only reduction proof. The external repository has
no license file, so no source or proof artifact is vendored here.

The pinned provenance and exact trust boundary are recorded in
`results/krenn_quantum_graph/n6_d3_external_nonimage_reference/reference.json`.
Our local static audit checked the theorem statement, the official
fifteen-matching bridge, and all 50 committed hashes. A redundant local full
rebuild was stopped and is not claimed.

Accepting the external theorem and combining it with the independently
replayed Laurent identity above gives the strict statement

\[
\operatorname{GHZ}_{6,3}\in
\overline{\operatorname{im}\Phi}\setminus\operatorname{im}\Phi.
\]

Color restriction also rules out canonical six-vertex solutions for every
`d >= 3`: any three colors would give the forbidden `d=3` system. This
settles six vertices, not the global conjecture.

The exact differential profile agrees with this degeneration. At the natural
seed, the full `729 x 135` Jacobian has rank 130 and nullity 5 over `Q`; its
kernel is exactly the five-dimensional vertex-scalar gauge

\[
W_{ij}^{ab}\longmapsto\lambda_i\lambda_jW_{ij}^{ab},
\qquad \prod_i\lambda_i=1.
\]

After deleting equation 70, the Jacobian has rank 129 and nullity 6. The
sixth direction is

```text
delta(W[0,1,0,0]) = -1
delta(W[2,3,0,0]) = +1
```

and satisfies \(J\delta=-e_{70}\). The ranks over `Q` are proved by nonzero
reductions of concrete integer minors modulo 31 together with the exact
kernel directions giving matching upper bounds; no finite-field solution is
promoted to characteristic zero.

The pole is not removable by a parameter-dependent vertex gauge. The
six-weight monomial

\[
Q=W_{02}^{11}W_{23}^{00}W_{03}^{22}
  W_{14}^{11}W_{45}^{00}W_{15}^{22}
\]

has vertex degree two everywhere and is therefore gauge invariant. Along the
border curve, \(Q=t^{-1}\). Equivalently, a nonnegative integer Farkas vector
selecting the same six coordinates turns the nine regularity inequalities
into the contradiction \(-1\geq0\). This remains valid for rational Puiseux
gauges and positive ramified reparameterizations.

Nor does a finite-order deformation obstruction appear. With \(t=1-s\), the
repair tangent lifts uniquely in a rational local gauge slice to

\[
W_{01}^{00}=1-s,\qquad
W_{23}^{00}=1+s+s^2+\cdots .
\]

Here “uniquely” has a precise local meaning. Fixing five declared input
coordinates cuts out a 130-dimensional linear slice through the natural
seed. The restriction of the five gauge directions to those coordinates has
determinant \(-2\). Since the full Jacobian kernel is exactly the
five-dimensional gauge space, the restricted Jacobian has rank 130 and
nullity zero over \(\mathbf Q\). At each power of \(s\), the next coefficient
therefore has at most one value in this fixed slice, and the displayed
geometric-series coefficient supplies it. This does not assert uniqueness of
all global branches or uniqueness in other gauge slices.

For every \(N\geq1\), the exact polynomial truncation satisfies all 729
coefficient identities

\[
\Phi(W^{(N)}(s))
=\operatorname{GHZ}_{6,3}+(1-s)e_{70}-s^{N+1}e_0.
\]

Thus the moving target lifts over \(\mathbf Q[[s]]\) to every order, but the
geometric series has a genuine gauge-invariant pole at \(s=1\). Local
Jacobian or finite-order cokernel calculations alone cannot decide affine
membership.

There is also a finite-support obstruction beyond the original nine-slot
ansatz. Retain all nine natural nonzero coordinates, but permit arbitrary
additional coordinates and arbitrary nonzero complex values on the resulting
support. Repairing equation 70 requires at least two new coordinates. Each of
the six minimum repairs creates two new singleton mixed equations.

A deterministic missing-set closure then follows the first singleton
equation and branches over every alternative perfect matching that could
cancel it. The raw, unquotiented traversal visits 1,632,189 supports through
total support size 21. No support of size at most 20 is singleton-free. At
size 21 exactly six minimal supports survive the singleton test. Each has 18
active mixed equations, all binomials, and each contains an odd
three-binomial exponent cycle. Multiplying the three resulting monomial
ratios gives

\[
1=(-1)^3=-1,
\]

so none is solvable over \(\mathbf C\) (or any field of characteristic not
two). Therefore any finite exact witness that retains all nine natural slots
must have at least 22 nonzero coordinates. This remains a support-conditional
lower bound: it does not exclude witnesses that omit a natural slot, and it
does not decide whether a support of size 22 or more can realize GHZ.

## Exact \(k=1\) source-ideal obstruction

For a coloring \(c=(c_0,\ldots,c_5)\in\{0,1,2\}^6\), write

\[
F_c(W)=\sum_{M\in\operatorname{PM}(K_6)}
       \prod_{\{i,j\}\in M}W_{ij}^{c_i c_j},
\qquad
D=F_{000000}F_{111111}F_{222222},
\]

and let \(J_{\mathrm{mix}}=\langle F_c:c\text{ is not all-equal}\rangle\).
The fine grading

\[
\deg W_{ij}^{ab}=e_{i,a}+e_{j,b}
\]

makes \(D\) homogeneous with one copy of every vertex-color degree. Thus any
identity \(D=\sum_c A_cF_c\) can be reduced without loss to degree-six
multipliers \(A_c\): at vertex \(i\), an admissible multiplier uses exactly
the two colors complementary to \(c_i\). There are 6,040 such monomials for
each of the 726 mixed colorings. The raw linear system has 4,385,040 columns
and 11,608,920 rows; exact \(S_6\times S_3\) Reynolds averaging reduces it to
1,314 domain orbits and 3,102 codomain orbits. The polynomial \(D\) has 3,375
monomials in eight codomain orbits.

In the orbit-total integer system \(Bx=b\), rows 550 and 568 correspond to

\[
\begin{aligned}
m_{550}={}&
W_{01}^{00}W_{01}^{11}W_{02}^{22}W_{13}^{22}
W_{24}^{00}W_{24}^{11}W_{35}^{00}W_{35}^{11}W_{45}^{22},\\
m_{568}={}&
W_{01}^{00}W_{01}^{11}W_{02}^{22}W_{13}^{22}
W_{24}^{00}W_{25}^{11}W_{34}^{11}W_{35}^{00}W_{45}^{22}.
\end{aligned}
\]

Their \(B\)-row profiles are identical: each has only column 142, with
coefficient 2,160. Their \(D\)-orbit totals are respectively 360 and 1,080.
Consequently the integer dual \(\lambda=e_{550}-e_{568}\) satisfies

\[
\lambda^{\mathsf T}B=0,
\qquad
\lambda^{\mathsf T}b=360-1080=-720.
\]

This is an exact certificate that \(D\notin J_{\mathrm{mix}}\) over
\(\mathbf Q\), and over every characteristic-zero field. Modular ranks
\(\operatorname{rank}(B)/\operatorname{rank}([B\mid b])=1193/1194\) at
\(p=31,1009,1000003\) are diagnostics only. The result does **not** prove
\(D\notin\sqrt{J_{\mathrm{mix}}}\), GHZ nonexistence, or nonmembership of the
GHZ tensor in the exact affine image.

```text
python -m experiments.krenn_quantum_graph.source_ideal
python -m unittest tests.test_krenn_source_ideal -v
```

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
- The pinned external Lean theorem decides affine nonmembership at
  `n=6,d=3`. Together with the local Laurent certificate it proves strict
  border membership for that case.
- Color restriction extends the six-vertex nonexistence conclusion to every
  canonical `d >= 3` target. It does not relate different vertex counts.
- Projective and local-diagonal-orbit modes are named explicitly but do not
  yet have solvers or proof certificates. Border mode now has the specific
  exact `n=6,d=3` Laurent certificate above, but no general solver.
- Forward local-projection equivariance is exact. Backward no-go propagation
  still requires an independently certified nonimage result.

The next unsettled even vertex count is eight. The structural program in
`results/krenn_quantum_graph/n_ge_8_structural_program` combines exact
simultaneous star/cofactor identities with the minimal-prime decomposition of
the perfect-matching monomial ideal. There are six blocker orbit types at
`n=8`, representing 1,408 labelled blockers. Two qutrit blocker systems are
square, with independently replayed top Chow coefficients 30 and 24. The
missing theorem is boundary escape: the EqSystem constraints must prevent
all projective blocker solutions from being confined to coordinate
boundaries. Positive intersection counts alone do not prove that statement.

## Historical bounded finite-counterexample campaign

This campaign predates and is superseded as decision work by the external
six-vertex nonimage theorem. It remains as a reproducible record of the
numerical and exact-search machinery; rerunning it cannot change the decided
`n=6,d=3` conclusion.

The native `n=6,d=3` campaign combines singleton closure, exact
`S_6 x S_3` support representatives, natural-coordinate omission strata,
support-size quotas, target-aware rank-adaptive gauge charts, hard
`L2`/`Linf` controls, complex deterministic multistart continuation, and a
dense all-135-coordinate comparison. The charts remove up to 14
color-diagonal directions common to `GHZ+t*e_002121`, and up to all 15
directions for direct GHZ searches. Large, generation-stamped optimizer
checkpoints remain outside the repository under
`D:\KrennScratch\counterexample_search`; interrupted jobs resume only after
their problem hashes replay. The committed result bundle stores selected
weights and all 729 floating residuals and verifies without the scratch
directory.

The released bounded run used master seed `60320260724`, eight workers,
100,000 orbit-closure nodes, a 2,000,000-state memory cap, support sizes
`22,22,23,23,24,24,79,81,82,83,84,84`, radii `8,16,32`, four starts per
sparse support, eight dense starts per radius, and 100 LM iterations per
solve. It completed 168 jobs. No numerical zero crossed `1e-8`, so exact
reconstruction was not triggered and no exact candidate was certified.

For radii `8,16,32`, respectively, the best residuals were
`0.567391,0.518492,0.505032` on natural-retaining sparse supports,
`0.615094,0.555793,0.520867` on natural-omitting sparse supports, and
`0.240291,0.524698,0.420280` in the dense comparison. All 27 retained
candidates had finite floating weights, but every one was active on its hard
norm boundary. Their weights therefore did not converge to a finite interior
point. The known `Q` monomial is invariant for the 14-dimensional
moving-target stabilizer but not for the extra direct-GHZ gauge direction;
the bundle records that qualification explicitly. This is a bounded search
miss, not a nonexistence proof.

Run the released campaign:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -B -m experiments.krenn_quantum_graph.counterexample_campaign `
  --scratch-directory D:\KrennScratch\counterexample_search\final_gauge_v2 `
  --results-directory results\krenn_quantum_graph\n6_d3_counterexample_search `
  --workers 8 --orbit-nodes 100000 `
  --support-state-cap 2000000 `
  --retained-supports 6 --omission-supports 6 `
  --starts-per-support 4 --dense-starts 8 `
  --radii 8 16 32 --iterations 100 --evaluations 900 `
  --checkpoint-interval 20 `
  --continuation-schedule 1 0.5 0.25 0.125 0 `
  --best-per-radius-class 3 --reconstruction-trigger 1e-8
```

Verify only the compact committed bundle:

```powershell
python -B -m experiments.krenn_quantum_graph.counterexample_campaign `
  --verify-only `
  --results-directory results\krenn_quantum_graph\n6_d3_counterexample_search
```

The exact reconstruction layer supports rational, Gaussian-rational, small
cyclotomic, and bounded degree-at-most-four number fields. Exact affine
membership is asserted only after the primary sparse system and an
independently enumerated perfect-matching evaluator both vanish in all 729
equations. A reconstruction miss, bounded numerical miss, or finite-field
point is never promoted to a proof.
