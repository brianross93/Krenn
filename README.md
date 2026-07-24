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
certificate. It does not yet prove membership in the strict boundary
\(\overline{\operatorname{im}\Phi}\setminus\operatorname{im}\Phi\): it is not
a finite witness and does not decide whether GHZ lies in the affine image
itself.

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

## Exact \(k=2\) source-ideal preflight

For \(D^2\), the codomain fine degree is \(2\delta=(2,\ldots,2)\).
The multiplier of a mixed \(F_c\) has degree

\[
2\delta-\deg F_c:
\quad
1\text{ at }(i,c_i),\qquad
2\text{ at the other two colors of vertex }i.
\]

An exact inclusion-exclusion recurrence counts loopless multigraphs on the
18 vertex-color tokens.  Every residual degree is at most two, so the
unrestricted components are paths, cycles, and double edges.  The resulting
fine-graded dimensions are

```text
multiplier monomials per mixed coloring     206,654,284,635
raw domain columns                     150,031,010,645,010
raw codomain rows                        74,680,326,909,360
raw nonzeros                           2,250,465,159,675,150
```

Burnside's lemma over the 33 conjugacy-type pairs of
\(S_6\times S_3\) gives

```text
compressed domain columns                 34,740,542,451
compressed codomain rows                   17,291,676,144
```

with the exact domain-orbit split

```text
5+1       1,723,078,477
4+2       4,307,241,637
4+1+1     4,307,033,332
3+3       2,871,521,344
3+2+1    17,224,970,227
2+2+2     4,306,697,434
```

The same recurrence independently reproduces every hard \(k=1\) count:
6,040 multiplier monomials, 11,608,920 codomain monomials, 1,314 domain
orbits with the published occupation split, and 3,102 codomain orbits.

The support of \(D^2\) has \(120^3=1,728,000\) monomials in 663 orbits.
Its coefficient and orbit censuses are

| coefficient | monomials | orbits |
|---:|---:|---:|
| 1 | 3,375 | 8 |
| 2 | 70,875 | 45 |
| 4 | 496,125 | 195 |
| 8 | 1,157,625 | 415 |

Full matrix construction is refused.  A conventional raw 64-bit CSC payload
would require about 37.2 PB.  Even before storing one compressed nonzero, its
64-bit column-pointer array alone needs 277,924,339,616 bytes (258.84 GiB),
above the reviewed 64 GiB bound.  At the 15-entry-per-column upper bound, the
compressed fixed-array payload would be about 8.62 TB; dense compressed
storage would be about 4.81 ZB.  These figures exclude allocator,
elimination, and temporary overhead.

### Small exact support-row search

The count preflight permits one controlled search on the 663 right-hand-side
orbits without constructing the full matrix.

- 632 orbits contain a rainbow perfect matching.  A `2+2+2` mixed generator
  has exactly one of its 15 terms in \(\operatorname{supp}(D^2)\), forcing
  the corresponding weight of any support-limited dual to vanish.
- The remaining 31 orbits, containing 34,560 raw monomials, yield 56
  distinct `4+2` matching-switch constraints from 291 predecessors.
- A retained \(31\times31\) integer minor has determinant \(-1\).

Thus the only invariant row functional supported on
\(\operatorname{supp}(D^2)\) that annihilates the source map is zero.
Reynolds averaging implies that no—not necessarily invariant—dual supported
only there can pair nontrivially with \(D^2\).  A separate standard-library
verifier reconstructs the matchings, all 663 orbits, all 632 singleton
columns, the retained full 15-term switch columns, and the determinant
without importing the producer.

This is a negative result about a certificate ansatz.  It does **not** decide
\(D^2\in J_{\rm mix}\), \(D\in\sqrt{J_{\rm mix}}\), affine GHZ membership, or
global GHZ nonexistence.  A global dual may use codomain monomials outside
\(\operatorname{supp}(D^2)\).

The matching evidence now has one common combinatorial description.  The
nine-edge natural support is a properly three-edge-colored triangular prism:
the defect is its rainbow rung matching and the pole monomial \(Q\) is the
complementary \(C_3\sqcup C_3\) two-factor.  The \(k=1\) separator uses a
\(C_6\) multiplier and one alternating four-cycle switch.  Each support-21
odd cycle is the corresponding three-switch parity circuit in a colored
\(K_{3,3}\).  Quadratic pairs of the 15 perfect matchings have no incidence
collisions; the first collisions occur for cubic triples and are exactly the
ten \(K_{3,3}\) bipartitions.  This explains both the shared mechanism and
why it does not by itself close the \(k=2\) radical question.

Two other propagation routes also fail cleanly.  For the first support-21
odd cycle (equations 16, 18, and 188), each full 15-term equation has exactly
two monomials on those 21 source coordinates.  The other 39 degree-three
monomials are pairwise distinct and spill outside that source support; this
is only a support-conditional mechanism audit.  For each rank stratum of a
bilinear two-vertex contraction, a replayed integer \(52\times52\) Jacobian
minor is nonzero.
This exceeds the 51-dimensional upper bound for the \(\Phi_{4,3}\) image,
so no nonzero such contraction universally preserves the matching image.
The bound \(51=54-3\) is replayed from the three independent reciprocal
rescalings of complementary \(K_4\) edge pairs on the dense source torus;
the matching-incidence law also replays the local-\(\mathrm{GL}_3\)
equivariance used to reduce contractions to their three rank normal forms.
The larger ranks 71, 81, and 81 at \(p=31,1009,1000003\) are retained only
as diagnostics.  The contracted GHZ targets themselves are diagonal
\(n=4\) tensors with explicit, symbolically replayed \(\Phi_{4,3}\)
witnesses.  Finally, the all-zero source valuation makes all 15 terms of
every mixed equation tie, so an additive valuation law alone cannot control
the required leading-coefficient cancellations.

A one-shell reconnaissance used every averaged column signature with a term
in \(\operatorname{supp}(D^2)\).  It has 4,493 columns, 39,033 output-row
orbits, and 60,850 nonzeros.  Ranks at
\(p=31,1009,1000003\) are \(4493/4494\) before/after adjoining \(D^2\).
Its deterministic column-orbit and complete sparse-layout fingerprints are
`f5ca49ab68e93c70640a16b857a2dccce7b995df357277f17a373e0e51a7692a`
and
`627ad8463b617c962b44c56af9e470dc8c2b685aaf81b63cf2ef0f5ca19bb156`.
These modular ranks are nonproof diagnostics.  An exact two-row shell
separator pairs to 1,080 with \(D^2\), but two explicit support-disjoint
columns each pair to \(-1\); symbolic replay therefore rejects it as a
global dual certificate.  A bounded extension search also did not close,
which is only a search boundary.  The opt-in deterministic rebuild is
`python -m experiments.krenn_quantum_graph.one_shell_full_recompute --run-full`;
its large cache, report, and log default to
`D:\KrennScratch\obstruction_certificate`.

The round-trippable exact certificate, independent verifier receipts, and
strict claim manifest are under
`results/krenn_quantum_graph/n6_d3_radical_obstruction/`.  The conclusion
remains undecided between affine membership and strict border membership:
the known border membership is exact, while neither \(D^2\in J_{\rm mix}\)
nor a global \(D^2\) separator has been proved.

```text
python -m experiments.krenn_quantum_graph.higher_power_source_ideal
python -m experiments.krenn_quantum_graph.mechanism_audit
python -m experiments.krenn_quantum_graph.radical_obstruction
python -m experiments.krenn_quantum_graph.radical_obstruction_artifact \
  results/krenn_quantum_graph/n6_d3_radical_obstruction --verify
python -m unittest tests.test_krenn_higher_power_source_ideal -v
python -m unittest tests.test_krenn_mechanism_audit -v
python -m unittest tests.test_krenn_one_shell_reconnaissance -v
python -m unittest tests.test_krenn_radical_obstruction -v
python -m unittest tests.test_krenn_radical_obstruction_artifact -v
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
- Projective and local-diagonal-orbit modes are named explicitly but do not
  yet have solvers or proof certificates. Border mode now has the specific
  exact `n=6,d=3` Laurent certificate above, but no general solver.
- Forward local-projection equivariance is exact. Backward no-go propagation
  still requires an independently certified nonimage result.

The next high-value mathematical outputs are a decision on exact affine
membership inside the now-certified border case, independently replayable
Nullstellensatz nonimage certificates if exact membership fails, certified
backward local-projection laws, and eventually a two-vertex contraction or
forbidden matching-minor theorem connecting different vertex counts.
