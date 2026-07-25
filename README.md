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
must have at least 22 nonzero coordinates. The later eight-root
\(S_6\times S_3\) closure removes the natural-slot assumption: every finite
exact \(n=6,d=3\) GHZ witness has support at least 22, including witnesses
that omit natural coordinates. Supports of size 22 and above remain open.

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

## Exact localized affine chart cover

Direct affine GHZ membership is equivalent to vanishing of all 726 mixed
outputs together with nonvanishing of the three pure outputs. Each nonzero
pure hafnian contains a nonzero perfect-matching monomial, giving
\(15^3=3375\) ordered seed charts. Their exact \(S_6\times S_3\) quotient has
eight representatives.

On each seed chart the full endpoint-color torus sets all nine selected
diagonal weights to one. This is a split rank-nine slice, verified by an
explicit identity minor; it does not equate symmetry-related weights. Adding
three inverse-amplitude variables gives an ordinary affine ideal

\[
K_r=\langle F_c|_{S_r=1}:c\ {\rm mixed}\rangle+
    \langle u_aF_{a^6}|_{S_r=1}-1:a=0,1,2\rangle .
\]

Every \(K_r\) has 129 variables, 729 generators, 10,938 sparse terms, and
maximum degree four. All eight systems are reconstructed bit-for-bit by a
second perfect-matching enumerator.

The residual rank-nine torus grading and each ordered-seed stabilizer give a
lossless bounded Nullstellensatz reduction. Through total certificate degree
six, exact averaged Koszul relations provide rational source-rank upper
bounds, and nonzero minors modulo 1009 attain those bounds while the
adjoined constant raises rank. Therefore none of the eight ideals has a
Nullstellensatz identity of total degree at most six over \(\mathbf Q\).
This is an exact bounded statement, not a unit-ideal or nonexistence proof.

The natural chart has one universal defect

\[
1+6\text{ quadratic repair monomials}
 +8\text{ cubic repair monomials}=0.
\]

A retained rational identity proves
\(1\in\langle f,\partial f\rangle\), so this defect hypersurface is smooth.
Its 12 derivative opens reduce to two exact symmetry representatives. A
second finite atlas uses the fact that one of the 14 repair monomials must be
nonzero; residual gauge and symmetry reduce those opens to four
representatives. Seven generic \(F_{31}\) Gröbner probes—the natural chart,
both derivative representatives, and all four repair representatives—each
parsed and then timed out after a full 600-second engine budget. Their
byte-exact logs and exit-124 receipts are retained as reconnaissance only.
The next campaign therefore uses the exact residual \(\mathbf Z^9\) grading
to decompose certificate and saturation computations instead of extending
the generic timeouts. After the repair-factor gauges, the four repair
charts have residual character ranks \(8,8,7,7\); exact replay verifies
that all 730 generators in every chart are homogeneous in the corresponding
split integral quotient. This grading supplies gauge lineality for later
Gröbner or tropical calculations, but it does not itself enumerate tropical
cones or decide an ideal. The two derivative opens also admit exact sparse
gauge slices \(d=1\): each keeps 129 variables, has 730 generators and
10,942 terms of degree at most four, replacing the older degree-six
elimination formulation.

Quotienting by the primitive derivative character gives a split
\(\mathbf Z^8\) grading on each sparse slice. There are 105 variable
character blocks and respectively 558 and 642 generator blocks. The
selected derivative stabilizer has order two; averaging certificates under
it never equates symmetry-related weights. Native character-zero Macaulay
systems exactly exclude rational Nullstellensatz identities through total
degree five on both slices. At degree six, the \(F_{1009}\) source and
augmented ranks still differ by one, but the exact relation bounds leave
rank gaps 123 and 117. Those degree-six rows are therefore reconnaissance,
not characteristic-zero proofs.

Two triangular two-pivot \(F_{1009}\) `std` probes and two leaf-free
\(F_{31}\) initial-\(A_0\) saturation probes each parsed and then consumed
their full 600-second engine budgets without completing. Removing inverse
leaves nevertheless gives exact smaller formulations: the derivative
systems have 126 variables, 727 generators, 10,894 terms, and degree at
most three; the four repair systems have 124 or 125 variables, 726
generators, 10,890 terms, and degree at most three. No saturation stage has
completed, and exact affine membership remains undecided.

These bounded-degree misses and engine timeouts carry no evidentiary weight
toward either emptiness or nonemptiness of the affine fiber. In particular,
degree seven is not a planned continuation: it would enlarge the same
unresolved Macaulay formulation without approaching an effective
Nullstellensatz bound.

Nor is the raw count of 729 equations versus 135 weights evidence for
emptiness. Exact border membership already puts GHZ in the closure of the
tensor-map image, so naive overdetermination does not distinguish finite
image membership from strict boundary membership.

The next reduction instead uses an exact star factorization. For any apex
vertex, all 729 equations are linear in its 45 incident weights. They split
into three systems with one shared \(243\times15\) quadratic coefficient
matrix and targets \(e_{00000},e_{11111},e_{22222}\). Primary and
independent perfect-matching enumerators replay this identity for all six
apices. One apex factorization is already equivalent to the full equation
system; the other five are simultaneous cross-checks, not independent
equations.

The tempting one-column differencing shortcut is false. Changing one
non-apex color changes the residual \(K_4\) tensor in the other four apex
partner columns as well: a full row difference has 30 signed monomials,
with 6 in the selected column and 24 omitted cross terms. Consequently the
claimed constant-or-singleton-support dichotomy and its finite case tree do
not follow.

There is nevertheless a smaller exact consequence. Restrict the shared
star matrix to the three target rows \(00000,11111,22222\). Any GHZ witness
gives this \(3\times15\) matrix a right inverse, hence rank three. Of its
455 three-column minors, 330 vanish structurally and the remaining
\(125=5^3\) factor as one residual \(K_4\) value in each color. These 125
nonzero opens form only three \(S_5\times S_3\) orbit types: the three
chosen partner vertices are all equal, exactly two are equal, or all are
distinct. Thus every finite witness lies in one of three exact pivot-chart
types for any fixed apex. This finite cover, not the failed differencing
case tree, is the next symbolic route.

Each selected residual factor is a semi-invariant for the 15-dimensional
direct-GHZ color gauge. On a pivot open, one product-one gauge parameter
per color sends the three nonzero factors to \(1\), explicitly and without
extracting roots. If \(B_{a',(v,b)}=\delta_{a'b}P_v(bbbb)\) and
\(Y_{(v,b),a}=w_{1v}^{ab}\), the nine monochromatic residual equations are
the single normal form \(BY=I_3\). After pivot normalization and column
reordering, \(B=[I_3\mid C]\), hence
\[
Y=\begin{pmatrix}I_3-CZ\\ Z\end{pmatrix},
\qquad Z\in\mathbb C^{12\times3}.
\]
This explains both the nine monic eliminations and the remaining
\(3\cdot12=36\) free star parameters. Consequently the 125 localized opens are
existence-equivalent to only three gauge-normalized polynomial slices—one
per orbit type—with no Rabinowitsch variable, colon ideal, or saturation.

The color-gauge action on the 90 nonstar weights has the generic stabilizer
\(\mu_2\): setting every endpoint-color multiplier to \(-1\) fixes every
edge weight. Its effective character lattice is therefore the index-two
even-sum lattice. A complete 15-character anchor matrix is a root-free basis
of the effective torus exactly when its determinant has absolute value two,
not merely when it has rank 15. For each of the three pivot representatives,
the exact audit supplies a 12-character nonstar complement with determinant
\(+2\) and Smith factors \((1,\ldots,1,2)\). Those 12 coordinates are a
lattice diagnostic only. Requiring them to be nonzero would shrink the
exhaustive pivot open, so they are not used in the retained search.

Keeping the pivots gives 135 variables, 732 generators, maximum degree
three, and 10,950 collected terms. Eliminating the nine monic weights gives
126 variables and 723 generators, but raises the maximum degree to five
and the term count to 35,292, so the retained cubic form is the preferred
first exact input. Both presentations now have deterministic sparse
exports and independent matching-enumerator reconstruction; no CAS result
has yet been promoted from them. The first retained-form \(F_{31}\)
`slimgb` probes on all three orbit representatives parsed successfully and
then timed out after 600 seconds each. These are performance receipts, not
chart decisions.

The corresponding numerical reformulation eliminates the 45 star weights
at every iterate. For \(U\in\mathbb C^{90}\), let
\(\Phi(U)\in\mathbb C^{243\times15}\) be the shared quadratic star matrix
and let \(E=[e_{00000},e_{11111},e_{22222}]\). Exact affine membership is
equivalent to the unrestricted equation \(\Phi(U)Y=E\). At radius \(r\),
the numerical campaign instead optimizes
\[
\min_{\|Y\|_F\le r}\|\Phi(U)Y-E\|_F^2
\]
and records the unrestricted span distance separately as telemetry. The
Euclidean distance itself is not color-gauge invariant. The search therefore
normalizes only the three guaranteed nonzero pivot factors, projects every
gradient into the tangent of \(P=1\) and orthogonally off the remaining 12
gauge directions, and retracts root-free after each trial step. It uses no
12-coordinate anchor slice and no intended mathematical cap on raw weights.
It does hard-reject raw \(U\) beyond enormous, radius-dependent floating
point overflow guards; these are enforced numerical domain cutoffs, not
gauge-invariant bounds.

Horizontal projection constrains each instantaneous velocity; it does not
put independently initialized runs into one common residual-gauge slice and
does not exclude accumulated second-order gauge drift. Consequently nonzero
objective values from different seeds or pivot representatives are not
intrinsically comparable. Their ordering below identifies only the smallest
value recorded in these particular deterministic lifts.

There is also no hidden norm-balancing shortcut. For every pivot color,
the residual cocharacter with exponent \(+1\) at the apex and \(-1\) at
the selected partner preserves \(P=1\) while weakly shrinking every
affected nonstar coordinate. Thus residual orbit-norm minimization has an
unattained recession direction generically. The known pole monomial \(Q\)
is invariant under vertex-scalar gauge but not under the full direct-GHZ
color gauge; its character remains independent of the three pivot
characters in all three orbit types. Since the known Laurent family is
itself a color-gauge one-parameter orbit, no invariant of that full gauge
can diverge along it. The retained campaign therefore logs \(Q\) only as a
chart-dependent path diagnostic. Laurent cross-ratio caps are reserved for
an optional dense-torus subcampaign because imposing them would exclude
zero-coordinate witnesses.

The deterministic variable-projection census used six seeds
2026072501--2026072506, cold starts on all three pivot orbit types, and
natural-seed repair starts on the all-distinct type: 24 trajectories in
total, with six workers, a 600-second per-trajectory deadline, 500 accepted
steps, and a radius-two bound on the recovered \(Y\) in the \(P=1\) slice.
The corresponding enforced numerical \(U\) guards were approximately
\(\|U\|_\infty\le6.30\cdot10^{39}\) and
\(\|U\|_2\le6.30\cdot10^{40}\). Twenty-two trajectories reached the
iteration budget and two exhausted their line searches; none formed a
detected two-cycle or reached a numerical zero. The smallest recorded
lift-dependent residual occurred in the exactly-two-equal chart at seed
2026072503: \(0.6467213054\), with \(\|U\|_2=97.22\),
\(\|Y\|_F=0.324\), and inactive \(Y\) control.

That telemetry-selected trajectory alone was continued for 5,000 accepted
steps. This selection is not a gauge-invariant claim that its basin was
globally best. It
finished in 230 seconds at residual \(0.6178675811198806\), with target
residuals \(0.2324310,0.2100928,0.5325384\),
\(\|U\|_2=97.1536\), \(\max|U|=85.4038\),
\(\|Y\|_F=0.832164\), and star-matrix condition number about 1215.
The last 100 steps improved at only
\(-3.38\cdot10^{-10}\) base-10 log residual per step, so the run was
stopped as a numerical plateau rather than extended arbitrarily. Both
perfect-matching enumerators replay all 729 outputs to \(1.12\cdot10^{-16}\)
agreement. The maximum GHZ equation residual is still about \(0.45\);
there is no numerical zero, no exact reconstruction, and no exact
counterexample.

The retained commands were:

```text
python -B -m experiments.krenn_quantum_graph.star_variable_projection_campaign --scratch-root D:\KrennScratch\counterexample_search\variable_projection_multistart_20260725_v1 --radii 2 --seeds 2026072501,2026072502,2026072503,2026072504,2026072505,2026072506 --initializations cold,natural-repair --workers 6 --maximum-iterations 500 --maximum-seconds-per-trajectory 600 --patience 500 --checkpoint-interval 25 --natural-repair-scale 0.01
python -B -m experiments.krenn_quantum_graph.numerical_continuation --parent-result D:\KrennScratch\counterexample_search\variable_projection_multistart_20260725_v1\orbit_1_seed_2026072503_radius_2_cold.result.json --output-directory D:\KrennScratch\counterexample_search\variable_projection_selected_continuation_20260725_v1 --label orbit1_seed2026072503_step5000 --maximum-iterations 5000 --maximum-seconds 600 --patience 5000 --checkpoint-interval 50
```

The portable 24-trajectory summary is committed under
`results/krenn_quantum_graph/n6_d3_counterexample_search/star_variable_projection`.
It retains all compact trajectory rows and embeds three orbitwise recorded
minima for scratch-independent replay with the current core. The archived
runtime campaign source hash is preserved separately and intentionally does
not equal the post-audit campaign source; the mathematical core source hash
does match. Verify the two-file bundle with:

```text
python -B -m experiments.krenn_quantum_graph.star_variable_projection_artifact results/krenn_quantum_graph/n6_d3_counterexample_search/star_variable_projection --verify
```

A finite support-stratified residual-gauge atlas exists in principle:
on each support, choose a lexicographically first basis of the supported
residual characters, record its Smith form, normalize those nonzero
coordinates, and quotient the remaining stabilizer. This handles zero
coordinates without pretending that one determinant-two complement covers
them. It still does not produce a coverage-complete fixed norm bound.
Same-character ratios remain noncompact, and the exact recession
cocharacters carry anchor charts into lower-support boundary strata.
Accordingly the horizontal \(P=1\) campaign is a candidate finder, not the
requested global bounded-135-variable comparison. A conclusive version of
the proposed “compare the infinities” idea should instead compactify the
remaining coordinates projectively or torically and separate the finite
chart from its boundary exactly (for example by a homogenizing coordinate
and saturation/initial-ideal checks).

The first bounded higher-structure audit now makes that comparison exact on
the natural series. On its nine moving coordinates the tropical valuation
kernel is six-dimensional and equals the direct-GHZ color-gauge lineality:
the known Laurent pole has no transverse direction after quotienting by
gauge. Of 104 fixed-victim nine-coordinate seed orbits, 103 have a singleton
equation and the four-node natural orbit is the only survivor. The residual
\(\mathbf Z^9\) grading assigns character zero to every term of the defect
equation, so it cannot separate the first repair choices.

The six pairwise unions of natural nodes form one victim-stabilizer orbit.
The canonical size-13 union has four singleton equations. All
\(14^4=38{,}416\) simultaneous first repairs were checked exactly; the
unique four-coordinate completion reaches support 17 but creates seven new
singletons. The complete canonical-pair singleton closure through support
21 contains 7,564 supports and one singleton-free terminal. Its 14 mixed
equations are binomial, and two exact odd exponent identities force the
corresponding pure-color sums to zero instead of one. Hence every pairwise
natural-node bridge through support 21 is excluded. This says nothing about
support 22+, other higher-support cones, or disconnected components with
zero defect parameter.

An exact projective preflight gives the next conclusive route without
claiming it will finish quickly. The four-factor Cox form has 139
coordinates, 732 generators, 10,950 terms, and maximum degree three; its 15
nonfinite boundary subsets reduce to 7, 11, and 7 symmetry orbits for the
three pivot types. The smaller first probe keeps \(U\) affine and
projectivizes the three star blocks. It has 138 coordinates with the same
generator and term counts. For the all-distinct pivot its first exact gate is
\[
  \operatorname{Sat}_{\langle Y^0_1,\ldots,Y^0_{15}\rangle}
  (J+(h_0)):(h_1h_2)^\infty,
\]
or equivalently the union of the 15 charts \(Y^0_j\ne0\). The
Cox-irrelevant saturation is essential: without it the spurious affine-cone
point \(Y^0=0\) survives at \(h_0=0\). Only the saturated gate represents
\(\operatorname{rank}A(U)\le14\) while the other two targets remain in
\(\operatorname{im}A(U)\). No large saturation has been launched: previous
600-second affine probes timed out, so there is currently no evidence that
this computation is only hours from a decision. A short characteristic-zero
growth preflight must pass before any longer CAS run.

A completed 24-trajectory legacy ALS baseline used one narrower
15-coordinate anchor chart, two initializations, seeds
2026072501--2026072503, four explicit \((L_\infty,L_2)\) cap pairs, and a
600-second budget per trajectory. All 24 runs ended by time budget; none
reached numerical tolerance. The best residual was
1.0309050035801723. All initial, final, and best vectors were replayed in
all 729 equations by both enumerators. Because the anchors and raw caps are
gauge-variant and define only one subopen, this baseline is retained solely
as basin and implementation reconnaissance.

Likewise, for an ideal \(I\) and polynomial \(q\), the two statements
\(q^m\in I\) and \(1\in I+(qz-1)\) are equivalent descriptions of the same
open-branch exclusion \(V(I)\cap D(q)=\varnothing\); together they do not
prove \(1\in I\). A valid divisor split must also decide the closed branch,
for example by proving \(1\in I+(q)\). Future exact work therefore uses
star-matrix pivot minors to branch into \(q\ne0\) and \(q=0\) rank strata
instead of routing through the known Laurent pole.

```text
python -B -m experiments.krenn_quantum_graph.localized_chart_artifact verify
python -B -m unittest tests.test_krenn_localized_chart_ideals -v
python -B -m unittest tests.test_krenn_localized_chart_macaulay -v
python -B -m unittest tests.test_krenn_localized_chart_graded_macaulay -v
python -B -m unittest tests.test_krenn_localized_chart_leaf_free -v
python -B -m unittest tests.test_krenn_star_linearization -v
python -B -m unittest tests.test_krenn_star_pivot_charts -v
python -B -m unittest tests.test_krenn_star_pivot_gauge -v
python -B -m unittest tests.test_krenn_star_pivot_affine_slices -v
python -B -m unittest tests.test_krenn_star_pivot_affine_cas_runner -v
python -B -m unittest tests.test_krenn_star_als_artifact -v
python -B -m unittest tests.test_krenn_star_variable_projection -v
python -B -m unittest tests.test_krenn_star_variable_projection_campaign -v
python -B -m unittest tests.test_krenn_star_variable_projection_artifact -v
python -B -m unittest tests.test_krenn_numerical_continuation -v
python -B -m unittest tests.test_krenn_tropical_series_structure -v
python -B -m unittest tests.test_krenn_star_pivot_compactification -v
```

The complete focused repository suite was replayed with

```text
python -B -m unittest discover -s tests -p "test_krenn_*.py" -v
```

and finished with 423 passing tests and 7 intentionally gated long tests
skipped.

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
