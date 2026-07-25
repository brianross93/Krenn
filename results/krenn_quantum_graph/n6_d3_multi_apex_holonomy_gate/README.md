# Exact multi-apex holonomy gate

This bundle records a bounded stopping test for the first simultaneous-star
holonomy proposal.  It is an exact negative preflight, not a Krenn--Gu
existence or nonexistence theorem.

## Exact outcome

The punctured Laurent family (`t != 0`) is exactly the full color-diagonal
GHZ-stabilizer one-parameter orbit with

```text
x_(0,0) = +1
x_(2,0) = -1
all other x_(v,a) = 0.
```

It sends the natural seed to

```text
w_01^(00) = t
w_23^(00) = t^-1
```

and gives the victim coloring `002121` character `+1`.  Consequently every
regular or rational invariant of that full stabilizer is constant wherever
it is defined for `t != 0`.  The previously known pole monomial is not an
invariant of this larger group.

The chart-free overlaps `B_r Y_s`, `r != s`, vanish identically on both the
`K4,d=3` witness and the entire natural `n=6,d=3` Laurent family.  Their
support consists of one monochromatic perfect matching per color, and the
matching partner required by `B_r` cannot simultaneously be the partner
required by `Y_s`.

The deterministic full-rank star minors along the `n=6` path are

```text
t^-9, -t^-3, -t^6, -t^6, -1, 1
```

Their valuations sum to zero and their product is exactly one.  These minors
use a declared lexicographic row chart; their product is a diagnostic, not a
claimed chart-independent invariant.

The gauge-invariant monochromatic Cauchy--Binet term at every apex is also
exactly one.  The marginal product `C^0 C^1 C^2` is `I_4` for `K4`; for the
natural `n=6` family it is the constant permutation

```text
[2, 4, 1, 3, 0, 5]
```

with cycles `(0 2 1 4)(3)(5)`, trace `2`, and determinant `-1`.
This separates the two sparse seeds but does not see Laurent divergence.
The certificate includes exact rational symmetric, zero-diagonal,
row-stochastic `6 x 6` matrices whose ordered product is the identity.  Thus
those four tested abstract marginal axioms alone have no six-vertex exclusion
power; realization by three blocks of one shared Krenn weighting is not
claimed.

## First mixed-equation repair

Let `A_a` be the all-color-`a` matching amplitude and

```text
C^a_rs = w_rs^(aa) H_(V minus {r,s})(a,...,a).
```

The homogeneous candidate tested here is

```text
F = C^0 C^1 C^2 - A_0 A_1 A_2 I_6.
```

It does not admit a degree-nine mixed-ideal identity.  For the invariant
trace `T=trace(F)`, the existing exact two-row dual annihilates the compressed
fine-graded source map and pairs to `2160`; Reynolds averaging over
`S6 x S3` lifts that obstruction losslessly to the raw source map over `Q`.
Therefore `T` is not in `J_mix` over `Q`.  Vertex symmetry and `F*1=0` then
imply that no individual entry `F_rs` lies in `J_mix`.

This does **not** decide membership in `radical(J_mix)`.  A higher-power
set-theoretic identity could still exist.

## Reproduction

```powershell
Set-Location 'C:\Users\brssn\Projects\Krenn-counterexample-search'

python -B -m unittest -v tests.test_krenn_multi_apex_holonomy

python -B -m experiments.krenn_quantum_graph.multi_apex_holonomy `
  --results-directory `
  results\krenn_quantum_graph\n6_d3_multi_apex_holonomy_gate `
  --verify-only
```

The computation is deterministic, exact over `Q[t,t^-1]`, serial, and uses
both native perfect-matching enumerators.  No floating point, cloud compute,
or large scratch cache is used.

## Claim boundary

The gate says to stop the tested cross-apex overlap, deterministic-minor,
monochromatic Pluecker/marginal-product, and degree-nine ideal candidates.
It neither finds a finite witness nor proves a new nonexistence result.
A continuation intended to distinguish this Laurent path in the same
framework would need mixed-color cofactor or vanishing-order data beyond the
`B_r X_r = I_3` normal form.
