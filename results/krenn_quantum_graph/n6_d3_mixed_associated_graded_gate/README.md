# Mixed associated-graded and rotor gate

This is an exact stopping and routing certificate for the known
`n=6,d=3` Laurent family.  It is not an affine membership or nonmembership
proof.

## What closes

All 1,215 deleted-pair cofactor coordinates were replayed with both
perfect-matching enumerators.  Exactly 18 are nonzero: nine monochromatic
and nine mixed.  The six `243 x 15` star matrices satisfy

```text
A_r(t) = diag(t^R(row)) A_r(1) diag(t^(-x_(v,b)))
```

entry by entry.  The same character telescope holds for all nonzero
coordinate coefficients of derivative orders 0, 1, 2, and 3 of the cubic
matching map.  Thus raw star, Jacobian, Hessian, and cubic contractions on
this particular orbit cannot reveal gauge-independent growth.

The Jacobian has 162 nonzeros, rank
130, nullity 5, and its
nonzero bipartite support graph has cycle rank zero.  There is no hidden
first-order holonomy cycle.

## What survives

For one fixed coloring, let `m` be its 15 perfect-matching amplitudes and
let `q=E*m` be the 15 edge/apex marginals.  The exact incidence matrix has
rank 10 and a saturated rank-five integer kernel.  Five
primitive `K3,3` parity circuits form a `Z`-basis; both displayed basis
minors have determinant `-1`.

The canonical projector

```text
P_rot = I - (E^T E)/4 + J/12
```

has image `ker(E)`.  Hence `(q, P_rot*m)`, not `q` alone, is the smallest
faithful state in this decomposition.  Cross-multiplied parity binomials
are the genuine chart-independent toric holonomies.

## Concrete departure from the known branch

For victim `002121`, the known path has only matching
`[[0, 1], [2, 4], [3, 5]]` active, with amplitude `t`.  Its
projective class is constant and is not on `sum_M m_M=0`.  Any exact repair
that stays in the `Mstar` chart must have at least one transverse ratio of
nonpositive valuation; it cannot converge projectively back to `e_Mstar`.

The exact matching repair-cost census is

```text
cost 0: 1
cost 1: 0
cost 2: 6
cost 3: 8
```

The six cost-two alternatives split into the two support-orbit classes
`[0,4,8]` and `[2,9,13]`.  Imposing the minimal relation `r_N=-1`
cancels the victim, but each size-11 branch creates exactly two non-target
singleton spills.  Those six branches are starting states, not witnesses.

## Reproduction

```powershell
Set-Location 'C:\Users\brssn\Projects\Krenn-counterexample-search'

C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m unittest -v `
  tests.test_krenn_mixed_associated_graded

C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m `
  experiments.krenn_quantum_graph.mixed_associated_graded `
  --results-directory `
  results\krenn_quantum_graph\n6_d3_mixed_associated_graded_gate `
  --verify-only
```

The computation is deterministic, serial, exact over `Q[t,t^-1]`, and
uses no floating point or cloud resources.
