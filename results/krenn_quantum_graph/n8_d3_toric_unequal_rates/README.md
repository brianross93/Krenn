# Exact `n=8,d=3` toric unequal-rate gate

This bundle tests whether unequal coordinate orders create new quotient
directions on the minimal H5/H6 cost-two repair skeletons.  It is a local
integral lattice calculation, not a construction of the full GIT quotient,
graph/Rees compactification, or an `n=8` existence/nonexistence proof.

## Exact result

| seed | raw decorations | symmetry orbits | constraint rank | gauge rank | quotient rays |
|---|---:|---:|---:|---:|---:|
| H5 | 144 | 43 | 5 | 11 | 0 |
| H6 | 1728 | 304 | 6 | 12 | 0 |

For every decorated skeleton, the repair-order constraint lattice equals the
restricted target-preserving gauge lattice over the integers.  Each equality
is certified by `C_S G_S = 0`, complementary exact ranks, and a displayed
maximal gauge minor of determinant `+1` or `-1`.  Therefore there are
0 nonzero
rational quotient rays.  Projectively the integral quotient is `Z/4`;
this finite torsion creates no ray, though normalization can require a
fourth-order base change.

The full `252 x 21` raw gauge lattice itself has saturation index two, as
expected from the generic `mu_2` ineffectivity.  The local determinant-one
minors are what make the stronger skeleton-level integral statement valid.

After gauge normalization, within-equation term-order differences and the
flat singleton obstruction are unchanged.  Hence unequal rates do not rescue
any of these 1872 minimal skeletons.

## Bounded H5 depth-two fan

All 20736 fixed-parent repair decorations
were ranked exactly.  The minimum positive quotient layer has four
support-22 decorations in two symmetry classes and four oriented rays.

| ray | noncancellable unique mixed minima at order <= target |
|---|---:|
| A+ | 14 |
| A- | 10 |
| B+ | 12 |
| B- | 12 |

Both perfect-matching enumerators replay all 6,561 equations for every ray.
All four rays are exactly excluded.  Support-24 positive-quotient branches
remain unclassified beyond their exact census.

## Boundary

This does not classify strata with multiple leading repairs, cost-three or
cost-four repairs, outside terms entering an initial form, deeper repair
closures, the other 29 seed orbits, or the full saturated initial ideal.

## Reproduce

```powershell
C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_unequal_rates --results-directory results\krenn_quantum_graph\n8_d3_toric_unequal_rates
```

```powershell
C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_unequal_rates --results-directory results\krenn_quantum_graph\n8_d3_toric_unequal_rates --verify-only
```
