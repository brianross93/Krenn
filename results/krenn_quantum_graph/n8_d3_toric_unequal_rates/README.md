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

The exclusion is gauge-invariant and comes from exact positive singleton
incidence circuits, not from absolute orders on selected ray lifts:

| class | positive singleton combination | tie-row-span witness |
|---|---|---|
| A | `3 E_571 + 2 E_851 + E_2438 + E_2493 + E_4048 + E_5792` | `[1,3,2,0,1,3]` |
| B | `E_853 + E_2430 + E_6557` | `[0,1,1,0,1,0]` |

Every listed mixed equation has exactly one support monomial under both
independent perfect-matching enumerators.  In each class, the positive
weighted sum of their nonseed incidence rows is exactly the displayed linear
combination of the tie rows over `Q`; adjoining that row does not increase
exact rank five.  Consequently the positive weighted sum of singleton orders
is zero everywhere on the tie lattice, so at least one unique mixed singleton
has order at most zero.

The target comparison is also exact.  Both enumerators find two active
monochromatic terms for color 0 and one each for colors 1 and 2.  Every one of
their four nonseed incidence rows has a displayed witness in the same tie-row
span.  Thus every active target support monomial has normalized order zero on
the full tie lattice.  The target polynomial valuation is therefore at least
zero (or infinite); cancellation among target terms can only raise it and
strengthen the mixed-singleton obstruction.

This excludes both quotient lines, hence all four orientations, throughout
the depth-two local declared family: either the exact support torus, where all
coordinates of the recorded support `S` have nonzero leading coefficient and
outside coordinates vanish, or the conditional degeneration extension where
every outside monomial stays strictly above every relevant active target and
recorded singleton order.  Outside-term-entry cones remain unclassified.

For audit only, the chosen normalized positive and negative lifts give these
gauge-dependent counts:

| chosen lift | unique mixed minima at order <= target |
|---|---:|
| A+ | 14 |
| A- | 10 |
| B+ | 12 |
| B- | 12 |

Those absolute-order counts are not residual-gauge invariant and are not used
as proof.  This unequal-rate bundle itself makes no support-24 classification.
The declared exact-support/no-outside-entry positive layer is handled by the
separate support-24 circuit gate.

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
