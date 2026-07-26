# Exact `n=8,d=3` toric first-shell gate

This bundle is an exact first-shell preflight for a future gauge-quotiented
graph compactification of the matching map.  That compactification and
quotient are not constructed here, and this is not an `n=8` existence or
nonexistence proof.

## Geometry

A common Hopf/projective normalization removes only one complex scaling.  The
GHZ-preserving color-diagonal torus has dimension 21.  On either twelve-weight
seed its restriction has saturated rank 9, so the seed orders can be
normalized to zero without root extraction.

The correct border object retains the limiting output direction: the closure
of `[w] -> [T(w)]`, or corresponding Rees/blow-up data.  Merely setting
`T(w)=h^4 GHZ` and then `h=0` loses that direction.

For one fixed coloring, the K8 edge--matching incidence matrix has rank 21 in
the 105-dimensional matching-amplitude space.  Its exact equivariant rotor
projector has rank 84 and retains the sector invisible to all edge/apex
marginals.

## Exact first-shell result

| seed | raw decorated strata | stabilizer orbits | minimum singleton spills |
|---|---:|---:|---:|
| H5 | 144 | 43 | 4 |
| H6 | 1728 | 304 | 6 |

All 347 minimal flat strata have a
zero-target singleton initial monomial, so none can land over the projective
GHZ direction.  The same replay excludes all
1872 corresponding exact nonzero
supports, because their singleton monomial cannot vanish in the support
torus.

Different minimal layers, negative quotient directions, deeper repairs,
higher-cost first repairs, the other 29 seed orbits, and the full saturated
initial ideal remain open.

## Reproduce

```powershell
C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_first_shell --results-directory results\krenn_quantum_graph\n8_d3_toric_first_shell
```

```powershell
C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_first_shell --results-directory results\krenn_quantum_graph\n8_d3_toric_first_shell --verify-only
```
