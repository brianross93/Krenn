# Exact bounded `n=8,d=3` support-24 circuit gate

This bundle covers one fixed H5 first-shell parent only.  It enumerates all
20,736 choices of one minimum-cost repair for each of the parent's four
mixed singleton spills.  The positive support-24 layer contains
860 raw branches in
444 classes under the true order-two
parent stabilizer.

## Exact exclusion

Every class has six repair-tie rows on twelve nonseed coordinates.  Every
active monochromatic target monomial has incidence row in their rational
row span, so its order is zero on the tie kernel.

For every class, `certificate.json` also records a primitive positive
integer combination of mixed support-singleton incidence rows that lies in
the same tie-row span.  Therefore at least one unique mixed support term has
order at most zero and cannot cancel on the declared leading-support chart.
Both independent perfect-matching enumerators replay every active signature,
and the parent involution transports every representative circuit to its
partner exactly.

| quotient dimension | raw branches | C2 classes | distinct supports |
|---:|---:|---:|---:|
| 1 | 852 | 440 | 850 |
| 2 | 8 | 4 | 8 |

The fact that 852 rank-one branches use only 850 supports is why the
certificate keys branches by complete decorated tie system, not support
alone.

## Boundary

This is not a global `n=8` nonexistence proof.  It does not exclude another
first-shell parent, a deeper repair closure, a zero-quotient support-24
branch, a different support, or a chart where an outside-support term enters
at leading order.  It neither finds nor rules out a finite counterexample.

The rank-two fan is not used as evidence here.  The positive-circuit theorem
is exact and basis-free.

## Reproduce

```powershell
C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_support24_circuits --results-directory results\krenn_quantum_graph\n8_d3_toric_support24_circuits
```

```powershell
C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m experiments.krenn_quantum_graph.n8_toric_support24_circuits --results-directory results\krenn_quantum_graph\n8_d3_toric_support24_circuits --verify-only
```
