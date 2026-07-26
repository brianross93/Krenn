# Projective victim-repair numerical campaign

This bundle records a bounded, deterministic direct-GHZ search from the two
projective victim-repair orbit representatives.  It does not use the known
Laurent initializer or continuation.

Status: `bounded-search-complete-no-exact-candidate`

- N=0 size-22 support found by this bounded preflight:
  `false`
- N=2 literal size-22 support used: `true`
- numerical jobs completed: `16`
- exact dual-enumerator candidate verified:
  `false`

The relation `r_N=-1` and the full victim hyperplane are satisfied only at
initialization (within the recorded floating tolerance).  They are not solver
constraints.  A bounded support miss or numerical miss proves nothing.

## Reproduction

```powershell
C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m experiments.krenn_quantum_graph.projective_repair_campaign --scratch-directory D:\KrennScratch\counterexample_search\projective_victim_v1 --results-directory C:\Users\brssn\Projects\Krenn-counterexample-search\results\krenn_quantum_graph\n6_d3_projective_repair_search --workers 8 --support-nodes 100000 --support-state-cap 2000000 --support-candidate-scan 16 --supports-per-orbit 1 --starts-per-support 8 --radii 8 32 --repair-scales .03 .1 .3 1 --iterations 250 --evaluations 2250 --checkpoint-interval 20 --master-seed 60320260725 --reconstruction-trigger 1e-8
```

Verify the compact bundle without scratch checkpoints:

```powershell
C:\tmp\Krenn-obstruction-venv\Scripts\python.exe -B -m experiments.krenn_quantum_graph.projective_repair_campaign --verify-only --scratch-directory D:\KrennScratch\counterexample_search\projective_victim_v1 --results-directory C:\Users\brssn\Projects\Krenn-counterexample-search\results\krenn_quantum_graph\n6_d3_projective_repair_search
```
