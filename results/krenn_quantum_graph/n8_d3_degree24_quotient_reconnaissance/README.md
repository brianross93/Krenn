# Degree-24 K5 quotient reconnaissance

This bundle validates the exact backend and independent-replay pipeline for
the square three-vertex Tutte-barrier chart at `n=8,d=3`.

The selected blocker leaves vertices `3,4,5,6,7`; its ten blocker edges form
`K5`. After normalizing color zero at those five vertices, the chart has ten
bi-affine equations in ten variables. Seed `80320260725` assigns 90 distinct
nonzero integers from `[-127,127]`, excluding multiples of 31, to the 90
edge-matrix slots. No equality is imposed between symmetry-related weights.

## Exact pilots

The standalone engine is Singular 4.3.2 in
`krenn-n8-singular:ubuntu24.04-v1`, image
`sha256:f9d3378746fb922f73802e52117c8e67893c45af2497275e50b5a940b1a751df`.
It was built from the repository Dockerfile, whose SHA-256 is
`5728a7eade496a9e558c8f9cad43add29d5f5ced81937e16154781da0919f278`.
Both pilots ran offline and serially with two CPUs.

| field | budget | RAM | engine subprocess wall | basis | cyclic quotient |
|---|---:|---:|---:|---:|---:|
| `F_31` | 600 s | 4 GiB | 0.324389 s | 56 | 22 |
| `Q` | 1800 s | 8 GiB | 73.255729 s | 58 | 24 |

The backend reported different `vdim` values. Independent replay certifies
only the two cyclic quotient modules, not the full quotient dimensions, so
the difference does not prove that 31 is a special fiber. The modular result
is not a lift or a statement over characteristic zero.

For each completed job, Singular returned a standard-monomial basis and all
ten multiplication matrices. Native exact replay checked:

- all 45 pairwise commutators;
- all ten blocker equations as matrix identities; and
- the 22 or 24 reported monomials against the cyclic vector `1`.

Thus the retained matrices define nonzero unital cyclic quotients. The
rational representation proves that this fixed specialized affine ideal is
proper, hence its variety has a point over
`Qbar`, and therefore over `C`. It does not exhibit a rational point.

The replay does not independently certify that either cyclic quotient is the
entire coordinate ring, that the ring is reduced or radical, or that its
length counts distinct points.

## Commands actually run

```powershell
Set-Location 'C:\Users\brssn\Projects\Krenn-counterexample-search'

docker build `
  --file experiments/krenn_quantum_graph/n8_k5_singular.Dockerfile `
  --tag krenn-n8-singular:ubuntu24.04-v1 .

docker image inspect '--format={{.Id}}' 'krenn-n8-singular:ubuntu24.04-v1'

docker run --rm --pull never --network none --cpus 1 --memory 1g `
  --entrypoint timeout `
  sha256:f9d3378746fb922f73802e52117c8e67893c45af2497275e50b5a940b1a751df `
  30s Singular -q -c `
  'ring r=31,(x),dp;ideal I=x;ideal G=slimgb(I);print(vdim(G));print(reduce(x,G));'

& 'D:\KrennScratch\counterexample_search\tools\krenn-tests-py313\Scripts\python.exe' `
  -B -m experiments.krenn_quantum_graph.blocker_quotient_runner `
  seed_80320260725_p31_slimgb

& 'D:\KrennScratch\counterexample_search\tools\krenn-tests-py313\Scripts\python.exe' `
  -B -m experiments.krenn_quantum_graph.blocker_quotient_runner `
  seed_80320260725_q_slimgb

$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

& 'D:\KrennScratch\counterexample_search\tools\krenn-tests-py313\Scripts\python.exe' `
  -B -m unittest -v `
  tests.test_krenn_blocker_quotient_artifact `
  tests.test_krenn_blocker_quotient_reconnaissance `
  tests.test_krenn_blocker_quotient_runner `
  tests.test_krenn_n8_seed_orbits

& 'D:\KrennScratch\counterexample_search\tools\krenn-tests-py313\Scripts\python.exe' `
  -B -m unittest discover -v -s tests -p 'test_krenn_*.py'
```

The smoke test returned quotient dimension `1` and normal form `0`.
The complete Krenn suite passed 287 tests, with three intentionally gated
long tests skipped.

An earlier SymPy 1.14 prototype on a superseded repeated-coefficient
specialization timed out after 600 seconds. That was a backend-performance
result only and is not part of either exact pilot above.

## Claim boundary

The specialization does not satisfy or impose the `n=8` Krenn EqSystem.
This is a pipeline validation, not evidence for a Krenn witness or for
boundary escape. It does not build the symbolic 90-parameter quotient,
handle all 56 labelled `K5` blockers, handle the degree-30 blocker type, or
prove either `n=8` existence or nonexistence. A backend unit-ideal result
would also need an independently replayed ideal-membership certificate; no
such claim is made here.

The full transcripts and matrices remain at
`D:\KrennScratch\counterexample_search\n8_k5_quotient_v2`. The rational
matrix representation is 5,601,312 bytes, so the scratch tree is required
for full matrix replay. Small run receipts and their hashes are retained in
the repository bundle.

## Next exact gate

Do not launch a universal 90-parameter Groebner basis. The unweighted
Hamilton-cycle proof suggests first reducing triples of monochromatic
perfect matchings under `S8 x S3`, then annotating their forced singleton
victims and matching-incidence circuits. Pilot exact branching on the
pairwise-Hamiltonian orbit types. Only unresolved multinomial branches
should be routed to the two square Tutte-blocker charts.
