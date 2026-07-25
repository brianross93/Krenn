# Exact two-chart modular prime sweep

This bundle records a deterministic arithmetic cross-check of both square
Tutte-blocker charts at `n=8,d=3`. It does not impose the Krenn EqSystem and
is not an existence, nonexistence, or boundary-escape certificate.

## Fixed charts

The two computations use separate fixed dense specializations with master
seed `80320260725` and coefficient height `127`. Coefficients are pairwise
distinct within each chart; no equality is imposed on symmetry-related
coefficient slots.

| key | blocker profile | labelled orbit | equations / variables | coefficient slots | projective Chow degree |
|---|---|---:|---:|---:|---:|
| `degree24_b3_11111` | barrier 3, components `1+1+1+1+1` | 56 | 10 / 10 | 90 | 24 |
| `degree30_b2_3111` | barrier 2, components `3+1+1+1` | 560 | 12 / 12 | 108 | 30 |

The degree-24 chart is exactly the chart from the earlier rational/P31
pilot, with SHA-256
`66a0e966a591ed1d28fd7cd98251f274940a746892a203603ff76e6e33595ab5`.
The degree-30 chart SHA-256 is
`cc020528fe57d893f874aef801e1f8d31adb235197266ef21ec85eb2dc666d99`.
The two coefficient assignments are not restrictions of one common global
`K8` weighting.

## Prime policy

The sweep was fixed before execution: the first twenty primes strictly
larger than `2*127 = 254`,

```text
257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
313, 317, 331, 337, 347, 349, 353, 359, 367, 373
```

At every selected prime, all 90 or 108 signed integer coefficients remain
nonzero and pairwise distinct. We call these primes
*coefficient-injective*, not algebraically good primes. No prime was added,
removed, or substituted after seeing a result.

The older `p=31` run is retained only as a colliding control. Its 90
distinct integer coefficients occupy 30 nonzero residue classes, and its
verified cyclic module has dimension 22. Over `Q`, the same degree-24 fixed
specialization has a verified cyclic module of dimension 24.

## Exact results

Every one of the forty new Singular transcripts produced a finite cyclic
quotient representation. Native exact replay checked all matrix
commutators, every blocker equation as a matrix identity, and every reported
standard monomial against the cyclic vector `1`.

| chart | primes completed | verified cyclic dimensions | checks per prime |
|---|---:|---:|---|
| degree 24 | 20 / 20 | `24` in all 20 fields | 45 commutators, 10 equations, 24 cyclic vectors |
| degree 30 | 20 / 20 | `30` in all 20 fields | 66 commutators, 12 equations, 30 cyclic vectors |

For degree 24 every Gröbner basis had 58 elements; for degree 30 every basis
had 83. The forty backend subprocesses used 12.859666 seconds in total, and
the slowest used 0.363167 seconds. The complete serial campaign, including
Python parsing, exact matrix verification, hashing, checkpointing, and the
two preflights, ran from `2026-07-25T20:01:37.771072+00:00` through the last
probe start at `2026-07-25T20:03:22.598429+00:00`. Backend stderr was empty.

This resolves the practical 22-versus-24 ambiguity: all twenty
coefficient-injective measurements agree with the rational dimension-24
module, while the colliding `p=31` control differs. It does **not** prove
that 31 is a special fiber, because the replay certifies a cyclic quotient
module rather than independently certifying the full coordinate ring,
flatness, or good reduction.

Likewise, the twenty degree-30 modules are the expected companion
measurement for the second square blocker. They are not a modular lift or a
statement over `Q` or `C`.

## Resources and commands

All probes ran locally, serially, with one worker, two CPUs, 4 GiB RAM, no
network, and a 600-second per-probe engine guard. The actual maximum was
under 0.364 seconds, so cloud RAM was neither used nor needed.

The pinned engine was Singular 4.3.2 in
`krenn-n8-singular:ubuntu24.04-v1`, image
`sha256:f9d3378746fb922f73802e52117c8e67893c45af2497275e50b5a940b1a751df`.

```powershell
Set-Location 'C:\Users\brssn\Projects\Krenn-counterexample-search'

& 'D:\KrennScratch\counterexample_search\tools\krenn-tests-py313\Scripts\python.exe' `
  -B -m unittest -v tests.test_krenn_blocker_quotient_prime_sweep

& 'D:\KrennScratch\counterexample_search\tools\krenn-tests-py313\Scripts\python.exe' `
  -B -m experiments.krenn_quantum_graph.blocker_quotient_prime_sweep `
  --resume

& 'D:\KrennScratch\counterexample_search\tools\krenn-tests-py313\Scripts\python.exe' `
  -B -c "from experiments.krenn_quantum_graph.blocker_quotient_prime_sweep import replay_sweep; replay_sweep()"
```

The first campaign command used a hidden local process only to avoid a host
shell timeout; its stdout and stderr were redirected inside the same scratch
root.

## Artifact layout and claim boundary

- `sweep.json` is the canonical forty-run summary.
- `receipts.json` retains all forty small, exactly replayed receipt records.
- `manifest.json` hashes the bundle and every source dependency.
- Full transcripts and multiplication matrices remain at
  `D:\KrennScratch\counterexample_search\n8_square_blocker_prime_sweep_v3`.

The bundle does not independently certify the backend Gröbner bases or full
quotient dimensions; count distinct points; prove radicality or reducedness;
prove a generic-prime theorem, flatness, or good reduction; lift a modular
result; check all labelled blockers or multiple coefficient
specializations; impose the EqSystem; or prove any `n=8` Krenn conclusion.

The next theorem-facing step is not another prime sweep or an open-ended H6
repair tree. The Gallagher Lean/SAT transfer must first be audited for a
finite closing rule. That audit is now recorded in
`../n8_d3_gallagher_transfer_audit`: the proof architecture is reusable, but
the six-vertex reduction and certificates do not transfer directly. The
bounded `n=10` pairwise-Hamiltonian census is now recorded in
`../n10_d3_pairwise_hamiltonian_seed_orbits`; it has ten full-symmetry orbit
types, so the two-type `n=8` pattern does not persist. H6-specific repair
branching remains deferred until it is stated with a terminating
certificate condition.
