# `n=8,d=3` common-support CNF gate

This bundle records the deterministic transfer gate for Gallagher's common
Boolean support calculus. It is deliberately weaker than a Krenn equation
solver: it remembers only which complex edge weights and perfect-matching
monomials are zero or nonzero.

## Frozen rule set

The CNF contains exactly these common families:

1. a matching term is active exactly when all four of its edge coordinates
   are active;
2. each of the three monochromatic target equations has an active term;
3. no non-target equation has exactly one active term;
4. every root/color has a pure diagonal star anchor;
5. every root/two-color pencil either has both pure output columns or never
   leaves the pencil;
6. every root/output color has a pure full-column anchor.

It excludes every orbit-specific exact-support, Laurent, equation-pattern,
closed-orbit, and generated branch-tail no-good used by the six-vertex
certificate.

The direct Gallagher-style encoding has 691,461 variables, 4,141,782 common
clauses, and 81,279,582 literal occurrences. A branch adds the twelve
positive colored edge coordinates selected by one of the 31 exact
`S_8 x S_3` seed representatives, giving 4,141,794 clauses and 81,279,594
literals.

## Fixed endpoint

Branches are processed in canonical census order with one deterministic
CaDiCaL 2.1.2 process and seed zero.

- The first SAT result is accepted only after the complete model is
  reconstructed from its 252 physical edge bits, replayed with both
  perfect-matching enumerators, compared against every generated DIMACS
  clause, and all semantic rule families pass. It terminates the gate with
  "common support calculus insufficient."
- An UNSAT status is not accepted by itself. That branch would require an
  immutable text LRAT certificate and kernel replay before continuing.
- A timeout, incomplete model, solver failure, hash mismatch, or unchecked
  UNSAT result makes no claim.

The large DIMACS, complete model, and logs remain under
`D:\KrennScratch\counterexample_search\n8_base_support_cnf_v1`. The
committed certificate stores the compact physical support, exact hashes,
solver receipt, semantic replay, and manifest.

## Result

The first canonical branch is SAT, so the frozen common support calculus is
conclusively insufficient to refute all 31 branches. The terminal branch and
model are:

| Item | Exact result |
|---|---:|
| Branch index | 0 |
| Matching representative | `(0, 0, 0)` |
| Positive branch anchors | `1, 5, 9, 118, 122, 126, 199, 203, 207, 244, 248, 252` |
| Active physical edge variables | 101 of 252 |
| Active matching-term variables | 23,958 of 688,905 |
| Active target terms | 4, 4, 12 |
| Non-target singleton equations | 0 |
| Star assertions | 24 of 24 |
| Pair assertions | 24 of 24 |
| Full-column assertions | 24 of 24 |
| Other census branches satisfied by this model | none |

The campaign stopped after branch 0, as fixed in advance; branches 1 through
30 were not run. One checked SAT branch is enough to show that the common
prefix cannot be an all-branch refutation.

The streamed branch input is 552,430,629 bytes with SHA-256
`dde9272be36632cea96f3b591817af96803bf3d0a70eebd753e65bf43b1b1bc3`.
It was generated from `21:29:33.126656Z` to `21:29:51.108104Z` on
2026-07-25. CaDiCaL ran from `21:29:51.124713Z` to
`21:31:54.617012Z`, returned code 10, and emitted a 5,619,090-byte model
with SHA-256
`0a3d695e5c569c3b43e984b6b75d1818f580dcc7bf803b04a4eface0d467b67c`.
The solver used one process, seed 0, internal checking, and a 7,200-second
safety guard.

The complete assignment was then derived independently from the 101 active
edge variables. Both exact perfect-matching enumerators agreed, every one of
the 4,141,794 generated clauses matched the stored DIMACS byte-for-byte, and
every clause evaluated true. No LRAT certificate exists or is needed because
the terminal result is SAT.

This is not a complex weighting and does not decide whether GHZ lies in the
affine image. It decides only the narrower transfer question: Gallagher's
common Boolean support prefix does not generalize into an `n=8` refutation
without additional, separately justified cuts.

## Commands

```powershell
Set-Location 'C:\Users\brssn\Projects\Krenn-counterexample-search'

& 'C:\tmp\Krenn-obstruction-venv\Scripts\python.exe' -B -m `
  experiments.krenn_quantum_graph.n8_base_support_cnf `
  --scratch-directory `
  D:\KrennScratch\counterexample_search\n8_base_support_cnf_v1 `
  --results-directory `
  results\krenn_quantum_graph\n8_d3_base_support_cnf_gate `
  --solver `
  D:\KrennScratch\counterexample_search\tools\elan\toolchains\leanprover--lean4---v4.27.0\bin\cadical.exe `
  --solver-timeout-seconds 7200

& 'C:\tmp\Krenn-obstruction-venv\Scripts\python.exe' -B -m `
  experiments.krenn_quantum_graph.n8_base_support_cnf `
  --results-directory `
  results\krenn_quantum_graph\n8_d3_base_support_cnf_gate `
  --verify-only

& 'C:\tmp\Krenn-obstruction-venv\Scripts\python.exe' -B -m unittest -v `
  tests.test_krenn_n8_base_support_cnf `
  tests.test_krenn_n8_seed_orbits `
  tests.test_krenn_n10_pairwise_hamiltonian_seed_orbits

& 'C:\tmp\Krenn-obstruction-venv\Scripts\python.exe' -B -m unittest `
  discover -s tests -p 'test_krenn_*.py' -v
```

## Claim boundary

A SAT support assignment is not a complex weighting and does not prove
existence. Failing to close one branch proves only that this frozen Boolean
calculus is insufficient. Global nonexistence would require all 31 branches
to be UNSAT with checked algebraic semantics and checked LRAT certificates.
