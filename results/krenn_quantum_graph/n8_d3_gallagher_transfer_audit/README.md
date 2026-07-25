# Gallagher Lean/SAT transfer audit

This is a read-only audit of revision
`c04696e515e0c02be140353fb52ea60c62e827b1` of
[`algal/krenn-gu-6x3-certificate`](https://github.com/algal/krenn-gu-6x3-certificate).
The external repository was inspected under
`D:\KrennScratch\counterexample_search\external_readonly`; it was not copied
into or built from this repository.

## Decision

The existing `n=6,d=3` Lean/SAT reduction does **not** transfer to
`n=8,d=3` as a parameter change. Its proof architecture transfers only
conditionally:

1. map each complex edge weight to a Boolean recording whether it is
   nonzero;
2. define each matching-term Boolean as the conjunction of its edge
   Booleans;
3. prove support clauses that every complex solution must satisfy;
4. cover every selected monochromatic matching triple by symmetry;
5. add exact algebraic no-good clauses;
6. refute each resulting CNF and replay its LRAT certificate in Lean.

Steps 3--6 are the decisive work, and Gallagher's implementations of them
are specific to six vertices. The current matching tables, support rules,
eight-orbit cover, Laurent identities, generated CNFs, and LRAT files prove
nothing directly at eight vertices.

The Boolean layer is a sound necessary-condition abstraction, not an exact
finite encoding of complex solvability. A target equation implies at least
one active matching term. A zero-target equation implies that exactly one
active term is impossible. Two or more active terms may or may not cancel
over the complex numbers, which the support Booleans do not record.
Therefore an LRAT refutation of sound clauses excludes complex solutions,
but a satisfying Boolean assignment would not produce a complex witness.

## Exact scope comparison

| quantity | `n=6,d=3` certificate | naive `n=8,d=3` analogue |
|---|---:|---:|
| physical edge-coordinate Booleans | 135 | 252 |
| perfect matchings per coloring | 15 | 105 |
| colorings | 729 | 6,561 |
| matching-term Booleans | 10,935 | 688,905 |
| total support/event variables | 12,330 | about 691,461 |
| ordered selected-target triples | 3,375 | 1,157,625 |
| target-triple symmetry orbits | 8 | 31 |

The 31-orbit count is the exhaustive `S_8 x S_3` seed census already
replayed in this repository. H5 and H6 are only the two orbits whose three
pairwise unions are Hamiltonian. The other 29 unrestricted repair branches
remain open, so a transferred exhaustive certificate would have 31 initial
branches, not two.

## Where six vertices are essential

- `SixThree.lean` fixes `Fin 6`, `Fin 3`, and the explicit 15 perfect
  matchings.
- `SupportEncoding.lean` hard-codes 135 edge variables, `729*15` term
  variables, and the six-vertex star, pair-pencil, and full-column events.
- `UniversalRules.lean` and the semantic wrappers assume
  `EqSystemN 6 3` and discharge finite cases against those 15 matchings.
- `TargetOrbitCoverage*.lean` encodes `Fin 3375`, eight branch bases, and
  generated six-vertex symmetry tables.
- `ExactSupportSemantics.lean`, `Orbit3PatternSemantics.lean`, and the
  generated branch files contain orbit-specific Laurent and support
  no-goods.
- `Unrestricted.lean` finishes by an explicit eight-way dispatch.

The generic reusable part is the final implication in
`BranchFramework.lean`: if a hypothetical complex solution induces a
Boolean assignment satisfying all proved semantic clauses, and an LRAT
certificate proves those clauses unsatisfiable, then that branch is
impossible. New sound `n=8` clauses and new branch certificates would still
have to be constructed.

## Consequence for the campaign

There is no justified shortcut from the two pairwise-Hamiltonian pilot
orbits to an exhaustive SAT/LRAT proof. A new `n=8` certificate campaign is
finite in principle, but it does not yet have a demonstrated closing rule:
the base support encoding may simply be satisfiable until enough new exact
algebraic cuts are proved.

The next bounded structural check was therefore the `n=10`
pairwise-Hamiltonian orbit census. Its exact result is recorded in
`../n10_d3_pairwise_hamiltonian_seed_orbits`: there are ten orbit types, so
the two-orbit `n=8` hard core does not persist. This removes the proposed
uniform-two-types route before any H6-specific repair branching is resumed.

## Claim boundary

This audit does not modify or recompile Gallagher's proof. It does not
challenge the checked `n=6,d=3` result. It does not prove an `n=8`
existence, nonexistence, boundary, or SAT result, and it does not claim that
a newly developed `n=8` support calculus could not eventually close all 31
branches.
