# Structural program after the six-vertex decision

This directory records the exact identities and the next open proof target.
It deliberately separates four statuses:

- **External formal theorem:** checked by a named external proof artifact.
- **Exact identity/corollary:** proved directly and replayed in this
  repository.
- **Finite computation:** true only for the enumerated finite instance.
- **Open route:** a proposed argument with its missing lemma stated.

## What is now closed

At proof-content commit
`105ffbc50b0443433fc53c248272617cc022f3e2`, the external Gallagher
repository proves in Lean

```text
not exists W : WeightsN 6 3 C, EqSystemN 6 3 W.
```

The exact statement, provenance, reported trust boundary, and the fact that
we did not complete a redundant local rebuild are recorded in
`../n6_d3_external_nonimage_reference/reference.json`. No unlicensed
external source or certificate is vendored here.

Accepting that theorem, this repository's independently replayed Laurent
identity

```text
Phi(W(t)) = GHZ_6_3 + t e_002121
```

proves that `GHZ_6_3` lies in the strict border: it is in the image closure
but not in the affine image. Restricting any canonical six-vertex
`d >= 3` solution to three colors would give the forbidden `d=3` solution.
Thus a counterexample to the global conjecture, if one exists, begins at an
even vertex count at least eight.

## Why `K4` is exceptional

`K4` has exactly three perfect matchings. They are pairwise edge-disjoint
and partition its six edges, so assigning one color to each matching creates
the exact three-color witness without a fourth mixed matching.

There is also a representation-theoretic reason the number three is
intrinsic. Label the vertices by `F_2^2`; the three matchings are the three
nonzero translation directions. The action of `S4` on them has kernel the
Klein four group and quotient

```text
S4 / V4 isomorphic to S3.
```

Equivalently, the perfect-matching permutation module is
`[4] + [2,2]`, of dimensions `1+2`. At six vertices it becomes
`[6] + [4,2] + [2,2,2]`, of dimensions `1+9+5`; the exceptional
three-object quotient is gone. This explains the `K4` construction. It is
not yet a proof that arbitrary nonsymmetric weights at larger `n` cannot use
cancellation. The quotient and representation dimensions are replayed in
`experiments/krenn_quantum_graph/symmetry_structure.py`.

## Simultaneous star structure

For every even vertex set and every root `r`, partitioning a perfect matching
by its unique edge at `r` gives

```text
H_V(c) =
  sum_(s != r) W_rs[c_r,c_s] H_(V minus {r,s})(c).
```

The implementation in
`experiments/krenn_quantum_graph/multi_star_identities.py` constructs this
linearization at every root. The two copies of every physical edge agree by
transpose, and a two-root expansion is checked coefficient by coefficient.
The machine-readable `identity_regressions.json` and tests replay all roots
for:

- the exact `K4,d=3` witness;
- the exact `K6,d=2` witness; and
- the natural `K6,d=3` seed, retaining its unique `002121` victim.

These are universal combinatorial identities, not extra equations and not an
induction: the residual cofactor tensor need not itself be a smaller GHZ
tensor.

## Product contraction and blocker primes

For product covectors `x_i`, define

```text
b_ij = x_i^T W_ij x_j.
```

The full tensor equation contracts exactly to

```text
haf(b) = sum_a product_i x_i(a).
```

If the nonzero graph of `b` has no perfect matching, its hafnian vanishes.
The squarefree perfect-matching monomial ideal therefore organizes all
zero-pattern obstructions. Its minimal primes are precisely the
inclusion-minimal edge blockers.

Tutte's theorem and edge maximality classify the complement of every minimal
blocker as

```text
K_s join (K_a1 disjoint_union ... disjoint_union K_a_(s+2)),
```

where the `a_i` are positive odd integers summing to `n-s`. The native exact
census is:

| vertices | symmetry types | labelled blockers |
|---:|---:|---:|
| 4 | 2 | 8 |
| 6 | 4 | 91 |
| 8 | 6 | 1,408 |

Symmetry reduces the structural types to six at `n=8`; it does not permit
equating weights or replacing the 1,408 labelled conditions for a fixed
nonsymmetric candidate.

## The first exact `n=8` target

There are two square qutrit blocker charts.

1. A two-vertex barrier with outside odd components `3+1+1+1` gives 12
   pullbacks of bilinear constraints in 12 affine variables. Its top Chow
   coefficient is exactly **30**.
2. A three-vertex barrier with five outside singletons gives all ten `K5`
   pullbacks of bilinear constraints in ten affine variables. Its top Chow
   coefficient is exactly **24**, the number of regular tournaments on five
   labelled vertices.

Both counts are replayed by two independent finite methods: truncated Chow
coefficient multiplication and brute-force endpoint orientations. The
nonzero top Chow products imply that every coefficient specialization has a
nonempty common projective zero locus. The numbers 30 and 24 are the lengths,
with multiplicity, of generic zero-dimensional fibers; a special fiber may
be nonreduced or positive-dimensional.

For a chosen color `a`, setting every barrier covector to `e_a` and
using independent projective rescalings to normalize every outside coordinate
`x_i(a)=1` makes the GHZ contraction exactly one. The normalized equations
may have constant, linear, and quadratic terms. Any solution of that affine
chart would therefore contradict exact GHZ membership.

The missing lemma is **boundary escape**:

> For every `W` satisfying `EqSystemN 8 3`, some labelled blocker of one of
> the two square types, some color `a`, and some projective blocker solution
> satisfy `product_(i not in S) x_i(a) != 0`.

The positive Chow counts alone do not prove this. Pulling a monomial prime
back through the bilinear forms can create reducible or nonreduced boundary
components. For a fixed labelled blocker, failure of boundary escape means
that every solution lies in the common locus
`g_0=g_1=g_2=0`, where `g_a=product_(i not in S)x_i(a)`.

## Next exact experiment

Work in the already-normalized affine charts; no Rabinowitsch variable is
needed.

1. Build the 12-by-12 degree-30 and 10-by-10 degree-24 quotient algebras for
   symbolic edge matrices.
2. Test each statement `g_a in radical(I_B)` separately, using saturation
   emptiness or nilpotence of multiplication by `g_a`. Testing only
   `g_0*g_1*g_2` would detect a weaker union of boundaries and is
   insufficient.
3. Reduce those confinement conditions using the simultaneous star/cofactor
   identities before any Groebner computation.
4. Intersect overlapping labelled barriers only through
   `S8 x S3` orbit representatives; never impose equal weights across an
   orbit.
5. Treat a timeout, modular point, numerical solution, or surviving branch
   as nonproof.

This program is aimed at a theorem for `n=8`, with a formulation that can
then be tested for general even `n`. It is not another bounded
counterexample search.
