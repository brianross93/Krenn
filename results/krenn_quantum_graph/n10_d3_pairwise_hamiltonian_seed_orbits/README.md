# Exact `n=10` pairwise-Hamiltonian seed census

This bundle tests whether the two pairwise-Hamiltonian seed orbits found at
`n=8,d=3` persist at the next even vertex count. They do not: at
`n=10,d=3` there are exactly **ten** `S_10 x S_3` orbit types in this
restricted locus.

## Bounded reduction

`K_10` has 945 perfect matchings, so the unrestricted ordered-triple space
has `945^3 = 843,908,625` elements. The computation does not traverse that
space.

Fix

```text
M0 = (01)(23)(45)(67)(89)
M1 = matching index 124 = (02)(14)(36)(58)(79).
```

Exactly 384 matchings form a Hamilton 10-cycle with `M0`. The stabilizer of
`M0` is transitive on these choices, so `M1` loses no generality. Exactly
148 matchings form a Hamilton cycle with both `M0` and `M1`.

For each of those 148 candidates, the canonicalizer tries all six color
orders. For each order it constructs the ten vertex maps carrying the first
two selected matchings to `(M0,M1)` by exact propagation around their
alternating 10-cycle. The least transported third matching is a complete
`S_10 x S_3` orbit key. Thus the core quotient evaluates only 8,880 small
normalizations.

As an independent mass audit, the implementation builds the 945-vertex
Hamilton-compatibility graph. It is 384-regular, has 181,440 edges, and
every edge has 148 common neighbors. Consequently it has 8,951,040
triangles, or 53,706,240 ordered pairwise-Hamiltonian triples.

## Exact result

There are 24 `S_10` orbits when the three colors remain ordered, and ten
orbits after the `S_3` color action. Stable IDs use the form `PH10-k`, where
`k` is the canonical third matching index; the suffix is not an ordinal
case number.

| ID | fixed-pair slice | full ordered orbit | stabilizer | internal PMs | singleton victims | exponent rank/nullity |
|---|---:|---:|---:|---:|---:|---:|
| `PH10-248` | 15 | 5,443,200 | 4 | 7 | 4 | 6 / 1 |
| `PH10-250` | 10 | 3,628,800 | 6 | 6 | 3 | 6 / 0 |
| `PH10-253` | 15 | 5,443,200 | 4 | 7 | 4 | 6 / 1 |
| `PH10-260` | 30 | 10,886,400 | 2 | 6 | 3 | 6 / 0 |
| `PH10-266` | 30 | 10,886,400 | 2 | 8 | 5 | 6 / 2 |
| `PH10-284` | 10 | 3,628,800 | 6 | 6 | 3 | 5 / 1 |
| `PH10-443` | 3 | 1,088,640 | 20 | 13 | 10 | 7 / 6 |
| `PH10-448` | 15 | 5,443,200 | 4 | 9 | 6 | 6 / 3 |
| `PH10-485` | 5 | 1,814,400 | 12 | 12 | 9 | 7 / 5 |
| `PH10-492` | 15 | 5,443,200 | 4 | 8 | 5 | 6 / 2 |

The fixed-pair slice sizes sum to 148. Multiplying each by 384 choices of
the second matching and 945 choices of the first gives the displayed full
orbits, whose masses sum to 53,706,240. Every orbit size divides
`10! * 3!`.

The ten cubic union graphs are separated by exact tuples consisting of
their internal-perfect-matching count, incidence rank, bipartiteness,
triangle count, four-cycle count, and girth. Internal PM counts range from
6 to 13; there is no stable two-type continuation of the `n=8` H5/H6
picture.

The bundle also enumerates every support-minimal rational dependence among
the internal perfect-matching incidence columns and primitive-normalizes
its integer coefficients. This records binomial circuit degrees without
using floating point or a finite field.

## Equation-system meaning

Every pairwise-Hamiltonian triple is edge-disjoint and selects 15 distinct
diagonal source coordinates. At each vertex and color there is exactly one
selected incident edge. Therefore a coloring supports at most one internal
perfect-matching term. The three selected matchings give the three
monochromatic targets; every other internal perfect matching gives a
distinct non-target singleton victim.

That proves only that the exact 15-coordinate seed support cannot itself be
a solution. A hypothetical complex solution may activate arbitrary outside
repair coordinates. All ten unrestricted repair branches remain open.

## Commands and resources

The run is deterministic, local, serial, and uses no network or cloud RAM.
The exact recomputation takes seconds and requires no scratch cache.

```powershell
Set-Location 'C:\Users\brssn\Projects\Krenn-counterexample-search'

python -B -m unittest -v `
  tests.test_krenn_n10_pairwise_hamiltonian_seed_orbits

python -B -m experiments.krenn_quantum_graph.n10_pairwise_hamiltonian_seed_orbits `
  --results-directory `
  results\krenn_quantum_graph\n10_d3_pairwise_hamiltonian_seed_orbits `
  --verify-only
```

`census.json` contains the ten representatives, exact orbit masses, graph
profiles, victim records, and primitive circuits. `manifest.json` hashes
the bundle and every source dependency.

## Claim boundary

This is exhaustive only among triples whose three pairwise unions are
Hamiltonian. It is not the unrestricted `n=10` seed census, a cover of all
hypothetical solutions, or a proof of `n=10` existence, nonexistence, or
border behavior. No equality is imposed on symmetry-related weights.
