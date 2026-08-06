# HNSW vs FLAT: when does the index earn its keep?

Valkey vector search offers two index types. FLAT is exact brute force:
every query compares against every stored vector. HNSW is an approximate
graph index. `valkey-faces` uses HNSW, but for seven faces that is
overkill, so the honest question is: at what scale does HNSW start to
win, and what do you give up?

I measured it. 512-D unit vectors (face-embedding shaped), cosine
distance, KNN, on `valkey/valkey-bundle`. Numbers are from one machine
and meant to show shape, not to be your benchmark. Reproduce with
`experiments/bench.py`.

## Latency: FLAT is linear, HNSW is flat

The clearest result, consistent across every run:

| Vectors | FLAT p50 | HNSW p50 | speedup |
|--------:|---------:|---------:|--------:|
| 1,000   | 0.18 ms  | 0.12 ms  | 1.5x |
| 5,000   | 0.58 ms  | 0.13 ms  | 4.3x |
| 10,000  | 1.2 ms   | 0.14 ms  | 9x |
| 25,000  | 4.1 ms   | 0.16 ms  | 26x |
| 50,000  | 8.4 ms   | 0.18 ms  | 46x |
| 100,000 | 17 ms    | 0.2 ms   | 74x |

FLAT scales with N, exactly as brute force must: double the vectors,
double the query time. HNSW stays near flat, a couple hundred
microseconds whether it holds a thousand vectors or a hundred thousand.

The crossover is around 1,000 to 2,000 vectors, and it is not close by
10,000. Below a few thousand, FLAT is fine and its answers are exact.
For a family photo library or a door that knows the household, use FLAT
and skip this whole discussion. Past ten thousand faces, HNSW is the
difference between a millisecond and a stall.

## Recall: the subtle part, and why recognition does not care

Approximate indexes trade accuracy for speed. Measuring that trade
honestly took three tries, and the failures were the interesting part.

**First surprise: random vectors are a trap.** My first data was random
Gaussian points. HNSW recall looked catastrophic, near zero. The cause
is the curse of dimensionality: in 512 dimensions, random points are all
nearly equidistant, so there is no neighborhood structure for the graph
to exploit and no meaningful "nearest" for FLAT to find either. Real
face embeddings are not random. The same person's photos cluster
tightly, and that structure is exactly what HNSW navigates. Benchmark on
data shaped like your real data or you will measure noise.

**Second surprise: recall@1 is the wrong metric for recognition.** With
realistic clusters (500 identities, 100 photos each, queries are
held-out photos of known people), at N=50,000:

| EF_RUNTIME | recall@1 | named right person | p50 ms |
|-----------:|---------:|-------------------:|-------:|
| default    | 0.33     | 0.99               | 0.18 |
| 20         | 0.33     | 1.00               | 0.20 |
| 50         | 0.33     | 1.00               | 0.31 |
| 100        | 0.33     | 1.00               | 0.52 |
| 200        | 0.36     | 1.00               | 2.95 |

FLAT named the right person 200 out of 200, exactly. HNSW found the
exact nearest vector only a third of the time, yet named the right
person 99 to 100 percent of the time. Those are not in tension. When a
person has a hundred photos in the index, HNSW returning photo #47
instead of the true-nearest photo #12 is a recall@1 "miss" and a
recognition success: both are the same face. You do not need the nearest
vector. You need a vector from the right cluster, and clusters are dense.

`EF_RUNTIME` is the dial if you ever do need exact-nearest: higher
values widen the graph search, buying recall at a latency cost. For
recognition you can leave it at the default and pay nothing.

## Build cost, the one place HNSW is worse

HNSW builds a navigation graph as it ingests, so loading is slower than
FLAT, and slowest exactly when clusters are tight (many near-equidistant
candidates to link). At 50,000 tightly clustered vectors, FLAT ingest
was about a second; HNSW took several. This rarely matters (you build
once and query forever), but if you rebuild a large index constantly, it
is real.

## The one-paragraph answer

Under a few thousand vectors, use FLAT: exact answers, no tuning, and the
latency is already sub-millisecond. Past ten thousand, use HNSW: query
time stays flat while FLAT climbs linearly, and for recognition the
approximation costs you nothing, because naming the right person only
needs a neighbor from the right cluster, not the single closest vector.
`valkey-faces` ships HNSW because it is built to scale past the demo, but
for the demo itself, FLAT would have been the honest choice.
