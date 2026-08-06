# HNSW vs FLAT: when does the index earn its keep?

Valkey vector search offers two index types. FLAT is exact brute force:
every query compares against every stored vector. HNSW is an approximate
graph index. `valkey-faces` uses HNSW, but for seven faces that is
overkill, so the honest question is: at what scale does HNSW start to
win, and what do you give up?

I measured it. 512-D unit vectors (face-embedding shaped: points cluster
around identities, queries are held-out photos of known people), cosine
distance, KNN, on `valkey/valkey-bundle`. Numbers are from one machine
and meant to show shape, not to be your benchmark. Reproduce with the
scripts in `experiments/`.

## Latency: FLAT is linear, HNSW is flat

FLAT scales with N; HNSW barely moves. Medians over thousands of queries:

| Vectors | FLAT p50 | HNSW p50 | median speedup |
|--------:|---------:|---------:|--------:|
| 100     | 0.077 ms | 0.084 ms | FLAT wins |
| 500     | 0.127 ms | 0.092 ms | 1.4x |
| 1,000   | 0.173 ms | 0.104 ms | 1.7x |
| 10,000  | 1.20 ms  | 0.133 ms | 9x |
| 50,000  | 8.33 ms  | 0.147 ms | 57x |

The crossover is low: around 250 vectors on the median. FLAT wins at 100,
they tie in the low hundreds, and HNSW wins everywhere above. FLAT's cost
doubles with the data; HNSW's holds near 0.15 ms whether it stores a
hundred vectors or fifty thousand.

## But look at the tails before you trust the median

p50 flatters HNSW. The picture at p99 and p999 is more honest:

| N | FLAT p50 / p99 / p999 | HNSW p50 / p99 / p999 |
|--:|:--|:--|
| 100    | 0.077 / 0.110 / 0.80 | 0.084 / 0.139 / 0.90 |
| 500    | 0.127 / 0.169 / 0.90 | 0.092 / 0.141 / 0.89 |
| 1,000  | 0.173 / 0.226 / 1.01 | 0.104 / 0.159 / 0.95 |
| 10,000 | 1.20 / 1.97 / 2.10   | 0.133 / 2.26 / 3.84 |
| 50,000 | 8.33 / 9.04 / 9.55   | 0.147 / 2.58 / 3.93 |

Two things fall out. FLAT's tail is tight: p50 and p999 nearly touch
(8.3 to 9.5 ms at 50k) because brute force does identical work every
query. HNSW's tail is fat relative to its median (0.15 ms to 3.9 ms, a
26x spread) because graph traversal is variable; some queries wander
farther than others. So at 50k HNSW is 57x faster at the median but only
about 2.4x faster at p999. If your SLA is written on the tail, plan
against that smaller number, not the headline.

At small N the ~0.8-0.9 ms p999 is identical for both, because it is not
the index at all. It is the redis round-trip, scheduling, and GC. Below
a thousand vectors you are measuring the harness, not the algorithm.

## Recall: high when small, approximate when large, right where it counts

Approximate indexes trade accuracy for speed. Three columns tell the
story: recall@1 (did HNSW return the exact nearest vector), recall@10
(overlap with the true top ten), and right-person (did it name the
correct identity, which is the only thing recognition asks).

| Vectors | recall@1 | recall@10 | right person |
|--------:|---------:|----------:|-------------:|
| 100     | 0.836    | 0.837     | 1.000 |
| 500     | 0.942    | 0.912     | 1.000 |
| 1,000   | 0.934    | 0.908     | 1.000 |
| 10,000  | 0.226    | 0.248     | 1.000 |
| 50,000  | 0.356    | 0.343     | 0.992 |

At small scale HNSW is nearly exact: 84 to 94 percent recall@1. As the
index grows, exact-nearest recall falls hard, because a bigger, denser
graph has more near-duplicate vectors for the search to lose the true
nearest among. And it does not matter. Right-person stays at 0.99 to
1.00 across every size. When an identity has a hundred photos in the
index, HNSW returning photo #47 instead of the true-nearest photo #12 is
a recall@1 miss and a recognition success: both are the same face. You
do not need the nearest vector. You need a vector from the right
cluster, and clusters are dense. `EF_RUNTIME` is the dial if you ever
need exact-nearest back (higher widens the graph search, buying recall
for latency), but recognition can leave it at the default and pay
nothing.

## Two methodology traps I hit, because they will bite you too

**Random vectors are a lie.** My first data was random Gaussian points,
and HNSW recall looked catastrophic, near zero. The curse of
dimensionality: in 512 dimensions random points are all nearly
equidistant, so there is no neighborhood for the graph to exploit and no
meaningful nearest for FLAT to find either. Real embeddings cluster.
Benchmark on data shaped like your real data or you measure noise.

**recall@1 is the wrong metric for recognition.** Chasing it would have
sent me tuning `EF_RUNTIME` up and paying latency for an accuracy the
application does not use. The metric that matches the task, right-person,
was already at 1.0.

## Build cost, the one place HNSW is plainly worse

HNSW builds a navigation graph as it ingests, so loading is slower than
FLAT, and slowest when clusters are tight (many near-equidistant
candidates to link). At 50,000 tightly clustered vectors FLAT ingest was
about a second; HNSW took several. You build once and query forever, so
this rarely matters, but a constantly rebuilt large index would feel it.

## The one-paragraph answer

Under a couple hundred vectors, use FLAT: exact, no tuning, already
sub-millisecond, and its predictable tail can beat HNSW's. Past a few
thousand, use HNSW: the median stays flat while FLAT climbs linearly,
and for recognition the approximation costs nothing, because naming the
right person needs a neighbor from the right cluster, not the single
closest vector. Watch the p999 if you live there. `valkey-faces` ships
HNSW because it is built to scale past the demo. For the demo itself,
FLAT would have been the honest choice.
