# Your cache has been a vector database all along

This is a post about teaching my laptop to greet me by name. Mostly,
though, it is a post about how little infrastructure that actually
requires in 2026, and about a fork that keeps earning its keep.

Last month I wrote about pushing Valkey to 200 Gbps on large GETs.
Line-rate benchmarks are one kind of fun. This weekend I wanted the other
kind: point my camera at my face, and have the terminal say
`+ Khawaja entered`. The stack for that used to be a vector database, a
Python environment, a GPU, and a weekend of glue. Here is what it is now:
one Rust binary, two small models, and the Valkey instance I already run.

## Vectors are just bytes

The insight that makes this whole thing small is almost embarrassing. A
face embedding is 512 floats. Packed little-endian, that is 2 KB. Valkey
stores it the way it stores everything else:

```
HSET face:khawaja:0 v <2KB of packed f32> name "Khawaja"
```

The search module (in the `valkey/valkey-bundle` image, one `docker run`)
turns those hashes into an HNSW index:

```
FT.CREATE faces ON HASH PREFIX 1 face: SCHEMA v VECTOR HNSW 6
  TYPE FLOAT32 DIM 512 DISTANCE_METRIC COSINE
```

And recognition is one command:

```
FT.SEARCH faces "*=>[KNN 1 @v $q AS dist]" PARAMS 2 q <query> DIALECT 2
```

No new database. No new operational surface. The same process that holds
your sessions and your queues will tell you, in about a millisecond,
whose face this is. Enrolling a new person is an HSET; the very next
query can name them. Firing someone from the index is a DEL. CRUD on
people. I find that genuinely delightful.

## The pipeline

ffmpeg reads the camera and rewrites a single JPEG atomically, twice a
second. A 610-line Rust binary watches that file's mtime, detects faces
with SeetaFace, embeds each one with a 13.6 MB ArcFace model running
under tract, and asks Valkey the only question that matters. Every piece
is pure Rust; `cargo build` is the entire setup. No Python, no GPU, no
cloud call.

The numbers, measured on my machine and not rounded for drama: an empty
frame costs 12 ms. A frame with a face costs 54 ms, and the KNN query
inside that is about one millisecond of it. The embedding model parses
once at startup; after that the camera is the bottleneck, and the camera
runs at 2 fps because faces do not move faster than that.

Accuracy has the same honest shape every embedding system does. Same
person across different photos: cosine similarity 0.35 to 0.93.
Different people: 0.16 and below. The threshold sits at 0.30, in the
gap, and you should measure your own gap rather than trust mine. The
detector is frontal-only. Look away and you are a stranger; that is the
scope, stated plainly.

## Why this matters more than a party trick

I keep coming back to one theme: consistency compounds. The Valkey
community shipped I/O threading until GETs hit line rate, and it shipped
a search module until a cache could answer nearest-neighbor questions.
Neither happened in one heroic release. The result is that the boring
instance you already operate keeps absorbing jobs that used to demand a
specialist system: pub/sub, streams, JSON, and now vectors.

That is what open source buys everyone. Not a feature checkbox, but the
steady collapse of your architecture diagram.

The code is at [github.com/ksmotiv8/valkey-faces](https://github.com/ksmotiv8/valkey-faces):
clone it, enroll your face from your own camera, and watch your terminal
learn your name in under ten minutes. If it does not, file an issue. And
if you want to push on what Valkey can hold next, come find us in the
Valkey Slack. The next capability is going to come from somebody who
showed up consistently.
