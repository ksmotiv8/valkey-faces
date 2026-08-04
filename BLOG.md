# Your cache has been a vector database all along

This is a post about teaching my laptop to greet me by name. Mostly,
though, it is a post about Valkey search, because the face part turned
out to be the easy part.

Last month I wrote about pushing Valkey to 200 Gbps on large GETs.
Line-rate benchmarks are one kind of fun. This weekend I wanted the
other kind: point my camera at my face and have the terminal print
`+ Khawaja entered`. That used to take a dedicated vector database, a
Python environment, a GPU, and a weekend of glue. It now takes one Rust
binary of about 600 lines, two small models, and the Valkey instance I
already run. The interesting question is not how the face models work.
It is what happened to the vector database.

## The centerpiece: Valkey search

The Valkey project ships a search module, and the
`valkey/valkey-bundle` image has it loaded out of the box. Getting a
vector database is now:

```bash
docker run -d --name valkey --rm -p 6379:6379 valkey/valkey-bundle
```

Everything that follows rests on one idea, and it is almost
embarrassing once you see it: a vector is just bytes. A face embedding
is 512 floats. Packed little-endian, that is 2 KB, and Valkey stores it
the way it stores everything else, as a field on a hash:

```
HSET face:khawaja:0 v <2KB packed f32> name "Khawaja"
```

The index declaration reads like a schema because it is one:

```
FT.CREATE faces ON HASH PREFIX 1 face: SCHEMA v VECTOR HNSW 6
  TYPE FLOAT32 DIM 512 DISTANCE_METRIC COSINE
```

Unpack that line and you have the whole design. `ON HASH PREFIX 1
face:` means every hash whose key starts with `face:` joins the index
automatically, at write time, no rebuild step. `VECTOR HNSW` picks the
index structure: HNSW keeps shortcut links between neighboring points
so a query hops toward its own neighborhood and inspects a tiny
fraction of the data instead of scanning all of it. That is the
difference between nearest-neighbor at five entries and at five
million. `DISTANCE_METRIC COSINE` decides what "near" means.

And the query, the entire "whose face is this" question, is one
command:

```
FT.SEARCH faces "*=>[KNN 1 @v $q AS dist]"
  PARAMS 2 q <packed query vector> RETURN 2 name dist DIALECT 2
```

It answers in about a millisecond. Inserts are incremental, so a hash
written this second is searchable the next. Deletes are just DEL. The
operational story is the story: there is no second system. The process
that holds your sessions and your queues will also tell you whose face
is at the door.

One trap deserves its own paragraph, because it will bite you exactly
once. Embedding papers speak in similarity, where 1.0 means identical.
Valkey returns cosine distance, where 0.0 means identical. Similarity
is 1 minus distance. Port a threshold between systems without
converting and one of two things happens: everything matches, or
nothing does. Both failure modes are silent. In this repo the
conversion lives in one line next to the query, where nobody can miss
it.

## Now it needs something to search

A vector index is only as interesting as its vectors, so let me compress
the face recognition background into the four facts that matter.

First, face recognition is two problems, not one. Detection answers
"where are the faces in this image" and knows nothing about identity.
Recognition answers "whose face is this" and never touches raw pixels.

Second, recognition works on embeddings. A pretrained model maps any
face to a point in 512-dimensional space, trained so the same person's
photos land close together and different people land far apart. No
training happens on your machine, ever. "Learning" a person means
storing one point with a name. This is the reframing that shrinks the
whole field: face recognition stops being an ML problem and becomes a
data problem, and the data problem is exactly the nearest-neighbor
question Valkey just answered.

Third, the models are small and pure Rust runs them. Detection is
SeetaFace via the `rustface` crate, a 1.2 MB model:

```rust
let mut detector = rustface::create_detector_with_model(model);
detector.set_min_face_size(20);   // >= 20, or the pyramid panics
detector.set_score_thresh(2.0);
```

Embedding is w600k_mbf, a MobileFaceNet trained with ArcFace on 600,000
identities, a 13.6 MB ONNX file executed by `tract`. Pixels in,
512 floats out, then one line of quiet importance:

```rust
let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
// L2-normalize: cosine similarity becomes a plain dot product
```

Unit-length vectors are what make COSINE in the index mean exactly what
you think it means.

Fourth, preprocessing is a contract, not a suggestion. Every face is
cropped the same way (box expanded 25 percent to a square, resized to
112x112) by one shared function, because embeddings only compare if
every image took the identical path. Enroll with one crop and recognize
with another, and your accuracy quietly falls apart.

## Registration is an HSET

Here is where the two halves meet, and where the old workflow collapses
into a cache write:

```rust
pub fn enroll(con: &mut Connection, name: &str, e: &[f32]) -> Result<String> {
    let key = format!("face:{}:{}", slug(name), next_free_n);
    con.hset_multiple(&key, &[("v", pack_f32le(e)), ("name", name)])?;
    Ok(key)
}
```

Detect the largest face in a photo, embed it, HSET it. The very next
query can name that person. Enroll a second photo and you get
`face:khawaja:1` next to `face:khawaja:0`; KNN returns the nearest
entry, so extra photos widen the poses you match under. Removing
someone is a DEL. Listing the enrolled is a SCAN. CRUD on people. I
find that genuinely delightful.

## The live loop

ffmpeg reads the camera and atomically rewrites one JPEG, twice a
second (`-update 1 -atomic_writing 1`, so the watcher never sees half a
frame):

```bash
ffmpeg -f avfoundation -framerate 30 -i "0" \
  -vf "fps=2,scale=512:-2" \
  -q:v 4 -f image2 -update 1 -atomic_writing 1 /tmp/tap.jpg
```

The binary polls that file's mtime, and on each new frame runs detect,
embed, KNN, then prints transitions: who entered, who left. The
numbers, measured and not rounded for drama: an empty frame costs
12 ms, a frame with a face costs 54 ms, and the Valkey query inside
that is about one millisecond. The cache is the fastest thing in the
pipeline. It usually is.

Accuracy has the honest shape every embedding system does. On this
repo's test set, the same person across different photos scores 0.35 to
0.93 similarity; different people score 0.16 and below. The threshold
sits at 0.30, in the gap. Swap models or preprocessing and you must
measure your own gap; thresholds do not travel. And SeetaFace is
frontal-only: turn your head and you are a stranger. Small, fast,
dependency free, honestly scoped.

## Consistency compounds, again

The Valkey community shipped I/O threading release after release until
large GETs hit line rate. It shipped a search module until the cache
could answer nearest-neighbor questions. Neither happened in one heroic
version, and that is the point I keep returning to: consistency
compounds. The boring instance you already operate keeps absorbing jobs
that used to demand a specialist system: pub/sub, streams, JSON, and
now vectors.

That is what open source buys everyone. Not a feature checkbox, but the
steady collapse of your architecture diagram.

The code is at
[github.com/ksmotiv8/valkey-faces](https://github.com/ksmotiv8/valkey-faces):
clone it, enroll your face from your own camera, and watch your
terminal learn your name in under ten minutes. If it does not, file an
issue. And if you want to push on what Valkey can hold next, come find
us in the Valkey Slack. The next capability is going to come from
somebody who showed up consistently.
