# valkey-faces

Local face recognition with Valkey as the vector store. Your camera in,
names out. One Rust binary (~600 lines), two small ML models, and a
Valkey instance doing KNN over 512-D face embeddings with
`FT.SEARCH ... KNN`. No cloud calls, no Python, no GPU.

```
camera ──ffmpeg──► frame tap ──► detect ──► embed ──► FT.SEARCH KNN ──► "+ Barack Obama entered"
                                (rustface)  (ArcFace         (Valkey HNSW,
                                             via tract)       cosine)
```

## Quickstart

```bash
git clone https://github.com/ksmotiv8/valkey-faces.git && cd valkey-faces

# 1. Valkey with the search module (plain valkey has no FT.* commands)
docker run -d --name valkey --rm -p 6379:6379 valkey/valkey-bundle

# 2. Models (~15 MB total; not committed - their own licenses)
./fetch-models.sh

# 3. Build (Rust; needs ffmpeg on PATH for `watch`)
cargo build --release
# if the build complains about rustc versions:
#   cargo update kstring@2.0.4 --precise 2.0.2

VF=./target/release/valkey-faces

# 4. Enroll someone and prove it
$VF enroll demo-faces/Barack_Obama.jpg "Barack Obama"
$VF recognize demo-faces/probe_obama.jpg     # different photo, same person

# 5. Live: the demo stream (no camera needed) ...
$VF watch --demo --duration 30

# ... or your actual camera
$VF watch                        # macOS: avfoundation device 0
$VF watch --camera /dev/video1   # Linux: pick your v4l2 device
```

Enroll yourself from your own camera:

```bash
ffmpeg -f avfoundation -i "0" -frames:v 1 me.jpg   # macOS
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 me.jpg   # Linux
$VF enroll me.jpg "Your Name"
$VF watch    # walk into frame; watch it greet you
```

## Commands

| Command | What it does |
|---|---|
| `enroll <img> <name>` | Detect the largest face, embed, `HSET face:<slug>:<n>` (multiple photos per person supported; nearest wins) |
| `recognize <img>` | Detect every face, KNN-match each, print names or "stranger" |
| `watch [--demo] [--camera] [--fps] [--duration]` | Live stream; prints `+ name entered` / `- name left` transitions |
| `list` | Who is enrolled, with entry counts |
| `forget <name>` | Delete all of a person's entries |

Global flags: `--url` (default `redis://127.0.0.1:6379`), `--threshold`
(default 0.30 cosine similarity), `--models` (default `models/`).

## How it works

- **Detection**: SeetaFace frontal detector via the pure-Rust `rustface`
  crate. Frontal faces only; that is the honest scope.
- **Embedding**: w600k_mbf (MobileFaceNet trained with ArcFace), a
  13.6 MB ONNX run by `tract`, pure Rust. 112x112 chip in, 512 floats
  out, L2-normalized. Parsed once per process; ~30 ms per face after.
- **Index**: `FT.CREATE faces ON HASH ... VECTOR HNSW ... DISTANCE_METRIC
  COSINE`. Each person is one or more hashes with a packed
  little-endian f32 `v` field and a `name` field. About 2 KB per face.
- **Matching**: `FT.SEARCH "*=>[KNN 1 @v $q AS dist]"`. Valkey returns
  cosine DISTANCE; similarity = 1 - dist. The 0.30 threshold was
  calibrated on this repo's test set: impostors measured <= 0.16,
  genuine pairs 0.35-0.93. Calibrate your own if you change models.
- **Live loop**: ffmpeg writes one JPEG atomically (`-update 1
  -atomic_writing 1`); the watcher polls its mtime natively in Rust and
  recognizes in-process. Measured: ~12 ms for an empty frame, ~54 ms
  per face, at 2 fps.

## Notes

- `demo-faces/` contains photos of public figures for testing only; all
  rights remain with their owners (see `demo-faces/NOTICE.md`).
- Code is MIT. The downloaded models keep their own licenses (the
  InsightFace embedder is distributed for research/educational use).
- Curious when HNSW beats brute-force FLAT? See
  [EXPERIMENTS.md](EXPERIMENTS.md): the latency crossover is ~1-2k
  vectors, and for recognition HNSW's approximation costs nothing.
- Born as a module of the
  [momento-face-workshop](https://github.com/ksmotiv8/momento-face-workshop),
  where an agent-driven curriculum builds this system step by step.
