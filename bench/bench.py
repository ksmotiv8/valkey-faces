#!/usr/bin/env python3
"""FLAT vs HNSW in valkey-search: where does the crossover happen?

512-D unit vectors (face-embedding shaped), KNN 10, cosine. For each N:
ingest into a FLAT and an HNSW index, time 100 held-out queries against
each, and score HNSW recall@10 against FLAT's exact results.

Usage: bench.py [redis_url] [sizes...]
"""
import sys, time, struct
import numpy as np
import redis

URL = sys.argv[1] if len(sys.argv) > 1 else "redis://127.0.0.1:16401"
SIZES = [int(s) for s in sys.argv[2:]] or [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
DIM, K, NQ = 512, 10, 100
rng = np.random.default_rng(7)

def unit(n):
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def pack(v):
    return v.astype("<f4").tobytes()

def create(r, name, prefix, algo):
    try: r.execute_command("FT.DROPINDEX", name)
    except redis.ResponseError: pass
    args = ["FT.CREATE", name, "ON", "HASH", "PREFIX", 1, prefix,
            "SCHEMA", "v", "VECTOR", algo, 6, "TYPE", "FLOAT32",
            "DIM", DIM, "DISTANCE_METRIC", "COSINE"]
    r.execute_command(*args)

def ingest(r, prefix, data):
    t0 = time.perf_counter()
    pipe = r.pipeline(transaction=False)
    for i, v in enumerate(data):
        pipe.hset(f"{prefix}{i}", mapping={"v": pack(v)})
        if i % 500