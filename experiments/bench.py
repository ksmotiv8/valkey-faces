#!/usr/bin/env python3
"""HNSW vs FLAT on CLUSTERED 512-D unit vectors (realistic for face embeddings:
many photos of fewer identities cluster tightly). Plus an EF_RUNTIME recall dial."""
import time, sys
import numpy as np
import redis

PORT = 16401
DIM = 512
K = 10
NQ = 100
SIZES = [1000, 5000, 10000, 25000, 50000, 100000]
CLUSTERS = 500          # identities; points cluster around these (face-like)
SEED = 42
r = redis.Redis(port=PORT)

def make_centroids(d, rng):
    c = rng.standard_normal((CLUSTERS, d)).astype(np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return c

def clustered(n, d, cents, rng):
    # points near shared centroids: queries are held-out photos of KNOWN people
    idx = rng.integers(0, CLUSTERS, n)
    v = cents[idx] + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def pack(v): return v.astype('<f4').tobytes()
def drop(idx):
    try: r.execute_command('FT.DROPINDEX', idx)
    except redis.ResponseError: pass
def flush_prefix(p):
    cur = 0
    while True:
        cur, keys = r.scan(cur, match=p+'*', count=2000)
        if keys: r.delete(*keys)
        if cur == 0: break
def load(prefix, vecs):
    pipe = r.pipeline(transaction=False)
    for i, v in enumerate(vecs):
        pipe.hset(f'{prefix}{i}', mapping={'v': pack(v)})
        if i % 2000 == 0: pipe.execute(); pipe = r.pipeline(transaction=False)
    pipe.execute()
def create(idx, prefix, algo):
    extra = ['M','16','EF_CONSTRUCTION','200'] if algo=='HNSW' else []
    r.execute_command('FT.CREATE', idx,'ON','HASH','PREFIX','1',prefix,'SCHEMA','v',
        'VECTOR',algo,str(6+len(extra)),'TYPE','FLOAT32','DIM',str(DIM),
        'DISTANCE_METRIC','COSINE',*extra)
def wait_indexed(idx, n):
    while True:
        info = r.execute_command('FT.INFO', idx); docs=None
        for i,tok in enumerate(info):
            if tok in (b'num_docs','num_docs') and i+1<len(info):
                try: docs=int(info[i+1])
                except: docs=None
                break
        if docs is None or docs>=n: return
        time.sleep(0.05)
def knn(idx, q, ef=None):
    clause = f'*=>[KNN {K} @v $q'+(f' EF_RUNTIME {ef}' if ef else '')+']'
    res = r.execute_command('FT.SEARCH', idx, clause,'PARAMS','2','q',pack(q),
        'NOCONTENT','DIALECT','2','LIMIT','0',str(K))
    return list(res[1:])
def ids_only(keys):
    return {k.split(b':')[1] if isinstance(k,bytes) else k.split(':')[1] for k in keys}

def bench_size(n, rng):
    cents = make_centroids(DIM, rng)
    base = clustered(n, DIM, cents, rng)
    qs = clustered(NQ, DIM, cents, rng)
    out={'n':n}
    for algo,idx,pfx in [('FLAT','i_flat','f:'),('HNSW','i_hnsw','h:')]:
        drop(idx); flush_prefix(pfx)
        t0=time.perf_counter(); load(pfx,base); create(idx,pfx,algo); wait_indexed(idx,n)
        out[f'{algo}_build']=time.perf_counter()-t0
        knn(idx,qs[0]); lat=[]; ids=[]
        for q in qs:
            t=time.perf_counter(); r_=knn(idx,q); lat.append((time.perf_counter()-t)*1000); ids.append(r_)
        lat.sort(); out[f'{algo}_p50']=lat[len(lat)//2]; out[f'{algo}_ids']=ids
        if algo=='FLAT': drop(idx); flush_prefix(pfx)  # keep HNSW for ef sweep at last size
    hits=tot=0
    for g,a in zip(out['FLAT_ids'],out['HNSW_ids']):
        gg=ids_only(g); hits+=len(gg&ids_only(a)); tot+=len(gg)
    out['recall']=hits/tot if tot else 0
    drop('i_hnsw'); flush_prefix('h:')
    del out['FLAT_ids'],out['HNSW_ids']
    return out

def ef_sweep(n, rng):
    print(f"\nEF_RUNTIME sweep at N={n} (HNSW recall/latency dial vs FLAT ground truth):")
    cents=make_centroids(DIM,rng); base=clustered(n,DIM,cents,rng); qs=clustered(NQ,DIM,cents,rng)
    drop('s_flat'); flush_prefix('sf:'); load('sf:',base); create('s_flat','sf:','FLAT'); wait_indexed('s_flat',n)
    gt=[knn('s_flat',q) for q in qs]
    drop('s_flat'); flush_prefix('sf:')
    drop('s_hnsw'); flush_prefix('sh:'); load('sh:',base); create('s_hnsw','sh:','HNSW'); wait_indexed('s_hnsw',n)
    print(f"  {'EF_RUNTIME':>10} {'recall@10':>9} {'p50 ms':>8}")
    for ef in [10,20,50,100,200,500]:
        knn('s_hnsw',qs[0],ef); lat=[]; hits=tot=0
        for q,g in zip(qs,gt):
            t=time.perf_counter(); a=knn('s_hnsw',q,ef); lat.append((time.perf_counter()-t)*1000)
            gg=ids_only(g); hits+=len(gg&ids_only(a)); tot+=len(gg)
        lat.sort()
        print(f"  {ef:>10} {hits/tot:>9.3f} {lat[len(lat)//2]:>8.2f}")
    drop('s_hnsw'); flush_prefix('sh:')

def main():
    rng=np.random.default_rng(SEED)
    print("Clustered data (500 identities, queries are held-out photos of known people). Lower is better for latency; recall is HNSW vs exact FLAT.")
    print(f"{'N':>7} {'FLAT p50':>9} {'HNSW p50':>9} {'speedup':>8} {'FLAT build':>10} {'HNSW build':>10} {'recall@10':>9}")
    for n in SIZES:
        o=bench_size(n,rng)
        sp=o['FLAT_p50']/o['HNSW_p50']
        print(f"{o['n']:>7} {o['FLAT_p50']:>8.2f}m {o['HNSW_p50']:>8.2f}m {sp:>7.1f}x "
              f"{o['FLAT_build']:>9.2f}s {o['HNSW_build']:>9.2f}s {o['recall']:>9.3f}")
        sys.stdout.flush()
    ef_sweep(50000, rng)

if __name__=='__main__': main()
