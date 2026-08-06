#!/usr/bin/env python3
# Clean recall study: well-populated clusters (100 pts/identity), recall@1 =
# "did HNSW find the exact nearest", right-person = "did it name the right
# identity". N=50000, EF_RUNTIME dial.
import time, numpy as np, redis
r = redis.Redis(port=16401); D=512; N=50000; PPC=100; NQ=200
rng = np.random.default_rng(7)
ncl = N//PPC
cents = rng.standard_normal((ncl,D)).astype('f4'); cents/=np.linalg.norm(cents,axis=1,keepdims=True)
labels = np.repeat(np.arange(ncl), PPC)
X = cents[labels] + 0.05*rng.standard_normal((N,D)).astype('f4'); X/=np.linalg.norm(X,axis=1,keepdims=True)
qcl = rng.integers(0,ncl,NQ)
Q = cents[qcl] + 0.05*rng.standard_normal((NQ,D)).astype('f4'); Q/=np.linalg.norm(Q,axis=1,keepdims=True)
pack = lambda v: v.astype('<f4').tobytes()

def drop(i):
    try: r.execute_command('FT.DROPINDEX', i)
    except: pass
def flush(p):
    c=0
    while True:
        c,k=r.scan(c,match=p+'*',count=3000)
        if k: r.delete(*k)
        if c==0: break
def load(p,V):
    pi=r.pipeline(transaction=False)
    for i,v in enumerate(V):
        pi.hset(f'{p}{i}',mapping={'v':pack(v)})
        if i%3000==0: pi.execute(); pi=r.pipeline(transaction=False)
    pi.execute()
def knn(idx,q,k,ef=None):
    cl=f'*=>[KNN {k} @v $q'+(f' EF_RUNTIME {ef}' if ef else '')+']'
    res=r.execute_command('FT.SEARCH',idx,cl,'PARAMS','2','q',pack(q),'NOCONTENT','DIALECT','2','LIMIT','0',str(k))
    return [int(x.split(b':')[1]) for x in res[1:]]
def waitn(idx,n):
    while True:
        info=r.execute_command('FT.INFO',idx)
        for i,t in enumerate(info):
            if t in (b'num_docs','num_docs'):
                if int(info[i+1])>=n: return
        time.sleep(0.1)

gt1 = np.argmax(Q@X.T, axis=1)   # exact nearest id per query (numpy ground truth)
print(f"N={N}, {ncl} identities x {PPC} photos, {NQ} held-out queries")

drop('rf'); flush('rf:'); load('rf:',X)
r.execute_command('FT.CREATE','rf','ON','HASH','PREFIX','1','rf:','SCHEMA','v','VECTOR','FLAT','6','TYPE','FLOAT32','DIM',str(D),'DISTANCE_METRIC','COSINE'); waitn('rf',N)
fcorrect=sum(labels[knn('rf',q,1)[0]]==labels[int(g)] for q,g in zip(Q,gt1))
print(f"FLAT (exact): named the right person {fcorrect}/{NQ}")
drop('rf'); flush('rf:')

drop('rh'); flush('rh:'); load('rh:',X)
r.execute_command('FT.CREATE','rh','ON','HASH','PREFIX','1','rh:','SCHEMA','v','VECTOR','HNSW','6','TYPE','FLOAT32','DIM',str(D),'DISTANCE_METRIC','COSINE'); waitn('rh',N)
print(f"{'EF_RUNTIME':>10} {'recall@1':>9} {'right person':>13} {'p50 ms':>8}")
for ef in [None,10,20,50,100,200]:
    knn('rh',Q[0],1,ef); lat=[]; r1=0; rp=0
    for q,g in zip(Q,gt1):
        t=time.perf_counter(); got=knn('rh',q,1,ef); lat.append((time.perf_counter()-t)*1000)
        if got[0]==int(g): r1+=1
        if labels[got[0]]==labels[int(g)]: rp+=1
    lat.sort()
    print(f"{str(ef):>10} {r1/NQ:>9.3f} {rp/NQ:>13.3f} {lat[len(lat)//2]:>8.2f}")
drop('rh'); flush('rh:')
