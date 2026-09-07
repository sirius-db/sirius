import os, numpy as np, json, datetime
# Pruning under different pin-time clustering strategies, at pin-chunk vs metadata-group
# granularity. SF10 lineitem (60M rows), real TPC-H scan predicates.
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
z=np.load(SP+'/li10.npz'); dicts=json.load(open(SP+'/li10_dicts.json'))
cols={k:z[k] for k in z.files}
for cn,vals in dicts.items():
    o=np.argsort(np.array(vals)); r=np.empty(len(vals),np.int64); r[o]=np.arange(len(vals))
    cols[cn]=r[cols[cn]]; dicts[cn]=[vals[i] for i in o]
def C(cn,v): return dicts[cn].index(v)
def D(s): return (datetime.date.fromisoformat(s)-datetime.date(1970,1,1)).days
Q={'q1':[('l_shipdate',None,D('1995-08-19'))],
   'q3':[('l_shipdate',D('1995-03-26'),None)],
   'q6':[('l_quantity',None,2399),('l_discount',2,4),('l_shipdate',D('1997-01-01'),D('1997-12-31'))],
   'q7':[('l_shipdate',D('1995-01-01'),D('1996-12-31'))],
   'q10':[('l_returnflag',C('l_returnflag','R'),C('l_returnflag','R'))],
   'q12':[('l_receiptdate',D('1994-01-01'),D('1994-12-31')),('l_shipdate',None,D('1994-12-31')),('l_commitdate',None,D('1994-12-31'))],
   'q14':[('l_shipdate',D('1994-08-01'),D('1994-08-31'))],
   'q15':[('l_shipdate',D('1993-05-01'),D('1993-07-31'))],
   'q20':[('l_shipdate',D('1993-01-01'),D('1993-12-31'))]}
N=len(cols['l_shipdate']); rng=np.random.default_rng(11)
KEY='l_shipdate'
PINCHUNK=int(N*0.031)   # SF1000 lineitem pin chunk = 3.1% of the table (8 GB batch)

def perm_global_sort():   return np.argsort(cols[KEY], kind='stable')
def perm_range_partition(K):
    """Assign rows to K key-range buckets (equi-depth); random order inside a bucket.
    One hash/partition pass, no comparison sort."""
    k=cols[KEY]; edges=np.quantile(k, np.linspace(0,1,K+1)[1:-1])
    b=np.searchsorted(edges, k, side='right')
    return np.argsort(b, kind='stable')          # stable: order within bucket = original (random)
def perm_local_sort(chunk):
    """Leave rows in their arriving pin chunk; sort only inside each chunk.
    Fits on one GPU, embarrassingly parallel, no shuffle."""
    p=np.arange(N)
    for s in range(0,N,chunk):
        seg=p[s:s+chunk]; seg[:]=seg[np.argsort(cols[KEY][seg], kind='stable')]
    return p

MODES={'0. none (as generated)': None,
       '1. local sort in pin chunk': perm_local_sort(PINCHUNK),
       '2. range-partition to pin chunks': perm_range_partition(max(N//PINCHUNK,1)),
       '3. 2 + local sort (= global sort)': perm_global_sort()}

def prune(Cs,cs):
    out={}
    M={}
    for k,v in Cs.items():
        n=(len(v)//cs)*cs; m=v[:n].reshape(-1,cs); mn,mx=m.min(1),m.max(1)
        if n<len(v): mn=np.append(mn,v[n:].min()); mx=np.append(mx,v[n:].max())
        M[k]=(mn,mx)
    nch=len(M[KEY][0])
    for q,ps in Q.items():
        keep=np.ones(nch,bool)
        for cn,lo,hi in ps:
            mn,mx=M[cn]; h=np.ones(nch,bool)
            if lo is not None: h&=(mx>=lo)
            if hi is not None: h&=(mn<=hi)
            keep&=h
        out[q]=100*(1-keep.mean())
    return out

GRAN=[('group G=8 (8192 rows)',8192),('group G=64 (65536)',65536),(f'pin chunk ({PINCHUNK} rows)',PINCHUNK)]
print(f"SF10 lineitem, N={N}; pin chunk sized to SF1000's 3.1% of table\n")
for gname,cs in GRAN:
    print(f"--- granularity: {gname}")
    print(f"{'clustering':<36}"+''.join(f"{q:>7}" for q in Q)+f"{'MEAN':>7}")
    for mname,perm in MODES.items():
        Cs={k:(v if perm is None else v[perm]) for k,v in cols.items()}
        r=prune(Cs,cs)
        print(f"{mname:<36}"+''.join(f"{r[q]:>7.1f}" for q in Q)+f"{np.mean(list(r.values())):>7.1f}")
    print()
