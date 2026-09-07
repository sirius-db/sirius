import os
import numpy as np, json, datetime, sys
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
z=np.load(SP+'/li10.npz'); dicts=json.load(open(SP+'/li10_dicts.json'))
cols={k:z[k] for k in z.files}
N=len(cols['l_orderkey'])
def D(s): return (datetime.date.fromisoformat(s)-datetime.date(1970,1,1)).days

# order-preserving dictionary codes for string cols
for cname,vals in dicts.items():
    order=np.argsort(np.array(vals))            # positions sorted by value
    remap=np.empty(len(vals),dtype=np.int64); remap[order]=np.arange(len(vals))
    cols[cname]=remap[cols[cname]]
    dicts[cname]=[vals[i] for i in order]       # now sorted
CARD={k:len(v) for k,v in dicts.items()}

def code(cn,v): return dicts[cn].index(v)

# predicates: list of (col, lo, hi) inclusive ranges; None = unbounded. equality => lo==hi
QUERIES={
 'q1' : [('l_shipdate',None,D('1995-08-19'))],
 'q3' : [('l_shipdate',D('1995-03-26'),None)],
 'q6' : [('l_quantity',None,2399),('l_discount',2,4),('l_shipdate',D('1997-01-01'),D('1997-12-31'))],
 'q7' : [('l_shipdate',D('1995-01-01'),D('1996-12-31'))],
 'q10': [('l_returnflag',code('l_returnflag','R'),code('l_returnflag','R'))],
 'q12': [('l_receiptdate',D('1994-01-01'),D('1994-12-31')),('l_shipdate',None,D('1994-12-31')),
         ('l_commitdate',None,D('1994-12-31'))],
 'q12b':[('l_shipmode',[code('l_shipmode','TRUCK'),code('l_shipmode','REG AIR')],'IN')],
 'q14': [('l_shipdate',D('1994-08-01'),D('1994-08-31'))],
 'q15': [('l_shipdate',D('1993-05-01'),D('1993-07-31'))],
 'q19': [('l_shipinstruct',code('l_shipinstruct','DELIVER IN PERSON'),code('l_shipinstruct','DELIVER IN PERSON')),
         ('l_shipmode',[code('l_shipmode','AIR'),code('l_shipmode','REG AIR')],'IN')],
 'q20': [('l_shipdate',D('1993-01-01'),D('1993-12-31'))],
}

LAYOUTS={'natural':None,
         'sort:shipdate':np.argsort(cols['l_shipdate'],kind='stable'),
         'sort:(returnflag,shipdate)':np.lexsort((cols['l_shipdate'],cols['l_returnflag'])),
         'sort:(shipmode,shipdate)':np.lexsort((cols['l_shipdate'],cols['l_shipmode'])),
        }

def chunk_minmax(a,cs):
    n=(len(a)//cs)*cs
    m=a[:n].reshape(-1,cs)
    mn,mx=m.min(1),m.max(1)
    if n<len(a):
        mn=np.append(mn,a[n:].min()); mx=np.append(mx,a[n:].max())
    return mn,mx

def chunk_bitset(a,cs,card):
    """returns bool matrix [nchunks, card] of present values"""
    nch=(len(a)+cs-1)//cs
    ch=np.arange(len(a))//cs
    out=np.zeros((nch,card),dtype=bool)
    out[ch,a]=True
    return out

CHUNKS=[1024,2048,4096,8192,16384,32768,65536,262144,1048576]
res={}
for lname,perm in LAYOUTS.items():
    C={k:(v if perm is None else v[perm]) for k,v in cols.items()}
    for cs in CHUNKS:
        mm={k:chunk_minmax(v,cs) for k,v in C.items()}
        bs={k:chunk_bitset(C[k],cs,CARD[k]) for k in dicts}
        nch=len(mm['l_shipdate'][0])
        for q,preds in QUERIES.items():
            for mode in ('minmax','minmax+bitset'):
                keep=np.ones(nch,dtype=bool)
                for p in preds:
                    cn=p[0]
                    if p[2]=='IN':
                        vals=p[1]
                        if mode=='minmax+bitset' and cn in bs:
                            hit=bs[cn][:,vals].any(1)
                        else:
                            mn,mx=mm[cn]; hit=np.zeros(nch,dtype=bool)
                            for v in vals: hit |= (mn<=v)&(v<=mx)
                    else:
                        lo,hi=p[1],p[2]
                        if mode=='minmax+bitset' and cn in bs and lo==hi:
                            hit=bs[cn][:,lo]
                        else:
                            mn,mx=mm[cn]
                            hit=np.ones(nch,dtype=bool)
                            if lo is not None: hit &= (mx>=lo)
                            if hi is not None: hit &= (mn<=hi)
                    keep &= hit
                res.setdefault((lname,cs,mode),{})[q]=1.0-keep.mean()
json.dump({f"{k[0]}|{k[1]}|{k[2]}":v for k,v in res.items()}, open(SP+'/sweep.json','w'), indent=1)

# print
qs=list(QUERIES)
for mode in ('minmax','minmax+bitset'):
    print('\n### mode =',mode,' (fraction of chunks pruned, SF10 lineitem 60M rows)')
    print(f"{'layout':<28}{'chunk':>8}"+''.join(f"{q:>7}" for q in qs)+f"{'MEAN':>7}")
    for lname in LAYOUTS:
        for cs in CHUNKS:
            r=res[(lname,cs,mode)]
            print(f"{lname:<28}{cs:>8}"+''.join(f"{100*r[q]:>7.1f}" for q in qs)+f"{100*np.mean([r[q] for q in qs]):>7.1f}")
