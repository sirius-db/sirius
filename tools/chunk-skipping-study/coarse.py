import os, numpy as np, json, datetime
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
z=np.load(SP+'/li10.npz'); dicts=json.load(open(SP+'/li10_dicts.json'))
cols={k:z[k] for k in z.files}
def D(s): return (datetime.date.fromisoformat(s)-datetime.date(1970,1,1)).days
for cn,vals in dicts.items():
    order=np.argsort(np.array(vals)); remap=np.empty(len(vals),dtype=np.int64); remap[order]=np.arange(len(vals))
    cols[cn]=remap[cols[cn]]; dicts[cn]=[vals[i] for i in order]
def C(cn,v): return dicts[cn].index(v)
Q={'q1':[('l_shipdate',None,D('1995-08-19'))],
   'q3':[('l_shipdate',D('1995-03-26'),None)],
   'q6':[('l_quantity',None,2399),('l_discount',2,4),('l_shipdate',D('1997-01-01'),D('1997-12-31'))],
   'q7':[('l_shipdate',D('1995-01-01'),D('1996-12-31'))],
   'q10':[('l_returnflag',C('l_returnflag','R'),)*1+ (C('l_returnflag','R'),)],
   'q12':[('l_receiptdate',D('1994-01-01'),D('1994-12-31')),('l_shipdate',None,D('1994-12-31')),('l_commitdate',None,D('1994-12-31'))],
   'q14':[('l_shipdate',D('1994-08-01'),D('1994-08-31'))],
   'q15':[('l_shipdate',D('1993-05-01'),D('1993-07-31'))],
   'q20':[('l_shipdate',D('1993-01-01'),D('1993-12-31'))]}
Q['q10']=[('l_returnflag',C('l_returnflag','R'),C('l_returnflag','R'))]
perm=np.argsort(cols['l_shipdate'],kind='stable')
Cs={k:v[perm] for k,v in cols.items()}
def mm(a,cs):
    n=(len(a)//cs)*cs; m=a[:n].reshape(-1,cs); mn,mx=m.min(1),m.max(1)
    if n<len(a): mn=np.append(mn,a[n:].min()); mx=np.append(mx,a[n:].max())
    return mn,mx
N=len(perm)
CH=[1024,8192,16384,65536,262144,1048576,2097152,4194304]
print(f"sorted by l_shipdate, SF10 lineitem N={N} (chunk/N in parens; SF1000 pin chunk ~= 2% of table)")
print(f"{'chunk':>9}{'%of tbl':>9}"+''.join(f"{q:>7}" for q in Q)+f"{'MEAN':>7}")
for cs in CH:
    M={k:mm(v,cs) for k,v in Cs.items()}
    nch=len(M['l_shipdate'][0]); row=[]
    for q,ps in Q.items():
        keep=np.ones(nch,bool)
        for cn,lo,hi in ps:
            mn,mx=M[cn]; h=np.ones(nch,bool)
            if lo is not None: h&=(mx>=lo)
            if hi is not None: h&=(mn<=hi)
            keep&=h
        row.append(100*(1-keep.mean()))
    print(f"{cs:>9}{100*cs/N:>8.2f}%"+''.join(f"{r:>7.1f}" for r in row)+f"{np.mean(row):>7.1f}")
