import os
import duckdb, numpy as np, time, json
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
c=duckdb.connect(); c.execute("SET threads=16")
t0=time.time()
tb=c.execute(f"select * from '{SP}/li10.parquet'").arrow().read_all()
print('load %.1fs'%(time.time()-t0), tb.num_rows)
D={}
for name in tb.column_names:
    a=tb.column(name).combine_chunks()
    if name in ('l_returnflag','l_shipmode','l_shipinstruct'):
        d=a.dictionary_encode()
        D[name]=('dict', np.asarray(d.indices).astype(np.int32), [str(x) for x in d.dictionary])
    else:
        D[name]=('num', np.asarray(a).astype(np.int64), None)
np.savez(SP+'/li10.npz', **{k:v[1] for k,v in D.items()})
json.dump({k:v[2] for k,v in D.items() if v[2]}, open(SP+'/li10_dicts.json','w'))
print('ok %.1fs'%(time.time()-t0))
