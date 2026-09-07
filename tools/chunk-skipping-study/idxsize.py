import os
import numpy as np, json, zlib, lzma
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
z=np.load(SP+'/li10.npz'); dicts=json.load(open(SP+'/li10_dicts.json'))
cols={k:z[k] for k in z.files}
N=len(cols['l_orderkey'])
# native widths of the full 16-col lineitem (SF1000 pinned, uncompressed): from metadata
WIDTH={'l_orderkey':8,'l_partkey':4,'l_suppkey':4,'l_linenumber':4,'l_quantity':16,'l_extendedprice':16,
 'l_discount':16,'l_tax':16,'l_returnflag':1,'l_linestatus':1,'l_shipdate':4,'l_commitdate':4,
 'l_receiptdate':4,'l_shipinstruct':1,'l_shipmode':1,'l_comment':1}
NCOL=len(WIDTH)
ROWBYTES=sum(WIDTH.values())   # rough fixed part; strings dict-coded to 1B
print('modelled bytes/row (dict-coded):',ROWBYTES)

def bitsize(a):
    """bits needed for FOR+bitpack of an int array"""
    r=int(a.max()-a.min())
    return (r.bit_length() or 1)

perm_sorted=np.argsort(cols['l_shipdate'],kind='stable')
print(f"\n{'chunk':>9}{'chunks/1e9row':>14}{'idx B/chunk':>13}{'idx MB @SF1000':>16}{'% of data':>11}{'idx zstd MB':>13}{'zstd %':>9}")
for cs in [1024,2048,4096,8192,16384,32768,65536,262144,1048576]:
    # raw index: min+max at native width + 4B null count, per column per chunk
    per_chunk=sum(2*w+4 for w in WIDTH.values()) + 16   # +16B chunk offset/length
    nch_sf1000 = (6_000_000_000+cs-1)//cs
    idx_mb = per_chunk*nch_sf1000/1e6
    data_mb = 6_000_000_000*ROWBYTES/1e6
    # compressed estimate: measure on real sorted+natural shipdate/orderkey mins
    blobs=b''
    for name,arr in cols.items():
        for perm in (None,):
            a=arr
            n=(len(a)//cs)*cs
            m=a[:n].reshape(-1,cs)
            mn,mx=m.min(1),m.max(1)
            blobs+=np.diff(mn,prepend=mn[:1]).astype(np.int64).tobytes()
            blobs+=(mx-mn).astype(np.int64).tobytes()
    ratio=len(zlib.compress(blobs,6))/len(blobs)
    print(f"{cs:>9}{nch_sf1000:>14}{per_chunk:>13}{idx_mb:>16.1f}{100*idx_mb/data_mb:>11.3f}{idx_mb*ratio:>13.1f}{100*ratio:>9.1f}")
