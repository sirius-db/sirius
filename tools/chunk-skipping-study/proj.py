import os
import json
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
TAG=os.environ.get('DATASET','/datasets/tpch_sf1000').rstrip('/').split('/')[-1]
SP=SP+'/'+TAG
os.makedirs(SP, exist_ok=True)
rep=json.load(open(SP+'/prune_report.json')); sw=json.load(open(SP+'/sweep.json'))
for layout in ('sort:shipdate','sort:(returnflag,shipdate)','sort:(shipmode,shipdate)'):
  for cs in (8192,65536,262144,1048576):
    k=f"{layout}|{cs}|minmax+bitset"; s=sw[k]
    # combine q12 (dates) and q12b (shipmode) conjunctively is not valid from marginals; use max as lower bound
    frac={q:s[q] for q in s}
    frac['q12']=max(s['q12'],s['q12b'])
    tot=0;pr=0;tl=0;pl=0
    for q,per in rep.items():
        for t,r in per.items():
            tot+=r['read_bytes']
            if t=='lineitem':
                tl+=r['read_bytes']; f=frac.get(q,0.0); pr+=r['read_bytes']*f; pl+=r['read_bytes']*f
    print(f"{layout:<28}{cs:>9}  lineitem scan {tl/1e9:6.1f} GB, skipped {pl/1e9:6.1f} GB ({100*pl/tl:4.1f}%)"
          f" | all-table total {tot/1e9:6.1f} GB, skipped {100*pr/tot:4.1f}%")
