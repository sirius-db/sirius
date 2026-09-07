import os
import duckdb, glob, os, re, json
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
TAG=os.environ.get('DATASET','/datasets/tpch_sf1000').rstrip('/').split('/')[-1]
SP=SP+'/'+TAG
os.makedirs(SP, exist_ok=True)
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
c=duckdb.connect()
base=os.environ.get('DATASET','/datasets/tpch_sf1000')
for t in os.listdir(base):
    p=os.path.join(base,t)
    if os.path.isdir(p):
        c.execute(f"create view {t} as select * from read_parquet('{p}/*.parquet')")
out={}
for f in sorted(glob.glob(REPO+'/test/tpch_performance/tpch_queries/orig/q*.sql'), key=lambda x:int(re.search(r'q(\d+)',x).group(1))):
    q=os.path.basename(f)[:-4]
    sql=open(f).read().strip().rstrip(';')
    try:
        plan=c.execute("EXPLAIN (FORMAT json) "+sql).fetchall()[0][1]
    except Exception as e:
        out[q]=['ERROR '+str(e)]; continue
    out[q]=plan
json.dump(out, open(SP+'/plans_json.json','w'))
