import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import duckdb
import performance_test as harness
from queries import queries_for_scale_factor

TABLES = ['region','nation','supplier','customer','part','partsupp','orders','lineitem']
REGION = 'us-east-1'
INPUT = 's3://sae-dataproc-aws-dev-sirius-scratch-us-east-1/datasets/tpch_sf10000_regenerated'

def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str)+'\n')

def connect(spill):
    assert duckdb.__version__ == '2.0.0'
    c=duckdb.connect(':memory:')
    assert c.execute('SELECT version()').fetchone()[0]=='v2.0.0'
    c.execute('LOAD httpfs; LOAD aws;')
    c.execute(f"CREATE SECRET benchmark_s3 (TYPE s3, PROVIDER credential_chain, REGION '{REGION}')")
    c.execute('SET enable_external_file_cache=true; SET parquet_metadata_cache=true;')
    c.execute(f"SET temp_directory='{str(spill).replace(chr(39),chr(39)*2)}'")
    assert c.execute("SELECT current_setting('enable_external_file_cache'),current_setting('parquet_metadata_cache')").fetchone()==(True,True)
    return c

def views(c):
    sql='\n'.join(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{INPUT}/{t}/*.parquet');" for t in TABLES)
    c.execute(sql)
    return sql

def cache(c):
    row=c.execute('SELECT count(*),coalesce(sum(nr_bytes),0),coalesce(sum(nr_bytes) FILTER (WHERE loaded),0),coalesce(sum(nr_bytes) FILTER (WHERE spilled),0) FROM duckdb_external_file_cache()').fetchone()
    return dict(zip(['entries','bytes','loaded_bytes','spilled_bytes'],row))

def main():
    p=argparse.ArgumentParser();p.add_argument('--query',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    def event(name,**kw):
        value=dict(event=name,query=a.query,pid=os.getpid(),wall_ns=time.time_ns(),**kw)
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(value,default=str)+'\n')
        print(json.dumps(value,default=str),flush=True)
    spill=Path('/mnt/nvme/duckdb-spill/sf10k-pairs-20261124')/f'q{a.query:02}'
    spill.mkdir(parents=True,exist_ok=False)
    c=connect(spill)
    settings=dict(c.execute("SELECT name,value FROM duckdb_settings() WHERE name IN ('enable_external_file_cache','parquet_metadata_cache','threads','memory_limit','temp_directory','max_temp_directory_size','cache_local_files','external_threads','pin_threads','enable_http_metadata_cache')").fetchall())
    assert int(settings['threads'])==96
    save(a.output/'settings.json',settings)
    (a.output/'setup.sql').write_text(views(c)+'\n')
    sql=queries_for_scale_factor(10000)[f'q{a.query}']
    (a.output/'query.sql').write_text(sql+';\n')
    event('engine_ready',version=duckdb.__version__,settings=settings,cache=cache(c))
    measured=[]
    try:
        for iteration in (1,2):
            event('query_begin',iteration=iteration)
            seconds,rows=harness.time_query(c,a.query,sql,False)
            event('query_end',iteration=iteration,seconds=seconds,rows=len(rows))
            directory=a.output/f'iteration{iteration}'
            harness._write_result(str(directory),'duckdb',a.query,rows)
            answer=directory/'duckdb'/f'q{a.query}'/'result.txt'
            measured.append(dict(iteration=iteration,seconds=seconds,rows=len(rows),answer=str(answer),answer_sha256=hashlib.sha256(answer.read_bytes()).hexdigest()))
            del rows
            gc.collect()
            event('cache_snapshot',iteration=iteration,cache=cache(c))
            save(a.output/'timings.json',measured)
    except Exception as e:
        event('query_error',error=str(e))
        raise
    finally:
        c.close()
        event('engine_closed')
    event('complete')

if __name__=='__main__':main()
