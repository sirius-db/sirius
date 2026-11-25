from pathlib import Path
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def save(path,obj):path.write_text(json.dumps(obj,indent=2,default=str)+'\n')
def meminfo():return Path('/proc/meminfo').read_text()

def main():
    OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',dict(version='2.0.0',scale_factor=10000,iterations=2,order='Q1 Q1 clear/restart Q2 Q2 clear/restart ... Q22 Q22',fresh_process='per query pair',os_cache_clear='before each pair, including Q1',warmup=False,timing='execute only; fetch/export/startup excluded',external_cache=True,parquet_metadata_cache=True,spill='/mnt/nvme/duckdb-spill',timeout_per_pair_s=3600,timeout_per_scale_s=14400,query_source_commit='26b5caba3e2c96290fdc63d4cf40ac42572eaab8'))
    records=[]
    deadline=time.monotonic()+14400
    for query in range(1,23):
        if time.monotonic()>=deadline:break
        save(OUT/'progress.json',dict(phase='clearing OS cache',query=query))
        before=meminfo();os.sync()
        subprocess.run(['sudo','sh','-c','echo 3 > /proc/sys/vm/drop_caches'],check=True)
        cleared=dict(query=query,wall_ns=time.time_ns(),before=before,after=meminfo(),command='sync; echo 3 > /proc/sys/vm/drop_caches')
        save(OUT/f'q{query:02}-cache-clear.json',cleared)
        pair=OUT/f'q{query:02}'
        save(OUT/'progress.json',dict(phase='running',query=query))
        with (OUT/f'q{query:02}.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'worker.py'),'--query',str(query),'--output',str(pair)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=process.wait(timeout=min(3600,max(1,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
        record=dict(query=query,pid=process.pid,exit_code=code,timed_out=timed_out,cache_clear_ns=cleared['wall_ns'],directory=str(pair))
        records.append(record);save(OUT/'processes.json',records)
        measured=[]
        for r in records:
            p=Path(r['directory'])/'timings.json'
            if p.is_file():
                measured.extend(dict(query=r['query'],**x) for x in json.loads(p.read_text()))
        totals=[]
        for i in (1,2):
            values=[x['seconds'] for x in measured if x['iteration']==i]
            totals.append(dict(iteration=i,queries=len(values),sum_s=math.fsum(values) if len(values)==22 else None,available_sum_s=math.fsum(values)))
        summary=dict(measurements=measured,totals=totals,complete=len(measured)==44 and all(r['exit_code']==0 and not r['timed_out'] for r in records))
        save(OUT/'summary.json',summary)
        with (OUT/'per-query.csv').open('w',newline='') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerow(['query','iteration1_s','iteration2_s'])
            for q in range(1,query+1):writer.writerow([q,*[next((x['seconds'] for x in measured if x['query']==q and x['iteration']==i),'') for i in (1,2)]])
        print(json.dumps(dict(query=query,exit_code=code,totals=totals)),flush=True)
    save(OUT/'progress.json',dict(phase='finished',totals=totals,complete=summary['complete']))
    return 0 if summary['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
