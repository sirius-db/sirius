from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
import boto3

root=Path(__file__).resolve().parent
before=json.loads((root/'preflight.json').read_text())
assert json.loads((root/'results/summary.json').read_text())['complete']
client=boto3.client('s3',region_name='us-east-1')
bucket='sae-dataproc-aws-dev-sirius-scratch-us-east-1'
checked={}
for sf,prefix in [(1000,'datasets/tpch_sf1000/'),(3000,'datasets/tpch_sf3000_regenerated/'),(10000,'datasets/tpch_sf10000_regenerated/')]:
 raw=client.get_object(Bucket=bucket,Key=prefix+'manifest.json')['Body'].read()
 assert hashlib.sha256(raw).hexdigest()==before['manifests'][str(sf)]['sha256']
 manifest=json.loads(raw)
 actual={o['Key'][len(prefix):]:o['Size'] for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix=prefix) for o in page.get('Contents',[]) if o['Key'].endswith('.parquet')}
 assert actual=={f['path']:f['bytes'] for f in manifest['files']}
 checked[sf]=dict(manifest_unchanged=True,inventory_matches=True,files=len(actual),bytes=sum(actual.values()))
for name,wanted in before['source_sha256'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==wanted,name
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
assert not any(str(root/'worker.py') in s for s in processes.splitlines())
proof=dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),datasets=checked,measured_sources_unchanged=True,no_benchmark_workers_remaining=True,storage_mount=json.loads(subprocess.check_output(['findmnt','-J','-T','/mnt/nvme/duckdb-spill'],text=True)))
(root/'postflight.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
