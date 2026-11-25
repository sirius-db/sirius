from pathlib import Path
import hashlib
import json
import os
import subprocess
import urllib.request

import boto3
from worker import connect,views,INPUT,TABLES

ROOT=Path(__file__).resolve().parent
client=boto3.client('s3',region_name='us-east-1')
bucket='sae-dataproc-aws-dev-sirius-scratch-us-east-1'
location=json.loads((ROOT/'bucket-location.json').read_text())
assert location['bucket']==bucket and (location['api_response']['LocationConstraint'] or 'us-east-1')==location['normalized_region']=='us-east-1'
manifests={}
for prefix,sf in [('datasets/tpch_sf1000/',1000),('datasets/tpch_sf3000_regenerated/',3000),('datasets/tpch_sf10000_regenerated/',10000)]:
    raw=client.get_object(Bucket=bucket,Key=prefix+'manifest.json')['Body'].read()
    m=json.loads(raw);assert m['scale_factor']==sf
    actual={o['Key'][len(prefix):]:o['Size'] for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix=prefix) for o in page.get('Contents',[]) if o['Key'].endswith('.parquet')}
    assert actual=={o['path']:o['bytes'] for o in m['files']}
    (ROOT/f'sf{sf}-manifest.json').write_bytes(raw)
    manifests[sf]=dict(sha256=hashlib.sha256(raw).hexdigest(),files=len(actual),bytes=sum(actual.values()),tables=m['tables'])
spill=Path('/mnt/nvme/duckdb-spill/preflight-sf10000');spill.mkdir()
c=connect(spill);views(c)
counts={t:c.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in TABLES}
assert counts==manifests[10000]['tables'],(counts,manifests[10000]['tables'])
c.close()
identity=None
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
token=opener.open(urllib.request.Request('http://169.254.169.254/latest/api/token',method='PUT',headers={'X-aws-ec2-metadata-token-ttl-seconds':'300'})).read().decode()
identity=json.loads(opener.open(urllib.request.Request('http://169.254.169.254/latest/dynamic/instance-identity/document',headers={'X-aws-ec2-metadata-token':token})).read())
assert identity['region']=='us-east-1' and identity['instanceType']=='m9gd.24xlarge'
proof=dict(scale_factor=10000,bucket_region_evidence=location,identity=identity,manifests=manifests,parquet_footer_counts=counts,counts_untimed=True,spilling_storage=json.loads(Path('/home/ec2-user/storage-proof.json').read_text()),hardware=subprocess.check_output(['lscpu'],text=True),memory=Path('/proc/meminfo').read_text(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
(ROOT/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(verified=True,counts=counts,manifests=manifests),indent=2),flush=True)
