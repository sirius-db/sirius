import json,subprocess,os
from pathlib import Path
def run(*a):return subprocess.check_output(a,text=True)
doc=json.loads(run('lsblk','-J','-o','NAME,PATH,TYPE,FSTYPE,MOUNTPOINTS,MODEL'))
disks=[d for d in doc['blockdevices'] if d['type']=='disk' and (d.get('model') or '').strip()=='Amazon EC2 NVMe Instance Storage']
assert len(disks)==3,doc
for d in disks:
 assert not d.get('children') and not d.get('fstype') and not any(d.get('mountpoints') or []),d
 signatures=json.loads(run('sudo','wipefs','-n','--json',d['path'])).get('signatures',[])
 assert not signatures,(d,signatures)
paths=[d['path'] for d in disks]
subprocess.run(['sudo','mdadm','--create','/dev/md/duckdb','--level=0','--raid-devices=3','--chunk=512','--run',*paths],check=True)
subprocess.run(['sudo','mkfs.xfs','/dev/md/duckdb'],check=True)
subprocess.run(['sudo','mkdir','-p','/mnt/nvme'],check=True)
subprocess.run(['sudo','mount','/dev/md/duckdb','/mnt/nvme'],check=True)
subprocess.run(['sudo','chown','ec2-user:ec2-user','/mnt/nvme'],check=True)
Path('/mnt/nvme/duckdb-spill').mkdir()
proof=dict(physical_disks=disks,raid=run('sudo','mdadm','--detail','/dev/md/duckdb'),mount=json.loads(run('findmnt','-J','-T','/mnt/nvme/duckdb-spill')),lsblk=json.loads(run('lsblk','-J','-o','NAME,PATH,TYPE,FSTYPE,MOUNTPOINTS,MODEL,SIZE')))
Path('/home/ec2-user/storage-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
