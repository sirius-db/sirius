from pathlib import Path
import json,hashlib,tarfile
root=Path(__file__).resolve().parent;archive=root.parent/(root.name+'-full-results.tar.gz');assert not archive.exists()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive.json','payload-sha256.json'])
manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files};index=root/'payload-sha256.json';index.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for p in files+[index]:t.add(p,arcname=str(p.relative_to(root)),recursive=False)
receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),'files':len(files),'payload_bytes':sum(x['bytes'] for x in manifest.values())};(root/'archive.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
