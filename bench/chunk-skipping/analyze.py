import os, re, csv, glob, collections, json
RES=os.environ.get('RES', os.path.dirname(os.path.abspath(__file__))+'/results')
runs={}
for d in sorted(glob.glob(RES+'/tpch_*')):
    m=re.search(r'_([0-9]+(?:GB|MB))_prune-(on|off)$', d)
    if not m: continue
    batch, arm = m.group(1), m.group(2)
    best=collections.defaultdict(lambda: 1e9)
    f=d+'/csv/runtimes.csv'
    if not os.path.exists(f): continue
    for r in csv.DictReader(open(f)):
        best[r['query']]=min(best[r['query']], float(r['runtime_s']))
    # pruning evidence
    pruned=0; total=0
    for lf in glob.glob(d+'/log_dir/*.log'):
        for line in open(lf, errors='ignore'):
            mm=re.search(r'zone-map pruning for pinned entry \'(\w+)\'.*?: (\d+)/(\d+) chunks', line)
            if mm: pruned+=int(mm.group(2)); total+=int(mm.group(3))
    runs[(batch,arm)]={'best':dict(best),'pruned':pruned,'total':total}

def order(b):
    n=float(b[:-2]); return n*1024 if b.endswith('GB') else n
batches=sorted({k[0] for k in runs}, key=order, reverse=True)
qs=sorted({q for v in runs.values() for q in v['best']}, key=lambda x:int(x[1:]))

print(f"{'batch':>8}{'suite ON':>11}{'suite OFF':>11}{'delta':>10}{'delta %':>9}   chunks pruned (ON)")
for b in batches:
    on,off=runs.get((b,'on')),runs.get((b,'off'))
    if not on or not off: continue
    so=sum(on['best'].get(q,0) for q in qs); sf=sum(off['best'].get(q,0) for q in qs)
    frac = f"{on['pruned']}/{on['total']} = {100*on['pruned']/max(on['total'],1):.0f}%"
    print(f"{b:>8}{so:>11.4f}{sf:>11.4f}{so-sf:>10.4f}{100*(so-sf)/sf:>8.1f}%   {frac}")

print(f"\nPer-query ON-minus-OFF delta (s), negative = pruning helped")
print(f"{'query':>7}"+''.join(f"{b:>10}" for b in batches))
for q in qs:
    row=[]
    for b in batches:
        on,off=runs.get((b,'on')),runs.get((b,'off'))
        row.append((on['best'].get(q,0)-off['best'].get(q,0)) if on and off else 0.0)
    if any(abs(x)>0.002 for x in row):
        print(f"{q:>7}"+''.join(f"{x:>10.4f}" for x in row))
