import os
import duckdb, json, re, os, collections
SP=os.environ.get('SCRATCH','/tmp/sirius-chunk-skipping')
c=duckdb.connect()

PREFIX=[('ps_','partsupp'),('l_','lineitem'),('o_','orders'),('c_','customer'),
        ('p_','part'),('s_','supplier'),('n_','nation'),('r_','region')]
def tbl_of(col):
    for p,t in PREFIX:
        if col.startswith(p): return t
    return None

# ---- load stats into memory: stats[table][(file,rg)] = {'rows':n, col:(mn,mx,nulls,csize,usize)}
stats={}
for t in ['customer','lineitem','nation','orders','part','partsupp','region','supplier']:
    rows=c.execute(f"select file_name,row_group_id,row_group_num_rows,col,mn,mx,nulls,csize,usize from '{SP}/stats_{t}.parquet'").fetchall()
    d=collections.defaultdict(dict)
    for fn,rg,nr,col,mn,mx,nulls,cs,us in rows:
        e=d[(fn,rg)]; e['rows']=nr; e[col]=(mn,mx,nulls,cs,us)
    stats[t]=dict(d)

def num(x):
    try: return float(x)
    except: return None

def cmp_pair(a,b):
    """return comparable versions of a,b"""
    na,nb=num(a),num(b)
    if na is not None and nb is not None: return na,nb
    return str(a),str(b)

# ---- filter AST: list of conjuncts; each conjunct is list of disjuncts; disjunct = predicate or UNSUPPORTED
UNSUP=object()
LIT=r"""('(?:[^']|'')*'(?:::\w+)?|-?\d+\.?\d*)"""
CMP=re.compile(r"^\s*(\w+)\s*(<=|>=|!=|=|<|>)\s*"+LIT+r"\s*$")
IN=re.compile(r"^\s*\(?\s*(\w+)\s+IN\s*\((.*)\)\s*\)?\s*$")

def unlit(s):
    s=s.strip()
    s=re.sub(r'::\w+$','',s)
    if s.startswith("'"): return s[1:-1].replace("''","'")
    return s

def parse_atom(s):
    s=s.strip()
    while s.startswith('(') and s.endswith(')') and s.count('(')==s.count(')'):
        inner=s[1:-1]
        if inner.count('(')==inner.count(')'): s=inner.strip()
        else: break
    m=CMP.match(s)
    if m: return (m.group(1), m.group(2), unlit(m.group(3)))
    m=IN.match(s)
    if m:
        vals=[unlit(v) for v in re.split(r",\s*(?=(?:[^']*'[^']*')*[^']*$)", m.group(2))]
        return (m.group(1),'IN',vals)
    return UNSUP

def parse_filter(f):
    f=f.strip()
    if f.startswith('optional:'): f=f[len('optional:'):].strip()
    # split on top-level AND
    parts=split_top(f,' AND ')
    conj=[]
    for p in parts:
        dis=[parse_atom(x) for x in split_top(p,' OR ')]
        conj.append(dis)
    return conj

def split_top(s,sep):
    out=[];depth=0;inq=False;cur='';i=0
    while i<len(s):
        ch=s[i]
        if ch=="'": inq=not inq
        if not inq:
            if ch=='(':depth+=1
            elif ch==')':depth-=1
            if depth==0 and s[i:i+len(sep)].upper()==sep:
                out.append(cur);cur='';i+=len(sep);continue
        cur+=ch;i+=1
    out.append(cur)
    return [x.strip() for x in out if x.strip()]

def atom_prunes(atom, e):
    """True if this atom guarantees zero matches in row group entry e."""
    if atom is UNSUP: return False
    col,op,val=atom
    if col not in e: return False
    mn,mx,nulls,_,_=e[col]
    if mn is None or mx is None: return False
    if op=='IN':
        lo=hi=None
        for v in val:
            a,b=cmp_pair(mn,v)
            lo=a  # min converted
        # prune if every value outside [mn,mx]
        for v in val:
            m1,vv=cmp_pair(mn,v); m2,vv2=cmp_pair(mx,v)
            if m1<=vv and vv2<=m2: return False
        return True
    a,v=cmp_pair(mn,val); b,_=cmp_pair(mx,val)
    if op=='=':  return not (a<=v<=b)
    if op=='<':  return not (a<v)
    if op=='<=': return not (a<=v)
    if op=='>':  return not (b>v)
    if op=='>=': return not (b>=v)
    if op=='!=': return a==b==v
    return False

def rg_prunable(conj, e):
    for dis in conj:
        if all(atom_prunes(a,e) for a in dis): return True
    return False

def cols_in(conj):
    s=set()
    for dis in conj:
        for a in dis:
            if a is not UNSUP: s.add(a[0])
    return s

# ---- run
plans=json.load(open(SP+'/plans_json.json'))
report={}
for q,v in plans.items():
    d=json.loads(v)
    scans=[]
    def walk(n):
        ei=n.get('extra_info',{})
        if ei.get('Function')=='READ_PARQUET':
            proj=ei.get('Projections',[])
            if isinstance(proj,str): proj=[proj]
            fl=ei.get('Filters',[])
            if isinstance(fl,str): fl=[fl]
            scans.append((proj,fl))
        for ch in n.get('children',[]): walk(ch)
    walk(d if isinstance(d,dict) else d[0])
    per={}
    for proj,fl in scans:
        # attribute
        allcols=set()
        conjs=[]
        unsup=[]
        for f in fl:
            cj=parse_filter(f)
            for dis in cj:
                if any(a is UNSUP for a in dis): unsup.append(f)
            conjs.append(cj)
            allcols|=cols_in(cj)
        for p in proj:
            for m in re.findall(r'\b([a-z]+_\w+)\b',p): allcols.add(m)
        tbls={tbl_of(x) for x in allcols}-{None}
        if not tbls: continue
        t=sorted(tbls, key=lambda x:-sum(1 for c2 in allcols if tbl_of(c2)==x))[0]
        readcols={x for x in allcols if tbl_of(x)==t}
        S=stats[t]
        tot_rg=len(S); pr_rg=0; tot_rows=0; pr_rows=0; tot_b=0; pr_b=0
        for k,e in S.items():
            b=sum(e[cc][4] for cc in readcols if cc in e)
            tot_rg+=0; tot_rows+=e['rows']; tot_b+=b
            if any(rg_prunable(cj,e) for cj in conjs):
                pr_rg+=1; pr_rows+=e['rows']; pr_b+=b
        per[t]={'rgs':tot_rg,'pruned_rgs':pr_rg,'rows':tot_rows,'pruned_rows':pr_rows,
                'read_bytes':tot_b,'pruned_bytes':pr_b,'filters':fl,'unsupported':unsup,
                'readcols':sorted(readcols)}
    report[q]=per
json.dump(report, open(SP+'/prune_report.json','w'), indent=1)

# print summary
print(f"{'q':<5}{'table':<10}{'rgs':>7}{'pruned':>8}{'%rg':>7}{'%rows':>8}{'GB read':>9}{'GB skip':>9}{'%bytes':>8}")
TR=TB=PB=0
for q in sorted(report,key=lambda x:int(x[1:])):
    for t,r in report[q].items():
        if r['rgs']==0: continue
        print(f"{q:<5}{t:<10}{r['rgs']:>7}{r['pruned_rgs']:>8}{100*r['pruned_rgs']/r['rgs']:>7.1f}"
              f"{100*r['pruned_rows']/max(r['rows'],1):>8.1f}{r['read_bytes']/1e9:>9.2f}{r['pruned_bytes']/1e9:>9.2f}"
              f"{100*r['pruned_bytes']/max(r['read_bytes'],1):>8.1f}")
        TB+=r['read_bytes']; PB+=r['pruned_bytes']
print(f"\nTOTAL uncompressed bytes scanned across 22 queries: {TB/1e9:.1f} GB, prunable {PB/1e9:.1f} GB = {100*PB/TB:.2f}%")

# --- coverage: how much scanned volume is even *under* a supported predicate
rep=json.load(open(SP+'/prune_report.json'))
tot=0; cov=0; unsup_b=0
per_t=collections.defaultdict(lambda:[0,0])
for q,per in rep.items():
    for t,r in per.items():
        tot+=r['read_bytes']
        has_supported = any(f not in r['unsupported'] for f in r['filters'])
        if r['filters'] and has_supported:
            cov+=r['read_bytes']; per_t[t][0]+=r['read_bytes']
        per_t[t][1]+=r['read_bytes']
print(f"\nScan volume under >=1 min/max-evaluable predicate: {cov/1e9:.1f} / {tot/1e9:.1f} GB = {100*cov/tot:.1f}%")
for t,(a,b) in sorted(per_t.items(), key=lambda x:-x[1][1]):
    print(f"  {t:<10} {b/1e9:8.1f} GB scanned, {100*a/max(b,1):5.1f}% under a supported predicate")
