"""Tally the FE's runtime filters in recorded TExecPlanFragmentParams dumps and judge each against
PR #2062's (f4780e74) acceptance rules.

Usage: python3 rf_tally.py <run_dir>
"""
import os, re, sys, pickle, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdebug

if len(sys.argv) < 2:
    sys.exit(__doc__)
RUN = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
fr = rdebug.load_all(os.path.join(RUN, 'frags'), os.path.join(RUN, 'frags.pkl'))

PLAN = {0:'OLAP_SCAN',4:'HASH_JOIN',6:'AGG',8:'SORT',9:'EXCHANGE',17:'FILE_SCAN',19:'UNION',27:'HDFS_SCAN',28:'PROJECT',33:'NESTLOOP_JOIN',12:'CROSS_JOIN',11:'SELECT',23:'ASSERT_NUM_ROWS',14:'ANALYTIC'}
MODE = {0:'NONE',1:'BROADCAST',2:'PARTITIONED',3:'LOCAL_HASH_BUCKET',4:'COLOCATE',5:'SHUFFLE_HASH_BUCKET',6:'REPLICATED'}
JDIST = {0:'NONE',1:'BROADCAST',2:'PARTITIONED',3:'LOCAL_HASH_BUCKET',4:'SHUFFLE_HASH_BUCKET',5:'COLOCATE',6:'REPLICATED'}
PRIM = ['INVALID','NULL','BOOLEAN','TINYINT','SMALLINT','INT','BIGINT','FLOAT','DOUBLE','DATE','DATETIME','BINARY','DECIMAL','CHAR','LARGEINT','VARCHAR','HLL','DECIMALV2','TIME','OBJECT','PERCENTILE','DECIMAL32','DECIMAL64','DECIMAL128','JSON','FUNCTION','VARBINARY','DECIMAL256','INT256','VARIANT']
EXPR = {16:'SLOT_REF',5:'CAST_EXPR',20:'FUNCTION_CALL',21:'COMPUTE_FUNCTION_CALL',1:'ARITHMETIC_EXPR'}
SIGNED_INT = {'TINYINT','SMALLINT','INT','BIGINT'}
TABLE_PREFIX = {'l':'lineitem','o':'orders','ps':'partsupp','p':'part','s':'supplier','c':'customer','n':'nation','r':'region'}
FACT = {'lineitem','orders','partsupp'}

def ev(e): return e['__v__'][0] if isinstance(e, dict) else e
def uid_hex(u): return '%016x-%016x' % (u['hi'] & (2**64-1), u['lo'] & (2**64-1))
def uuid(u):
    h = uid_hex(u).replace('-', '')
    return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'

def expr_kind(texpr):
    nodes = texpr['nodes']
    first = nodes[0]
    t = first['type_']['types'][0].get('scalar_type')
    ty = PRIM[ev(t['type_'])] if t else 'complex'
    if t and ty.startswith('DECIMAL'): ty += f"({t['precision']},{t['scale']})"
    nt = ev(first['node_type'])
    if len(nodes) == 1 and nt == 16:
        return 'slot', ty, (first['slot_ref']['tuple_id'], first['slot_ref']['slot_id'])
    return 'expr:' + EXPR.get(nt, str(nt)) + f'[{len(nodes)}n]', ty, None

def children_of(nodes):
    ch = [[] for _ in nodes]
    def walk(at):
        nxt = at + 1
        for _ in range(nodes[at]['num_children']):
            ch[at].append(nxt); nxt = walk(nxt)
        return nxt
    walk(0)
    return ch

# ---- per-query fragment index ----
queries = collections.OrderedDict()   # qid -> {'frags': {root: info}, 'files': [...]}
for fn in sorted(fr):
    x = fr[fn]
    p = x['params']; qid = uuid(p['query_id'])
    nodes = x['fragment']['plan']['nodes']
    root = nodes[0]['node_id']
    q = queries.setdefault(qid, {'frags': {}, 'files': []})
    q['files'].append(fn)
    if root in q['frags']:
        q['frags'][root]['instances'].add(uuid(p['fragment_instance_id']))
        continue
    slots = {s['id']: s for s in x['desc_tbl']['slot_descriptors']}
    by_tuple = collections.defaultdict(list)
    for s in x['desc_tbl']['slot_descriptors']: by_tuple[s['parent']].append(s['col_name'])
    ch = children_of(nodes)
    sink = x['fragment']['output_sink']
    sinfo = None
    if sink and sink.get('stream_sink'):
        ss = sink['stream_sink']; sinfo = (ss['dest_node_id'], ['UNPARTITIONED','RANDOM','HASH_PARTITIONED','RANGE_PARTITIONED','BUCKET_SHUFFLE_HASH_PARTITIONED'][ev(ss['output_partition']['type_'])] if ev(ss['output_partition']['type_'])<5 else ev(ss['output_partition']['type_']))
    q['frags'][root] = dict(file=fn, nodes=nodes, children=ch, by_tuple=by_tuple, sink=sinfo,
                            has_exchange=any(ev(n['node_type']) == 9 for n in nodes),
                            instances={uuid(p['fragment_instance_id'])},
                            node_ids={n['node_id'] for n in nodes})

def table_of(q, node):
    for frag in q['frags'].values():
        pass
    return None

def locate(q, node_id):
    for root, f in q['frags'].items():
        for i, n in enumerate(f['nodes']):
            if n['node_id'] == node_id: return root, f, i, n
    return None

def scan_table(f, n):
    cols = [c for t in n['row_tuples'] for c in f['by_tuple'].get(t, [])]
    for c in cols:
        pre = c.split('_')[0]
        if pre in TABLE_PREFIX: return TABLE_PREFIX[pre]
    return '?'

# ---- collect filters ----
filters = collections.OrderedDict()   # (qid, fid) -> info
for qid, q in queries.items():
    for root, f in q['frags'].items():
        for i, n in enumerate(f['nodes']):
            hj = n.get('hash_join_node')
            for rf in (hj or {}).get('build_runtime_filters') or []:
                key = (qid, rf['filter_id'])
                info = filters.setdefault(key, {'desc': rf, 'join_seen': False})
                info['desc'] = rf; info['join_seen'] = True
                info['join_frag'] = root; info['join_node'] = n['node_id']
                info['join_dist'] = JDIST.get(ev(hj.get('distribution_mode')), '?') if hj.get('distribution_mode') else None
                info['join_op'] = ev(hj['join_op'])
                bc = f['children'][i][1] if len(f['children'][i]) > 1 else None
                info['build_child'] = PLAN.get(ev(f['nodes'][bc]['node_type']), str(ev(f['nodes'][bc]['node_type']))) if bc is not None else None
                info['build_child_id'] = f['nodes'][bc]['node_id'] if bc is not None else None
            for rf in n.get('probe_runtime_filters') or []:
                key = (qid, rf['filter_id'])
                info = filters.setdefault(key, {'desc': rf, 'join_seen': False})

# ---- judge ----
rows = []
for (qid, fid), info in filters.items():
    q = queries[qid]; d = info['desc']
    mode = MODE.get(ev(d.get('build_join_mode')), 'unset') if d.get('build_join_mode') is not None else 'unset'
    ftype = ev(d['filter_type']) if d.get('filter_type') is not None else 0
    bkind, bty, bslot = expr_kind(d['build_expr'])
    join_loc = locate(q, d.get('build_plan_node_id'))
    join_frag = join_loc[0] if join_loc else None
    if not info['join_seen'] and join_loc:
        info['join_frag'] = join_loc[0]
    targets = []
    for tnode, texpr in (d.get('plan_node_id_to_target_expr') or {}).items():
        pk, pty, _ = expr_kind(texpr)
        loc = locate(q, tnode)
        if loc:
            troot, tf, ti, tn = loc
            ttype = PLAN.get(ev(tn['node_type']), str(ev(tn['node_type'])))
            table = scan_table(tf, tn)
            same = (troot == join_frag) if join_frag is not None else (None)
            leaf = not tf['has_exchange']
        else:
            ttype = table = '?'; same = None; leaf = None
        targets.append(dict(node=tnode, kind=pk, ty=pty, scan=ttype, table=table, same_frag=same, leaf=leaf))
    # build-side verdict (built_filters)
    build_reason = None
    if ftype != 0: build_reason = 'not JOIN_FILTER'
    elif mode != 'BROADCAST': build_reason = f'mode {mode}'
    elif info.get('join_seen') and info.get('build_child') != 'EXCHANGE': build_reason = f"build child {info.get('build_child')}"
    elif bkind != 'slot': build_reason = 'build key not bare slot'
    key_ok = bty in SIGNED_INT
    tverdicts = []
    for t in targets:
        if build_reason: v = 'rej:' + build_reason
        elif t['scan'] not in ('FILE_SCAN', 'HDFS_SCAN'): v = 'rej:target not file scan' if t['scan'] != '?' else 'unk:target frag not dumped'
        elif t['leaf'] is False: v = 'rej:probe scan in non-leaf frag' + (' (same frag as join)' if t['same_frag'] else '')
        elif not key_ok: v = f'rej:key type {bty} (key_stats needs signed int)'
        elif t['ty'] != bty: v = f'rej:probe type {t["ty"]} != key {bty}'
        else: v = 'ok' + ('' if info.get('join_seen') else '(join frag not dumped)')
        t['verdict'] = v; tverdicts.append(v)
    rows.append(dict(qid=qid, fid=fid, mode=mode, has_remote=d.get('has_remote_targets'), join_dist=info.get('join_dist'),
                     build_child=info.get('build_child') if info.get('join_seen') else '?', bkind=bkind, bty=bty,
                     targets=targets, ftype=ftype, join_node=d.get('build_plan_node_id')))

# ---- map query ids to TPC-H numbers through the FE audit log (statement order) ----
order = []
for line in open(os.path.join(RUN, 'fe', 'log', 'fe.audit.log')):
    m = re.search(r'\|QueryId=([0-9a-f-]+)\|.*?\|Stmt=(.*?)\|Digest', line)
    if m and 'tpch_sf1000' in m.group(2) and m.group(1) not in order: order.append(m.group(1))
qname = {qid: 'q%02d' % (i + 1) for i, qid in enumerate(order)}

def label(qid): return qname.get(qid, 'other:' + qid[:18])

# ---- CN log evidence ----
ev_log = collections.defaultdict(lambda: collections.Counter())
pat = [('built', re.compile(r'receiver builds runtime filters receiver=(\S+) filters=(.*)')),
       ('deferred', re.compile(r'deferring a scan until its runtime filters arrive instance=(\S+) filters=\[(.*?)\]')),
       ('not_on_cn', re.compile(r'no runtime filter of this scan is built on this CN instance=(\S+) probed=\[(.*?)\]')),
       ('applied', re.compile(r'applying runtime filter filter_id=(\d+)')),
       ('dense_skip', re.compile(r'runtime filter keys fill their range; skipping it filter_id=(\d+)')),
       ('keys_unreadable', re.compile(r'cannot read runtime filter keys; skipping it filter_id=(\d+)')),
       ('timeout', re.compile(r'runtime filters did not arrive in time.*instance=(\S+)')),
       ('taken', re.compile(r'runtime filter keys were taken before the scan ran filter_id=(\d+)')),
       ('translate_fail', re.compile(r'cannot translate the scan with its runtime filters')),
       ('engine_fallback', re.compile(r'runtime filter keys unavailable; running unfiltered'))]
for cn in sorted(f for f in os.listdir(RUN) if re.match(r'cn\d+\.log', f)):
    for line in open(os.path.join(RUN, cn), errors='replace'):
        if 'runtime filter' not in line: continue
        for name, rx in pat:
            m = rx.search(line)
            if m:
                inst = re.search(r'(?:receiver|instance)=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4})', line)
                qhi = inst.group(1) if inst else None
                ev_log[qhi][name] += 1
                ev_log['__all__'][name] += 1
                break

if __name__ == '__main__':
    print('# per-filter detail')
    for r in sorted(rows, key=lambda r: (label(r['qid']), r['fid'])):
        ts = '; '.join(f"n{t['node']}:{t['scan']}/{t['table']} {t['kind']} {t['ty']} same={t['same_frag']} leaf={t['leaf']} -> {t['verdict']}" for t in r['targets'])
        print(f"{label(r['qid'])} rf{r['fid']} join=n{r['join_node']} mode={r['mode']} dist={r['join_dist']} remote={r['has_remote']} buildchild={r['build_child']} key={r['bkind']}:{r['bty']} | {ts}")
    print()
    print('# per-query tally (filters; a filter counts as accepted if any target is ok)')
    agg = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: label(r['qid'])):
        a = agg.setdefault(label(r['qid']), collections.Counter())
        a['total'] += 1
        vs = [t['verdict'] for t in r['targets']]
        if r['mode'] == 'BROADCAST':
            if any(v.startswith('ok') for v in vs): a['bcast_ok'] += 1
            elif any(v.startswith('unk') for v in vs): a['bcast_ok_target_not_dumped'] += 1
            else:
                reasons = sorted({v for v in vs})
                a['bcast_rej'] += 1
                for v in reasons: a['R:' + v] += 1
        elif r['mode'] in ('PARTITIONED', 'SHUFFLE_HASH_BUCKET'): a['partitioned'] += 1
        else: a['other:' + r['mode']] += 1
        for t in r['targets']:
            if t['table'] in FACT and not t['verdict'].startswith('ok'):
                a['fact_unapplied:' + t['table']] += 1
    for k, a in agg.items():
        print(k, dict(a))
    print()
    print('# fragments dumped per query (distinct fragments / files)')
    for qid, q in queries.items():
        print(label(qid), qid, len(q['frags']), len(q['files']))
    print()
    print('# CN-log runtime-filter evidence by query hi')
    for qid in queries:
        hi = qid[:18]
        if hi in ev_log: print(label(qid), dict(ev_log[hi]))
    print('ALL', dict(ev_log['__all__']))
    pickle.dump(rows, open(os.path.join(RUN, 'rows.pkl'), 'wb'))
