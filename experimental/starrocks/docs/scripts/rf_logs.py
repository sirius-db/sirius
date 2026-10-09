"""Attribute #2062's runtime-filter log lines in cn*.log to queries (by the latest query id seen in
that CN's log; queries ran one at a time)."""
import os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_tally as T
RUN = T.RUN
hi2q = {qid[:18]: T.label(qid) for qid in T.queries}
for line in open(os.path.join(RUN, 'fe', 'log', 'fe.audit.log')):
    m = re.search(r'\|QueryId=([0-9a-f-]+)\|', line)
res = collections.defaultdict(lambda: collections.defaultdict(list))
UUID = re.compile(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4})-[0-9a-f]{4}-[0-9a-f]{12}')
for cn in sorted(f for f in os.listdir(RUN) if re.match(r'cn\d+\.log', f)):
    cur = None
    for line in open(os.path.join(RUN, cn), errors='replace'):
        m = UUID.search(line)
        if m and m.group(1) in hi2q: cur = hi2q[m.group(1)]
        if 'runtime filter' not in line and 'filtered scan' not in line: continue
        q = cur or '?'
        if (m2 := re.search(r'receiver builds runtime filters .*filters=\[(.*)\]$', line.strip())):
            ids = re.findall(r'filter_id: (\d+).*?column_type: "([^"]+)"', m2.group(1))
            res[q]['built'].append((cn, tuple(ids)))
        elif (m2 := re.search(r'deferring a scan.*filters=\[(.*?)\]', line)): res[q]['deferred'].append((cn, m2.group(1)))
        elif (m2 := re.search(r'no runtime filter of this scan is built on this CN.*probed=\[(.*?)\]', line)): res[q]['not_on_cn'].append((cn, m2.group(1)))
        elif (m2 := re.search(r'applying runtime filter filter_id=(\d+) stats=KeyStats \{ rows: (\d+), min: (-?\d+), max: (-?\d+) \} waited_ms=(\d+)', line)):
            res[q]['applied'].append((cn,) + m2.groups())
        elif (m2 := re.search(r'fill their range; skipping it filter_id=(\d+) stats=KeyStats \{ rows: (\d+), min: (-?\d+), max: (-?\d+)', line)):
            res[q]['dense'].append((cn,) + m2.groups())
        elif 'cannot read runtime filter keys' in line: res[q]['unreadable'].append((cn, line.strip()[-200:]))
        elif 'did not arrive in time' in line: res[q]['timeout'].append(cn)
        elif 'taken before the scan ran' in line: res[q]['taken'].append(cn)
        elif 'cannot translate the scan with its runtime filters' in line: res[q]['translate_fail'].append((cn, line.strip()[-250:]))
        elif 'running unfiltered' in line: res[q]['engine_fallback'].append((cn, line.strip()[-200:]))
        elif 'copied runtime filter keys' in line:
            m3 = re.search(r'stream_id=(\d+) rows=(\d+) batches=(\d+)', line); res[q]['copied'].append((cn,) + m3.groups())
for q in sorted(res):
    print('==', q)
    for k, v in res[q].items():
        if k in ('applied', 'dense'):
            agg = collections.Counter((x[1], x[2], x[3], x[4]) for x in v)
            print(f'  {k} x{len(v)}:', '; '.join(f'rf{a} rows={b} [{c},{d}] x{n}' for (a, b, c, d), n in sorted(agg.items())),
                  ('waited_ms=' + ','.join(sorted({x[5] for x in v}, key=int))) if k == 'applied' else '')
        elif k == 'built':
            print(f'  built x{len(v)}:', collections.Counter(x[1] for x in v))
        elif k in ('deferred', 'not_on_cn'):
            print(f'  {k} x{len(v)}:', dict(collections.Counter(f'{x[0]}:[{x[1]}]' for x in v)))
        else:
            print(f'  {k} x{len(v)}:', v[:3])
