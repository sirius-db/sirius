#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Fail-closed qualification of joint receipts. No temporal/URL identity inference."""
import argparse, collections, json
from pathlib import Path
SCENARIOS = ('reuse', 'stale', 'precondition', 'off', 'disabled', 'weak')
EVENTS = {'select_begin', 'select_end', 'submitted', 'finish', 'classification', 'invalidated', 'seed_begin', 'seed_end'}
FIELDS = {'kind', 'id', 'physical', 'open', 'context', 'size', 'current', 'selected', 'attempt', 'auth', 'enabled', 'reuse', 'strong', 'delegated', 'cached', 'started', 'refreshed', 'status', 'rc', 'bytes', 'result', 'seeds', 'hits', 'refreshes'}

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def load(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

def validate(rows, scenario):
    require(scenario in SCENARIOS, 'unknown scenario')
    require(rows and set(rows[0]) == {'kind', 'failed', 'events'} and (rows[0]['kind'] == 'observer'), 'missing observer header')
    require(rows[0]['failed'] == 0, 'observer failure')
    events = [r for r in rows[1:] if r.get('kind') in EVENTS]
    terminals = [r for r in rows[1:] if r.get('kind') == 'terminal']
    wires = [r for r in rows[1:] if r.get('kind') == 'wire']
    require(len(events) == rows[0]['events'], 'missing observer event')
    require(len(events) + len(terminals) + len(wires) == len(rows) - 1, 'unknown/duplicate record type')
    ids = [0, 64, 128] if scenario == 'stale' else [0, 64]
    require(sorted((t['id'] for t in terminals)) == ids, 'missing/duplicate terminal')
    require(set((e['id'] for e in events)) == set(ids), 'unexpected operation')
    for e in events:
        require(set(e) == FIELDS, 'event schema')
        require(e['physical'] == ids.index(e['id']) + 1 and e['open'] == e['context'] == 1, 'operation/open/context identity')
        require(e['size'] == 32 and e['auth'] == 0 and (e['rc'] == 0), 'unsupported request/transport')
        require(all((isinstance(e[x], int) and e[x] >= 0 for x in ['current', 'selected', 'attempt', 'seeds', 'hits', 'refreshes'])), 'invalid state/counter')
        for x in ['enabled', 'reuse', 'strong', 'delegated', 'cached', 'started', 'refreshed']:
            require(e[x] in (0, 1), 'invalid boolean')
        require((e['enabled'], e['reuse'], e['strong']) == (int(scenario != 'disabled'), int(scenario != 'off'), int(scenario != 'weak')), 'config/validator gate')
    prev = {'seeds': 0, 'hits': 0, 'refreshes': 0}
    for e in events:
        for name in prev:
            delta = e[name] - prev[name]
            require(delta in (0, 1), 'counter jump')
            if delta:
                permitted = name == 'hits' and e['kind'] == 'select_end' and e['cached'] or (name == 'seeds' and e['kind'] == 'seed_end' and (e['result'] == 1)) or (name == 'refreshes' and e['kind'] == 'select_begin' and e['refreshed'])
                require(permitted, 'counter changed at wrong boundary')
            prev[name] = e[name]
    eligible = scenario not in ('off', 'disabled', 'weak')
    normalized = []
    for id_ in ids:
        group = [e for e in events if e['id'] == id_]
        legs = []
        i = 0
        while i < len(group):
            require([e['kind'] for e in group[i:i + 5]] == ['select_begin', 'select_end', 'submitted', 'finish', 'classification'], 'missing/reordered leg events')
            (a, b, s, f, c) = group[i:i + 5]
            i += 5
            for e in [b, s, f, c]:
                require(e['attempt'] == a['attempt'] and e['auth'] == a['auth'], 'attempt association')
            gate = a['enabled'] and a['reuse'] and a['strong'] and (not a['delegated']) and (not a['refreshed']) and (a['attempt'] == a['auth'] == 0)
            selected = a['current'] if gate else a['selected']
            require(b['current'] == a['current'], 'selection publication changed')
            require(b['selected'] == selected, 'selected publication mismatch')
            for field in ['attempt', 'auth', 'enabled', 'reuse', 'strong', 'refreshed', 'seeds', 'refreshes']:
                require(a[field] == b[field], 'selection changed unrelated state')
            require({k: v for (k, v) in b.items() if k != 'kind'} == {k: v for (k, v) in s.items() if k != 'kind'}, 'submission changed selection')
            require({k: v for (k, v) in f.items() if k not in ('kind', 'result')} == {k: v for (k, v) in c.items() if k not in ('kind', 'result')}, 'classifier result source')
            require(b['cached'] == int(bool(selected)), 'selection outcome')
            require(b['hits'] - a['hits'] == int(bool(gate and a['current'])), 'selection hit count')
            for e in [s, f, c]:
                for field in ['selected', 'cached', 'delegated', 'started', 'refreshed']:
                    require(e[field] == b[field], 'submission/finish association')
            require(s['status'] == 0 and f['status'] == c['status'], 'wire result association')
            refused = bool(c['cached'] and c['status'] in (307, 401, 403, 404, 503))
            require(c['result'] == int(refused), 'actual recovery classifier')
            if refused:
                require(i < len(group) and group[i]['kind'] == 'invalidated', 'missing invalidation')
                inv = group[i]
                i += 1
                require(inv['selected'] == c['selected'], 'wrong invalidation identity')
                require(inv['current'] == (0 if c['current'] == c['selected'] else c['current']), 'stale publication removed')
            elif f['status'] == 206 and f['delegated'] and (not f['cached']):
                require([e['kind'] for e in group[i:i + 2]] == ['seed_begin', 'seed_end'], 'missing validated publication')
                (sb, se) = group[i:i + 2]
                i += 2
                should_seed = eligible and sb['current'] == 0
                require(se['result'] == int(should_seed), 'seed result')
                require(se['current'] > 0 if should_seed else se['current'] == sb['current'], 'publication identity')
            legs.append((s, f, c))
        expected = [('origin', 307, 0), ('target', 206, 0)]
        if scenario == 'disabled':
            expected = [('origin', 206, 0)]
        elif eligible and id_:
            expected = [('target', 412 if scenario == 'precondition' else 206, 0)]
            if scenario == 'stale':
                expected = [('target', 404, 0), ('origin', 307, 1), ('target', 206, 1)]
        actual = [('target' if s['delegated'] else 'origin', f['status'], s['attempt']) for (s, f, c) in legs]
        require(actual == expected, 'wire-leg path/attempt')
        for (s, f, c) in legs:
            require(f['bytes'] == (3 if f['status'] == 307 else 32 if f['status'] == 206 else 4 if scenario == 'stale' and id_ == 64 and (f['status'] == 404) else 0), 'body bytes')
        terminal = next((t for t in terminals if t['id'] == id_))
        fail = scenario == 'precondition' and id_ == 64
        require(terminal == {'kind': 'terminal', 'id': id_, 'completions': 1, 'bytes': 0 if fail else 32, 'result': 'object_changed' if fail else 'success'}, 'terminal result/completion')
        ws = [w for w in wires if w.get('range') == f'bytes={id_}-{id_ + 31}']
        require(collections.Counter((w['role'] for w in ws)) == collections.Counter((x[0] for x in expected)), 'wire/submission mismatch')
        require(all((set(w) == {'kind', 'role', 'method', 'range', 'if_match'} and w['method'] == 'GET' and (w['if_match'] == int(scenario != 'weak')) for w in ws)), 'wire headers')
        normalized.append({'id': id_, 'events': group, 'terminal': terminal, 'wire': sorted(ws, key=lambda w: w['role'])})
    require(len(wires) == sum((len([e for e in events if e['kind'] == 'submitted' and e['id'] == id_]) for id_ in ids)), 'unknown wire operation')
    seeds = [e for e in events if e['kind'] == 'seed_end' and e['result'] == 1]
    require([e['current'] for e in seeds] == ([1, 2] if scenario == 'stale' else [1] if eligible else []), 'publication sequence')
    if scenario == 'stale':
        stale = next((e for e in events if e['id'] == 64 and e['kind'] == 'invalidated'))
        require(stale['selected'] == 1 and stale['current'] == 2, 'P1 invalidation must preserve P2')
        require(events.index(seeds[1]) < events.index(stale), 'publication/barrier ordering')
    return normalized

def main():
    p = argparse.ArgumentParser()
    p.add_argument('reference', type=Path, nargs='?')
    p.add_argument('candidate', type=Path, nargs='?')
    p.add_argument('--out', type=Path)
    p.add_argument('--single', type=Path)
    p.add_argument('--scenario', choices=SCENARIOS)
    a = p.parse_args()
    result = []
    if a.single:
        require(a.scenario is not None, '--single requires --scenario')
        validate(load(a.single), a.scenario)
        print('receipt PASS')
        return
    require(a.reference is not None and a.candidate is not None and (a.out is not None), 'pair requires reference candidate --out')
    for scenario in SCENARIOS:
        lhs = validate(load(a.reference / (scenario + '.jsonl')), scenario)
        rhs = validate(load(a.candidate / (scenario + '.jsonl')), scenario)
        require(lhs == rhs, 'old/new difference: ' + scenario)
        result.append({'scenario': scenario, 'operations': len(lhs), 'status': 'PASS'})
    for path in [a.reference, a.candidate]:
        for fault in ['observer_failure', 'observer_overflow', 'observer_exception']:
            rows = load(path / (fault + '.jsonl'))
            require(rows[0]['failed'] == 1, 'fault was not injected')
            try:
                validate(rows, 'reuse')
            except ValueError:
                pass
            else:
                raise ValueError('observer failure accepted')
    a.out.write_text(json.dumps({'results': result, 'observer_failures_rejected': 6}, indent=2) + '\n')
    print('D1 paired qualification PASS')
if __name__ == '__main__':
    main()
