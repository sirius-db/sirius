#!/usr/bin/env python3
"""Key-matched drift analysis: join Sirius and oracle rows on their key column(s) so a
reordering caused by wrong values does not masquerade as a huge positional difference.
"""
import sys

KEYS = {  # query -> index of the key column(s) in the result
    "q03": [0], "q10": [0], "q15": [0], "q05": [0], "q07": [0, 1, 2],
    "q01": [0, 1], "q09": [0, 1], "q18": [1], "q21": [0],
}


def read(p):
    with open(p) as f:
        ls = [l.rstrip("\n") for l in f if l.strip()]
    return ls[0].split("\t"), [l.split("\t") for l in ls[1:]]


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


q, spath, opath = sys.argv[1], sys.argv[2], sys.argv[3]
ki = KEYS.get(q, [0])
shdr, srows = read(spath)
ohdr, orows = read(opath)
omap = {tuple(r[i] for i in ki): r for r in orows}

print(f"{q}: sirius={len(srows)} oracle={len(orows)} rows; key cols {[shdr[i] for i in ki]}")
worst, nbad, nmiss = 0.0, 0, 0
for si, s in enumerate(srows):
    k = tuple(s[i] for i in ki)
    o = omap.get(k)
    if o is None:
        nmiss += 1
        print(f"  row {si}: key {k} NOT IN ORACLE")
        continue
    oi = orows.index(o)
    for c, (a, b) in enumerate(zip(s, o)):
        fa, fb = num(a), num(b)
        if fa is None or fb is None:
            continue
        if fb == 0:
            continue
        d = abs(fa - fb) / abs(fb)
        if d > 1e-9:
            worst = max(worst, d)
            nbad += 1
            sign = "LOW " if fa < fb else "HIGH"
            print(f"  {shdr[c]:16} key={k[0]:>12}  sirius={a:>22} oracle={b:>22} {sign} {d*100:.4f}%"
                  + ("   [POSITION %d->%d]" % (si, oi) if si != oi else ""))
print(f"  => {nbad} differing cells, worst {worst*100:.4f}%, {nmiss} missing keys")
