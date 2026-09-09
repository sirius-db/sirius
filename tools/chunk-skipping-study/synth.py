import os
import numpy as np
rng=np.random.default_rng(7)
N=1<<24            # 16.7M rows
base=np.sort(rng.integers(0,1<<20,N))   # globally sorted key
def scramble(a,W):
    """shuffle within sliding windows of W rows -> clustering window W"""
    if W<=1: return a
    b=a.copy()
    for s in range(0,len(b),W):
        seg=b[s:s+W]; rng.shuffle(seg); b[s:s+W]=seg
    return b
def prune_frac(a,cs,lo,hi):
    n=(len(a)//cs)*cs
    m=a[:n].reshape(-1,cs)
    mn,mx=m.min(1),m.max(1)
    return 1.0-((mx>=lo)&(mn<=hi)).mean()

WINDOWS=[1,1024,8192,65536,262144,1048576,1<<24]
CHUNKS=[1024,4096,16384,65536,262144,1048576]
for sel in (0.01,0.1,0.5):
    span=int((1<<20)*sel); lo=int((1<<20)*0.3); hi=lo+span
    print(f"\n### selectivity {sel:.0%} : rows pruned (%) — rows are sorted then shuffled within a window W")
    print(f"{'W (clustering)':>16}"+''.join(f"{c:>9}" for c in CHUNKS))
    for W in WINDOWS:
        a=scramble(base,W)
        print(f"{('random' if W>=N else W):>16}"+''.join(f"{100*prune_frac(a,c,lo,hi):>9.1f}" for c in CHUNKS))
