#!/usr/bin/env python3
"""
P1b Calibration Script
Measures GPU sort vs hash groupby performance to derive:
  - K  (SIRIUS_P1B_K): minimum rows-per-group for P1b overhead to matter
  - DEDUP_THRESHOLD (SIRIUS_P1B_DEDUP_THRESHOLD): dedup ratio below which P1b helps

Run on the target GPU before deploying Sirius:
    python3 benchmark_results/calibrate_p1b.py

Then set the output env vars in your run script.

CAVEAT: This uses cudf Python API which has ~0.5ms overhead per call vs the C++ API
used in Sirius. Results are conservative for small inputs (< 100K rows) but accurate
for the large-input regime where the P1b guard matters most.
"""

import cudf
import cupy as cp
import numpy as np
import time
import sys

NUM_TRIALS = 5
WARMUP = 2


def cuda_time(fn, trials=NUM_TRIALS, warmup=WARMUP):
    """Time a cudf operation using CUDA events. Returns min time in seconds."""
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(trials):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end) / 1000.0)
    return min(times)


def make_df(n, g):
    """n rows, g distinct group keys, random INT64 values."""
    keys = cudf.Series(cp.random.randint(0, g, n, dtype=cp.int64), name="k")
    vals = cudf.Series(cp.random.randint(0, 10_000_000, n, dtype=cp.int64), name="v")
    return cudf.DataFrame({"k": keys, "v": vals})


def make_dedup_df(n, g, dedup_ratio):
    """
    n rows, g group keys, dedup_ratio controls distinct (k,v) pairs.
    dedup_ratio=1.0 means every (k,v) pair is unique (no dedup benefit).
    dedup_ratio=0.1 means 90% of (k,v) pairs are duplicates.
    """
    if dedup_ratio >= 1.0:
        # Truly unique pairs: use sequential values to guarantee no collisions
        # (random values would have birthday-paradox collisions)
        keys = cudf.Series(cp.random.randint(0, g, n, dtype=cp.int64), name="k")
        vals = cudf.Series(cp.arange(n, dtype=cp.int64), name="v")
        return cudf.DataFrame({"k": keys, "v": vals})

    n_distinct = max(g, int(n * dedup_ratio))
    # Generate n_distinct unique (k, v) pairs
    unique_k = cp.random.randint(0, g, n_distinct, dtype=cp.int64)
    unique_v = cp.random.randint(0, 10_000_000, n_distinct, dtype=cp.int64)
    # Sample n rows from the distinct pairs (with replacement to get dedup)
    idx = cp.random.randint(0, n_distinct, n, dtype=cp.int64)
    keys = cudf.Series(unique_k[idx], name="k")
    vals = cudf.Series(unique_v[idx], name="v")
    return cudf.DataFrame({"k": keys, "v": vals})


def time_nunique(df):
    return cuda_time(lambda: df.groupby("k").agg({"v": "nunique"}))


def time_p1b(df, g):
    """Simulate P1b: regular groupby for SUM + distinct + count groupby + merge."""
    def run():
        # Step 1: regular groupby for other aggregates
        regular = df.groupby("k").agg({"v": "sum"})
        # Step 2: distinct (k, v) pairs
        distinct = df.drop_duplicates()
        # Step 3: count groupby on distinct pairs
        counts = distinct.groupby("k").agg({"v": "count"})
        # Step 4: merge
        regular.merge(counts, on="k", how="left")
    return cuda_time(run)


def gpu_mem_free_gb():
    """Return free GPU memory in GB."""
    free, total = cp.cuda.runtime.memGetInfo()
    return free / (1024**3)


def safe_row_count(n, safety_factor=3):
    """Check if we can allocate n rows of 2x INT64 with safety margin."""
    # 2 INT64 columns = 16 bytes/row, safety_factor accounts for intermediate buffers
    needed_gb = (n * 16 * safety_factor) / (1024**3)
    return needed_gb < gpu_mem_free_gb()


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Find K — break-even rows-per-group at dedup_ratio=1.0 (worst case)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("Part 1: Finding K (break-even rows/group at zero dedup benefit)")
print("="*65)
print(f"  {'Config':<45} {'nunique':>10} {'p1b':>10} {'ratio':>8}")
print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*8}")

results_part1 = []
for n in [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]:
    if not safe_row_count(n):
        print(f"  Skipping n={n:,} (insufficient GPU memory)")
        continue
    for g in [100, 1_000, 10_000]:
        rows_per_group = n // g
        if rows_per_group < 10:
            continue
        df = make_dedup_df(n, g, dedup_ratio=1.0)
        t_nu = time_nunique(df)
        t_p1 = time_p1b(df, g)
        ratio = t_p1 / t_nu
        label = f"n={n:>10,} g={g:>6,} rows/g={rows_per_group:>6,}"
        print(f"  {label:<45} {t_nu*1000:>9.1f}ms {t_p1*1000:>9.1f}ms {ratio:>7.2f}x")
        results_part1.append((rows_per_group, ratio, n, g))
    sys.stdout.flush()

# Find K: smallest rows_per_group where P1b becomes meaningfully slower (ratio > 1.1)
# At dedup_ratio=1.0, P1b should always be slower (no dedup benefit + extra overhead).
# We want the threshold where the slowdown becomes significant (>10%).
# Small ratios near 1.0 may just be noise.
break_even_rpg = None
for rpg, ratio, n, g in sorted(results_part1):
    if ratio > 1.1:  # 10% significance threshold
        break_even_rpg = rpg
        break

print()
if break_even_rpg:
    K = break_even_rpg
    print(f"  -> K = {K:,}  (P1b >10% slower when rows/group > {K:,} and dedup_ratio=1.0)")
else:
    K = 100  # safe default
    print(f"  -> K not found in tested range; using conservative default K={K:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Find DEDUP_THRESHOLD — at large n, find dedup ratio where P1b wins
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("Part 2: Finding DEDUP_THRESHOLD (at large n, fixed g=1000)")
print("="*65)
print(f"  {'Config':<45} {'nunique':>10} {'p1b':>10} {'ratio':>8}")
print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*8}")

# Use the largest size that fits in GPU memory
N_LARGE = 50_000_000
while N_LARGE > 1_000_000 and not safe_row_count(N_LARGE):
    N_LARGE //= 2
    print(f"  Reducing N_LARGE to {N_LARGE:,} due to GPU memory constraints")

G_FIXED = 1_000

results_part2 = []
for dedup_ratio in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05]:
    df = make_dedup_df(N_LARGE, G_FIXED, dedup_ratio)
    t_nu = time_nunique(df)
    t_p1 = time_p1b(df, G_FIXED)
    ratio = t_p1 / t_nu
    label = f"n={N_LARGE:>10,} g={G_FIXED:>4,} dedup={dedup_ratio:.2f}"
    print(f"  {label:<45} {t_nu*1000:>9.1f}ms {t_p1*1000:>9.1f}ms {ratio:>7.2f}x")
    results_part2.append((dedup_ratio, ratio))
    sys.stdout.flush()

# Find DEDUP_THRESHOLD: highest dedup_ratio where P1b is still faster (ratio < 1.0)
dedup_threshold = 0.0
for dedup_ratio, ratio in results_part2:
    if ratio < 1.0:
        dedup_threshold = max(dedup_threshold, dedup_ratio)

if dedup_threshold == 0.0:
    dedup_threshold = 0.5  # safe default
    print(f"\n  -> P1b never faster in tested range; conservative DEDUP_THRESHOLD={dedup_threshold}")
else:
    # Add a small safety margin above the measured break-even
    dedup_threshold = min(1.0, dedup_threshold + 0.1)
    print(f"\n  -> DEDUP_THRESHOLD = {dedup_threshold:.2f}  (P1b wins when dedup_ratio < {dedup_threshold:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("Summary — set these env vars before running Sirius:")
print("="*65)
print(f"\n  export SIRIUS_P1B_K={K}")
print(f"  export SIRIUS_P1B_DEDUP_THRESHOLD={dedup_threshold:.2f}")
print()
print("Interpretation:")
print(f"  P1b guard skips the optimization when BOTH:")
print(f"    - input_rows > {K} x estimated_output_groups  (large input)")
print(f"    - expected_distinct_pairs / input_rows > {dedup_threshold:.2f}  (low dedup benefit)")
print()
print("To use with Sirius benchmark:")
print(f"  SIRIUS_P1B_K={K} SIRIUS_P1B_DEDUP_THRESHOLD={dedup_threshold:.2f} \\")
print(f"  bash benchmark_results/run_benchmark.sh clickbench_100m.duckdb")
print()
print("NOTE: These measurements use Python cudf API which adds ~0.5ms overhead")
print("per operation vs C++ cudf. For the large-input regime where the guard")
print("matters, this overhead is negligible relative to the operation time.")
print()
