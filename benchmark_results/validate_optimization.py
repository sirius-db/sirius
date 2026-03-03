#!/usr/bin/env python3
"""
Validate a Sirius optimization by comparing GPU outputs against CPU golden reference.

Usage:
    python3 validate_optimization.py <gpu_output_file>

Compares GPU query outputs against DuckDB CPU golden reference.
Reports correctness matches, timing improvements, and regressions.

Expected input format (from run_benchmark.sh):
    Each query block starts with "Q<NN>:" line, followed by:
    - gpu_buffer_init Success block + timing
    - 3x query result + timing (cold, hot1, hot2)
    HOT = min(hot1, hot2)
"""
import sys, re, json, csv, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_JSON = os.path.join(SCRIPT_DIR, "golden_outputs_10m.json")
BASELINE_CSV = os.path.join(SCRIPT_DIR, "rtx6000_10m_baseline.csv")

# Pre-existing known mismatches (don't flag these as new regressions)
KNOWN_MISMATCHES = {
    "Q03": "FP precision",
    "Q17": "tie-breaking (no ORDER BY)",
    "Q18": "tie-breaking (tied counts)",
    "Q21": "tie-breaking (tied counts)",
    "Q22": "pre-existing data diff",
    "Q24": "pre-existing data diff",
    "Q31": "tie-breaking (all c=1)",
    "Q32": "tie-breaking (all c=1)",
    "Q38": "pre-existing GPU OFFSET bug",
    "Q39": "pre-existing GPU OFFSET bug",
    "Q40": "pre-existing GPU OFFSET bug",
    "Q42": "pre-existing GPU OFFSET bug",
}

def extract_results(filepath):
    """Extract query results from benchmark output.

    Splits on Q<NN> markers, strips timings and Success blocks,
    takes the last table block from each query (handles 3 runs).
    """
    with open(filepath) as f:
        content = f.read()
    results = {}
    # Split on "Q00:" or "Q00" markers (with optional colon and trailing text)
    parts = re.split(r'(Q\d{2})(?::.*\n|\n)', content)
    i = 1
    while i < len(parts) - 1:
        qid = parts[i]
        block = parts[i+1]
        block = re.sub(r'Run Time \(s\):.*\n?', '', block)
        block = re.sub(r'┌─+┐\n│\s*Success\s*│\n│\s*boolean\s*│\n├─+┤\n│\s*0 rows\s*│\n└─+┘\n?', '', block)
        # Multiple runs: take the last table block (they should all be identical)
        table_blocks = re.findall(r'(┌[^┌]*┘)', block, re.DOTALL)
        if table_blocks:
            results[qid] = table_blocks[-1].strip()
        else:
            results[qid] = block.strip()
        i += 2
    return results

def extract_timings(filepath):
    """Extract HOT timings from benchmark output.

    Format per query: gpu_buffer_init timing, cold run timing, hot1 timing, hot2 timing
    = 4 timings per query. HOT = min(hot1, hot2).
    """
    with open(filepath) as f:
        content = f.read()

    # Split by Q markers to get per-query blocks
    parts = re.split(r'Q(\d{2})(?::.*\n|\n)', content)
    result = {}
    i = 1
    while i < len(parts) - 1:
        qid = f"Q{parts[i]}"
        block = parts[i+1]
        times = re.findall(r'Run Time \(s\): real ([\d.]+)', block)
        times = [float(t) for t in times]
        # times[0] = gpu_buffer_init, times[1] = cold, times[2] = hot1, times[3] = hot2
        if len(times) >= 4:
            result[qid] = min(times[2], times[3])
        elif len(times) == 3:
            # 3 timings: init + cold + 1 hot (or no init + 3 runs)
            result[qid] = times[2]
        i += 2
    return result

def load_baseline():
    baseline = {}
    with open(BASELINE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            baseline[row["query"]] = float(row["hot_s"])
    return baseline

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <gpu_raw_output_file>")
        sys.exit(1)

    gpu_output_file = sys.argv[1]

    # Load references
    with open(GOLDEN_JSON) as f:
        golden = json.load(f)
    cpu_golden = golden["cpu"]
    baseline = load_baseline()

    # Extract new GPU results
    new_gpu = extract_results(gpu_output_file)
    new_timings = extract_timings(gpu_output_file)

    print("=" * 70)
    print("OPTIMIZATION VALIDATION REPORT")
    print("=" * 70)

    # Correctness check
    print("\n--- CORRECTNESS (vs DuckDB CPU golden) ---\n")
    new_mismatches = []
    for qid in sorted(cpu_golden.keys()):
        if qid not in new_gpu:
            print(f"  {qid}: MISSING from GPU output!")
            new_mismatches.append(qid)
            continue
        if cpu_golden[qid] == new_gpu[qid]:
            continue  # exact match
        if qid in KNOWN_MISMATCHES:
            continue  # known pre-existing mismatch
        # New mismatch!
        new_mismatches.append(qid)
        print(f"  {qid}: NEW MISMATCH (was previously matching!)")
        print(f"    CPU: {cpu_golden[qid][:100]}...")
        print(f"    GPU: {new_gpu[qid][:100]}...")

    if not new_mismatches:
        print("  All queries: PASS (no new mismatches)")
    else:
        print(f"\n  FAIL: {len(new_mismatches)} NEW mismatches: {new_mismatches}")

    # Performance check
    print("\n--- PERFORMANCE (vs baseline) ---\n")
    regressions = []
    improvements = []
    for qid in sorted(baseline.keys()):
        if qid not in new_timings:
            continue
        old = baseline[qid]
        new = new_timings[qid]
        if old == 0:
            continue
        ratio = new / old
        change = (new - old) / old * 100
        marker = ""
        if ratio < 0.9:
            marker = " <<<< IMPROVED"
            improvements.append((qid, old, new, ratio))
        elif ratio > 1.15:
            marker = " !!!! REGRESSION"
            regressions.append((qid, old, new, ratio))
        print(f"  {qid}: {old:.3f}s -> {new:.3f}s ({change:+.1f}%){marker}")

    print()
    if improvements:
        print(f"  Improvements ({len(improvements)}):")
        for qid, old, new, ratio in improvements:
            print(f"    {qid}: {old:.3f}s -> {new:.3f}s ({ratio:.2f}x)")
    if regressions:
        print(f"\n  REGRESSIONS ({len(regressions)}):")
        for qid, old, new, ratio in regressions:
            print(f"    {qid}: {old:.3f}s -> {new:.3f}s ({ratio:.2f}x)")

    # Summary
    print("\n" + "=" * 70)
    total_old = sum(baseline.values())
    total_new = sum(new_timings.get(q, baseline.get(q, 0)) for q in baseline)
    print(f"Total: {total_old:.3f}s -> {total_new:.3f}s ({(total_new-total_old)/total_old*100:+.1f}%)")
    if new_mismatches:
        print("VERDICT: FAIL (new correctness issues)")
    elif regressions:
        print("VERDICT: WARNING (performance regressions detected)")
    else:
        print("VERDICT: PASS")
    print("=" * 70)

if __name__ == "__main__":
    main()
