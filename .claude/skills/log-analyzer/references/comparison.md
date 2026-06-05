# Two-run comparison playbook

The comparison workflow is the highest-leverage use of this skill — it turns "this query is wrong/slow now" into "operator X is where things first go off the rails."

This document is dense on the matching rules because they're the part that most often goes wrong.

## Step 0: Identify the two runs

Two runs can live in:

- **The same log file** — common in validation testing. A workload runs the same SQL many times; one of them fails. Read `_summary.json -> queries.complete_queries` and `incomplete_queries`; ask the user which timestamp is the "bad" run.
- **Two different log files** — re-run after a code change, before/after a config flip. Parse both with `parse_logs.py` to separate output dirs.

## Step 1: Confirm the runs are comparable (DO NOT SKIP)

Two queries are comparable only if **both** of the following are true. If either fails, *stop* and tell the user the runs are not comparable.

### Rule 1: Full SQL string matches

The `sql_preview` in `_index.csv` is a 120-char convenience field the parser derives for display. The Sirius log itself captures the full SQL on `QueryBegin`, so the authoritative text lives at `<folder>/query.sql` — don't rely on the preview as proof of equivalence.

To verify the full SQL matches:
```bash
diff <out_dir>/<good_ts>/query.sql <out_dir>/<bad_ts>/query.sql
```

If diff returns non-empty, the SQL differs — even by a single character. Two queries that look identical in the preview often differ in WHERE constants, projection lists, or LIMIT clauses.

### Rule 2: Operator list matches

Same SQL doesn't guarantee same plan. Compare the operator inventory between the two runs:

```bash
# Quick visual check:
python3 -c "
import json
a = json.load(open('<good_folder>/pipeline_plan.json'))
b = json.load(open('<bad_folder>/pipeline_plan.json'))
print('good:', a['counts']['operator_types'])
print(' bad:', b['counts']['operator_types'])
"
```

If `counts.operator_types` (a dict of `{op_type: count}`) differs between the two, the *plan* differs — even if the SQL string is identical.

### When no comparable run exists in the current log

If the bad run's operator inventory (`counts.operator_types`) doesn't match any other run of the same SQL in the current log, **stop and ask the user before doing anything else**:

> "I found N other runs of this SQL in the log, but none have a matching operator inventory (e.g. the bad run uses LEFT_DELIM_JOIN + GPU_PARQUET_SCAN, the others use RIGHT_DELIM_JOIN + DUCKDB_SCAN). Do you have another log — a successful test run, or a different failure log that happened to also run this query — where the same query executed with the same operators? If so, share the path and I'll parse it and compare."

**Do not** silently fall back to comparing across mismatched plans. Trap B (next section) explains why scan-path differences make cross-path diffs untrustworthy. The same logic applies to other plan differences (LEFT vs RIGHT DELIM_JOIN, missing or extra operators) — these are legitimate planner decisions, not bugs, and a row-count divergence across them tells you nothing.

The common case where this works: a project usually has many log files (one per test run, one per failure). A workload that runs Q4 on the parquet path with a particular set of operators is likely to have run with the *same* operator set in another log. Parse that other log and pair the runs across the two parser-output directories. The matching rule (full SQL + operator inventory) is the same; only the file paths differ.

## Step 2: Watch for two common false-match traps

### Trap A: `sql_preview` collision

The Sirius log captures the full SQL on `QueryBegin`, but the parser derives a 120-char `sql_preview` field for display in `_index.csv` (and in some report tables). Two different queries that share the same first ~120 characters will look identical in the preview even though their full SQL differs. Always diff `<folder>/query.sql` (the authoritative full text) before declaring two runs equivalent — and run the operator-list check (Rule 2) regardless, since identical SQL can still yield different plans.

### Trap B: Different scan source

`GPU_PARQUET_SCAN` (Sirius native Parquet path) vs `DUCKDB_SCAN` (DuckDB read with table scan on top) vs `TABLE_SCAN` (forwarded from a DuckDB child) all answer "where does the data come from." Same SQL can produce different scan op types based on:

- Whether the file is registered as a Parquet view vs a native table
- The `gpu_execution` config flag and other Sirius config
- Fallback (Sirius punted the scan back to DuckDB)

If the two runs differ only in their scan-op type, they're the *same logical query on a different execution path*. Differences in time and memory are *expected*, not a bug. Surface this as a finding (not a diagnosis) and ask the user whether they want a within-path comparison.

## Step 3: Compare aggregates pipeline-by-pipeline

Once Step 1 and Step 2 pass, you can confidently diff.

**Inputs:** `_pipeline_aggregates.csv` from both runs (or both queries within the same parse output), filtered to the two `query_begin_ts` values.

Pair pipelines across the runs by matching **both** `pipeline_num` AND the operator-chain string (e.g. `"HASH_JOIN -> FILTER -> PROJECTION"`) — both must be equal. For two runs of the same SQL on the same plan, pipeline numbers are stable, and the chain check protects against mis-pairing when a run actually has a different plan slipping through. Don't pair by chain alone — chains repeat (multiple PARTITIONs, multiple CONCATs) and a chain-only match silently pairs the wrong pipelines.

For each paired pipeline, compute the deltas on these columns:

| Column | Why it matters |
| --- | --- |
| `sum_output_num_rows` | Wrong-result diagnosis. If outputs differ in row count, the operator is producing different data. |
| `sum_output_size_bytes` | Same as above but catches changes in column widths / encoding. |
| `sum_execution_time_ms` | Perf diff. |
| `max_execution_time_ms` | Catches new straggler-task behavior. |
| `sum_history_peak_bytes_to_materialize_input` | Consumer-side downgrade behavior (tasks re-reading evicted inputs) — see single_query.md #4. |
| `min_memory_available` | New memory pressure. |

Also compare the per-query downgrades directly: total bytes evicted, source tier breakdown, and host-vs-disk destination. These come from `<folder>/downgrades.csv` in each run (sum `total_bytes` grouped by `source_tier`; sum `to_host_bytes` / `to_disk_bytes`). A run that's slower because it now spills to disk where the other run didn't is a common pattern and shows up here, not in the per-pipeline aggregate.

## Step 4: The validation-error walk

This is the killer use case. Two runs, same SQL + same operators, but the bad run produces wrong results.

The goal is to find the **earliest pipeline whose output diverges between the two runs.** That pipeline contains the operator responsible for the bug.

Algorithm:

1. From `pipeline_plan.json`, get `leaves` (scan pipelines) and `dependencies` for each pipeline.
2. Do a topological walk from leaves toward `root_pipeline` (the RESULT_COLLECTOR pipeline).
3. At each pipeline, compare `sum_output_num_rows` and `sum_output_size_bytes` between the two runs.
4. The first pipeline where these diverge is the suspect.

Things to watch for:

- If `sum_output_num_rows` diverges but `sum_output_size_bytes` does not — likely a row-ordering issue downstream of a partitioner. (Could be benign for SQL that doesn't enforce ORDER BY.)
- If `sum_output_size_bytes` diverges but `sum_output_num_rows` matches — same row count, different content. Often a value-corruption bug or wrong projection.
- If the divergent pipeline contains a HASH_JOIN and the bad run has fewer output rows — likely a build-side hash collision or a key-type mismatch.
- If divergence first appears at a scan pipeline — the input was read differently. Check whether scan op types match (Trap B).

Surface the divergent pipeline and its operators to the user. Don't claim "this operator has a bug" — claim "this is the first place outputs diverge; it's the most likely source." The user (or a follow-up skill like race-check) takes it from there.

### If the walk finds no divergence

If every paired pipeline matches on rows and bytes, just report that plainly: "No divergence found in the captured aggregate metrics between these two runs." Don't speculate about what kind of bug it could be (e.g. don't claim it must be a content/value bug). The most likely explanation is that the comparison pair wasn't right — perhaps a different good run would show the divergence. Tell the user the negative result and ask whether there are other good runs (in other logs, or other matching timestamps in the same log) worth trying.

## Step 5: Comparing against an incomplete run

If the bad run is `status: incomplete` (likely a crash or hang), it didn't reach RESULT_COLLECTOR. You can still diagnose:

1. From the incomplete run's `_pipeline_aggregates.csv`, enumerate which pipeline IDs have any rows (i.e. produced any task output).
2. For those same pipelines (matched by pipeline_num + operator chain) in the good run, compare aggregates as in Step 3.
3. The last pipeline the incomplete run touched is the most informative: did it produce output? Did it diverge from the good run? Did it have unusual memory pressure?

For hangs specifically, the *gap* between the last task in the incomplete run and the end-of-log timestamp is roughly the hang duration. Hand off to **runtime-errors** at that point.

## Step 6: Report

Save a comparison report at `<parser_output>/COMPARE_<good_ts>_vs_<bad_ts>.md`. Use the template in the main SKILL.md. Key sections:

- **Matched?** Confirm both rules passed.
- **Earliest divergence** — the operator/pipeline.
- **Per-pipeline deltas** — a table with the key columns from Step 3.
- **Recommended next step** — race-check / runtime-errors / optimization-advisor / validate.
