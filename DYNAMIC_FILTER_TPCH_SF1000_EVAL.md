# TPC-H SF1000 Evaluation of Global Multi-Partition Dynamic Filters

**Date:** 2026-08-12
**Scope:** single-GPU correctness and performance evaluation of the global
multi-partition Bloom extension built on PR #1277
**Status:** complete; multi-GPU evaluation remains deferred

## 1. Executive conclusion

The global multi-partition filter is correct and measurably useful under the
controlled SF1000 configuration used for this study.

Relative to PR #1277's existing one-shot dynamic filters, enabling global
multi-partition construction reduced the sum of the 22 pooled query medians
from 27.4995 seconds to 25.8402 seconds, a 6.03% reduction. The corresponding
paired-block geometric-mean ratio was 0.94072, corresponding to 5.93% lower
runtime, and all three balanced timing blocks favored the global mode.

The queries that proved publication from more than one original build batch
were q8, q10, and q16. Their equal-query cohort C/B ratio was 0.90694,
corresponding to 9.31% lower runtime. The cohort-level geometric mean favored C
in all three blocks, although q8 individually was slower in blocks 2 and 3.
This aggregate needs qualification:

- q10 had clearly and consistently lower runtime: 19.74% by the paired-block
  ratio;
- q8 was neutral: 0.48% lower runtime, within control noise;
- q16 had 6.60% lower runtime in the point estimate, but its three-block interval
  crossed parity.

q5 improved by 53.02% and supplied 73.1% of the total seconds saved by C over
B. It used the multi-partition publication path, but its complete build arrived
as one original contribution, so it is evidence for extending publication to a
partitioned build, not evidence for the accumulator's multi-batch merge cost.
q10 supplied another 23.2% of the saved seconds. The true multi-batch cohort
supplied 25.5% of the net C-over-B time reduction.

The 256 MiB per-GPU policy cap rejected global Blooms for q3, q7, q9, q19, and
q21. That cohort was neutral versus one-shot mode: 0.06% higher runtime by its
paired-block geometric mean. The near-null timing is consistent with the
explicitly verified fail-open behavior.

The result supersedes the earlier statement that no TPC-H benefit had been
measured on this machine. It does not justify enabling the feature by default:
the study used deliberate 1 GB (1,000,000,000-byte) partition and CONCAT
settings that exposed the global path, ran on one GPU, and did not test a
production-natural config, a Bloom-cap sweep, or reduction and replication across GPUs. Keep
`enable_dynamic_filter_multi_partition=false` by default until those remaining
evaluations pass.

## 2. Compared modes

One accepted Sirius extension binary was used for all primary comparisons. The
three YAML files were mechanically checked to differ semantically only in the
two filter flags.

| Arm | Meaning | `enable_dynamic_filter` | `enable_dynamic_filter_multi_partition` | Config SHA256 |
| --- | --- | ---: | ---: | --- |
| A | All dynamic filters off | `false` | `false` | `a5b7d7e16ecf91dcad4ec02dce7cb7654299c4a5b1ca84c59db51aa6cf000740` |
| B | PR #1277 one-shot filters only | `true` | `false` | `cfa16ee8c6606b9c3e7d35bbaeb170c7b0ab4be0fdf9db096e87f77c71f89649` |
| C | One-shot plus global multi-partition filters | `true` | `true` | `458afd19ca2271ca4310b4412ea7c6baf28aec22d329656bfafe981fc507ab59` |

The primary incremental comparison is C/B. B/A measures the existing PR #1277
stack, while C/A measures the complete dynamic-filter stack. This separation is
important: comparing only C with A would incorrectly attribute PR #1277's
one-shot gains to the multi-partition extension.

Common operator settings included:

```yaml
scan_task_batch_size: 1GB
hash_partition_bytes: 1GB
concat_batch_bytes: 1GB
max_build_hash_table_bytes: 900MB
max_dynamic_filter_bloom_bytes_per_gpu: 268435456
dynamic_filter_domain_coverage_threshold: 0.9
enable_dynamic_zone_map_filter: false
```

The 1 GB (1,000,000,000-byte) partition and CONCAT sizes are study settings
chosen to expose non-broadcast, multi-partition builds. They should not be
interpreted as the machine's production-natural tuning.

## 3. Code and binary identity

The benchmark ran the accepted but uncommitted API-refactor worktree based on:

- branch: `codex/pr1277-multi-partition-dynamic-filter`;
- recorded HEAD: `13e0eb4078451f4b9a666e0bcf4a1d90dfb805f8`;
- Python DuckDB module: version 1.5.5 at
  `.pixi/envs/default/lib/python3.12/site-packages/_duckdb.cpython-312-aarch64-linux-gnu.so`;
- Python DuckDB module SHA256:
  `29d655c0e415dd964228122a267c13ecad56d2a2be4f378e7b93b9d70ee1317d`;
- Sirius extension SHA256:
  `053737e2824b6a527b1c4c4262cb34f99be23d1223ec102cd02c222f8bfa861d`.

Because the accepted refactor was not committed before measurement, the HEAD
alone does not identify the tested source, and the retained artifacts do not
contain the uncommitted diff. Exact reproduction requires that source diff and
build environment, or runtime binaries matching both hashes above. The hash
`6bcffc3b4e47b6e5ae903d5cda8f779b5810c46b529e37c3180c0bcb1f8286b3`
covers only the `git status --short --branch` listing; its equality before and
after the run proves that the dirty path/status set was unchanged, not that it
hashes the modified file contents.

## 4. Machine and dataset

| Item | Value |
| --- | --- |
| GPU | 1 x NVIDIA GB300, 256,703 MiB HBM |
| Driver / CUDA reported by `nvidia-smi` | 595.84 / 13.2 |
| CPU | 72-core Arm Neoverse-V2 |
| Host memory | 736 GiB, no swap |
| Storage | local NVMe; 3.7 TiB filesystem, 3.2 TiB free at preflight |
| OS | Ubuntu kernel 6.8.0-136-generic, aarch64 |
| Dataset | TPC-H SF1000 Parquet |
| Dataset inventory | 90 Parquet files plus metadata, approximately 265 GiB |
| Input location | `/localhome/local-kkristensen/Code/sirius/test_datasets/tpch_parquet_sf1000` |

The GPU was idle before each timing process and had no remaining compute
process and reported zero volatile uncorrected ECC errors after the run. Tables were host-pinned before each timed query; the
pinning operation itself was outside the recorded query interval.

## 5. Experimental method

The canonical runner was `test/tpch_performance/performance_test.py`, using
Parquet input, grouped mode, host pinning, and q1-q22.

The evaluation had three phases:

1. One debug-logged C run over all 22 queries established which joins armed,
   how many exact build contributions completed, and whether publication or a
   policy-cap skip occurred.
2. One C run with `--engine both --validation` checked Sirius against DuckDB for
   all 22 queries.
3. Nine separate GPU timing processes ran seven iterations of every query.
   The three balanced blocks used orders `B,C,A`, `C,A,B`, and `A,B,C`.

The runner successfully dropped the OS page cache once at the start of each of
the 11 benchmark processes; it did not reset the cache between grouped queries
or iterations.

Iteration 0 of each process was discarded. Each query and mode therefore has 18
warm observations: six iterations in each of three separate processes. The
report uses:

- a pooled median over those 18 observations for each query and mode;
- sums of pooled query medians as a duration-weighted suite description;
- within-block mode ratios, then their geometric mean, for the primary paired
  comparisons;
- a conventional log-ratio t interval over the three paired block ratios,
  assuming approximately independent, normally distributed log ratios;
- query-geometric-mean ratios for cohorts so one long query does not dominate.

With only three blocks, uncertainty estimates are low resolution. All three
suite C/B directions agree, but the exact two-sided sign test is still
`p=0.25`. The intervals should therefore be read as repeatability evidence on
this machine, not as a broad population claim.

## 6. Activation and policy outcomes

The activation run found candidate paths in nine queries, representing ten
Bloom targets because q9 had two. The final accepted-contribution count matched
the frozen build-batch set in every case; arm and terminal records also reported
the same frozen snapshot row count.

| Query | Exact contributions | Build rows | Hash partitions | Published filters | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| q3 | 7 | 147,060,214 | 3 | 0 | skipped: 294,120,448 B candidate exceeds cap |
| q5 | 1 | 45,508,350 | 2 | 1 | published; single original contribution |
| q7 | 12 | 145,884,740 | 5 | 0 | skipped: 291,769,600 B candidate exceeds cap |
| q8 | 3 | 91,139,462 | 3 | 1 | published; true multi-batch |
| q9 | 13 | 800,000,000 | 19 | 0 | skipped: two 1,600,000,000 B candidates exceed cap |
| q10 | 7 | 114,709,814 | 2 | 1 | published; true multi-batch |
| q16 | 10 | 29,700,215 | 2 | 1 | published; true multi-batch |
| q19 | 332 | 214,260,122 | 4 | 0 | skipped: 428,520,448 B candidate exceeds cap |
| q21 | 20 | 730,806,711 | 6 | 0 | skipped: 1,461,613,568 B candidate exceeds cap |

No contribution was missing at completion; every final accepted-contribution
count matched the armed batch set. There was no publication abort, snapshot
freeze failure, cache miss, execution error, or runtime CPU fallback. A cap skip is an expected eligibility outcome and leaves
the authoritative join unchanged.

## 7. Correctness

The canonical CPU/GPU validation passed 22 of 22 queries.

- q2-q22 were byte-identical between DuckDB and Sirius.
- q1 differed only in the final decimal representation
  (`0.049999601148178884` versus `0.04999960114817889`), approximately
  `6e-18`, and passed the runner's `1e-10` absolute tolerance.
- All 198 saved last-iteration timing result files, covering 22 queries x 3 modes
  x 3 blocks, were byte-identical per query across A, B, and C and to the Sirius
  correctness reference.

## 8. Whole-suite performance

| Metric | A: off | B: one-shot | C: global |
| --- | ---: | ---: | ---: |
| Sum of pooled query medians | 40.1688 s | 27.4995 s | 25.8402 s |
| Ratio to A, paired-block geometric mean | 1.00000 | 0.68521 | 0.64459 |
| Runtime delta from A, paired-block geometric mean | - | -31.48% | -35.54% |

| Comparison | Block 1 | Block 2 | Block 3 | Geometric mean | Runtime delta | Log-t 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C/B: incremental global | 0.9307 | 0.9427 | 0.9488 | 0.94072 | -5.93% | [0.9181, 0.9638] |
| B/A: one-shot stack | 0.6906 | 0.6841 | 0.6809 | 0.68521 | -31.48% | [0.6731, 0.6976] |
| C/A: complete stack | 0.6428 | 0.6450 | 0.6460 | 0.64459 | -35.54% | [0.6405, 0.6487] |

The all-query geometric mean, which weights each query equally rather than by
duration, measured C/B at 0.95646, or 4.35% lower runtime. The duration-weighted
and equal-query views therefore agree on direction.

## 9. Per-query performance

The seconds columns are pooled medians over 18 warm observations. The percentage
columns are paired-block geometric-mean runtime deltas; negative is faster.

| Query | A seconds | B seconds | C seconds | B/A | C/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| q1 | 1.493 | 1.490 | 1.492 | -0.38% | +0.37% |
| q2 | 0.498 | 0.385 | 0.395 | -21.73% | +0.85% |
| q3 | 1.997 | 1.900 | 1.900 | -4.67% | +0.55% |
| q4 | 1.309 | 0.870 | 0.856 | -33.70% | -1.46% |
| q5 | 2.298 | 2.282 | 1.070 | -0.94% | -53.02% |
| q6 | 0.640 | 0.640 | 0.643 | +0.01% | +0.71% |
| q7 | 1.759 | 1.115 | 1.127 | -36.88% | +1.31% |
| q8 | 2.869 | 1.220 | 1.215 | -57.45% | -0.48% |
| q9 | 3.749 | 2.492 | 2.483 | -33.48% | -0.27% |
| q10 | 2.120 | 1.944 | 1.559 | -8.68% | -19.74% |
| q11 | 0.396 | 0.226 | 0.229 | -42.13% | +2.57% |
| q12 | 1.360 | 1.166 | 1.161 | -14.30% | -0.28% |
| q13 | 1.424 | 1.418 | 1.425 | -0.23% | +0.28% |
| q14 | 0.818 | 0.821 | 0.819 | +0.97% | +0.04% |
| q15 | 0.737 | 0.733 | 0.750 | -0.34% | +1.93% |
| q16 | 0.565 | 0.561 | 0.527 | +0.23% | -6.60% |
| q17 | 2.190 | 0.815 | 0.810 | -62.92% | +0.28% |
| q18 | 2.311 | 1.626 | 1.628 | -29.67% | +0.13% |
| q19 | 1.844 | 1.843 | 1.856 | -0.34% | +1.08% |
| q20 | 0.959 | 0.798 | 0.803 | -17.07% | +0.77% |
| q21 | 8.396 | 2.871 | 2.810 | -65.57% | -2.33% |
| q22 | 0.437 | 0.284 | 0.283 | -35.19% | +0.48% |

Noncandidate C/B values are null controls for ordinary run variation.
Cap-skipped C/B values are negative controls for published-filter benefit and
include any global candidate-construction and cap-check overhead in addition to
run variation.

## 10. Published-query and cohort analysis

| Cohort or query | C/B block ratios | Geometric mean | Runtime delta | Interpretation |
| --- | --- | ---: | ---: | --- |
| q5 | 0.470 / 0.465 / 0.474 | 0.46982 | -53.02% | strong, stable; single contribution |
| q8 | 0.975 / 1.002 / 1.009 | 0.99516 | -0.48% | neutral at control-noise scale |
| q10 | 0.810 / 0.800 / 0.798 | 0.80260 | -19.74% | strong, stable, true multi-batch |
| q16 | 0.959 / 0.893 / 0.952 | 0.93401 | -6.60% | all blocks faster, block-variable |
| Published at default cap: q5/q8/q10/q16 | - | 0.76943 | -23.06% | equal-query geometric mean |
| True multi-batch: q8/q10/q16 | 0.912 / 0.894 / 0.915 | 0.90694 | -9.31% | all blocks faster |
| Cap skipped: q3/q7/q9/q19/q21 | 0.986 / 1.009 / 1.008 | 1.00060 | +0.06% | near-null; includes cap-path overhead |
| Noncandidate controls | 0.998 / 1.003 / 1.014 | 1.00509 | +0.51% | ordinary run noise |

For individual published queries, the low-df log-t ratio intervals were:

- q5: [0.459, 0.481];
- q8: [0.952, 1.040];
- q10: [0.786, 0.819];
- q16: [0.847, 1.030].

Among noncandidate controls, the median absolute C/B per-query delta was 0.48%,
the 90th percentile was 1.84%, and the maximum was 2.57%. The no-apply B/A
control was 1.00003. The no-apply cohort is q1, q6, q13, q14, and q15, as
classified by the separate C activation run. Within-process CV among
noncandidate controls had median 1.50%, p90 4.98%, and maximum 9.66%; for the
no-apply cohort these were 1.40%, 2.52%, and 4.67%. These controls support
treating q5 and q10 as material, q8 as neutral, and q16 as promising but not
conclusive.

## 11. Drift and outlier sensitivity

The balanced process order constrained systematic warm-up and position effects:

- last-position versus first-position drift was -0.03% on noncandidate controls;
- the same comparison was +0.30% on no-apply controls;
- warm iteration 6 had 0.75% lower runtime than warm iteration 1 overall.

Robust rules flagged 61 of 1,188 warm observations. Removing every flagged
observation changed the paired C/B suite result by only 0.034 percentage points.
The headline conclusion is therefore not driven by the flagged samples.

## 12. Limitations

This evaluation does not establish:

- multi-GPU correctness, reduction route behavior, or replication cost;
- performance with production-natural batch and partition settings;
- the cap threshold at which each skipped query becomes useful;
- memory-pressure behavior with concurrent joins;
- publication latency, probe keep ratio, exchange-byte savings, or per-device
  reduction and replication timing;
- whether q5's large gain generalizes beyond this physical plan and data shape.

Timing used `SIRIUS_LOG_LEVEL=error`; all nine raw timing Sirius logs are
zero-byte files because no error was emitted. Full consoles and controller logs
prove normal completion and contain no runtime CPU fallback banner, but
info-level plan-time fallback cannot be independently excluded from those
timing logs alone. Separate debug/info activation and correctness runs showed no
plan-time or runtime fallback. Activation cohorts were assigned from the
separate debug C run; error-level timing logs do not independently confirm
publication or cap-skip outcomes in each C timing process.

The activation and correctness logs contain 3,577 nonfatal warnings: 2,316 GPU
device-to-slot fallback diagnostics, 1,246 schema diagnostics, 13
already-drained/no-schedulable-partition warnings, and two host-space/NUMA-count
warnings. These deserve separate cleanup; result parity nevertheless passed.

## 13. Engineering recommendation

This evaluation supports retaining the global implementation, but it does not
establish superiority over the partition-specific CONCAT prototype; that
preference continues to rest on architectural considerations pending a direct
comparison. On this machine the global implementation produced a real
incremental benefit over PR #1277 without changing results. For q10 under this
configuration, the end-to-end C/B result is consistent with the merged global
Bloom's benefit exceeding all added global-path overheads. The study did not
isolate accumulation cost.

Retain the feature and its API refactor, but keep the subordinate switch off by
default for now. Before changing that default:

1. run the prepared multi-GPU matrix, including P2P and host-staged routes;
2. repeat A/B/C under production-natural settings and report how often the path
   activates without forced 1 GB (1,000,000,000-byte) partitioning;
3. sweep the Bloom cap around each candidate's required footprint, especially
   q3, q7, q19, and q21;
4. collect publication latency, probe rows kept, exchange bytes, and HBM peaks;
5. require q10-like wins to persist without material regressions on q8-like
   neutral workloads.

If multi-GPU reduction and strict replication erase the q10-class benefit, the
partition-specific CONCAT prototype remains a useful comparison. This
single-GPU evaluation, however, provides no evidence to replace the global
design.

## 14. Retained artifacts

The complete artifact tree is under `/tmp/tpch-pr1451-sf1000-eval` on the test
machine. Important entries are:

- `README.md`: execution handoff;
- `analysis/run-index.tsv`: exact run paths and order;
- `analysis/activation-summary.tsv`: contribution and publication evidence;
- `analysis/comprehensive-audit.txt`: 78 checks passed, zero failed;
- `analysis/result-cross-compare.tsv`: cross-mode byte comparison;
- `analysis/sf1000warm_20260812T025127Z_sf1000stats_summary.json`: statistical
  summary and cohort definitions;
- `analysis/sf1000warm_20260812T025127Z_sf1000stats_pooled_stats.tsv`: derived
  pooled per-query statistics;
- `analysis/sf1000warm_20260812T025127Z_sf1000stats_paired_query_ratios.tsv`:
  paired query effects and intervals;
- `analysis/sf1000warm_20260812T025127Z_sf1000stats_cohort_ratios.tsv`: cohort
  effects;
- `analysis/sf1000warm_20260812T025127Z_sf1000stats_outlier_sensitivity.tsv`:
  flagged-observation sensitivity analysis;
- `MANIFEST.sha256`: 420-file execution snapshot, whose own SHA256 is
  `701ecaf42f1d2b250275a340cc08f55c597d5e4d4754908063b5dddaf2d33ae5`.

The 420-entry manifest covers every other retained file, including all
statistics and raw execution artifacts; only `MANIFEST.sha256` itself is
excluded. A final `sha256sum -c MANIFEST.sha256` verification passed.
