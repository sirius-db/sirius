# Q21 comparison: main vs PR #2468 vs kept-side semi/anti build

`run.sh` builds three Sirius versions and runs TPC-H q21 and the full q1-q22 suite on each, under
`delim_join_as_cte = true` and `false`, then writes `results/summary.md`.

| name | branch | commit measured on the GB300 |
|---|---|---|
| `main` | `sirius-db/sirius` `main` | `e3935a660` |
| `pr2468` | [PR #2468](https://github.com/sirius-db/sirius/pull/2468), `feat/exists-minmax-rewrite-bindings` | `3a7c5712f` |
| `kept` | `joosthooz/sirius` `feat/semi-anti-build-left` | see `git log` on that branch |

`pr2468` merges sibling correlated `EXISTS` / `NOT EXISTS` on one table (q21's two `lineitem`
subqueries) into one scan and a per-key MIN/MAX aggregate. With `delim_join_as_cte = true` it keeps
the joins unmerged when every subquery scan is pinned in GPU memory and the estimated join fits the
GPU budget; with `false` it always merges. Its decisions are logged as
`[sirius_plan_exists_rewrite]`, which the summary counts.

`kept` builds non-delim RIGHT_SEMI / RIGHT_ANTI joins (`delim_join_as_cte = true`) on the small kept
side and streams the large side through, instead of concatenating and hashing the large side whole.
`SET semi_join_kept_build_ratio = 0` restores main's plan on that build.

## Running

On a fresh machine with the NVIDIA driver, `git` and [pixi](https://pixi.sh):

```bash
git clone https://github.com/joosthooz/sirius.git && cd sirius
git checkout feat/semi-anti-build-left
# SF3000 from S3, the RTX 4500 configuration that blew up on main:
bench/q21-compare/run.sh --sf 3000 --data s3://<bucket>/tpch_sf3000 --work ~/q21-work
# SF1000 with tables pinned in GPU memory (GB300-class GPUs only):
for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
  export SIRIUS_PIN_TIER_$t=gpu
done
bench/q21-compare/run.sh --sf 1000 --data /datasets/tpch_sf1000 --pin host \
  --config bench/q21-compare/sirius-gb300-sf1000.yaml
# A 32 GB budget on a larger GPU, to reproduce memory pressure:
bench/q21-compare/run.sh --sf 1000 --data /datasets/tpch_sf1000 --gpu-limit 32GB
```

- Without `--data`, TPC-H parquet is generated into `<work>/data/tpch_sf<N>` with `tpchgen-cli`
  when it is on `PATH` (`cargo install tpchgen-cli`), else with DuckDB's `dbgen`, which is slow
  and memory-hungry above SF100.
- S3 input uses the AWS credentials in the environment, as `test/tpch_performance/performance_test.py`
  does.
- `--branches "main kept"` limits the set; `--skip-full` runs only q21; `--skip-build` reuses
  builds in `<work>/src/<name>`.
- `sirius-small-gpu.yaml` (the default config) keeps Sirius' device-derived memory defaults and a
  1 GB scan batch. `sirius-gb300-sf1000.yaml` is the GB300 SF1000 configuration; its host and disk
  settings are specific to that machine.
- Each configuration takes 3 iterations per query. The summary reports the median per query, the
  sum of medians (TOTAL rows are ignored), and the number of OOM reschedules
  (`reschedule (retry` log lines).

## Reference numbers

The original blow-up: RTX 4500 (32 GB), SF3000 from S3, `delim_join_as_cte = true`: q21 took 271 s
on main with 329 OOM retries, and 57 s with #2468's merge.

GB300 (256 GB), SF1000, median of 3, q21 (s):

| configuration | cte | main | #2468 | kept |
|---|---|---|---|---|
| tables pinned in GPU memory | true | 0.477 | 0.477 | 0.436 |
| tables pinned in GPU memory | false | 0.526 | 0.489 | 0.525 |
| tables pinned in host memory | true | 0.674 | 0.633 | 0.643 |
| tables pinned in host memory | false | 0.665 | 0.569 | 0.725 |
| host-pinned, 32 GB GPU limit | true | 0.590 | 0.620 | 0.545 |
| host-pinned, 32 GB GPU limit | false | 0.712 | 0.568 | 0.718 |
| not pinned (parquet scan each run) | true | 8.395 | 6.667 | 8.296 |
| not pinned (parquet scan each run) | false | 8.222 | 6.796 | 8.317 |

Sum of q1-q22 medians (s): GPU-pinned cte=true 4.084 / 4.084 / 4.052, cte=false 4.137 / 4.100 /
4.141; host-pinned cte=true 7.320 / 7.253 / 7.255, cte=false 7.299 / 7.205 / 7.379. #2468 merged
q21 in every configuration except GPU-pinned cte=true. No run on the GB300 logged an OOM
reschedule, including with the 32 GB limit, so the RTX 4500 blow-up has to be checked on that
hardware. The host-pinned runs vary by about 0.05 s between repeats; `kept` executes main's plan
when `delim_join_as_cte = false`.
