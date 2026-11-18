---
name: distributed-sirius-bench
description: >
  Run a query set on Distributed Sirius through its gateway, one rank per GPU, on one node or
  several, with or without Slurm: TPC-H by default, or any dataset with a folder of .sql files.
  Plan, preflight, build, a smoke test, then the full benchmark, with the Sirius configuration
  shown and approved before anything starts. Use it for a distributed smoke test or benchmark of
  a branch. For a single-process run, use the benchmark skill.
disable-model-invocation: true
---

# distributed-sirius-bench

Runs a query set on Distributed Sirius: one gateway, one engine process (a rank) per GPU, SQL sent
through the client. The query set is TPC-H by default, or any dataset you point it at (section
1a). `scripts/bench.py`, next to this file, does the mechanical work and keeps
everything about a run in one run folder. Your part is to collect the inputs, get the user's
approval at each gate, and read the results critically.

This skill is for runs through the gateway ([Distributed experiments](../../../docs/super-sirius/distributed-experiments.md)).
For a single-process run with `performance_test.py`, use the `benchmark` skill.

The gates, in order: **collect, confirm, preflight, build, show the config, smoke test, full
benchmark, report.** Do not skip one. GPU nodes are shared and a full benchmark can hold them for
hours, so nothing that uses a node starts before the user has agreed to it in this conversation.
The script enforces this: without a terminal, every such step refuses to run unless it gets
`--yes`, and you pass `--yes` only after the user approved that step.

## 1. Collect the inputs

Ask the user for each of these. Do not guess, and do not reuse a value from an earlier run or from
memory without confirming it: a wrong dataset or branch produces a number that looks valid.
Where a question tool exists (AskUserQuestion in Claude Code), use it; otherwise ask in plain text.

| Input | Flag | Notes |
|---|---|---|
| Branch to test | `--branch` | |
| Benchmark | `--benchmark tpch` or `custom` | TPC-H (default), or any dataset and queries: section 1a |
| Dataset location | `--data` | A folder with one subfolder of Parquet files per table |
| Scale factor | `--sf` | TPC-H: required; the smoke test checks it against the `supplier` row count, and Q11 uses it. Custom: optional, shown in the reports |
| Execution mode | `--mode single` or `--mode multi` | Single: one node, all its GPUs. Multi: several nodes, all GPUs of each. For multi also ask which nodes (`--nodes a,b`) or, with Slurm, how many (`--num-nodes`) |
| Sirius configuration | `--config` | Run `bench.py configs` and show the list: a preset, the user's own YAML file, or `engine-defaults` |

You also need to know where things go. Ask if the conversation does not already say:

| Input | Flag | Notes |
|---|---|---|
| Checkout folder | `--checkout` | Used if it exists, cloned there if not. On a file system every node mounts. The checkout you are working in is the natural choice |
| Run folder | `--out` | New, visible from this host and from the nodes, usually under `$HOME` |
| Launcher | `--launcher slurm` or `local` | Slurm whenever Slurm schedules the nodes. `local` is only for machines it does not |
| Slurm partition, site options | `--partition`, `--sbatch-args`, `--time-limit` | |
| Spill folder | `--spill-dir` | Needed by the presets; on a disk with free space |
| Large output | `--big-dir` | The client output reaches gigabytes at large scale factors. Keep it off a `$HOME` near its quota |
| Extra environment | `--env KEY=VALUE` | For the gateway and the ranks. Where `$HOME` has a quota, `--env PIXI_CACHE_DIR=<a folder on scratch>` keeps Pixi's cache out of it |

Other flags have defaults that rarely change: `--runs 2` (cold, then warm), `--queries` (TPC-H:
`1-22`), `--smoke-queries` (TPC-H: `1,6,12`), `--statement-timeout`, `--setup-sql`,
`--env KEY=VALUE`. `bench.py plan --help` lists them all.

## 1a. Any other dataset

With `--benchmark custom` the script makes no assumption about the tables or the queries:

| Input | Flag | Notes |
|---|---|---|
| Queries | `--queries-dir` | Required. A folder of `.sql` files, one query per file. A query's name is its file name without `.sql` (letters, digits, `_`, `.`, `-`) |
| Which queries | `--queries` | `all` (default), names (`q7,q12`), or numbers and ranges (`1-5`), which match the number at the end of a file name |
| Smoke test | `--smoke-queries` | Same forms. Default: the first three queries, in natural order |
| Tables | `--tables` | Default `auto`: one view per subfolder of `--data` that holds Parquet files, named after the subfolder. Or a comma-separated list |
| Views | `--views-sql` | Instead of the generated views, a file whose statements create the tables or views: CSV, another layout, a DuckDB file. `--data` is then optional |
| Report name | `--name` | Default: the dataset folder's name |

The TPC-H specifics (the Q11 fraction and the `supplier` row-count check) apply only to
`--benchmark tpch`. A custom smoke test passes on the other checks: no error, a time for every
query, every rank up and ready. So ask the user how they will know the answers are right (a
reference engine, expected files), and say in the report if nothing checked them.

Run the script on a host where it can start jobs: with Slurm, one where `sbatch` works. If that is
not the host you are on, run every command below there (for example over `ssh` to the head node,
never to a compute node). The skill folder and the run folder must be visible from that host.

## 2. Plan and confirm

```bash
python3 <this folder>/scripts/bench.py plan --branch ... --data ... --sf ... --mode ... \
  --config ... --checkout ... --out ... [--nodes ...] [--partition ...] [--spill-dir ...] --yes
```

It validates the inputs, writes `plan.json` and the resolved `config.yaml` into the run folder, and
prints the plan. Show that output to the user and ask whether every line is right. If one is not,
run `plan` again with `--replace`. A person at a terminal can run `plan` with no flags and answer
its questions instead.

## 3. Preflight and build

```bash
python3 .../bench.py preflight --out <run folder> --yes   # short checks on the nodes
python3 .../bench.py build     --out <run folder> --yes   # clone if absent, then build
```

- Preflight checks the tools on this host and, on each node, that every table has Parquet files
  and that the run folder is visible. It counts the GPUs, which gives the number of ranks. For a
  custom benchmark it also finds the tables (with `--tables auto`) and resolves the query
  selection against the files in `--queries-dir`; show the user the lists it prints.
- Build reuses a build that is already there and says which commit it is. It builds only when
  there is none, or with `--rebuild`. A fresh build takes 35 to 40 minutes on 144 cores, so under
  Slurm it runs as its own job on one node.
- The script never switches the branch of an existing checkout. If the checkout is on another
  branch it stops and says so; let the user decide.
- If the clone or `pixi` fails for lack of network on the compute nodes, clone and run
  `pixi install` where there is network, then run `build` again.

## 4. Show the Sirius configuration, then the smoke test

```bash
python3 .../bench.py show  --out <run folder>
python3 .../bench.py smoke --out <run folder> --yes
```

`show` prints the plan again, now with the rank count and the build's commit, followed by the full
Sirius configuration: the YAML file and the environment variables the gateway and ranks will get.
Put that in front of the user and wait for a yes before the smoke test.

The smoke test starts the gateway and the ranks, runs a few queries once each, and passes only if
all of these hold: the client exited cleanly, no statement returned an error, every query has a
time, every rank is `up` and `ready` in the final status, and, for TPC-H, `supplier` has
`10000 x SF` rows.
`--dry-run` writes the launch script and the SQL without starting anything, which is a good way to
show the user exactly what will run.

If it fails, stop and diagnose (section 7). Do not go on to the full benchmark, and do not rerun
in a loop: after a job whose ranks were killed, wait until the node is idle again.

## 5. Full benchmark

Tell the user the smoke result and ask before starting:

```bash
python3 .../bench.py full --out <run folder> --yes
```

It refuses to run unless the smoke test passed for this plan. A full run can take from minutes to
hours, longer than a tool call may wait. Start it in the background, with its output in a file,
and look at progress with `report`, which works while the run is going:

```bash
nohup python3 .../bench.py full --out <run folder> --yes > <run folder>/full.log 2>&1 &
python3 .../bench.py report --out <run folder>
```

Interrupting the script cancels the Slurm job (or stops the gateway and ranks). If you stop it any
other way, check that nothing is left: `squeue -u $USER`, or on a machine without Slurm
`nvidia-smi --query-compute-apps=pid --format=csv,noheader` must print nothing.

## 6. Report

`summary.md` and `timings.csv` are in `<run folder>/smoke/` and `<run folder>/full/`. Give the user:

1. Pass or fail, and every line under "Problems", quoted.
2. The totals: run 1 (cold), run 2 (warm) and best of both, with the per-query table.
3. What was run: branch and commit, nodes and ranks, dataset and scale factor, the Sirius config.
4. What was not checked. By default that includes the answers.

Read the numbers before passing them on. A query far slower than its neighbours, or than an
earlier run, is more often storage or memory than the engine: look in the engine logs under
`<run folder>/<kind>/rank-logs/` for `[gds] held:` lines (a storage stall) and for the exchange
path of each pair of ranks (on GB200 NVL it should read `NIXL over the NVLink fabric`). Single runs
on shared storage differ by around ten percent between days and nodes, so do not call a smaller
difference a change.

**Check the answers.** Times mean little if the answers are wrong, and the script does not compare
them. With a reference for the same data and the same query texts, use the branch's own tool:

```bash
python3 <checkout>/scripts/compare_answers.py <client.txt> --ref-client <reference client output> \
  --ties 3:revenue,o_orderdate --ties 18:o_totalprice,o_orderdate
```

The reference can be the same `run.sql` (without its last line) run through the build's plain
DuckDB on the CPU; at SF10000 that needs about 15 minutes and 1 TB of memory, so run it as its own
job and ask first. If no reference exists, say in the report that the answers were not checked.

## 7. When something fails

| Symptom | Where to look |
|---|---|
| `preflight failed` | The lines under it name the node and what is missing |
| `job ... ended before the gateway was up` | `<run folder>/<kind>/job.out`. `rank exited` repeating there means the ranks die at start, usually from the config |
| The client output says `the gateway starts once N ranks are registered` | Fewer ranks joined than expected: `job.out`, then the engine logs |
| An `Error` line for a query | The line itself; CPU fallback is off, so a query Sirius cannot run on the GPU fails |
| Fewer ranks `up,ready` than expected | A rank died during the run: its log under `rank-logs/` |
| `supplier has ... rows` | The dataset is not the scale factor the user gave. Ask which is right |
| `sbatch failed`, or a job that never starts | The site needs other options: ask the user and pass them with `--sbatch-args` |
| Times far slower than expected | CPUs per rank (`scontrol show job <id>`), storage stalls, the exchange path |

## Rules that come from the cluster, not from this skill

- On nodes that Slurm schedules, reach them only through Slurm. A process started over `ssh` can
  get the node drained. The `local` launcher uses `ssh` to other machines and is only for machines
  Slurm does not schedule.
- One job per node at a time. Ask before holding four or more nodes.
- Start narrow: the smoke test, then a few queries (`--queries 1,6,12`), before the whole set.
- The `gb200-gds-one-node` preset reads through GPUDirect Storage. Read
  `docs/super-sirius/gds-runbook.md` in the checkout first, probe the node, and tell the storage
  admins before the first GDS run on a node. It was written for one node.
- TPC-H query texts come from `<checkout>/test/tpch_performance/tpch_queries/orig/` unless
  `--queries-dir` says otherwise. Say in the report which set ran: numbers from different query
  parameters do not compare.

## What the script has and has not been tested on

- **Run for real once** (2026-11-18): one GB200 node under Slurm, an existing build, the
  `gb200-sf10000` preset with the default scan backend, SF10000. Plan, preflight, build reuse and
  the smoke test all worked, and the smoke test passed.
- **Tested only against stand-ins** for Slurm and for the client: the full benchmark, failure
  detection, a build from a fresh clone, and custom datasets (tables found automatically, a
  views file, queries chosen by name and by number).
- **Not run at all:** several nodes, the `local` launcher, the GDS preset.

For any of the untested paths, run the step with `--dry-run` first, read the generated script with
the user, and treat the first smoke test as the test of that path.
