# Created Draft PR: TPC-H benchmark harness

- PR: [#1738](https://github.com/sirius-db/sirius/pull/1738)
- Title: `chore(bench): add TPC-H harness and GPU/MIG placement`
- Base / head: `sirius-db:sirius/dev` ← `aocsa:codex/tpch-mig-benchmark-harness`
- State: Draft; labels `benchmarking`, `starrocks`, `multi-gpu`
- Source baseline: `upstream/dev` `ea1c2783`

## Exact PR body

## Summary

Add one reproducible StarRocks/Sirius TPC-H workflow:

- a parquet generator, FE/CN launcher, DuckDB oracle, result comparator, and per-CN distribution report;
- all 22 `FILES()`-based query texts, including the scale-correct q11 and the CTE-reuse guidance;
- `GPU_DEVICES` for shared-GPU development and future `MIG_DEVICES` launcher plumbing.

`cluster8.sh` documents the resource contract explicitly: each CN consumes `GPU_MEM` plus its `STAGING` arena outside that limit, plus CUDA context overhead. The README now states that an actual multi-CN run still requires #1714's CN bring-up/flags, #1693's stable staging arena, and the follow-on distributed exchange runtime.
It also marks UUID-based `MIG_DEVICES` unsupported on the current engine because its NVML device-count check fails under UUID-only visibility; use whole-GPU ordinals such as `GPU_DEVICES=0,1` on that box for now.

## Why one PR

The scripts, query kit, oracle/comparator, and topology launcher form one reproducible measurement and correctness workflow. Splitting them would leave a reviewer unable to run or validate the workflow end to end. The source commits are preserved as six cherry-picks, followed by a documentation-only prerequisite clarification.

## Validation

- `pixi run bash -n` for `gen-tpch.sh`, `cluster8.sh`, and `bench.sh`
- `pixi run python -m py_compile` for `cn-distribution.py`, `compare.py`, and `oracle.py`
- Comparator fixture: a wrong cold run (`r0`) is rejected while a correct warm run (`r1`) passes; the overall comparator exits non-zero.

Known gate limitation, intentionally not changed here: `compare.py` currently treats `nan` versus a finite oracle value as a match because its `d > tolerance` comparison is false for NaN. A focused follow-up should reject non-finite numeric values before using this as a strict correctness gate.

## Draft status and runtime scope

This remains a Draft because `dev` does not yet contain the distributed runtime needed for real multi-CN execution. The harness and documentation are ready to review independently; runtime measurements should wait for the prerequisite series.

## Validation evidence

`bash -n` and Python compilation passed using `pixi run` from the source worktree. The comparator fixture had an oracle value of `1.0`, a wrong cold run of `2.0`, and a correct warm run of `1.0`; it reported `VALUES-DIFFER` for cold, `MATCH` for warm, and exited 1. A second fixture reproduced the known `nan` versus finite false match; that limitation is disclosed in the PR and was intentionally left for a focused correctness follow-up.
