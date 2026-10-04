---
name: sirius-fuzz
description: Run bounded Sirius SQL correctness checks with the repository's siriusfuzz CLI. Use when a developer asks to run the fuzzer, find what falls back to the CPU, check fuzzing readiness, recheck findings after a fix, or summarize a fuzz run. Select among the developer's available local or remote environments and distinguish CPU harness checks from GPU validation.
---

# Sirius Fuzz

Use the existing CLI; do not recreate its generator, comparator, supervisor or reporting logic.
Read the checkout's `AGENTS.md` and the [fuzzer guide](../../../test/fuzz/README.md); use
`--help` for flags at the selected revision.

## Pick the environment

Fuzzing needs a compatible Linux GPU host with the intended source built there (`pixi run make`,
which produces the `build/release/duckdb` shell the fuzzer drives; no Python module is needed).
Prefer the host the user names; otherwise an accessible development GPU with the least setup.
Inspect `nvidia-smi`, active workloads, Pixi availability and checkout state; a remote connection
alone does not establish GPU availability. If no compatible GPU is accessible, report execution
as blocked with the missing prerequisite. Do not substitute CPU `selftest` or the unit tests
unless they were explicitly requested; they prove nothing about the GPU. Preserve working
changes and existing workloads; use a separate worktree for an isolated revision.

## Run

Use absolute paths for the shell, YAML and output root so the recorded command is unambiguous
on a remote host. `run` performs the doctor's readiness checks first and stops on
failure; `fuzz doctor` on its own is where to look when setup is in doubt.

```bash
pixi run fuzz run --seed 42 --queries 100 --duration 3m \
  --no-reduce --max-respawns 0 --out "$out/campaign"          # default: a short readiness sample
pixi run fuzz run --mode gaps --duration 30m --out "$out/gaps"
pixi run fuzz recheck "$run_dir" --shell "$shell"            # after a fix
```

Respect a supplied budget rather than adding these defaults to it. For a request to find
unsupported features or what falls back to the CPU at runtime, use `--mode gaps` and keep
reduction on: the reduced query is what names the unsupported feature. Add
`--features configured` when the question is about the supported surface only. For long runs use a
persistent session and reconnect to it rather than starting a duplicate campaign.

## Report

Read the run directory the command printed. Report the host, tested extension identity, exact
command and limits, completion status and exit code, verdict counts, and paths to `summary.txt`
and the finding bundles. Separate setup failures, skipped CPU errors, gaps and findings; dedup
signatures are grouping hints, not confirmed defect counts. Exit 0 does not mean zero findings.
In a gaps run, report findings first, then the runtime-fallback table (reason, GPU time thrown
away, smallest query, features), then the plan-time table. After a `recheck`, report which findings cleared,
which changed verdict, and which still reproduce. Preserve the evidence; do not edit the engine
or publish issues as part of a fuzzing request.
