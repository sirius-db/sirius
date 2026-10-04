---
name: sirius-fuzz
description: Run bounded Sirius SQL correctness checks with the repository's siriusfuzz CLI. Use when a developer asks to run the fuzzer, check fuzzing readiness, or summarize a fuzz run. Select among the developer's available local or remote environments and distinguish CPU harness checks from GPU validation.
---

# Sirius Fuzz

Use the existing CLI; do not recreate its generator, comparator, supervisor or reporting logic.
Read the checkout's `AGENTS.md` and the relevant sections of the
[fuzzer guide](../../../test/fuzz/README.md). Use CLI `--help` for flags at the selected revision.
This skill has no dependency on a particular workstation, cloud provider, account or connector.

## Select the task and execution environment

Use the user's requested revision, configuration, mode, seed, duration and host when supplied.
For an unspecified run, start with a short campaign on the default configuration: seed 42, one
worker, at most 100 queries or three minutes, and no automatic reduction. This is a readiness
and correctness sample, not comprehensive coverage. A request to find unsupported features or
operators, or to see what falls back to the CPU at runtime, is a `--mode gaps` run; keep
reduction on there, since the reduced query is what names the unsupported feature. Requests to run fuzzing require a
compatible GPU and a passing doctor. Run CPU harness tests or selftest only when explicitly
requested; they are not a substitute for GPU fuzzing. Inspecting saved reports needs no GPU.

Inspect the current checkout and the execution environments already available to this task.
Consider a remote development host when the user names it or an existing connection provides
access. Use available SSH/container/remote-execution tools and existing authentication. Do not
assume a hostname, credential store, project path, GPU model or host-specific skill exists.

Prefer the requested environment; otherwise choose an available compatible development GPU
with the intended source and least setup. Inspect OS/architecture, `nvidia-smi`, visible devices,
active GPU workloads, Pixi availability, checkout status and submodule revisions. Check the
current `pixi.toml` for supported platforms. A remote connection alone does not establish GPU
availability. Explain the selection before starting work. Ask only for a missing decision that
affects execution, such as which allocation to use when ownership or availability is unclear.

For a GPU fuzzing request, if no compatible GPU is accessible, stop before building or launching
the workload. Report execution as blocked, identify the observed missing prerequisite, and state
what environment is needed to continue. Do not automatically run CPU harness tests or selftest,
even when Python/DuckDB is available. An explicit CPU-only request can proceed without a GPU;
label its results CPU-only. Provisioning a new machine or paid allocation is a separate decision,
not an implicit part of selecting an existing environment.

## Prepare a reproducible run

- Resolve the intended source state before building. Preserve working changes and existing
  workloads. For an isolated revision, use a separate worktree and initialize its submodules;
  for requested uncommitted changes, record the dirty state and ensure the tested build includes it.
- On another host, prefer fetching a published branch and checking out its exact commit.
  Unpublished commits require an explicit transfer, such as a Git bundle. Do not silently test
  a different revision or push a branch merely to make it fetchable.
- Use the host's repository build instructions and `pixi run`. Build Sirius and the DuckDB
  Python module from matching sources with `pixi run make` and
  `pixi run -e duckdb-python build-duckdb-python` when those artifacts are missing or stale.
  A stock DuckDB wheel can support CPU harness tests, but is not a supported Sirius GPU runtime.
- Select an existing host-appropriate Sirius YAML, verifying memory settings and writable spill
  paths. Do not copy another machine's resource limits. Keep metadata checks enabled; a mismatch
  is a setup problem to investigate, not a reason to automatically bypass the check.
- Choose absolute paths for the configuration, extension, YAML and a fresh output root, so the
  recorded command is unambiguous on a remote host. Record host, source state, submodules, binary
  identity, device selection, configuration and overrides without dumping credentials or the full environment.

Use one available GPU and one worker initially. For long runs, use an available persistent
session/job mechanism, preserve logs and exit codes, and reconnect to that job after a
disconnect. Do not launch a duplicate campaign because a client lost its connection.

## Check readiness and execute

Set `repo`, `config`, `extension`, `yaml` and `out` to verified absolute paths on the selected
host. The default configuration is `$repo/test/fuzz/config/default.toml`.
Run commands from that checkout and capture each command's output and actual exit code.

```bash
pixi run -e duckdb-python fuzz doctor \
  --config "$config" --extension "$extension" --sirius-config "$yaml" \
  --no-allow-metadata-mismatch --timeout 120 --out "$out/doctor"
```

Require the doctor's successful outcome, `gpu_verified=true`, and interception evidence before
starting GPU comparisons. `run` repeats these checks first and stops on failure, so the separate
`doctor` step is where to look when setup is in doubt, not an extra prerequisite. If doctor fails, inspect its logs and identify the setup blocker.
Resume only after addressing that cause; do not switch to CPU fallback to make the check pass.
Sirius intercepts ordinary SQL. The harness uses file-backed tables and checkpointing to reach
the GPU scan path; an ad hoc in-memory query is not an equivalent readiness check.

For a short run when the user has not specified other limits:

```bash
pixi run -e duckdb-python fuzz run \
  --config "$config" --extension "$extension" --sirius-config "$yaml" \
  --no-allow-metadata-mismatch --seed 42 --queries 100 --duration 3m \
  --workers 1 --no-reduce --max-respawns 0 --out "$out/campaign"
```

The first query/time budget reached stops the campaign; provenance and cleanup can add elapsed
time. This initial run stops after a worker death rather than repeatedly restarting it. Preserve
the evidence and report the incomplete run. Do not extend a completed budget or start repeated
campaigns without a corresponding user request. Respect a supplied budget instead of adding
these example limits to it blindly.

Only for an explicit request to test the harness or run CPU selftest, use the relevant command:

```bash
pixi run -e duckdb-python fuzz-test
pixi run -e duckdb-python fuzz selftest --out "$out/selftest"
```

These examples require that Pixi environment to be supported on the execution host. On an
unsupported platform, use an already available compatible CPU Python environment with the
required DuckDB module, or report the missing setup. From `$repo/test/fuzz`, the equivalent
entrypoints are `python -m unittest discover -s tests -t .` and
`python -m siriusfuzz selftest --out "$out/selftest"`; invoke them through that environment
according to the host's instructions. Keep CPU selftest's bounded
defaults; explicit workload overrides change what is being validated. Plain `selftest` exercises
its own default seed and query count; record them from its artifacts rather than hardcoding a
test count or substituting a different seed without explanation.

## Inspect evidence and hand back results

Read the emitted run directory, not an assumed newest directory from another job. Inspect
`summary.json`, `summary.txt`, `queries.jsonl`, runtime/interception evidence and relevant worker
logs. Exit 0 alone does not mean zero findings: campaigns only fail on findings when requested
with `--fail-on-findings`. Exit 2 means setup/incomplete work, and 130 means cancellation;
interpret other exit codes using the guide for that command.

Separate setup failures, CPU reference errors, coverage gaps, result discrepancies and process
failures. Dedup signatures are grouping hints, not counts of confirmed defects. A counted CPU
range/conversion error is not a GPU result comparison. Report skipped queries and incomplete
work explicitly. Do not infer GPU success from a CPU selftest. In a gaps run, report the
runtime-fallback table first (reason, GPU time thrown away, smallest query, features), then the
plan-time table, separately from findings; gaps are its expected output and do not fail the run.

Preserve original finding bundles and their datasets/configuration. A request to run fuzzing
ends with saved evidence and a summary; it does not imply engine edits, repeated crash/hang
replay, or issue publication. After a fix, `fuzz recheck <run-dir>` replays every finding of a
run against the current build and reports which ones cleared. Follow repository approval rules
before any external publication.

Return the selected host and tested source/binary identity, exact command and limits, completion
status and exit code, query/verdict totals, GPU evidence or CPU-only limitation, and paths to the
summary, logs and finding bundles. Explain blockers and the smallest next step. Remote evidence
stays on that host unless copied; provide an accessible report or explicitly identify the remote path.
