# siriusfuzz: differential fuzzing for Sirius

`siriusfuzz` generates SQL, runs it on both DuckDB (CPU) and Sirius (GPU), and saves every
disagreement as a replayable finding. It is a plain command-line tool; no AI agent is required.

For the reasoning behind generation, execution, comparison and reduction, read
[Understanding the Sirius fuzzer](../../docs/fuzzer/README.md). This page is the operating manual.

```text
            ┌────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   TOML     │  generate  │ SQL  │   execute   │ rows │   compare   │ diff │    save     │
  config ──▶│ tables +   │─────▶│ CPU, then   │─────▶│ multiset /  │─────▶│ finding     │
            │ data + SQL │      │ GPU strict  │      │ ordered     │      │ + reduce    │
            └────────────┘      └──────┬──────┘      └─────────────┘      └──────┬──────┘
                                       │ on match: rerun under                   │
                                       │ randomized Sirius settings,             ▼
                                       └ compare with GPU baseline    findings/<n>-<verdict>-<hash>/
                                                                     replay ▸ fix ▸ recheck
```

**Contents:** [Quick start](#quick-start) · [Commands](#commands) · [How it works](#how-it-works) ·
[Verdicts](#verdicts) · [Configuration](#configuration) ·
[Finding unsupported features](#finding-unsupported-features) · [Saved evidence](#saved-evidence) ·
[Replay and recheck](#replay-and-recheck) · [Fine print](#fine-print)

---

## Quick start

**Requirements.** A supported Linux GPU host (x86-64 or aarch64) with the repository's CUDA
environment. The fuzzer drives the DuckDB shell that the Sirius build produces
(`build/release/duckdb`, with Sirius built in) over pipes; nothing else has to be built or
installed. For CPU-only use, any DuckDB CLI works (`--shell`, or `SIRIUSFUZZ_SHELL`).

**1. Build** Sirius:

```bash
git submodule update --init --recursive
pixi run make
```

**2. Fuzz.** With no options, `run` checks that the shell, Sirius and the GPU are ready, then
fuzzes the enabled features for 10 minutes with one worker and prints the summary:

```bash
pixi run fuzz run                                     # correctness: GPU answers against CPU answers
pixi run fuzz run --mode gaps                         # what Sirius hands back to the CPU at runtime
pixi run fuzz run --mode gaps --features configured   # the same, over the supported features only
```

Results land in `test/fuzz/out/run-<timestamp>-seed<seed>-<unique>/`. The summary names the
directory and ends with the command to replay a finding.

**3. Shape the run** when you need to:

```bash
pixi run fuzz run --seed 42 --queries 100 --no-reduce      # quick sample
pixi run fuzz run --seed 42 --duration 30m --workers 2     # longer campaign
pixi run fuzz run --shell /path/to/other/build/release/duckdb \
  --sirius-config /path/to/sirius.yaml --out /path/to/results               # another build or YAML
pixi run fuzz run --set features.scalar_functions.enabled=substring,like
```

`--no-doctor` skips the readiness check on repeat runs. Check shared GPU availability before
adding workers; each one holds a Sirius session.

**4. Fix, then recheck.** After a change, replay every finding of a run against the new build:

```bash
pixi run fuzz recheck test/fuzz/out/run-<...>
```

**No GPU handy?** `fuzz show-config` prints the effective configuration, `fuzz-test` runs the
harness unit tests, and `fuzz selftest --queries 100` is a small CPU-only end-to-end check with
8–16 rows per table.

---

## Commands

All commands are invoked as `pixi run fuzz <command>`.

| Command | What it does | GPU? |
|---------|--------------|------|
| `run` | Generate, execute and compare queries; save findings. `--mode gaps` hunts runtime fallbacks | yes |
| `doctor` | The readiness check on its own: shell, Sirius, interception and a GPU answer | yes |
| `replay <finding-or-sql>` | Re-run one finding directory or your own `.sql` file | yes (`--cpu-only` for CPU) |
| `recheck <run-dir>` | Replay every finding of a run against the current build; print old and new verdicts | yes |
| `show-config` | Print the effective TOML configuration | no |
| `selftest` | CPU-only end-to-end harness check | no |

Also: `pixi run fuzz-test` runs the harness unit tests.

**Paths.** A relative path on the command line resolves from the directory you run `pixi` in,
then from the repository root. Paths inside the configuration file resolve from the repository
root. Outputs default to `test/fuzz/out/`.

**The shell.** `--shell` names the DuckDB binary to drive; the default is the configuration's
`sirius.shell` (`build/release/duckdb`), then `$SIRIUSFUZZ_SHELL`, then `duckdb` on `PATH`. The
build's shell has Sirius built in. A plain DuckDB CLI of the same version works with
`--extension` pointing at the loadable `sirius.duckdb_extension`, or on its own for `--cpu-only`.

**Exit codes.** `0` completed (findings may exist), doctor passed, replay matched, recheck all
clear. `1` findings with `--fail-on-findings`, a replay that still fails, a recheck with findings
left. `2` invalid setup or input, incomplete campaign, failed readiness check. `130` Ctrl-C.

---

## How it works

1. **Generate.** Typed tables, data and SQL come from the TOML configuration, including NULLs and
   edge values. The summary flags enabled features that were never generated.
2. **Execute.** Each worker drives one DuckDB shell process over pipes in JSON mode. Each query
   runs on DuckDB CPU, then on Sirius with `enable_duckdb_fallback = false`, so a fallback
   surfaces as an error instead of a silent CPU run. After a match, the query is rerun under
   randomized Sirius settings and compared with the GPU baseline.
3. **Compare.** Rows compare as a multiset unless `ORDER BY` covers every output column.
   FLOAT/DOUBLE use relative and absolute tolerances; everything else compares exactly, including
   DECIMAL and NULL. A mismatch whose CPU result changes when the input rows are reordered is
   classified as *ambiguous* rather than a bug.
4. **Save.** Findings are grouped by signature, shrunk, and reported with verdict counts.

Generated tables live in an `ATTACH`ed file-backed database and are `CHECKPOINT`ed, because
in-memory tables never reach the GPU native scan. A GPU fault or a query that overruns its
timeout kills the shell process; the worker records the crash or timeout with the shell's
stderr, starts a fresh shell, re-attaches the dataset files and carries on. The orchestrator
respawns a worker only if the Python process itself dies or stalls (`--max-respawns`, default
200). With both `--queries` and `--duration`, whichever is reached first stops the run; Ctrl-C
keeps completed findings and the summary; a worker that fails at startup or exhausts its
respawns marks the run `incomplete`.

---

## Verdicts

| Verdict | Meaning | Finding? |
|---------|---------|----------|
| `ok` | GPU rows match CPU rows, and every setting variant matched the GPU baseline | no |
| `cpu_error` / `cpu_timeout` | the reference run failed; the query is skipped and counted | no |
| `ambiguous` | mismatch that changes under permuted row order (nondeterministic query) | no |
| `known_issue` | matches an entry in `known_issues.toml`; counted, not failed | no |
| `mismatch` | GPU rows differ from CPU rows | **yes** |
| `variant_mismatch` | same query, one Sirius setting changed, different rows | **yes** |
| `plan_fallback` | Sirius declined the plan; the reason is recorded. A *gap* | **yes** |
| `runtime_fallback` | the GPU run raised an error that says the operation is not supported; with fallback enabled the query would have run on the CPU. A *gap* | **yes** |
| `gpu_error` / `gpu_internal_error` / `gpu_oom` | the GPU run raised | **yes** |
| `timeout` | the GPU run exceeded `oracle.query_timeout_seconds` | **yes** (hang candidate) |
| `crash` | the worker process died during the query | **yes** |

---

## Configuration

There is one configuration file, `config/default.toml`; every command loads it unless `--config`
names another. Its `[features]` flags mark what Sirius runs on the GPU today: every enabled
feature is expected to stay on the GPU, so a plan-time fallback is a finding. When Sirius gains a
feature, flip its flag to `true` there. `--set key.path=value` overrides any key for one run, and
`show-config` prints the effective result.

`known_issues.toml` quarantines confirmed divergences by regex; every entry carries an issue link.

**Sirius YAML.** `--sirius-config` selects it (repeat the flag to alternate across workers);
otherwise the configuration's `sirius.configs` applies, and with `sirius.configs = []` the run
uses whatever Sirius itself would pick up (`SIRIUS_CONFIG_FILE`, `./sirius.yaml`,
`~/.sirius/sirius.yaml`, else built-in defaults). Whichever file is used is copied into the run
and into every finding, so a replay elsewhere restores it.

---

## Finding unsupported features

`fuzz run --mode gaps` finds the queries Sirius accepts at plan time and then hands to the CPU at
runtime. A plan-time rejection costs one failed translation; a runtime fallback runs the GPU
pipeline up to the failing operator and then runs the CPU plan from scratch, so the GPU work is
thrown away. The mode exists to find the runtime ones so their checks can move to plan time. It
changes three things:

- **Everything is generated, unless you say otherwise.** Every feature switch the configuration
  keeps off because Sirius does not run it yet is turned on, and setting variants are skipped.
  Known-unsupported features stay on because the shapes that slip past the planner are exactly
  what is being looked for. `--features configured` keeps the configuration's own flags instead:
  every query then reaches runtime, so the whole budget probes the supported surface for runtime
  fallbacks and plan rejections appear only when something inside that surface is declined.
  `--set` still applies afterwards; `show-config --mode gaps` prints the result.
- **Runtime fallbacks are reduced to the unsupported feature.** The reducer accepts a smaller
  query as long as it still fails at runtime for the same *kind* of reason, with the rejected
  expression's function set allowed to shrink, so `regexp_matches(concat(..))` reduces to the
  function Sirius cannot translate. A candidate that turns into a plan rejection is refused, so
  every reproducer still passes the planner: once the check moves to plan time, replaying it
  flips from `runtime_fallback` to `plan_fallback`, which makes it the regression test for that
  check. Plan-time rejections are not reduced in this mode; their reason already names the
  operator.
- **Gaps do not fail the run.** `--fail-on-findings` ignores the two gap verdicts in this mode.

Everything else works as in a correctness run. Where each outcome ends up:

| Outcome | Verdict | In the summary | `--fail-on-findings` |
|---------|---------|----------------|----------------------|
| the GPU raised an error that says the operation is not supported | `runtime_fallback` | the `runtime fallbacks` table, reduced, with the GPU time thrown away | ignored |
| Sirius declined the plan | `plan_fallback` | the `plan-time fallbacks` table, by reason, not reduced | ignored |
| GPU rows differ from CPU rows | `mismatch` | `findings`, reduced | fails |
| any other GPU error | `gpu_error`, `gpu_internal_error`, `gpu_oom` | `findings`, reduced | fails |
| the GPU run hung, or the worker died | `timeout`, `crash` | `findings` | fails |
| the query failed on the CPU | `cpu_error`, `cpu_timeout` | skipped and counted; the top reasons are listed | ignored |
| matches `known_issues.toml` | `known_issue` | a known gap stays in its fallback table, tagged `known: <issue>`; anything else under `findings` with the issue link | ignored |

A runtime error counts as a gap only when its message says *not supported*, *unsupported* or
*not implemented*; any other GPU error would also fall back in production but looks like a bug,
so it is listed under `findings`. If one turns out to be an unsupported feature phrased
differently, widen the pattern in `siriusfuzz/classify.py`.

Runtime fallbacks come first, ordered by the GPU time they threw away, which is the order in which
to move their checks to plan time (fuzz datasets are small, so read the figures as relative). Both
tables group by the reason the smallest reproducer reports:

```text
runtime fallbacks (1 reason, 5 queries, 3.2s of GPU work thrown away); smallest query that passes the planner and still fails, and its features:
  x5      3.2s  Distinct aggregates not supported in GPU path yet   known: sirius-db/sirius#1218
      SELECT count(DISTINCT "a0"."c1") AS c0 FROM "t2" AS "a0"
      features: Agg(count,distinct), ColumnRef, Select, TableRef   findings: 007-runtime_fallback-2b3c4d5e

plan-time fallbacks (2 reasons, 207 queries):
  x180  Window not supported   findings: 000-plan_fallback-1f2e3d4c
  x27   Unsupported expression in projection: {concat, regexp_matches}   findings: 003-plan_fallback-9a8b7c6d, 011-plan_fallback-5e6f7a8b

findings (1 unique):
  [mismatch] x1    012-mismatch-7c8d9e0f  row count 41 vs 40
```

`summary.json` carries the same tables under `gaps.runtime` and `gaps.plan`. A query is reported
for the first rejection Sirius hits, so a long run finds more than a short one; the "features
emitted" line shows how much of the generator's surface a run covered.

---

## Saved evidence

```text
run-<timestamp>-seed<seed>-<unique>/
├── config.toml, sirius-<n>.yaml        effective configuration snapshots
├── environment.json, invocation.json   source revisions, binary hashes, GPU/runtime, CLI arguments
├── summary.json, summary.txt           verdict counts, gap tables, findings, coverage
├── queries.jsonl                       one record per query (every observation is here)
├── datasets/w<worker>-s<spawn>-d<n>.sql
├── logs/                               native stderr, active operation, runtime per worker
└── findings/<n>-<verdict>-<hash>/
    ├── FINDING.md                      what happened, the query, how to replay
    ├── query.sql, dataset.sql          the original inputs
    ├── reduced.sql, reduction.json     when reduction made progress
    ├── config.toml, sirius.yaml        configuration for this worker
    ├── meta.json                       the full record, provenance and session settings
    ├── worker.stderr                   crashes and hangs only
    └── more/<n>/                       up to five further query/dataset pairs with the same signature
```

Start with `FINDING.md`: the verdict and reason, the smallest query that reproduces, the
differing rows or error, and the replay command for that directory.

Errors and fallbacks group by normalized reason; mismatches and timeouts are kept apart per
query, dataset, comparison mode and setting variant. Each error signature is reduced once per
worker; later queries with the same signature go under `more/`. Reduction never replaces the
original: `query.sql` is what was observed, `reduced.sql` is the smallest query that still failed
within the budget. The optional sqlsmith reducer loads an already-installed extension and never
runs `INSTALL`; AST reduction works without it.

---

## Replay and recheck

Copy the **entire** finding directory; it replays from any compatible Sirius checkout.

```bash
pixi run fuzz replay /path/finding              # reduced.sql if present
pixi run fuzz replay /path/finding --original   # the full query
pixi run fuzz replay /path/finding \
  --extension /path/new/sirius.duckdb_extension                  # another build (recorded)
pixi run fuzz replay /path/query.sql --dataset /path/dataset.sql
pixi run fuzz recheck /path/run-dir             # every finding of a run
```

Replay restores the saved TOML, YAML, baseline session settings, comparison mode and the exact
failing setting variant; it never picks a new variant or invents a dataset. Each replay runs in
its own shell with `--timeout` (default 180 s) as the query deadline, writes a new attempt
directory and leaves the finding unchanged. Custom SQL compares as a multiset unless `--ordered`;
`--cpu-only` runs without a GPU and proves nothing about it. The ambiguity filter is not rerun
for SQL-only replays, so a replay is evidence for investigation, not automatic confirmation.

`recheck` prints one line per finding with the recorded verdict and the new one, writes
`recheck.json`, and exits 0 only when every finding now passes. A gap that moved to plan time
shows as `runtime_fallback -> plan_fallback`; a fixed mismatch shows as `mismatch -> ok`;
anything that changed to a different failure is worth a look.

Portability: machine-specific YAML paths (spill directories) may need `--sirius-config` on another
host; CUDA device visibility comes from the current host; reproduction across different builds,
GPUs or runtimes is not guaranteed; a bundle without a saved YAML needs an explicit one.

---

## Fine print

**Agents.** The [sirius-fuzz skill](../../.agents/skills/sirius-fuzz/SKILL.md) (also linked from
`.claude/skills/`) runs the same commands on an available GPU host and reports the saved evidence.

**Versions.** The shell and the extension must come from the same DuckDB version; the build's
shell guarantees that. A plain DuckDB CLI that refuses a `LOAD` names the version it wants. The
`duckdb-python` Pixi environment is no longer needed by the fuzzer.

**Shell protocol.** Every statement is followed by a sentinel on each stream, so results (JSON
on stdout) and errors (stderr) are delimited exactly. DECIMAL and HUGEINT values arrive as
strings and compare exactly; floating-point values arrive as numbers and compare with the
tolerances. A query whose output columns share a name is still compared positionally.
