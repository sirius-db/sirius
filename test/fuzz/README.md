# siriusfuzz: differential fuzzing for Sirius

`siriusfuzz` generates SQL, runs it on both DuckDB (CPU) and Sirius (GPU), and saves every
disagreement as a replayable finding. It is a plain command-line tool; no AI agent is required.

For a detailed explanation of generation, execution, comparison, reduction, and the design
decisions behind the harness, read [Understanding the Sirius fuzzer](../../docs/fuzzer/README.md).

```text
            ┌────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   TOML     │  generate  │ SQL  │   execute   │ rows │   compare   │ diff │    save     │
 profile ──▶│ tables +   │─────▶│ CPU, then   │─────▶│ multiset /  │─────▶│ finding     │
            │ data + SQL │      │ GPU strict  │      │ ordered     │      │ + reduce    │
            └────────────┘      └──────┬──────┘      └─────────────┘      └──────┬──────┘
                                       │ on match: rerun under                   │
                                       │ randomized Sirius settings,             ▼
                                       └ compare with GPU baseline    findings/<n>-<verdict>-<hash>/
                                                                     replay ▸ fix ▸ recheck
```

**Contents**

- [Quick start](#quick-start)
- [Command cheat sheet](#command-cheat-sheet)
- [How it works](#how-it-works)
- [Verdicts](#verdicts)
- [Running a campaign](#running-a-campaign)
- [Configuration](#configuration)
- [Finding unsupported features](#finding-unsupported-features)
- [Saved evidence](#saved-evidence)
- [Replaying a finding](#replaying-a-finding)
- [Agent-assisted use](#agent-assisted-use)
- [Fine print](#fine-print)

---

## Quick start

**Requirements.** A supported Linux GPU host (x86-64 or aarch64) with the repository's CUDA
environment. The `duckdb-python` Pixi environment does not support macOS. A stock DuckDB wheel
is fine for CPU-only harness tests but is not a supported runtime for loading Sirius.

**1. Build** the extension and the Python module from the same checkout:

```bash
git submodule update --init --recursive
pixi run make
pixi run -e duckdb-python build-duckdb-python
```

**2. Fuzz.** With no options, `run` first checks that the build, the Python module and the GPU
are ready (the same checks as `doctor`), then fuzzes the enabled features for 10 minutes with
one worker and prints the summary:

```bash
pixi run -e duckdb-python fuzz run               # correctness: GPU answers against CPU answers
pixi run -e duckdb-python fuzz run --mode gaps   # what Sirius hands back to the CPU at runtime
```

Results land in `test/fuzz/out/run-<timestamp>-seed<seed>-<unique>/`; the summary names the
directory and ends with the command to replay a finding. See [Saved evidence](#saved-evidence).

**3. Shape the run** when you need to:

```bash
# Fixed seed, query budget, no reduction: a quick sample
pixi run -e duckdb-python fuzz run --seed 42 --queries 100 --no-reduce

# Longer campaign; check shared GPU availability before adding workers
pixi run -e duckdb-python fuzz run --seed 42 --duration 30m --workers 2

# Another build, Sirius YAML or output root
pixi run -e duckdb-python fuzz run --extension /path/to/sirius.duckdb_extension \
  --sirius-config /path/to/sirius.yaml --out /path/to/fuzz-results

# Steer generation toward specific features (others that are enabled can still appear)
pixi run -e duckdb-python fuzz run --set features.scalar_functions.enabled=substring,like
```

`--no-doctor` skips the readiness check on repeat runs. `fuzz doctor` runs only that check: it
validates the configuration and output location, imports DuckDB, loads Sirius, builds a tiny
file-backed dataset, proves GPU interception with a canary query and checks a GPU query's answer,
all in a disposable subprocess with a 120-second deadline. It leaves a diagnostic directory with
logs, configuration, runtime info and `outcome.json`; temporary database files are removed on
normal exit. It never installs dependencies, rebuilds the engine, switches branches or downloads
extensions.

**No GPU handy?** These need only a CPU:

```bash
pixi run -e duckdb-python fuzz show-config           # print the effective configuration
pixi run -e duckdb-python fuzz-test                  # unit tests for the harness
pixi run -e duckdb-python fuzz selftest --queries 100  # CPU-only end-to-end harness check
```

The selftest defaults to 8–16 rows per table, at most two joined tables per `SELECT`, and one
subquery level, so it stays short. Raise those bounds with explicit `--set` values if you want
a larger CPU-only run.

---

## Command cheat sheet

All commands are invoked as `pixi run -e duckdb-python fuzz <command>`.

| Command | What it does | GPU? |
|---------|--------------|------|
| `doctor` | Verify build, config, interception and a GPU answer (`run` does this first unless `--no-doctor`) | yes |
| `run` | Generate, execute and compare queries; save findings. `--mode gaps` hunts runtime fallbacks | yes |
| `replay <finding-or-sql>` | Re-run one finding directory or your own `.sql` file | yes (`--cpu-only` for CPU) |
| `recheck <run-dir>` | Replay every finding of a run against the current build; print old and new verdicts | yes |
| `show-config` | Print the effective TOML configuration | no |
| `selftest` | CPU-only end-to-end harness check | no |

Also: `pixi run -e duckdb-python fuzz-test` runs the harness unit tests.

**Paths:** a relative path on the command line resolves from the directory you run `pixi` in,
then from the repository root, so repo-relative paths work from anywhere in the checkout. Paths
inside the configuration file resolve from the repository root. Outputs default to
`test/fuzz/out/`.

---

## How it works

1. **Generate.** Typed tables, data and SQL come from the TOML configuration, including NULLs and
   edge values. The run summary flags enabled features that were never generated.
2. **Execute.** Each query runs on DuckDB CPU, then on Sirius with
   `enable_duckdb_fallback = false`. After a match, the query is rerun under randomized Sirius
   settings and compared with the GPU baseline.
3. **Compare.** Rows compare as a multiset unless `ORDER BY` covers every output column.
   FLOAT/DOUBLE use relative and absolute tolerances; everything else compares exactly, including
   DECIMAL and NULL. NaN/inf must match exactly. A mismatch whose CPU result changes when the
   input rows are reordered is classified as *ambiguous* rather than a bug.
4. **Save.** Findings are grouped by signature, optionally shrunk, and reported alongside verdict
   counts and coverage gaps.

Two implementation details worth knowing:

- Generated tables live in an `ATTACH`ed file-backed database and are `CHECKPOINT`ed, because
  in-memory tables never reach the GPU native scan.
- Each worker is a separate process. A GPU fault or hang costs one worker: the orchestrator
  records the in-flight query and respawns it.

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

## Running a campaign

### Budgets and workers

- Defaults: **1 worker, 10 minutes**, after a readiness check. The banner states the budget in
  effect.
- With both `--queries` and `--duration`, whichever budget is reached first stops the campaign.
  Workers already mid-query can overshoot the query count slightly.
- `--duration` includes worker startup and execution but excludes initial provenance collection.
  Work in flight at the boundary is stopped and is *not* reported as a hang.
- `--max-respawns` (default 200) caps worker restarts after crashes or hangs.

### Stopping and completion status

| How the run ended | Summary status | What survives |
|-------------------|----------------|---------------|
| Budget reached | `complete` | everything |
| Ctrl-C | `cancelled` | completed findings and summary |
| Worker startup failure, failed interception check, or exhausted respawn budget | `incomplete` | completed findings and summary |
| SIGKILL or machine failure | no summary guaranteed | completed finding bundles and the flushed query log |

Worker database files from forcibly terminated processes can remain under the run directory;
remove them after inspection.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Run completed (findings may exist); doctor passed; replay matched |
| `1` | Findings with `--fail-on-findings`; or replay discrepancy, crash, timeout, or inconclusive CPU outcome |
| `2` | Invalid setup or input, incomplete campaign, or failed doctor |
| `130` | Cancelled with Ctrl-C |

### Choosing the Sirius YAML configuration

- `--sirius-config` selects the YAML; the harness sets `SIRIUS_CONFIG_FILE` from it. Repeat the
  flag to alternate configurations across worker processes.
- Otherwise the configuration file's `sirius.configs` applies; the default names the integration
  test YAML. With `sirius.configs = []`, the run uses whatever Sirius itself would pick up:
  `SIRIUS_CONFIG_FILE`, then `./sirius.yaml`, then `~/.sirius/sirius.yaml`, else built-in
  defaults.
- Whichever file is used is copied into the run directory and into every finding, and the session
  records how it was chosen, so a replay on another host restores the same settings.

---

## Configuration

There is one configuration file, `config/default.toml`; every command loads it unless `--config`
names another. Its `[features]` flags mark what Sirius runs on the GPU today: every enabled
feature is expected to stay on the GPU, so a plan-time fallback is a finding. When Sirius gains a
feature, flip its flag to `true` there. `--set key.path=value` overrides any key for one run, and
`show-config` prints the effective result.

`known_issues.toml` quarantines confirmed divergences by regex. Every entry carries an issue link.

---

## Finding unsupported features

`fuzz run --mode gaps` finds the queries Sirius accepts at plan time and then hands to the CPU at
runtime. The two kinds of fallback cost very different amounts: a plan-time rejection costs one
failed translation, after which DuckDB's own plan runs; a runtime fallback runs the GPU pipeline up
to the failing operator and then runs the stored CPU plan from scratch, so all the GPU work is
thrown away. The mode exists to find the runtime ones so their checks can move to plan time. It
changes three things:

- **Everything is generated.** Every feature switch the configuration keeps off because Sirius
  does not run it yet (window functions, `DISTINCT`, grouping sets, `FULL`/`CROSS` joins,
  `UNION`/`EXCEPT`/`INTERSECT`, uncorrelated subqueries, ungrouped `COUNT(DISTINCT)`, `TRY`,
  temporal-numeric casts) is turned on, and setting variants are skipped. Known-unsupported
  features stay on because the shapes of them that slip past the planner are exactly the runtime
  fallbacks being looked for. `--set` still applies afterwards, so
  `--set features.window_functions=false` narrows the survey; `show-config --mode gaps` prints the
  effective configuration.
- **Runtime fallbacks are reduced to the unsupported feature.** A `runtime_fallback` is shrunk
  like any other finding, but the reducer accepts a smaller query as long as it still fails at
  runtime for the same *kind* of reason, with the rejected expression's function set allowed to
  shrink. Dropping a supported `concat` from an unsupported `regexp_matches(concat(..))` therefore
  keeps going until only the function Sirius cannot translate is left. A candidate that turns into
  a plan rejection is refused, so every reduced reproducer still passes the planner: once the check
  moves to plan time, replaying it should flip from `runtime_fallback` to `plan_fallback`, which
  makes it the regression test for that check. Plan-time rejections are not reduced in this mode;
  their reason already names the operator, and they cost no GPU time.
- **Gaps do not fail the run.** `--fail-on-findings` ignores the two gap verdicts in this mode.

Everything else works as in a correctness run: every query is still compared with the CPU, and a
query that fails for any other reason is still reported. Where each outcome ends up:

| Outcome | Verdict | In the summary | `--fail-on-findings` |
|---------|---------|----------------|----------------------|
| the GPU raised an error that says the operation is not supported | `runtime_fallback` | the `runtime fallbacks` table, reduced, with the GPU time thrown away | ignored |
| Sirius declined the plan | `plan_fallback` | the `plan-time fallbacks` table, by reason, not reduced | ignored |
| GPU rows differ from CPU rows | `mismatch` | `findings`, reduced as in a correctness run | fails |
| any other GPU error | `gpu_error`, `gpu_internal_error`, `gpu_oom` | `findings`, reduced as in a correctness run | fails |
| the GPU run hung, or the worker died | `timeout`, `crash` | `findings` | fails |
| the query failed on the CPU | `cpu_error`, `cpu_timeout` | skipped and counted; the top reasons are listed | ignored |
| matches `known_issues.toml` | `known_issue` | a known gap stays in its fallback table, tagged `known: <issue>`; anything else under `findings` with the issue link | ignored |

So a bug that only shows up once window functions or `EXCEPT` are generated is not lost: it sits
in the `findings` section of the same summary with its own replayable bundle, below the gaps.
And a runtime fallback you have already filed stays in the runtime table with its issue, so the
table remains the complete list of what still falls back.

The line between a gap and an error is the message. A runtime error counts as a gap only when it
says *not supported*, *unsupported* or *not implemented*; any other GPU error would also fall back
to the CPU in production, but it looks like a bug rather than a missing feature, so it is listed
under `findings` with its reason. If one of those turns out to be an unsupported feature phrased
differently, widen the pattern in `siriusfuzz/classify.py`.

Runtime fallbacks come first, ordered by the GPU time they threw away, which is the order in which
to move their checks to plan time. A reason's time is summed over every query that hit it; fuzz
datasets are small, so read the figures as relative. Both tables group by the reason the smallest
reproducer reports, so rejections of different expressions around the same function merge into one
line:

```text
runtime fallbacks (1 reason, 5 queries, 3.2s of GPU work thrown away); smallest query that passes the planner and still fails, and its features:
  x5      3.2s  Distinct aggregates not supported in GPU path yet
      SELECT count(DISTINCT "a0"."c1") AS c0 FROM "t2" AS "a0"
      features: Agg(count,distinct), ColumnRef, Select, TableRef   findings: 007-runtime_fallback-2b3c4d5e

plan-time fallbacks (2 reasons, 207 queries):
  x180  Window not supported   findings: 000-plan_fallback-1f2e3d4c
  x27   Unsupported expression in projection: {concat, regexp_matches}   findings: 003-plan_fallback-9a8b7c6d, 011-plan_fallback-5e6f7a8b

findings (1 unique):
  [mismatch] x1    012-mismatch-7c8d9e0f  row count 41 vs 40
```

`summary.json` carries the same tables under `gaps.runtime` and `gaps.plan`, with every
contributing finding directory. Each runtime finding keeps its own `reduced.sql` and
`reduction.json` (which records the reduced query's reason) and replays like any other finding. A
query is reported for the first rejection Sirius hits, so a long run finds more than a short one;
the "features emitted" line in the summary shows how much of the generator's surface a run covered.

---

## Saved evidence

Outputs default to unique directories below `test/fuzz/out/`; use `--out` for another root.

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

Start with `FINDING.md`: it shows the verdict and reason, the smallest query that reproduces,
the differing rows or error, and the replay command for that directory.

**How findings are grouped.** Signatures are grouping heuristics, not confirmed root causes.
Mismatch and timeout signatures include the original SQL, dataset identity, comparison mode,
variant value and result evidence, so distinct inputs stay separate even when their operator
labels match. Repeated errors group by normalized reason. This favours keeping evidence over
compact counts.

**Reduction never replaces the original.** The original observation is saved before reduction
starts. Each error signature is reduced once per worker; later queries with the same signature
are kept under `more/` without spending another reduction budget. A reducer failure is recorded
separately with `stage=reduction` and the active candidate. Reduced output is additional
evidence and must itself be replayed; to test an edited query, replay it as a SQL file with the
bundle's `dataset.sql`.

**Provenance.** `meta.json` carries the source revisions and binary hashes of the run and the
session settings of the worker. Source revisions describe the checkout at run time; the
extension fingerprint identifies the actual binary, and the revision alone does not prove how it
was built. Findings carry small result samples and comparator differences, not full result dumps.

**Naming.** Dataset filenames include the worker incarnation, so a respawn cannot overwrite
earlier data. Every entry under `more/` carries its own dataset.

The optional sqlsmith reducer loads an already-installed extension and never runs `INSTALL`.
AST reduction works without sqlsmith.

---

## Replaying a finding

Copy the **entire** finding directory; it can then be replayed from any compatible Sirius checkout.

```bash
# Replay reduced.sql if present, otherwise query.sql
pixi run -e duckdb-python fuzz replay /absolute/path/finding

# Force the original query
pixi run -e duckdb-python fuzz replay /absolute/path/finding --original

# Deliberately test another binary/config; the attempt records the overrides
pixi run -e duckdb-python fuzz replay /absolute/path/finding \
  --extension /absolute/path/new/sirius.duckdb_extension \
  --sirius-config /absolute/path/local.yaml --timeout 180

# Your own SELECT/WITH query with its CREATE/INSERT dataset
pixi run -e duckdb-python fuzz replay /absolute/path/query.sql \
  --dataset /absolute/path/dataset.sql
```

**After a fix, recheck the whole run.** `recheck` replays every finding directory of a run
(reduced query by default, `--original` for the full one) against the current build and prints
one line per finding with the recorded verdict and the new one. It exits 0 when every finding
now passes, 1 otherwise, and writes `recheck.json` next to the per-finding replay evidence:

```bash
pixi run -e duckdb-python fuzz recheck /absolute/path/run-20261002-121126-seed42-m7lb_mer \
  --extension /path/to/new/sirius.duckdb_extension
```

A gap that moved to plan time shows as `runtime_fallback -> plan_fallback`; a fixed mismatch
shows as `mismatch -> ok`; anything that changed to a different failure is worth a look.

**What replay restores:** the saved TOML, YAML, captured baseline session settings, comparison
mode and the exact failing setting variant. It never picks a new random variant and never
invents a replacement dataset; missing required inputs fail explicitly. For custom SQL only,
`--dataset-seed N` requests a generated dataset instead of `--dataset`.

**Supervision.** Each replay runs under a supervising process with a hard deadline
(`--timeout`, default 180 s), so a native crash or an uninterruptible GPU call cannot kill or
block the CLI. The supervisor records the last query, CPU/GPU phase, settings, signal/exit code
and logs.

**Comparison options.** Custom SQL defaults to multiset comparison; `--ordered` requires
deterministic output ordering. `--cpu-only` runs without a GPU and does not establish GPU
correctness. The generated-AST ambiguity filter is not rerun for SQL-only replays, so replay is
evidence for manual investigation, not automatic bug confirmation.

**Attempts are append-only.** Replay writes a new attempt directory and never modifies the
source bundle. Each attempt records its source, inputs, effective settings, environment and
outcome, and announces a changed extension fingerprint.

**Portability caveats:**

- Machine-specific YAML paths (spill directories, for example) may need an explicit
  `--sirius-config` on another host.
- CUDA device visibility comes from the current host's environment and is recorded.
- Reproducibility across different builds, GPUs or runtimes is not guaranteed.
- Bundles without a saved YAML require an explicit `--sirius-config`.

---

## Agent-assisted use

Use the [sirius-fuzz skill](../../.agents/skills/sirius-fuzz/SKILL.md) in Codex, or the shared
`.claude/skills/sirius-fuzz` entry in Claude:

> Use $sirius-fuzz to run a short correctness check on an available development GPU.

The skill picks an accessible development host, requires a passing GPU `doctor`, runs within the
requested budget and summarizes the saved evidence. Without a GPU it reports the blocker; CPU
harness tests and `selftest` run only when explicitly requested. Reading saved reports needs no
GPU.

---

## Fine print

**Build metadata.** `build-duckdb-python` uses the extension preset's
`OVERRIDE_GIT_DESCRIBE=v1.5.6`, and version checks stay enabled. Use
`--allow-metadata-mismatch` only for independently verified matching builds; it cannot fix an
ABI mismatch. Its allowance and use are recorded, and replay announces a restored allowance.
`--no-allow-metadata-mismatch` disables it for that attempt.
