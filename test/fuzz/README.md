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
                                                                     replay ▸ triage ▸ issue-draft
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
- [Triage and issue drafts](#triage-and-issue-drafts)
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
| `replay-file <file.sql>` | Classify every `SELECT` in a file, each in its own supervised process | yes |
| `triage <run-or-finding>...` | Re-validate findings, shrink them, write `REPORT.md` and issue drafts | yes |
| `triage-report <triage-dir>` | Regenerate reports and drafts from saved evidence, no execution | no |
| `review <triage-dir> F-ID` | Record a human disposition for a candidate | no |
| `group <triage-dir> F-ID...` | Group candidates that look related | no |
| `issue-draft <triage-dir> [F-ID\|--all]` | Export copy-ready Markdown issue drafts | no |
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

- `--sirius-config` overrides the profile's YAML; the harness sets `SIRIUS_CONFIG_FILE` from it.
  Repeat the flag to alternate configurations across worker processes.
- With `sirius.configs = []` and no `--sirius-config`, GPU sessions use built-in defaults.
- Sessions **refuse to start** if an ambient configuration would silently take effect: when
  `SIRIUS_CONFIG_FILE` is set (even to an empty string), or when `sirius.yaml` exists in the
  working directory or `$HOME/.sirius/`. Either select that file explicitly with
  `--sirius-config` so it is saved with the run, or remove it to use defaults.
- The same check applies when replaying a finding without YAML on another host. Every session
  records whether it selected a YAML file or verified that no ambient configuration was present.

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
| matches `known_issues.toml` | `known_issue` | `findings`, with the issue link | ignored |

So a bug that only shows up once window functions or `EXCEPT` are generated is not lost: it sits
in the `findings` section of the same summary with its own replayable bundle, below the gaps.

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
├── summary.json, summary.txt           verdict counts, completion status, coverage histogram
├── queries.jsonl                       flushed per-query records (every observation is here)
├── datasets/w<worker>-s<spawn>-d<n>.sql
├── logs/                               native stderr, active operation, runtime per worker
└── findings/<n>-<verdict>-<hash>/
    ├── query.sql, dataset.sql          immutable original inputs
    ├── config.toml, sirius.yaml        configuration for this particular worker
    ├── meta.json                       outcome, variant, comparison, timing, sample evidence
    ├── environment.json, runtime.json  binary/source provenance and session settings
    ├── worker.stderr                   log snapshot at observation
    ├── bundle.json                     SHA-256 integrity manifest for the original inputs
    ├── reduced.sql, reduction.json     optional reduction evidence
    ├── REPLAY.md, repro.sql            CLI and standalone shell reproduction instructions
    ├── repro_catch2.cpp                regression-test starting point (needs developer review)
    └── additional/<n>/                 up to five more complete bundles with the same signature
```

**How findings are grouped.** Signatures are grouping heuristics, not confirmed root causes.
Mismatch and timeout signatures include the original SQL, dataset identity, comparison mode,
variant value and result evidence, so distinct inputs stay separate even when their operator
labels match. Repeated errors group by normalized reason. This favours keeping
evidence over compact counts; use triage groups to merge candidates after investigation.

**Reduction never replaces the original.** The original observation is saved before reduction
starts. Each error signature is reduced once per worker; later queries with the same signature
are kept as additional reproducers without spending another reduction budget. A reducer failure is recorded separately with `stage=reduction` and the active
candidate. Reduced output is additional evidence and must itself be replayed. `bundle.json`
detects missing or edited original inputs; to test an edited query, use SQL-file replay with an
explicit dataset.

**Provenance.** Source revisions describe the checkout at run time; the extension fingerprint
identifies the actual binary, and the revision alone does not prove how it was built. Findings
carry small, labelled result samples and comparator differences rather than full result dumps.
`repro.sql` prints CPU/GPU results plus any recorded variant result; the CLI does the comparison.

**Naming.** Dataset filenames include the worker incarnation, so a respawn cannot overwrite
earlier data. Every retained additional reproducer carries its own dataset and settings.

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

# A stream of SELECT statements, each in its own supervised process
pixi run -e duckdb-python fuzz replay-file /absolute/path/queries.sql \
  --dataset /absolute/path/dataset.sql
```

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
- Legacy bundles without saved YAML require an explicit `--sirius-config`; bundles without
  integrity metadata emit a warning.
- Additional query files from older runs may lack their own dataset; recover the association
  from that run's `queries.jsonl` first.

---

## Triage and issue drafts

Triage re-validates saved findings, optionally shrinks them, and writes copy-ready Markdown
issue drafts for candidates that pass an automated policy. Nothing is created on GitHub and
there is no CI integration; publishing is a separate manual step. Human review is optional.

```text
  run dir / finding dirs
          │
          ▼
  ┌──────────────┐   3 replays + row-order permutation,   ┌──────────────┐
  │ fuzz triage  │──▶ optional reduction, per candidate ──▶│  REPORT.md   │  candidate queue
  └──────────────┘                                        │  report.json │  + batches/ history
          │                                               └──────────────┘
          │  optional                                             │
          ▼                                                       ▼
  ┌──────────────┐                                        ┌──────────────┐
  │ fuzz review  │  human disposition, evidence link      │ fuzz         │  DRAFTS.md
  │ fuzz group   │──────────────────────────────────────▶ │ issue-draft  │  drafts.json
  └──────────────┘                                        └──────────────┘
```

Start with a small selection; each replay initializes a fresh GPU process.

```bash
# One or more finding directories or campaign run directories
pixi run -e duckdb-python fuzz triage /absolute/path/run-or-finding \
  --out /absolute/path/triage --attempts 3 --timeout 90 --duration 30m

# Compare against a second, explicitly selected compatible binary
pixi run -e duckdb-python fuzz triage /absolute/path/finding \
  --out /absolute/path/triage-comparison --no-reduce \
  --extension /absolute/path/current/sirius.duckdb_extension \
  --compare-extension /absolute/path/other/sirius.duckdb_extension

# Refresh reports and eligible drafts from saved evidence, without running queries
pixi run -e duckdb-python fuzz triage-report /absolute/path/triage

# Export all automatically eligible candidates, or one candidate
pixi run -e duckdb-python fuzz issue-draft /absolute/path/triage --all
pixi run -e duckdb-python fuzz issue-draft /absolute/path/triage F-CANDIDATE_ID
```

### Triage outputs

| File | Contents |
|------|----------|
| `REPORT.md` | candidate queue with links to SQL, data, outcomes, logs and provenance |
| `report.json` | the same queue for other tools |
| `batches/` | each candidate's replay and reduction history |
| `DRAFTS.md` | links to eligible local drafts and any separate `.sql` dataset attachments |
| `drafts.json` | validation checks and reasons a candidate needs investigation |

Original inputs are copied and fingerprinted. Only current links indicate eligibility.
Regeneration preserves user edits and reports conflicts.

### What triage does per candidate

- **Replays.** Default is three independent replays plus one replay with rows reversed within
  each supported `INSERT ... VALUES` batch. Exact CPU fingerprints check stability across runs
  and insertion orders. Floating-point accumulation can change fingerprints, and one permutation
  cannot prove determinism.
- **Classifies.** Automatic results: `automatically_reproduced`, `intermittent`,
  `not_reproduced`, `changed_failure`, `unstable_reference`, `needs_investigation`.
  Reproduction alone does not qualify a candidate for a draft. Timeouts record their deadline
  and CPU/GPU phase. Plan rejections are coverage gaps until reviewed.
- **Suggests groups** from signatures and query structure. These are hints only; they do not
  establish a shared root cause or transfer review status.
- **Reduces (optional).** SQL and data are shrunk in fresh supervised processes, including for
  native crash and timeout candidates. Defaults: 20 proposed edits and 120 s per candidate;
  tune with `--reduce-steps`, `--reduce-seconds`, or disable with `--no-reduce`. A reduction is
  accepted only if it preserves the failure signature twice, has a stable CPU reference, and
  passes the available row-order check. The reducer removes selected clauses, projections and
  `INSERT` rows; it leaves ordered-query SQL unchanged and declines unsupported quoting. The
  result is the smallest reproducer seen within budget, not a guaranteed minimum. An unrelated
  crash cannot substitute for a mismatch.

### Budgets, exit codes and resuming

- Each attempt needs enough remaining budget for its full `--timeout`; choose
  `--reduce-seconds` greater than `--timeout` for reduction to run. Provenance and report I/O
  add overhead to `--duration`.
- Ctrl-C preserves evidence and exits 130. Incomplete work or budget exhaustion exits 2.
  Completed triage exits 0 even when discrepancies were found.
- A workspace lock prevents concurrent writers.
- **Resume** by repeating the same command and output directory. Saved evidence is checked before
  reuse; edited or missing inputs, outcomes or reductions block reuse and export. An expired
  reduction time budget leaves an incomplete trial that resumes with a fresh budget; a completed
  step-limited pass stays complete.
- A changed binary, runtime, host, harness or configuration starts a **separate batch**. Use
  `--rerun` to force fresh attempts, including after changes to unrecorded system libraries.
  Keep binaries unchanged during an investigation.

### Automated validation policy (`saved-evidence-v1`)

The policy recomputes observations from fingerprinted evidence. A candidate is eligible for an
automatic draft only if **all** of the following hold:

- **Stable reproduction.** At least three distinct original replays with the same failure, a
  stable CPU fingerprint and successful interception probes. Mismatch evidence must include a
  completed GPU operation.
- **Matching provenance.** Extension and DuckDB fingerprints, effective configuration and
  baseline session settings all match. Replay YAML must match the original finding; adding or
  changing YAML needs investigation even if every replay uses the same file. A no-YAML source
  and all its replays must record verified built-in-defaults selection; older no-YAML evidence
  without that provenance needs investigation, since new replays cannot establish which
  configuration an older run used. Runs that bypassed extension version metadata need
  compatibility investigation.
- **Eligible failure type.** A mismatch or GPU-phase crash under multiset comparison. Timeouts,
  setup errors, memory exhaustion, plan rejections, intermittent results and unstable references
  need investigation.
- **Conservative SQL screening.** Unknown functions, volatile functions, windows, limits,
  sampling and ordered-result semantics need investigation. Screening does not prove SQL
  semantics.
- **Insertion-order check.** Must preserve both the failure and the CPU fingerprint. Only a
  recognized empty or one-row literal dataset may skip it.
- **Completed reduction pass.** The selected reduction must preserve the failure in two distinct
  replays, with a stable CPU fingerprint and the applicable insertion-order check.

Eligible drafts say **automatically validated** and never claim a human reviewed them. Any
recorded human objection or uncertain disposition blocks automatic drafting, even if its batch
is stale. Grouping does not transfer eligibility. A current explicit human verification can
still support a per-candidate draft when the automatic policy declines it.

### Optional human verification

For each candidate: inspect the original and best reduced inputs, the CPU/GPU differences,
settings, binary hashes and logs. Check expected SQL semantics, NULLs, types, ordering and
floating-point tolerance. Then run a **separate replay yourself** from the original bundle or a
retained reduction attempt bundle (each carries its exact query, dataset and settings), and
inspect that new evidence before recording a disposition. Automated reproduction does not
perform this step.

```bash
pixi run -e duckdb-python fuzz replay /absolute/path/candidate/source --original

# Record uncertainty, expected behaviour, a harness problem, or another disposition
pixi run -e duckdb-python fuzz review /absolute/path/triage F-CANDIDATE_ID \
  --disposition needs_investigation --reviewer 'Your name' \
  --notes 'Explain what you checked and what remains unresolved.'

# Only a person who has actually performed the verification should run this
pixi run -e duckdb-python fuzz review /absolute/path/triage F-CANDIDATE_ID \
  --disposition manually_verified --reviewer 'Your name' \
  --evidence /absolute/path/new-replay-attempt \
  --expected 'Explain the correct CPU result and SQL semantics.' \
  --actual 'Explain the observed GPU discrepancy.' \
  --notes 'Describe the checks you personally performed.' \
  --acknowledge-manual-verification

# Export a draft that incorporates the recorded human verification
pixi run -e duckdb-python fuzz issue-draft /absolute/path/triage F-CANDIDATE_ID
```

| Disposition | Notes |
|-------------|-------|
| `manually_verified` | requires a completed triage batch, `--evidence` pointing at matching replay evidence (same extension, DuckDB runtime, configuration, CPU fingerprint, and original or accepted reduced inputs), and `--acknowledge-manual-verification` |
| `needs_investigation` | records uncertainty; blocks automatic drafting |
| `intermittent`, `not_reproduced` | reproduction outcome as judged by the reviewer |
| `harness_problem`, `expected_behavior` | not a Sirius bug |
| `duplicate` | requires `--duplicate-of F-OTHER_ID` |

Review history and evidence are retained. New batches make prior reviews stale; edited or
missing evidence prevents export. The attestation records the reviewer's statement only:
software cannot prove that a person inspected the results, and **agents must not supply it on
the user's behalf**.

**Grouping.** Assign related candidates to a named group; reassign a subset to split it.
Grouping preserves each candidate's evidence and verification status.

```bash
pixi run -e duckdb-python fuzz group /absolute/path/triage F-ID1 F-ID2 \
  --name 'Possible cause' --reviewer 'Your name' --notes 'Why these appear related'
```

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
