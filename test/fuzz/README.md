# siriusfuzz: differential fuzzing for Sirius

A command-line tool that generates SQL, compares Sirius GPU results with DuckDB CPU results,
and saves findings for replay and triage. No AI agent is required.

## How it works

1. Generate typed tables, data and SQL using a TOML feature profile, including NULLs and edge
   values. The summary flags enabled features that were never generated.
2. Run each query on DuckDB CPU, then on Sirius with `enable_duckdb_fallback = false`.
   After a match, rerun under randomized Sirius settings and compare with the GPU baseline.
3. Compare rows as a multiset unless `ORDER BY` covers every output column. FLOAT/DOUBLE use
   relative and absolute tolerances; other values compare exactly, including DECIMAL and NULL.
   NaN/inf require exact matches. Mismatches whose CPU result changes under reordered input
   are classified as ambiguous.
4. Save findings by signature, optionally shrink the query, and report verdicts and coverage gaps.

Generated tables live in an ATTACHed file-backed database and are CHECKPOINTed, because
in-memory tables never reach the GPU native scan. Each worker is a separate process, so a GPU
fault or hang costs one worker (the orchestrator records the in-flight query and respawns).

## Verdicts

| Verdict | Meaning | Finding? |
|---------|---------|----------|
| `ok` | GPU rows match CPU rows, and every setting variant matched the GPU baseline | no |
| `cpu_error` / `cpu_timeout` | the reference run failed; the query is skipped and counted | no |
| `ambiguous` | mismatch that changes under permuted row order (nondeterministic query) | no |
| `mismatch` | GPU rows differ from CPU rows | **yes** |
| `variant_mismatch` | same query, one Sirius setting changed, different rows | **yes** |
| `plan_fallback` | Sirius declined the plan; the reason is recorded | strict: yes; frontier: counted |
| `fallback_mismatch` | frontier only: the CPU fallback path returned wrong rows | **yes** |
| `gpu_error` / `gpu_internal_error` / `gpu_oom` | the GPU run raised | **yes** |
| `timeout` | the GPU run exceeded `oracle.query_timeout_seconds` | **yes** (hang candidate) |
| `crash` | the worker process died during the query | **yes** |
| `known_issue` | a finding matching `known_issues.toml`; counted, not failed | no |

## Developer quick start

Use a supported Linux GPU host (x86-64 or aarch64 with the repository's CUDA environment).
The `duckdb-python` Pixi environment does not support macOS. A stock DuckDB wheel is suitable
for CPU-only harness tests, but is not a supported runtime for loading Sirius.

Build the extension and Python module from the same checkout and initialized submodules:

```bash
git submodule update --init --recursive
pixi run make
pixi run -e duckdb-python build-duckdb-python
pixi run -e duckdb-python fuzz doctor
```

`doctor` validates the TOML profile and output location, imports DuckDB, loads Sirius,
creates a tiny file-backed dataset, proves interception with the canary, and checks a GPU
query's answer. It runs in a disposable subprocess with a 120-second hard deadline, including
initialization and cleanup. It leaves a diagnostic directory with logs, configuration, runtime
information and `outcome.json`; temporary database files are cleaned up on normal exit.
It does not install dependencies, rebuild the engine, switch branches or download extensions.

```bash
# Choose an existing build, configuration and output location explicitly.
pixi run -e duckdb-python fuzz doctor \
  --extension /absolute/path/sirius.duckdb_extension \
  --sirius-config /absolute/path/sirius.yaml --out /absolute/path/fuzz-results

# A short, bounded exploration: one worker by default.
pixi run -e duckdb-python fuzz run --seed 42 --queries 100 --duration 3m --no-reduce

# A longer campaign; check shared GPU availability before increasing workers.
pixi run -e duckdb-python fuzz run --seed 42 --duration 30m --workers 2

# Targeted generation; other enabled expressions/operators can still appear.
pixi run -e duckdb-python fuzz run --queries 100 \
  --set 'features.scalar_functions.enabled=["substring","like"]'

pixi run -e duckdb-python fuzz show-config
pixi run -e duckdb-python fuzz-test
pixi run -e duckdb-python fuzz selftest --queries 100
```

The CPU-only selftest defaults to 8–16 rows per table, at most two joined tables
per SELECT, and one subquery level. These bounds keep the developer check short;
explicit `--set` values can increase them for a larger CPU-only run.

Use absolute paths for input/output outside the repository: the Pixi `fuzz` task runs from
`test/fuzz`. Profile-relative engine paths are resolved by the existing configuration loader.
`--sirius-config` overrides the profile's YAML; the harness sets `SIRIUS_CONFIG_FILE` from that
selection. Repeat the flag to alternate configurations across worker processes.

With `sirius.configs = []` and no `--sirius-config`, GPU sessions use built-in defaults.
They stop before connecting if `SIRIUS_CONFIG_FILE` is set (even empty), or if `sirius.yaml`
exists in the working directory or `$HOME/.sirius/`. Select an intended ambient file with
`--sirius-config` so the harness saves it, or remove that ambient configuration to use defaults.
The same check applies when replaying a finding without YAML on another host. Each session
records whether it selected a YAML file or verified that no ambient configuration was present.

A run defaults to one worker and 500 queries. If both `--queries` and `--duration` are supplied,
the first reached budget stops the campaign. Workers already executing a query may produce a
small query-count overshoot. The duration includes worker startup and execution, but excludes
initial provenance collection. In-flight work is stopped at the duration boundary and is not
reported as an engine hang merely because the campaign budget expired.

Ctrl-C stops workers, preserves completed findings and writes a summary marked `cancelled`.
Worker startup failures, failed interception checks and exhausted restart budgets produce an
`incomplete` summary. SIGKILL or machine failure cannot guarantee a final summary; completed
finding bundles and the flushed query log remain on disk. Worker database files from forcibly
terminated processes can remain under the run directory and can be removed after inspection.

| Exit code | Meaning |
|-----------|---------|
| `0` | Run completed (findings may exist); doctor passed; replay matched |
| `1` | Run findings with `--fail-on-findings`, or replay discrepancy/crash/timeout/inconclusive CPU outcome |
| `2` | Invalid setup/input, incomplete campaign, or failed doctor |
| `130` | Cancelled with Ctrl-C |

### Agent-assisted use

Use the [sirius-fuzz skill](../../.agents/skills/sirius-fuzz/SKILL.md) in Codex or the shared
`.claude/skills/sirius-fuzz` entry in Claude:

> Use $sirius-fuzz to run a short correctness check on an available development GPU.

The skill selects an accessible development host, requires a passing GPU doctor, runs within
the requested budget and summarizes saved evidence. Without a GPU it reports the blocker;
CPU harness tests and selftest run only when explicitly requested. Saved reports need no GPU.

### Build metadata

`build-duckdb-python` uses the extension preset's `OVERRIDE_GIT_DESCRIBE=v1.5.6`.
Version checks stay enabled. Use `--allow-metadata-mismatch` only for independently verified
matching builds; it cannot fix an ABI mismatch. Its allowance and use are recorded, and replay
announces a restored allowance. `--no-allow-metadata-mismatch` disables it for that attempt.

### Replay a finding or your own SQL

```bash
# Copy the entire finding directory, then replay it from any compatible Sirius checkout.
pixi run -e duckdb-python fuzz replay /absolute/path/finding
pixi run -e duckdb-python fuzz replay /absolute/path/finding --original

# Deliberately test another binary/config; the new attempt records the overrides.
pixi run -e duckdb-python fuzz replay /absolute/path/finding \
  --extension /absolute/path/new/sirius.duckdb_extension \
  --sirius-config /absolute/path/local.yaml --timeout 180

# A custom SELECT/WITH query and its CREATE/INSERT dataset.
pixi run -e duckdb-python fuzz replay /absolute/path/query.sql \
  --dataset /absolute/path/dataset.sql

# A stream of SELECT statements, each in its own supervised process.
pixi run -e duckdb-python fuzz replay-file /absolute/path/queries.sql \
  --dataset /absolute/path/dataset.sql
```

Replay restores the saved TOML, YAML, captured baseline session settings, comparison mode and
exact failing setting variant. It never chooses a new random variant. It uses `reduced.sql`
when available, or `query.sql` with `--original`. Missing required inputs fail explicitly;
replay never silently invents a replacement dataset for a finding. For custom SQL only,
`--dataset-seed N` explicitly requests a generated dataset instead of `--dataset`.

Each replay runs under a supervising process with a hard deadline (`--timeout`, 180 seconds
by default), so a native crash or an uninterruptible GPU call cannot kill or indefinitely block
the CLI. The supervisor records the last query, CPU/GPU phase, settings, signal/exit code and
logs. CPU-only replay is available with `--cpu-only` and does not establish GPU correctness.
Custom SQL defaults to multiset comparison; `--ordered` requires deterministic output ordering.
The generated-AST ambiguity filter is not rerun for SQL-only replays; replay is evidence for
manual investigation, not automatic bug confirmation.

Replay writes a new attempt directory and never modifies the source bundle. Each attempt records
its source, inputs, effective settings, environment and outcome. Changed extension fingerprints
are announced. Machine-specific YAML paths (for example spill directories) may need an explicit
`--sirius-config` override on another host. CUDA device visibility is recorded and uses the current
host's environment. Reproducibility across different builds, GPUs or runtimes is not guaranteed.
Legacy bundles without saved YAML require an explicit `--sirius-config`; bundles without integrity
metadata emit a warning. Additional query files from older runs may not have their own dataset;
recover the association from that run's `queries.jsonl` before attempting reproduction.

### Saved evidence

Outputs default to unique directories below `test/fuzz/out/`. Use `--out` to choose another root.

```text
run-<timestamp>-seed<seed>-<unique>/
  config.toml, sirius-<n>.yaml       effective configuration snapshots
  environment.json, invocation.json source revisions, binary hashes, GPU/runtime, CLI arguments
  summary.json, summary.txt         verdict counts, completion status, coverage histogram
  queries.jsonl                    flushed per-query records
  datasets/w<worker>-s<spawn>-d<n>.sql
  logs/                            native stderr, active operation, runtime per worker
  findings/<n>-<verdict>-<hash>/
    query.sql, dataset.sql          immutable original inputs
    config.toml, sirius.yaml        configuration for this particular worker
    meta.json                      original outcome, variant, comparison, timing and sample evidence
    environment.json, runtime.json binary/source provenance and session settings
    worker.stderr                  log snapshot at observation
    bundle.json                    SHA-256 integrity manifest for original inputs
    reduced.sql, reduction.json    optional additional reduction evidence
    REPLAY.md, repro.sql            CLI and standalone shell reproduction instructions
    repro_catch2.cpp               regression-test starting point, requiring developer review
    additional/<n>/               up to five additional complete bundles in the same signature group
```

Datasets include the worker incarnation in their filenames: a respawn cannot overwrite earlier
data. Every retained additional reproducer carries its own dataset and settings. All observations
are still in `queries.jsonl`; signatures are grouping heuristics, not confirmed root causes.
Mismatch and timeout signatures include the original SQL, dataset identity, comparison mode,
variant value, execution path and result evidence. Distinct inputs remain separate even when
their operator labels match;
repeated errors still group by normalized reason. This favors retaining evidence over compact
finding counts. Use triage groups to combine candidates after investigation.

The original observation is saved before reduction starts. A reducer failure is recorded separately
with the active candidate and `stage=reduction`; it does not replace the original mismatch.
Reduction output remains additional evidence and must be replayed. `bundle.json` detects missing
or edited original inputs. To test an edited query, use SQL-file replay with an explicit dataset.

Source revisions describe the checkout at run time. The extension fingerprint identifies the
actual binary; source revision alone is not proof of how that binary was built. Findings include
small, clearly labelled result samples and comparator differences rather than unbounded result dumps.
The standalone `repro.sql` prints CPU/GPU results and any recorded fallback or variant results;
the CLI performs the comparison. For fallback findings the script continues past the expected
strict plan rejection and reruns with fallback enabled.

The optional sqlsmith reducer loads an already installed extension; it never runs `INSTALL`.
AST reduction remains available without sqlsmith.

## Triage and local issue drafts

Triage validates saved findings and writes copy-ready Markdown issue drafts for eligible
candidates. Human review is optional. Nothing is created on GitHub. Start with a small selection;
each replay initializes a fresh GPU process.

```bash
# Accepts one or more finding directories or campaign run directories.
pixi run -e duckdb-python fuzz triage /absolute/path/run-or-finding \
  --out /absolute/path/triage --attempts 3 --timeout 90 --duration 30m

# Optionally compare an explicitly selected second compatible binary.
pixi run -e duckdb-python fuzz triage /absolute/path/finding \
  --out /absolute/path/triage-comparison --no-reduce \
  --extension /absolute/path/current/sirius.duckdb_extension \
  --compare-extension /absolute/path/other/sirius.duckdb_extension

# Refresh reports and eligible drafts from saved evidence without executing queries.
pixi run -e duckdb-python fuzz triage-report /absolute/path/triage

# Export all automatically eligible candidates, or one candidate with an optional title.
pixi run -e duckdb-python fuzz issue-draft /absolute/path/triage --all
pixi run -e duckdb-python fuzz issue-draft /absolute/path/triage F-CANDIDATE_ID
```

Open `REPORT.md` for the candidate queue and links to SQL, data, outcomes, logs and provenance;
`report.json` provides the same queue for other tools. Original inputs are copied and fingerprinted,
and each candidate retains its replay and reduction history under `batches/`.
`DRAFTS.md` links eligible local drafts and any separate `.sql` dataset attachments;
`drafts.json` records validation checks and reasons for investigation. Only current links indicate
eligibility. Regeneration preserves user edits and reports conflicts.

The default is three independent replays and a replay with rows reversed within each supported
`INSERT ... VALUES` batch. Exact CPU fingerprints check stability across runs and insertion orders.
Floating-point accumulation can change fingerprints, and one permutation cannot prove determinism.
Timeouts record their deadline and CPU/GPU phase. Plan rejections are coverage gaps until reviewed.

Automatic results include `automatically_reproduced`, `intermittent`, `not_reproduced`,
`changed_failure`, `unstable_reference` and `needs_investigation`. Reproduction alone does not
qualify a candidate for a draft. Suggested groups use signatures and query structure as hints;
they do not establish a shared root cause or transfer review status.

Triage optionally shrinks SQL and data with fresh supervised processes, including native crash
and timeout candidates. Defaults are 20 proposed edits and 120 seconds per candidate; use
`--reduce-steps`, `--reduce-seconds` or `--no-reduce` to control this. A reduction must preserve
the failure signature and strict/fallback execution path twice, have a stable CPU reference,
and pass the available row-order check.
The reducer removes selected clauses, projections and INSERT rows, leaving ordered-query SQL
unchanged and declining unsupported quoting. Results are the smallest observed reproducer within
the budget, not a guaranteed minimum. An unrelated crash cannot substitute for a mismatch.

Each attempt needs enough remaining budget for its full `--timeout`; choose `--reduce-seconds`
greater than `--timeout` for reduction. Provenance and report I/O add overhead to `--duration`.
Ctrl-C preserves evidence and exits 130; incomplete work or budget exhaustion exits 2;
completed triage exits 0 even with discrepancies. A workspace lock prevents concurrent writers.

Repeat the same command and output directory to resume. Saved evidence is checked before reuse;
edited or missing inputs, outcomes or reductions block reuse and export. An expired reduction
time budget leaves an incomplete trial that can resume with a fresh budget; a completed
step-limited pass remains complete. Changed binaries, runtime, host, harness or configuration
start a separate batch. Use `--rerun` to force fresh attempts, including after changes to
unrecorded system libraries. Keep binaries unchanged during an investigation.

### Automated validation policy

The `saved-evidence-v1` policy recomputes observations from fingerprinted evidence and requires:

- At least three distinct original replays with the same failure, a stable CPU fingerprint and
  successful interception probes; mismatch evidence must include a completed GPU operation.
- Matching extension and DuckDB fingerprints, effective configuration and baseline session settings.
  Replay YAML must also match the original finding; adding or changing YAML requires investigation
  even if every replay uses the same file. A no-YAML source and all its replays must record verified
  built-in defaults selection. Older no-YAML evidence without that provenance needs investigation;
  new replays cannot establish which configuration an older run used.
  Runs that bypassed extension version metadata need compatibility investigation.
- A mismatch or GPU-phase crash using multiset comparison. Timeouts, setup errors, memory exhaustion,
  plan rejections, intermittent results and unstable references need investigation.
- Conservative SQL screening: unknown functions, volatile functions, windows, limits, sampling and
  ordered-result semantics need investigation. This screening does not prove SQL semantics.
- The available insertion-order check must preserve both failure and CPU fingerprint. Only a
  recognized empty or one-row literal dataset can omit that replay.
- A completed bounded reduction pass. The selected reduction must preserve the failure in two
  distinct replays, with a stable CPU fingerprint and the applicable insertion-order check.

Eligible drafts say **automatically validated** and never claim a human reviewed them. Any
recorded human objection or uncertain disposition blocks automatic drafting, even if its batch
is stale. Grouping does not transfer eligibility. A current explicit human verification can
still support a per-candidate draft when the conservative automatic policy declines it.

### Optional human verification

For each candidate, inspect the original and best reduced inputs, CPU/GPU differences, settings,
binary hashes and logs. Check expected SQL semantics, NULLs, types, ordering and floating-point
tolerance. Run a separate replay yourself using the original bundle or a retained reduction
attempt bundle, which carries its exact query, dataset and settings. Inspect the new replay
evidence before recording a disposition. Automated reproduction does not perform this step.

```bash
pixi run -e duckdb-python fuzz replay /absolute/path/candidate/source --original

# Record uncertainty, expected behavior, a harness problem, or another disposition.
pixi run -e duckdb-python fuzz review /absolute/path/triage F-CANDIDATE_ID \
  --disposition needs_investigation --reviewer 'Your name' \
  --notes 'Explain what you checked and what remains unresolved.'

# Only a person who has actually performed the verification should run this command.
pixi run -e duckdb-python fuzz review /absolute/path/triage F-CANDIDATE_ID \
  --disposition manually_verified --reviewer 'Your name' \
  --evidence /absolute/path/new-replay-attempt \
  --expected 'Explain the correct CPU result and SQL semantics.' \
  --actual 'Explain the observed GPU discrepancy.' \
  --notes 'Describe the checks you personally performed.' \
  --acknowledge-manual-verification

# Export a local draft incorporating the recorded human verification.
pixi run -e duckdb-python fuzz issue-draft /absolute/path/triage F-CANDIDATE_ID
```

Review dispositions also include `intermittent`, `not_reproduced`, `harness_problem`,
`expected_behavior` and `duplicate` (requires `--duplicate-of F-OTHER_ID`). Manual verification
requires a completed triage batch and saved matching replay evidence with the same extension,
DuckDB runtime, configuration, CPU result fingerprint and original or accepted reduced inputs.
Review history and evidence are retained. New batches make prior reviews stale;
edited or missing evidence prevents export.
The attestation records the reviewer's statement; software cannot prove that a person inspected
the results. Agents must not supply it on the user's behalf.

Use `fuzz group /absolute/path/triage F-ID1 F-ID2 --name 'Possible cause' --reviewer 'Your name'
--notes 'Why these appear related'` to assign a group; reassign selected candidates to split it.
Grouping preserves each candidate's evidence and verification status. Publishing drafts is a
separate manual action; this workflow has no GitHub integration or CI.

## Profiles

`config/strict.toml` enables only what runs on the GPU today; it is identical to the built-in
defaults (a unit test enforces this) and treats any plan-time fallback as a finding.
`config/frontier.toml` also enables window functions, `DISTINCT`, grouping sets, `FULL`/`CROSS`
joins, `UNION`/`EXCEPT`/`INTERSECT`, uncorrelated subqueries, `TRY`, temporal-numeric and
to-VARCHAR casts; fallbacks are counted and the fallback path is checked for correct results.
When a feature lands on the GPU, flip its key in `strict.toml`.

`oracle.on_plan_fallback` controls plan rejections: `fail` saves a finding; `count` records
the coverage gap and checks the fallback result against CPU; `skip` records the gap without
executing the fallback query. Coverage gaps under `count` or `skip` do not trigger
`--fail-on-findings` or reduction. Fallback mismatches and execution errors remain findings.

`known_issues.toml` quarantines confirmed divergences by regex; every entry carries an issue link.
