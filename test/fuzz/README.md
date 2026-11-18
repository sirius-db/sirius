# siriusfuzz: differential fuzzing for Sirius

## Regression corpus and yield

`pixi run fuzz corpus` compares every small SQL reproducer in `test/fuzz/corpus/`
on DuckDB CPU and strict Sirius GPU. It reports `still differs`, `now matches`,
`declined`, or `error` for each file. A `now matches` result is a prompt to check
whether that family's quarantine can be removed. Dataset-dependent failures are
listed in [the corpus README](corpus/README.md).

`fuzz corpus` exits 1 unless every file still differs. A file that starts
matching means the engine fix landed: remove the file and lift its switch.

`oracle.tlp_share` (default 0.1) adds a GPU-only check to that share of plain
SELECT queries whose CPU comparison passed: the query is rerun under an integer
predicate, its negation, and its IS NULL form, and the three results together
must equal the original. A difference is a `tlp_mismatch`. A partition the GPU
declines or fails leaves the query's verdict unchanged and is counted as
`partition_failed` in the summary.

Every campaign writes a coverage and budget table to `summary.txt`, machine-readable
`coverage`, `accounting`, and `budget` sections to `summary.json`, and one row to
`yield.jsonl` in the output root. Run `pixi run fuzz coverage-check <run-dir>`
against a completed campaign to require every claimed feature at least once and
at least one emission per 1,000 queries. `test/fuzz/scripts/nightly.sh 15m
<output-root>` runs doctor, corpus, a campaign, and the coverage check. The campaign
seed is the UTC date unless `FUZZ_SEED` is set. The script
is run manually; it is not installed in a scheduler.

| Gap switch | Withheld behavior | Issue | Corpus |
|---|---|---|---|
| `features.joins.any_mixed` | mixed arithmetic ANY operands | #2153 | bundle only |
| `features.casts.float_to_int` | FLOAT to integer rounding | #2152 | `2152-float-to-integer.sql` |
| `features.casts.integer_narrowing`, `integer_sign_change`, `integer_out_of_range` | integer casts outside the target range | #2580 | `2580-try-cast-wrap.sql` |
| `features.casts.decimal_scale` | explicit and implicit DECIMAL scale reduction | #2590 | `2590-decimal-scale.sql`, `2590-decimal-coalesce.sql`, `2590-decimal-case.sql` |
| `features.casts.timestamp_units` | casts to second/millisecond timestamp units | #2594 | `2594-timestamp-unit.sql` |
| `data.parquet.timestamp_ms` | Parquet tables containing `TIMESTAMP_MS` | #2595 | saved bundle |
| `data.edge_values.nan_inf`, `features.aggregates.float_division_minmax` | nonfinite MIN/MAX inputs, including computed NaN | #2085, #2592 | `2085-nan-grouped-minmax.sql`, `2592-nan-ungrouped-max.sql` |
| `data.edge_values.unicode_case` | Unicode case mappings | #2598 | `2598-unicode-case.sql` |
| `data.edge_values.date_beyond_ns` | DATE values outside the nanosecond timestamp range (years 1 and 9999) | #2631 | `2631-date-trunc-far-date.sql` |
| `features.scalar_functions.regexp_backreferences` | repeated match backreferences after multibyte text | #2650 | `2650-regex-backreference.sql` |

The filtered MIN (#2593), floating AVG (#2591), and EXCEPT ALL/TopN (#2596)
families have no dedicated generation quarantine; their saved reproducer or
bundle is checked separately.

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
(`sirius-duckdb/build/release/duckdb`) over pipes and loads the matching
`sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension` automatically.
For CPU-only use, any DuckDB CLI works (`--shell`, or `SIRIUSFUZZ_SHELL`).

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
pixi run fuzz run --shell /path/to/other/sirius-duckdb/build/release/duckdb \
  --extension /path/to/other/sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension \
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

**The shell.** `--shell` names the DuckDB binary to drive; otherwise the fuzzer tries
`$SIRIUSFUZZ_SHELL`, the configuration's `sirius.shell`
(`sirius-duckdb/build/release/duckdb`), then `duckdb` on `PATH`. The build's shell loads Sirius
from `sirius.extension`, which defaults to
`sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension`.
When selecting another build, pass both `--shell` and `--extension` to keep them matched.
A plain DuckDB CLI of the same version also works with `--extension`, or on its own for
`--cpu-only`. Custom TOML files using the old `build/release/` paths need both paths updated.

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
3. **Compare.** Rows compare as a multiset unless `ORDER BY` covers every output column and
   none of its keys is a computed FLOAT/DOUBLE (an average or arithmetic result sorts near-ties
   differently on the two sides, so such queries also get no `LIMIT`). FLOAT/DOUBLE use relative
   and absolute tolerances; everything else compares exactly, including DECIMAL and NULL. A
   mismatch whose CPU result changes when the input rows are reordered is classified as
   *ambiguous* rather than a bug.
4. **Save.** Findings are grouped by signature, shrunk, and reported with verdict counts.

Generated tables live in an `ATTACH`ed file-backed database and are `CHECKPOINT`ed, because
in-memory tables never reach the GPU native scan. A GPU fault or a query that overruns its
timeout kills the shell process; the worker records the crash or timeout with the shell's
stderr, starts a fresh shell, re-attaches the dataset files and carries on. The orchestrator
respawns a worker only if the Python process itself dies or stalls (`--max-respawns`, default
200); a stalled or cancelled worker is killed together with its shell. With both `--queries` and `--duration`, whichever is reached first stops the run; Ctrl-C
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

Join sampling uses `features.subqueries.correlated_in_share` (0.2 by default,
1.0 in gaps mode) to retain correlated `IN` while limiting repeated mixed-key
MARK declines. `features.joins.cartesian_max_rows` bounds cross and inequality-only
joins by the product of their input row counts; derived inputs require a `LIMIT`.
Negated subqueries are counted separately as `subquery:not_in` and
`subquery:not_exists`.
The join switches `any_join`, `any_mixed`, `self_join`, `grouped_inputs`, `expression_keys`,
and `on_filters` add those shapes when enabled. `any_mixed` and `on_filters` are
gaps-only because mixed arithmetic ANY joins can hit a GPU AST type error
(#2153) and outer ON filters can hit an arbitrary-join planner decline.
`max_tables` permits rare five
and six table chains, and `star_share` chooses joins back to the first table.
Four-or-more-table chains use semi/anti joins on `k` and no fact table: default
tables repeat `k`, so equality alone does not bound their fan-out.
`any_join_share` controls the share of eligible INNER joins that use an ANY
condition, so a focused acceptance run can collect enough samples.
`self_join_share` similarly controls explicit self joins. Grouped joins carry
`join:grouped_g` or `join:grouped_agg` for their exact join key.
`aggregates.float_division_minmax` is gaps-only: finite inputs can produce NaN
through division, and GPU MIN/MAX then differ from DuckDB (#2085, #2592).
Each `queries.jsonl` record includes `generator_tags`, allowing per-shape
verdict and fallback rates to be counted without inferring them from SQL text.

There is one configuration file, `config/default.toml`; every command loads it unless `--config`
names another. Its `[features]` flags mark what Sirius runs on the GPU today: every enabled
feature is expected to stay on the GPU, so a plan-time fallback is a finding. When Sirius gains a
feature, flip its flag to `true` there. `--set key.path=value` overrides any key for one run, and
`show-config` prints the effective result.

`variants.per_query` selects how many Sirius settings to vary on a sampled
comparison; `variants.share` is the fraction of successful CPU/GPU comparisons
sampled at all (default 1.0). For example, `--set variants.share=0.5` runs
variants on about half the compared queries.

`known_issues.toml` quarantines confirmed divergences by regex; every entry carries an issue link.

**Window functions, aggregate filters and set operations.**
`features.window_functions` generates whole-partition `sum`, `avg`,
`count`, `min`, and `max`; ordered `rank`, `dense_rank`, and `row_number`; and running
`sum`, `count`, `min`, and `max` with an explicit unbounded-preceding ROWS frame. It also
generates multiple window specifications and a derived-table top-N-per-group filter. These
queries use direct base-table inputs. `features.window_cast_tiebreaks` adds exact cast
order keys for unsigned and narrow DECIMAL columns; otherwise scopes without
complete order keys use `rank` instead of order-sensitive running windows.
`features.window_gaps` adds declined default and bounded frames and running `avg` in gaps mode.

`features.aggregates.filter` adds `FILTER (WHERE ...)` to roughly one fifth of grouped and
ungrouped aggregates, including `count(DISTINCT x)`. The generated predicate uses the same
input scope as the aggregate; `features.aggregates.filter_subquery` allows experimental
`IN`/`EXISTS` predicates in gaps mode. `features.set_ops.except` and `.intersect` each cover
distinct and `ALL` forms, including pairs of projections over the same table. By default the
generator keeps computed floating-point results out of exact DISTINCT, set-membership,
grouping, and window keys. `features.float_exact_keys` restores those shapes in gaps mode so
the known last-bit rounding issue stays measurable. `EXCEPT ALL` and `INTERSECT ALL` exclude
floating-point keys because Sirius does not admit those keys for the `ALL` forms.

**Float-to-integer casts.** `features.casts.float_to_int` is off by default and on in gaps mode. A one-expression `CAST(FLOAT AS BIGINT)` check differs at negative half values (#2152), so default runs do not emit those casts.

**Expressions and edge-value data.** The scalar function list includes `upper`,
`round` (FLOAT/DOUBLE with optional constant INTEGER precision), and date arithmetic.
`features.scalar_functions.gap_enabled` adds `lower`, `abs`, and unary minus in gaps
mode; those missed the initial per-shape 95% admission gate.
`features.expressions.simple_case` adds `CASE x WHEN value`;
`features.aggregates.expression_args` adds `sum` and `count` over CASE and `min`/`max`
over casts in gaps mode while aggregate-expression acceptance is investigated. Accepted functions also reach filters, CASE branches, and grouping keys;
`features.scalar_functions.window_args` adds them to window arguments in gaps mode.
`features.scalar_functions.join_neighbours` tries `upper` on
non-key join comparisons in gaps mode. The generator keeps computed floating-point
values out of exact grouping and ordering keys by default.

`features.casts.integer_to_varchar`, `.decimal_to_integer`, `.integer_to_decimal`,
`.decimal_to_double`, `.date_timestamp`, and `.try_cast` control cast boundary
shapes. Integer narrowing and sign changes remain in gaps mode while overflow
behavior differs; decimal scale reduction and timestamp second/millisecond
casts also remain in gaps mode because one-expression checks found mismatches.
`data.edge_values.unicode_case` exposes Unicode case-mapping differences in
gaps mode. NaN/infinity data is also gaps-only because floating aggregates
mismatch on it (#2085, #2592); negative zero and date extremes stay on
by default. Rounding ties such as `0.125` and strings over 64 KiB have their
own gaps switches. The long-string switch inserts at most one such value per small dataset;
DuckDB-native GPU scans currently decline overflow strings.

**Type switches.** `types.decimal_widths` chooses the concrete DECIMAL
widths used for generated columns. The type probe found native scan, filter,
group, and join matches for `(9,2)`, `(9,9)`, `(12,2)`, `(18,0)`, `(18,4)`, and
`(18,18)`; widths of 19 digits or more and `(4,0)` declined at scan. The
shipped list includes all six widths after DECIMAL acceptance. Mixed
scale arithmetic, comparisons, and join keys use DuckDB's exact result width.
Default DECIMAL `COALESCE` and `CASE` result arms share one exact type:
mixing scales introduces an implicit scale-reducing cast that reproduces
the #2590 rounding mismatch (`a3dc45ec` for `CASE`).
Gaps mode also selects `(4,0)` and widths of 19–38 digits to measure their
native-scan declines.
`types.hugeint` and `types.uhugeint` add 128-bit integer columns in gaps mode;
native scan declines both. `features.aggregates.hugeint_sum` leaves
`sum(BIGINT)` at DuckDB's natural HUGEINT result in gaps mode. Its isolated
acceptance had 55 OK and no planner declines in 56 tagged queries, but a
combined default run had six planner declines in 100 tags (94% non-fallback),
below the 95% admission gate. The declines involved 128-bit GPU VALUES carriers.
`features.joins.mixed_integer_keys` tests INTEGER/BIGINT,
SMALLINT/UINTEGER, and BIGINT/UBIGINT equality keys in gaps mode. The
BIGINT/UBIGINT pair had 35/38 non-fallback, below the 95% gate.
`types.array` adds fixed `INTEGER[3]` columns by default and projects them only in a simple scan; the probe found
array scan/projection matches, while array range filters declined. Focused
acceptance had 31/31 tagged projections OK.

`data.strings.charset=utf8` includes accented Latin, CJK, and emoji. The
repeated-match regexp backreference path is quarantined under #2650; the
separate case-mapping gaps remain controlled by
`data.edge_values.unicode_case`.
`data.strings.max_len` is a byte ceiling; lengths near 4,095 are rare so most
strings remain short. The probe accepted 4,095 bytes and declined 4,096.
UTF-8 and long-string tags require the referenced source column to contain
that content in the generated dataset.
`data.null_density_extremes` adds rare all-NULL and no-NULL columns by
default, tagged when referenced.
`features.scalar_functions.variable_divisors` allows column or expression
denominators for integer division and modulo in gaps mode. Default queries
use nonzero literal divisors to avoid CPU division-by-zero skips.
`features.scalar_functions.regexp_backreferences` keeps repeated match
backreferences such as `\0\0` in gaps mode. The CPU duplicates the match but
the GPU does not when multi-byte characters precede it (#2650).
TIME, TIMESTAMPTZ, INTERVAL, UUID, BLOB, ENUM, LIST, and STRUCT are not
generated: the native scan declined every one of them in the
probe, so their `types.*` switches are reserved for a future decoder pass.

**Sirius YAML.** `--sirius-config` selects it (repeat the flag to alternate across workers);
otherwise the configuration's `sirius.configs` applies (by default the integration YAML and
`config/partition16k.yaml`, whose 16 KiB budgets split the small fuzz tables into several hash
partitions), and with `sirius.configs = []` the run
uses whatever Sirius itself would pick up (`SIRIUS_CONFIG_FILE`, `./sirius.yaml`,
`~/.sirius/sirius.yaml`, else built-in defaults). Whichever file is used is copied into the run
and into every finding, so a replay elsewhere restores it. `config/` also holds
profiles for specific regimes: `micro16k.yaml`, `micro64k.yaml` and
`micro256k.yaml` shrink scan, partition, concat and sort batches to that size so
small tables span many batches; `lowmem.yaml` and `lowmem-disk.yaml` restrict
memory, the latter with a disk tier.

**Dataset layouts.** `[data.pin] tables_fraction`, `tiers`, and `compression` select a
seeded subset of base tables to pin after `CHECKPOINT`. `[data.parquet]
tables_fraction` and `row_group_size` replace a separate seeded subset with
views over Parquet files after `CHECKPOINT`. Pinned and Parquet subsets do not
overlap. The selected tables, tiers, compression choices, and row-group size are
recorded in a comment in each saved `dataset.sql`; `fuzz replay` recreates the
Parquet files in its own work directory and reapplies pins after loading the
dataset. CPU-only replay omits the Sirius pin calls. A storage campaign is the
default configuration with, for example,
`--set data.pin.tables_fraction=0.5 --set data.parquet.tables_fraction=0.5`.

---

### Large dimension/fact datasets

`data.large.fraction` selects the large tier per dataset. It defaults to `0.0`,
so existing campaigns keep the small tables. Set it to `1.0` to use only the
large tier or `0.3` for a mixed campaign. The large tier has `data.large.dims`
dimension tables (`t0`, ...), each with a dense unique `k`, and
`data.large.fact_tables` fact tables (`f0`, ...). Each fact has a unique `k`,
one `fk_tN` per dimension, and typed payload columns. `data.large.fact_rows`
sets the row-count range. The SQL-generated facts use seeded arithmetic and
pool values, so saved `dataset.sql` files replay without materializing Python
tuples. `Dataset.data_sql(permutation_seed)` inserts the same fact rows in a
different order for ambiguity checks.

`data.large.null_fk_ratio`, `dangling_fk_ratio`, and `skew` control foreign-key
NULLs, keys absent from their dimension, and concentration on hot dimension
keys. `data.large.fact_bias` is the chance that a top-level FROM clause starts
at a fact table. Large-tier joins use the `fk_tN = tN.k` relationship. The fuzzer skips
fact cross joins and inequality-only fact joins because they make the CPU
reference unbounded.

For a bounded large-tier run, use `--set data.large.fraction=1.0` with one of
`large-default.yaml`, `large-8m.yaml`, `large-lowmem.yaml`, or
`large-lowmem-disk.yaml` through `--sirius-config`. The `large-8m` profile also
needs `--set 'variants.settings.hash_partition_bytes=[8388608]'` and
`--set 'variants.settings.max_sort_partition_bytes=[8388608]'` so randomized
per-query settings do not override the YAML. Use
`--set data.parquet.row_group_size=20000` with the Parquet variant to create
multiple row groups on large fact tables.

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

The summary lists findings first, since a crash or a wrong answer found during a gaps run still
outranks any gap, then runtime fallbacks ordered by the GPU time they threw away, which is the
order in which to move their checks to plan time (fuzz datasets are small, so read the figures as
relative), then plan-time fallbacks. Every entry shows its smallest query; both gap tables group by
the reason that query reports:

```text
findings (1 unique); smallest query that reproduces:
  [mismatch] x1    012-mismatch-7c8d9e0f  row count 41 vs 40
      SELECT "a0"."k" AS c0 FROM "t0" AS "a0" LEFT JOIN "t1" AS "a1" ON ("a0"."k" > "a1"."k")
      features: ColumnRef, Compare(>), Join(left), Select, TableRef

runtime fallbacks (1 reason, 5 queries, 3.2s of GPU work thrown away); smallest query that passes the planner and still fails:
  x5      3.2s  Distinct aggregates not supported in GPU path yet   known: sirius-db/sirius#1218
      SELECT count(DISTINCT "a0"."c1") AS c0 FROM "t2" AS "a0"
      features: Agg(count,distinct), ColumnRef, Select, TableRef   findings: 007-runtime_fallback-2b3c4d5e

plan-time fallbacks (2 reasons, 207 queries):
  x180  Window not supported   findings: 000-plan_fallback-1f2e3d4c
  x27   Unsupported expression in projection: {concat, regexp_matches}   findings: 003-plan_fallback-9a8b7c6d, 011-plan_fallback-5e6f7a8b
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
shell guarantees that. A plain DuckDB CLI that refuses a `LOAD` names the version it wants.

**Shell protocol.** Every statement is followed by a marker on each stream, written with the
shell's own dot commands so they never pass through SQL or Sirius; results (JSON on stdout) and
errors (stderr) are delimited exactly, and the banner Sirius prints on stdout when it falls back
is kept out of the result. The shell streams rows as they arrive, so a query that fails midway
leaves a partial result on stdout; the error on stderr decides the outcome. DECIMAL and HUGEINT values arrive as
strings and compare exactly; floating-point values arrive as numbers and compare with the
tolerances. A query whose output columns share a name is still compared positionally.
