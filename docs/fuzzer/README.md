# Understanding the Sirius fuzzer

`siriusfuzz` tests whether Sirius executes SQL correctly on the GPU. It generates
tables, fills them with data, builds SQL queries, and compares Sirius's answers
with DuckDB's CPU answers. When something disagrees, crashes, hangs, or falls
back, it saves the inputs and execution evidence so a developer can investigate
the observation later.

The work goes beyond generating random SQL. It addresses four practical problems:

1. Generate queries that are valid and exercise meaningful combinations of operators.
2. Establish that the GPU path was actually reached.
3. Compare answers without mistaking legal SQL behavior or floating-point noise for bugs.
4. Turn a failure in a large random query into a small, reproducible investigation.

This document explains the current implementation and the reasoning behind its
design. The [command and usage guide](../../test/fuzz/README.md) covers installation,
CLI flags, campaign management, and output formats. Examples below illustrate
semantics; they are not claims about bugs observed in a campaign.

## 1. The foundation: differential testing

A correctness test needs an **oracle**: a way to decide what the answer should be.
For a handwritten test, the developer supplies expected rows. A random query might
contain several joins, nested expressions, NULLs, and aggregates, making a
handwritten expected result impractical.

The fuzzer uses **differential testing** instead. It executes the same query over
the same data through two execution paths:

```text
                       Same SQL + same stored tables
                                    |
                         +----------+----------+
                         |                     |
                 DuckDB CPU execution    Sirius GPU execution
                         |                     |
                    reference rows         observed rows
                         +----------+----------+
                                    |
                            semantic comparison
```

DuckDB is a useful reference because Sirius is a DuckDB extension: the SQL
frontend and host environment are shared, while execution differs. The harness
switches `gpu_execution` off for the reference and on for the GPU run. Ordinary
SQL is sufficient because Sirius intercepts it transparently.

The core condition is:

```text
result_CPU(query, data) ≈ result_GPU(query, data)
```

Here, `≈` means SQL-aware equivalence, including duplicate rows and configured
floating-point tolerances. It does not simply mean that two printed outputs look
the same.

This choice avoids implementing a second SQL engine inside the harness. Its limit
is that agreement is not a proof of correctness: the paths share frontend behavior,
and a defect common to both can escape detection. A failing CPU reference also
cannot provide a trustworthy answer for that query.

## 2. One query's complete journey

The main campaign loop lives in [runner.py](../../test/fuzz/siriusfuzz/runner.py).
Its important steps are:

1. A worker opens a session: one DuckDB shell process, the one built with Sirius, driven
   over pipes. It creates a generated dataset in a file-backed database.
2. It checks Sirius interception on its first dataset.
3. The query generator builds an abstract syntax tree and renders it as SQL.
4. Before execution, the worker records the current SQL and dataset for crash attribution.
5. DuckDB runs the query on the CPU. A CPU error or timeout stops this query's comparison.
6. Sirius runs the query with CPU fallback disabled. Errors receive specific verdicts; a
   shell that dies or overruns its deadline is recorded as a crash or timeout and replaced.
7. Successful results are compared. A mismatch may trigger a reference-stability check.
8. After a match, selected Sirius settings are changed individually and the query is rerun.
9. The worker sends the original observation to the orchestrator for persistence.
10. Eligible findings are reduced, with reduced evidence saved separately.

Workers reuse a dataset for several queries before creating another one. The
shipped profile sets `sirius.dataset_queries = 200`. This amortizes table creation,
insertion, and checkpoint costs while still varying schemas and values across
the campaign. Reusing a dataset does mean adjacent queries share data; the saved
dataset makes that dependency explicit.

## 3. Generation starts with data that makes queries interesting

[schema_gen.py](../../test/fuzz/siriusfuzz/schema_gen.py) generates both schemas and
values. Purely uniform randomness would waste much of the campaign. For example,
joining two tables whose keys are independently drawn from enormous ranges often
produces no rows. Many faulty implementations also produce no rows, so such a test
has little power to expose them.

The generator therefore biases data toward useful situations:

| Choice | What it exercises | Why it matters |
|---|---|---|
| A numeric `k` column and shared small-value pools | Matching and repeated join keys | Joins produce meaningful matches and duplicate multiplicities. |
| Shared value pools mixed with independently generated values | Repetition and variety | Repeated values exercise grouping and matching; varied values exercise expression behavior. |
| A string and a non-key numeric column in every table | String functions and arithmetic | These families have inputs available even when other columns vary. |
| Per-column NULL probabilities and occasional all-NULL columns | SQL NULL propagation and aggregates | Entirely missing columns expose cases that occasional NULLs may miss. |
| Edge values under configurable switches | Integer boundaries and other special values | Representation and conversion defects often appear near boundaries. |
| Bounded ordinary numeric values | Successful reference execution | Excessive overflow would turn the run into mostly CPU errors. |

The default profile generates 3–6 tables with 50–2,000 rows per table. Its common
value ratio is 0.4, and its edge-value probability is 0.02. NULL probability is
assigned per column; the configured ratio is not a promise that every column has
exactly that fraction of NULLs.

NULL deserves special attention. SQL uses three-valued logic: a predicate can be
true, false, or unknown. `NULL = NULL` is unknown, and `WHERE` retains only true
rows. `IS NOT DISTINCT FROM` instead supports NULL-safe equality. Aggregates also
distinguish `COUNT(*)`, which counts rows, from `COUNT(column)`, which counts
non-NULL values. Combining these rules with joins is a productive source of
correctness tests.

The configuration exposes the tradeoff between reaching more successful
comparisons and testing more extreme inputs. The shipped profile favors small
ordinary numbers but still enables occasional integer extremes. It leaves some
other special values, such as NaN and infinity, disabled by default.

## 4. SQL is built as a typed tree

The generator does not assemble arbitrary text fragments. It constructs an
**abstract syntax tree (AST)** using [sqlast.py](../../test/fuzz/siriusfuzz/sqlast.py).
An AST represents SQL as nested objects: a `Select` contains projected expressions,
a source table or join, predicates, grouping, ordering, and optional limits.

For example:

```sql
SELECT a.k, SUM(b.amount) AS total
FROM t0 AS a
JOIN t1 AS b ON a.k = b.k
WHERE b.amount > 0
GROUP BY a.k;
```

Conceptually, its tree is:

```text
Select
  projected expressions: ColumnRef(a.k), Agg(sum, ColumnRef(b.amount))
  source: Join(TableRef(t0), TableRef(t1), Compare(a.k = b.k))
  predicate: Compare(b.amount > Literal(0))
  grouping: ColumnRef(a.k)
```

The renderer turns that structure into SQL, with identifier quoting handled by
the AST. The same tree also supports feature labels, ordering checks, and
structural reduction later.

### Types constrain expression construction

[query_gen.py](../../test/fuzz/siriusfuzz/query_gen.py) builds expressions toward a
requested result kind, such as numeric, string, Boolean, date, or timestamp.
String functions receive appropriate string inputs; predicates produce Boolean
expressions; set-operation arms are constructed with compatible projected types.
[sqltypes.py](../../test/fuzz/siriusfuzz/sqltypes.py) supplies type descriptions and
literal rendering, including exact decimal literals.

This reduces parser and binder failures. It does not guarantee successful
execution: a valid cast can still fail on a particular value, and a valid arithmetic
expression can still overflow. Those failures are counted as CPU reference errors
rather than treated as evidence against Sirius.

### Scope constrains column references

The generator tracks which aliases and columns are visible in the current query
and which belong to an outer query. That is necessary for nested SQL. A correlated
subquery may legitimately refer to an outer alias; an unrelated inner alias may
not be available to the outer projection. Semi and anti joins also have different
output visibility from ordinary joins.

Tracking scope avoids spending most of the budget on invalid name references.
Scalar subqueries use an ungrouped aggregate shape to produce a scalar result
without routinely violating the one-row requirement.

### Complexity is bounded, and features are gated

The generator composes joins, predicates, grouping, aggregates, CTEs, set
operations, subqueries, and scalar expressions subject to the configuration.
Depth limits and weighted choices keep queries manageable. Generation retries
when a selected construct cannot be built in its scope.

The one shipped [default profile](../../test/fuzz/config/default.toml) describes
the surface expected to run on the GPU. For example, it enables `UNION ALL`, but
disables several other set operations and window functions. An unexpected
plan rejection in a correctness campaign is therefore useful evidence: either
support is incomplete or the generator combined features in an unsupported way.

A switch enables a feature for sampling; it does not ensure every query uses it.
The summary reports generation statistics and enabled features never generated.
These are generation coverage signals, not code coverage or proof that a GPU
operator executed: the planner can simplify SQL or reject it before execution.

## 5. Proving that execution reaches Sirius

An extension can load successfully while queries still execute on the CPU. If the
harness compared CPU with CPU and called that GPU validation, a broken GPU path
could appear healthy.

[session.py](../../test/fuzz/siriusfuzz/session.py) addresses this in two ways.

### Stored tables reach the native scan path

Generated tables live in an `ATTACH`ed, file-backed DuckDB database. The harness
loads them with GPU execution off and runs `CHECKPOINT` before testing queries.
Checkpointing persists their state so Sirius's native GPU scan can see it.
In-memory tables do not reach that path, so a convenient in-memory test would
not establish the same readiness.

### A canary makes interception observable

When available, the harness enables the test-only
`sirius_test_inject_transparent_gpu_error` setting and runs a simple query. Seeing
the injected `fuzz-canary` error demonstrates that the query passed through the
GPU operator where the injection occurs. The setting is then cleared.

For builds without that option, the harness uses a probe expected to receive a
Sirius-specific plan rejection. That is weaker evidence of interception at
planning time. `doctor` additionally executes a GPU aggregate and checks its
expected answer, so its readiness check includes successful execution.

`run` performs these checks before fuzzing, and the standalone `doctor` command
performs only them; each step is bounded by the shell's own deadlines. They check
setup; they do not establish correctness across the generated SQL surface.

## 6. Fallback must be visible during testing

Production Sirius can fall back to DuckDB when a plan or runtime operation cannot
be handled on the GPU. This is helpful to users, but it can hide defects from a
differential test. A failed GPU attempt followed by a correct CPU answer would
look like a successful comparison.

The harness sets:

```sql
SET enable_duckdb_fallback = false;
```

This makes rejection and runtime failure observable. It exposes two distinct
stages:

```text
Plan-time rejection:
  SQL → Sirius planning rejects the query
  production would execute the CPU plan

Runtime fallback:
  SQL → Sirius accepts the plan → GPU work begins → unsupported operation fails
  production would execute the stored CPU plan from scratch
```

The second case throws away the GPU work already performed. It is consequently
useful to find unsupported shapes accepted by the planner and move their checks
to planning time, even before implementing GPU support.

[classify.py](../../test/fuzz/siriusfuzz/classify.py) distinguishes plan rejections,
unsupported runtime operations, other GPU errors, internal errors, and memory
exhaustion. Classification uses error prefixes and message patterns; it does not
inspect an authoritative engine error enum. Changes to engine wording can
therefore require classifier changes.

## 7. Comparing answers requires SQL semantics

[compare.py](../../test/fuzz/siriusfuzz/compare.py) is a central correctness component.
A poor comparator could report legal differences as bugs or accept wrong results.

### Unordered results are multisets

SQL generally does not guarantee row order without an appropriate `ORDER BY`.
These two outputs are equivalent:

```text
CPU: (1), (2), (2)
GPU: (2), (1), (2)
```

They are compared as a **multiset**: an unordered collection that retains duplicate
counts. Comparing ordinary sets would incorrectly accept `(1), (2)` as equivalent,
losing evidence of a missing duplicate row. Duplicate counts matter especially
for joins and `UNION ALL`.

### An ORDER BY alone does not imply deterministic rows

Consider:

```sql
SELECT k, payload FROM t ORDER BY k;
```

Two rows with the same `k` can appear in either order if their payloads differ.
The AST ordering check conservatively selects ordered comparison only when
`ORDER BY` covers every projected output alias. Equal ties then have identical
projected rows and are indistinguishable.

The default generator also requires total ordering for `LIMIT`. Otherwise, a
limit can change which rows belong to the result, rather than merely reorder
them. This structural rule is conservative: the harness does not infer unique
keys to prove additional cases deterministic. Custom SQL replay defaults to
multiset comparison; `--ordered` explicitly requests ordered semantics.

### Floating-point values use tolerance; exact values remain exact

Parallel arithmetic can change the order of operations. Floating-point addition
is not associative: `(a + b) + c` can differ from `a + (b + c)` because intermediate
values are rounded. CPU and GPU aggregates may therefore differ slightly without
a correctness defect.

For finite floating-point values, equality accepts either condition:

```text
|a - b| ≤ relative_tolerance × max(|a|, |b|)
|a - b| ≤ absolute_tolerance
```

The absolute floor handles values close to zero. Defaults are `1e-4` for FLOAT,
`1e-9` for DOUBLE, and `1e-12` absolute tolerance. These are testing policy choices,
not universal guarantees about numerical accuracy. Loose tolerances can conceal
small defects; tight ones can report harmless arithmetic variation.

Non-floating values compare exactly, including DECIMAL. NULL matches only NULL.
The comparator canonicalizes NaNs into a matching special category and requires
infinities to match their exact value and sign. Non-finite values are not accepted
through the finite-value tolerance rule.

### Approximate row matching needs more than sorting

Approximate equality is not transitive. Under some tolerance, `a` can match `b`
and `b` can match `c` while `a` does not match `c`. Consequently, approximate
values cannot safely be collapsed into ordinary hash keys.

The comparator groups rows by their exact columns, retaining multiplicity. It
tries sorted row pairs as a fast path within each group. If those pairs do not
match, it searches for a one-to-one matching of rows using the floating-point
tolerances.

Conceptually, this is a bipartite graph: CPU rows are on one side, GPU rows on the
other, and an edge means two rows can match. An augmenting-path search can revise
earlier pairings. Greedy pairing alone can consume the only compatible partner
for a later row and incorrectly declare a mismatch.

This is more expensive for large groups with floating-point columns. Small,
bounded fuzz datasets make that cost practical.

### Unstable references are filtered

After a mismatch, the evaluator can rebuild the same dataset with a different
insertion order and rerun the query on the CPU. If the reference answer changes
under the applicable comparison mode, the observation becomes `ambiguous`.

This avoids treating input-order-dependent behavior as a straightforward GPU bug.
It is a heuristic: one permutation cannot prove determinism, and filtering an
unstable reference can also hide a real GPU defect. SQL-only replay lacks the
generated dataset object used by this filter, so a replay is evidence for
investigation rather than automatic confirmation.

The evaluator uses CPU-described logical column types to interpret GPU values.
The main comparison checks result dimensions and cell values; it is not an
independent check of CPU/GPU result metadata equality.

## 8. Settings provide a second correctness check

Once the CPU and baseline GPU answers match, the evaluator selects configured
Sirius settings, changes one setting, reruns the query, and compares the result
with the baseline GPU answer. Each setting is restored before the next variant.

This is **metamorphic testing**: change something that should preserve meaning,
then check that the answer stays the same.

```text
CPU reference == GPU baseline
GPU baseline == GPU with one selected setting changed
```

The shipped choices include expression evaluation strategies (`materialize`,
`ast_interpret`, `ast_jit`), hash partition sizes, build hash-table limits, and
sort partition limits. These settings can select different execution paths
without changing the SQL's intended result.

Changing one setting at a time makes a failure easier to attribute and replay.
The default requests two selected settings per query, with a sampled value for
each. It does not exhaustively test values or combinations, and a sampled value
may equal the baseline. Unavailable settings are logged and skipped. Gaps mode
skips these variants to focus its execution budget on unsupported query shapes.

## 9. The shell is the isolation boundary

Native C++ or CUDA faults terminate the process they run in without raising a
catchable exception, and a hung GPU call may ignore a normal query interruption.
The harness therefore never runs Sirius inside the Python process. Each session
drives the DuckDB shell built with Sirius (`build/release/duckdb`) over pipes, in
batch JSON mode. Every statement is followed by a sentinel on each stream, a
`SELECT` on stdout and an `error()` call on stderr, so results and error text are
delimited exactly. The shell's stderr, including Sirius's backtrace on a fault, is
kept with the worker's log.

A query that overruns its deadline gets the shell killed; the shell exits on an
interrupt when its input is a pipe, so interrupting would end it anyway. A GPU
fault ends the shell the same way. Either case is recorded as a `timeout` or a
`crash` with the shell's exit status and stderr tail, and the next statement
starts a fresh shell, re-attaches the dataset files and re-applies the session's
settings. Datasets are files on disk, so a restart costs a process start rather
than regenerated data.

This choice also removes a build step: the harness needs no DuckDB Python
package, and the shell and the extension cannot disagree about versions because
they come from the same build. JSON output loses some type information, so exact
types come from `DESCRIBE`; DECIMAL and HUGEINT values arrive as strings and are
compared exactly, and floating-point values arrive as numbers.

Campaign workers remain separate `spawn` processes so that several shells run in
parallel and a Python-level failure in one worker cannot stop the run. The
orchestrator tracks active queries, persists results, and respawns a worker whose
Python process dies or stalls, within a configured budget; with the shell
handling query faults, that path is a backstop. The active-operation file still
records SQL, phase, settings and stage before each operation, so a worker death
can be attributed to a query, and worker messages still use atomic JSON files in
an on-disk mailbox so a writer dying mid-message cannot corrupt the queue.

`doctor` and replay run in the calling process with the same session machinery,
each step bounded by the shell's deadlines. Campaign duration exhaustion is
different from a query timeout: deliberately stopping in-flight work at the
campaign boundary must not manufacture a hang finding.

## 10. Verdicts preserve different meanings

| Verdict family | What was observed | How to interpret it |
|---|---|---|
| `ok` | CPU/GPU match and selected variants match | A successful sampled comparison. |
| `cpu_error`, `cpu_timeout` | The reference could not supply an answer | Skipped and counted; no successful GPU result comparison. |
| `ambiguous` | The reference changes with insertion order | Semantics or reference stability need investigation. |
| `mismatch` | Successful CPU/GPU executions return different rows | A wrong-answer candidate. |
| `variant_mismatch` | A setting change alters a correct GPU baseline | A candidate defect in an alternate execution path. |
| `plan_fallback` | Sirius declines the plan | A support gap before GPU execution. |
| `runtime_fallback` | An execution error says an operation is unsupported | A support gap discovered after planning. |
| `gpu_error`, `gpu_internal_error`, `gpu_oom` | GPU-side execution fails | Inspect the error, inputs, and resources. |
| `timeout`, `crash` | The query exceeds a deadline or the shell dies | Inspect phase, stage, the shell's stderr, and the exit status. |
| `known_issue` | An observation matches a configured quarantine rule | Acknowledged evidence counted separately. |

[known_issues.toml](../../test/fuzz/known_issues.toml) associates quarantine rules
with issue links. The rules avoid repeatedly presenting acknowledged divergences
as new findings, but overly broad patterns can conceal unrelated observations.

Completion status and verdict totals answer different questions. A completed
campaign can contain findings. Exit zero alone does not mean zero discrepancies;
`--fail-on-findings` requests that behavior. An incomplete run retains useful
evidence but must not be presented as a completed campaign.

## 11. Gaps mode targets late support checks

`--mode gaps` broadens generation by enabling feature switches marked for the
mode, including constructs disabled in the default correctness profile. It still
uses the generator's implemented surface; it does not generate arbitrary SQL or
every possible function. `--features configured` keeps the configuration's own
flags instead, so every query reaches runtime and the budget goes to the
supported surface; `--features all` enables the same switches in a correctness
run. Explicit `--set` overrides apply afterwards.

The goal is to discover shapes that pass planning but fail at runtime. For example,
a query might contain an unsupported expression nested inside several supported
ones. A reducer should remove the surrounding expressions until the late support
check is exposed clearly.

Three policy choices follow from that goal:

1. Runtime fallbacks are reduced; plan-time gaps are not reduced in this mode.
2. A smaller runtime reproducer must remain a runtime fallback. A plan rejection
   cannot substitute for it.
3. Both gap verdicts are expected survey results and do not fail
   `--fail-on-findings`. Wrong answers and other defects still do.

For unsupported-expression reasons, the failure-preservation rule permits the
function set to shrink. Removing a supported wrapper around an unsupported
function should not prevent useful reduction merely because the printed reason
changes. The rejection kind must still match, and new unrelated functions cannot
replace the original ones.

Reports put runtime gaps first and sum elapsed GPU-attempt time by reason. This
is a relative prioritization signal for wasted execution on small fuzz inputs,
not a benchmark of production performance or pure device-kernel time. Queries
show only the first rejection reached, so later unsupported operations can remain
hidden until earlier checks change.

A reduced runtime fallback is useful as a regression reproducer: after its check
moves to planning time, replay should classify it as `plan_fallback`.

## 12. Reduction preserves the failure, not the original query's meaning

A random query may be too large to diagnose directly. Reduction asks whether
parts can be removed while the relevant failure still occurs.

An illustrative result could be:

```sql
-- Original shape: join, filters, several projections, grouping, ordering.
SELECT a.k, COUNT(DISTINCT b.v), SUM(b.amount)
FROM t0 AS a JOIN t1 AS b ON a.k = b.k
WHERE b.amount > 0
GROUP BY a.k
ORDER BY a.k;

-- A possible smaller reproducer, if it preserves the observed failure:
SELECT COUNT(DISTINCT v) FROM t1;
```

The smaller query need not produce the same answer as the original. It must
preserve the failure under its own valid CPU reference. The example does not
assert that these two queries actually share a failure.

### Campaign reduction uses the AST

[reduce.py](../../test/fuzz/siriusfuzz/reduce.py) proposes structural simplifications,
such as dropping clauses or projected items, simplifying expressions, removing
join branches, and shrinking set operations. A greedy search accepts an edit
when a failure predicate approves it, then searches again from the accepted tree.
The default budget is 150 candidate checks.

If an available SQLSmith extension supplies `reduce_sql_statement`, a subsequent
string-level pass can propose additional simplifications. Once a string candidate
is accepted, it no longer has a corresponding AST. String-only candidates use
multiset comparison.

The preservation checks depend on the finding:

- A mismatch requires CPU and GPU executions to succeed and still disagree,
  with comparison mode recomputed and the ambiguity filter applied.
- A variant mismatch also requires the baseline GPU result to keep matching CPU
  before the saved setting change causes a disagreement.
- An error requires CPU success and the same verdict and compatible reason.
  A mismatch cannot shrink into an unrelated crash and still count as success.

Reduction skips crash and timeout queries: every candidate could cost another
shell restart. Repeated error signatures are reduced once per worker to avoid
spending the budget again on an equivalent reported reason; in gaps mode plan
rejections are not reduced at all. The original observation is saved before
reduction starts.

The reducer finds a smaller reproducer within its search and budget. It does not
guarantee a global minimum. Reduced SQL remains additional evidence; it never
replaces the original query.

## 13. Reproducibility includes the environment

A seed helps reconstruct generated inputs, but it is not sufficient by itself.
Worker count, respawns, generator changes, binary builds, runtime versions,
settings, and machine configuration can all affect what was generated or observed.
The harness therefore saves actual SQL and data as well as seeds.

[report.py](../../test/fuzz/siriusfuzz/report.py) and
[artifacts.py](../../test/fuzz/siriusfuzz/artifacts.py) preserve, per finding:

- `FINDING.md` for the reader: verdict, reason, the smallest query that reproduces,
  the differing rows or error, and the replay command for that directory.
- Original query and dataset SQL, and the reduced query when reduction made progress.
- Effective TOML and the worker's Sirius YAML.
- `meta.json` for tools: the full record, the run's source revisions and binary
  fingerprints, and the worker's session settings.
- The shell's stderr for crashes and hangs.
- Further query/dataset pairs with the same signature, without repeating the rest.

If no Sirius YAML is selected explicitly, the harness discovers the one Sirius
itself would use (`SIRIUS_CONFIG_FILE`, then `./sirius.yaml`, then
`~/.sirius/sirius.yaml`), copies it into the run and into every finding, and
records where it came from. A replay on another host then restores the same
settings instead of silently picking up that host's.

The shell and the extension come from one build, so there is no version metadata
to reconcile. The run's provenance records the shell binary and its hash as well
as the source revision; the revision alone does not prove how the binary was built.

Replay restores the saved comparison mode, baseline settings, and exact failing
variant. It does not sample a replacement variant or invent missing data. Each
attempt writes separate evidence without modifying its source. A different GPU,
build, or runtime may still change the outcome; provenance makes that change visible.

## 14. Grouping is a convenience, not a root-cause conclusion

Repeated runtime errors often vary only in column names, numeric values, or
addresses. The report normalizes these details to group similar reasons and
keeps up to five additional complete reproducers per signature.

For mismatches, variant mismatches, and timeouts, grouping retains the inputs:
SQL, dataset, comparison mode, and variant. Grouping only by operator names would
risk discarding different defects that happen to use the same operator.

This is an intentional tradeoff: preserve distinct correctness evidence even if
the finding list becomes longer. Conversely, normalized error signatures can
combine different causes that print similar messages. A signature and a finding
count are therefore not counts of confirmed engine bugs. The gaps tables are the
one place where findings are merged after the fact, by the reason their reduced
query reports, because there the reduction has already identified the cause.

## 15. Recheck closes the loop

After a fix, `recheck` replays every finding directory of a run against the
current build and prints the recorded verdict next to the new one. A mismatch
that now reads `ok` was fixed; a runtime fallback that now reads `plan_fallback`
had its check moved to planning time; anything that changed to a different
failure deserves a look. The command exits non-zero while any finding still
reproduces, which makes it usable as a gate.

Replaying a single finding restores the saved configuration, YAML, baseline
settings, comparison mode and the exact failing setting variant, and writes a new
directory without touching the finding. Neither command confirms a bug by itself:
a replay is evidence for a person to read, and what sections 7 and 14 say about
what a match or a signature does and does not prove still applies.

## 16. What the tests establish

The [harness tests](../../test/fuzz/tests) cover the machinery that decides which
observations are meaningful. Examples include duplicate-aware and tolerant row
comparison, scope-aware generation, reduction predicates, precise gap
classification, the shell protocol with its timeouts, restarts and crashes,
worker death, interrupted mailbox writes, configuration restoration, and recheck
reporting. Tests that execute SQL use a DuckDB shell and are skipped when none is
available.

CPU-only `selftest` exercises generation, execution, comparison, and reporting
end to end by comparing repeated CPU runs. It can expose harness failures, but
it cannot validate GPU execution. GPU correctness requires a compatible host,
passing interception checks, and actual CPU/GPU comparisons.

Several limits follow directly from the design:

- Sampling covers the generator's implemented grammar, not all SQL.
- Feature emission does not prove GPU operator or source-code coverage.
- Small datasets favor quick diagnosis over exhaustive memory-pressure testing.
- Setting variants do not exhaustively cover interacting settings.
- Message-based classification and signature grouping remain heuristics.
- Reduction gives a smaller observed reproducer, not a proven minimum.
- A finite successful campaign cannot prove the absence of defects.

These limits are why the outputs preserve counts, skips, completion state,
provenance, and original inputs alongside reduced findings.

## 17. Source map for deeper reading

| Module | Responsibility |
|---|---|
| [cli.py](../../test/fuzz/siriusfuzz/cli.py) | Commands, configuration loading, paths, and command outcomes. |
| [config.py](../../test/fuzz/siriusfuzz/config.py) | Typed configuration, validation, overrides, and mode switches. |
| [sqltypes.py](../../test/fuzz/siriusfuzz/sqltypes.py) | SQL type descriptions and literal serialization. |
| [schema_gen.py](../../test/fuzz/siriusfuzz/schema_gen.py) | Schemas, data pools, NULLs, edge values, and dataset SQL. |
| [sqlast.py](../../test/fuzz/siriusfuzz/sqlast.py) | Query structure, rendering, traversal, and feature labels. |
| [query_gen.py](../../test/fuzz/siriusfuzz/query_gen.py) | Typed, scope-aware query construction. |
| [session.py](../../test/fuzz/siriusfuzz/session.py) | The DuckDB shell process, stored datasets, GPU switches, interception, settings, and operation evidence. |
| [compare.py](../../test/fuzz/siriusfuzz/compare.py) | Exact/tolerant cell equality, multiset matching, and ordering checks. |
| [classify.py](../../test/fuzz/siriusfuzz/classify.py) | Verdicts, message classification, and compatible gap reasons. |
| [runner.py](../../test/fuzz/siriusfuzz/runner.py) | Evaluation, variants, workers, the orchestrator backstop, and campaign budgets. |
| [reduce.py](../../test/fuzz/siriusfuzz/reduce.py) | AST reduction and optional SQLSmith candidates. |
| [report.py](../../test/fuzz/siriusfuzz/report.py) | Query logs, signatures, findings, `FINDING.md`, and summaries. |
| [artifacts.py](../../test/fuzz/siriusfuzz/artifacts.py) | Atomic JSON writes, provenance, and fingerprints. |
| [probe.py](../../test/fuzz/siriusfuzz/probe.py) | The doctor check and single-query replay, in the calling process. |

For an implementation walkthrough, start with `Evaluator.evaluate()` in
`runner.py`, follow its calls into `Session.run()` and `compare_results()`, then
read query generation and reduction. That order connects the testing contract to
the code responsible for producing and interpreting each observation.
