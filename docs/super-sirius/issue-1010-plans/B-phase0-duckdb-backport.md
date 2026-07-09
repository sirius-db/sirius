# PR B1 Implementation Plan — Phase 0: DuckDB #22963 backport + explicit-oracle regressions

> **Status (2026-07-09): deferred pin-bump playbook.** Track A is complete through merged #1134
> (`1eecaf97`). B1 blocks nothing in Track C. See the
> [status reconciliation](../issue-1010-implementation-plan.md#status-reconciliation-and-track-c-re-evaluation-2026-07-09).

Companion to [issue-1010-dynamic-filter-sip-design.md](../issue-1010-dynamic-filter-sip-design.md);
current `dev` baseline `fac81e87`; historical file:line anchors at `506a1d9f`.
PR IDs covered: **B1**.

> **STATUS: DEFERRED (decision 2026-07-08).** No Sirius-owned fork or backport will be created.
> The latent exposure is accepted; the fix arrives at the next pin bump to a released DuckDB
> containing duckdb/duckdb#22963, at which point this document is the playbook: skip the fork
> sections (§ fork/gitlink mechanics become a plain pin advance) and ship the sentinel
> regressions and metadata assertions as specified. Until then, nothing here blocks Track A or C;
> the standing rule that survives deferral: LIMIT/TOP-N-shaped tests use explicit expected rows
> or a filters-disabled reference run — never an unpatched CPU run. Tracked in
> sirius-db/sirius#1123.

> **Empirical status (adversarial review, executed on the unpatched pin):** at default optimizer
> settings the pinned DuckDB returns **correct** results for all five query shapes below — the bug
> is *latent*, masked by `late_materialization`, `compressed_materialization`, and pipeline-timing
> races. The deterministic red→green evidence for this backport is the **logical-metadata
> assertion** (`probe_info` non-empty on the unpatched pin, empty on the patched pin) with
> `late_materialization` disabled; the only *runtime* CPU red sentinel is Q-S3 with
> `late_materialization,compressed_materialization` disabled, and even that is schedule-dependent.
> Every section below is structured around that reality. This plan involves no timing/perf
> measurement passes; the one CI perf-adjacent check (TPC-H snapshot) runs at the CI default INFO
> log level with no new DEBUG/TRACE logging introduced.

## 1. GOAL + NON-GOALS

At the next qualifying pin bump, make the DuckDB candidate source correct by adopting the released
version containing upstream `444f473a12` ("Avoid join filter pushdown below limits", the
single-parent change commit behind merge `4c8c90db44`) and land regressions with explicit oracles —
upstream's two SQL cases kept as pin-upgrade sentinels, Sirius variants with join-sourced
LIMIT/TOP-N inputs, and per-plan assertions that no outer producer crosses a row selector while a
join wholly inside the selector input keeps its route (design
`docs/super-sirius/issue-1010-dynamic-filter-sip-design.md:224-246`, historical phasing row B1).
The former claim that B1 gated C1/C3 is superseded: B1 blocks no Track C merge, experiment, or
enablement decision.

Non-goals: no changes to the Sirius placement walk or any `src/` production code ("Do not
reimplement this target walk", design `:244-246`); no new config flags; no A1 telemetry IDs (plan
assertions are keyed by plan position now, re-keyed by `dynamic_filter_publication_plan_id` when A1
lands — query-relative monotonic per the design's "Publication, target, channel, and filter
identity" section); no SIP-on test config yet (C3 appends it); no pin advance to v1.5.5+/v1.6. On
the "no released tag contains the fix" claim: `git tag --contains 4c8c90db44` is empty in the
submodule store, but that store tops out at `v1.5.4` while the fix merged upstream 2026-06-04 — the
claim is plausible but **must be re-verified against fresh upstream refs before it is written into
the PR** (`git -C duckdb fetch https://github.com/duckdb/duckdb --tags && git -C duckdb tag
--contains 444f473a12`). If a released tag does contain it, evaluate a pin advance instead of the
fork before proceeding.

## 2. DELIVERABLES

**Repo/infra (no Sirius C++ API changes; zero `src/` diff):**
- Public fork `sirius-db/duckdb` (does not exist today; verified `gh repo view sirius-db/duckdb` →
  resolution error), branch `sirius/v1.5.4-backports` =
  `08e34c447bae34eaee3723cac61f2878b6bdf787` (== tag `v1.5.4` == current gitlink) +
  `git cherry-pick -x 444f473a12` (verified clean via `merge-tree`, +33/−2, 2 files). Tag the head
  `v1.5.4-sirius.1` for reachability independent of the branch ref.
- `.gitmodules` duckdb entry repointed; gitlink bumped to the new SHA.
- GitHub issue documenting the merged-Phase-1 **latent** exposure (filed before the PR; PR closes
  it) — honest text in §3.8.
- Small follow-up issue (or in-PR code comment) for the pre-existing uninitialized-read hazard in
  §3.9.

**Test-only C++ (anonymous namespaces in the two new test files):**
```cpp
// test/cpp/planner/test_join_filter_pushdown_limit_plan.cpp
//
// Transaction ownership: the CALLER opens one transaction (con.Query("BEGIN TRANSACTION"))
// before build_optimized_logical_plan, keeps it open across the logical-plan walk and
// to_sirius_plan, and commits/rolls back itself. Helpers never BEGIN/COMMIT/ROLLBACK —
// Parser→Optimizer→create_plan and every interleaved inspection must share one open
// transaction (same discipline as generate_sirius_plan,
// test/cpp/planner/test_distinct_hash_join_detection.cpp:53-92).
duckdb::unique_ptr<duckdb::LogicalOperator> build_optimized_logical_plan(
  duckdb::Connection& con, const std::string& query);                 // Parser→Planner→Optimizer→ResolveOperatorTypes→ColumnBindingResolver
void collect_logical_comparison_joins(duckdb::LogicalOperator& root,
  std::vector<duckdb::LogicalComparisonJoin*>& out);                   // pre-order
bool join_build_side_is_table(duckdb::LogicalComparisonJoin& join,
  const std::string& table_name);                                      // RHS-binding check: every condition's right expression resolves
                                                                       // into a binding produced by children[1], and children[1]'s subtree
                                                                       // contains a LogicalGet on `table_name` owning that table_index
duckdb::unique_ptr<sirius::op::sirius_physical_operator> to_sirius_plan(
  sirius::planner::sirius_physical_plan_generator& gen,
  duckdb::unique_ptr<duckdb::LogicalOperator> plan);                   // gen passed in so gen.dynamic_filter_channels stays inspectable
void collect_hash_joins(sirius::op::sirius_physical_operator* root,
  std::vector<sirius::op::sirius_physical_hash_join*>& out);           // pre-order (outer join first)
bool subtree_contains(sirius::op::sirius_physical_operator const& op,
  sirius::op::SiriusPhysicalOperatorType type);                        // identify the join above the LIMIT/TOP_N (shape guard only —
                                                                       // NOT sufficient alone, see Risk 4; pair with join_build_side_is_table)

// test/cpp/integration/test_gpu_execution_join_filter_limit.cpp
struct pushdown_setting_guard {                                        // RAII: capture current enable_dynamic_filter_pushdown from the
  explicit pushdown_setting_guard(duckdb::Connection& con);            // shared process-global SiriusContext operator_params, restore the
  ~pushdown_setting_guard();                                           // CAPTURED value (not hardcoded true) in the destructor
private:
  duckdb::Connection& con_;
  bool prior_value_;
};
struct disabled_optimizers_guard {                                     // RAII: capture disabled_optimizers, SET the requested list,
  disabled_optimizers_guard(duckdb::Connection& con, const std::string& list);
  ~disabled_optimizers_guard();                                        // restore captured prior set
};
void require_rows(sirius::test::GpuExecutionFixture& fx, const std::string& query,
  std::vector<std::vector<std::string>> expected_sorted);              // uses GpuExecutionFixture::collect_rows
```
No guard for `gpu_execution`: `SetEnableGpuExecution` is a log-only no-op on shared state
(`src/sirius_extension.cpp:1599-1602`); the option value itself is per-session DuckDB option state
that dies with the fixture connection.

**New test assets:** `test/sql/join_filter_pushdown_limit.test` (manual sqllogic sentinel), the two
Catch2 files above, two `TEST_SOURCES` lines.

## 3. STEP-BY-STEP CHANGES

### 3.1 Fork + backport branch (outside the superproject; do FIRST — push order is fork-then-superproject)
```bash
gh repo fork duckdb/duckdb --org sirius-db --clone=false      # must be PUBLIC (CI checkout uses default token)
git -C duckdb worktree add --detach /tmp/ddb-b1 08e34c447bae34eaee3723cac61f2878b6bdf787
cd /tmp/ddb-b1
git switch -c sirius/v1.5.4-backports
git cherry-pick -x 444f473a12                                  # clean; -x records the upstream SHA
git tag v1.5.4-sirius.1
git remote add sirius https://github.com/sirius-db/duckdb.git
git push sirius sirius/v1.5.4-backports v1.5.4-sirius.1
NEW_SHA=$(git rev-parse HEAD)
cd - && git -C duckdb worktree remove --force /tmp/ddb-b1
```
If `gh repo fork --org` lacks permission, fallback: create empty repo `sirius-db/duckdb`,
`git -C duckdb push <url> 08e34c447b:refs/heads/sirius/v1.5.4-backports`, then push the cherry-pick
commit. Protect the branch (no force-push/delete). The cherry-pick touches only
`src/optimizer/join_filter_pushdown_optimizer.cpp` (splits `LOGICAL_LIMIT`/`LOGICAL_TOP_N` out of
the transparent fall-through group at pinned lines 63-69 into a terminal `break`) and
`test/sql/join/inner/test_inner_join_filter_pushdown.test` (+30). No headers, no ABI, no
serialization.

### 3.2 `.gitmodules` (repo root, entry at `.gitmodules:1-5`: currently `url = https://github.com/duckdb/duckdb`, `branch = main`)
```
[submodule "duckdb"]
	path = duckdb
	url = https://github.com/sirius-db/duckdb
	branch = sirius/v1.5.4-backports
	ignore = dirty
```
Then `git submodule sync duckdb`. Precedent for org-fork submodules: `substrait` →
`sirius-db/duckdb-substrait-extension` (`.gitmodules:6-9`).

### 3.3 Gitlink bump
```bash
git -C duckdb fetch https://github.com/sirius-db/duckdb sirius/v1.5.4-backports
git -C duckdb checkout $NEW_SHA
git add .gitmodules duckdb
```
Consumers that follow the gitlink automatically (verified, no edits needed): `Makefile:16`
(`DUCKDB_DIR ?= duckdb`; `set_duckdb_version` stubbed at `Makefile:90-91`), CI checkouts with
`submodules: recursive` at `test.yml:57-58`, `check.yml:32-33`, `check.yml:107-109`, Python API via
`-DDUCKDB_SOURCE_PATH` (`pixi.toml:115`). **Leave unchanged:** `distribution.yml:33` and `:57`
`duckdb_version: v1.5.4` (label into extension-ci-tools only; extension ABI still v1.5.4).
`experimental.yml` inits only starrocks submodules (`experimental.yml:42-46`) — unaffected. sccache
is content-hash keyed (`test.yml:66-76`, `check.yml:124-137`) — one TU recompile.

### 3.4 `test/sql/join_filter_pushdown_limit.test` (new; manual-only — CI runs no sqllogic, grep of `.github/workflows/` for `test/sql` is empty)
Header per repo convention (`test/sql/bugfix.test:15-20`): `# name:` / `# description:` /
`# group: [sirius]` / `require sirius`. Add a header comment block documenting **runner environment
sensitivity**: the sirius-linked binary reads `SIRIUS_CONFIG_FILE` / `~/.sirius/sirius.yaml` at
startup and a stale user yaml aborts the process before any SQL runs (observed:
`unknown config key: 'duckdb_scan'` from a stale `~/.sirius/sirius.yaml`); document the required
invocation (`SIRIUS_CONFIG_FILE=test/cpp/config/data/minimal.yaml`) and **smoke-run the file once
before landing** — there is no existing transparent-plain-SQL sqllogic precedent (existing files
use `call gpu_buffer_init` + legacy `call gpu_processing`, `test/sql/tpch-sirius.test:151,155`,
`test/sql/bugfix.test:24`), so the "no `gpu_buffer_init` needed" assumption must be validated by
that smoke run. Use transparent plain SQL; `SET` via plain `statement ok` (options registered at
`src/sirius_extension.cpp:1834-1848`; `require sirius` loads the extension first so the "SET before
context" gotcha does not bite).

Sections, each running the full query set against **hardcoded expected rows** (the oracle — design
`:238-239`). Every bug-exercising section wraps its queries in
`SET disabled_optimizers = 'late_materialization,compressed_materialization';` … restore with
`SET disabled_optimizers = '';` afterwards — at defaults those two optimizers mask the bug
(late materialization rewrites LIMIT/TOP-N-over-scan into `TOP_N(rowid) → SEMI join → full scan`,
making the filter target the rowid-preserving base scan, a legal route; compressed materialization
inserts `__internal_compress_integral_*` projections that fail `PushdownJoinFilterExpression` for
these small-int columns):
1. `SET gpu_execution = false;` — **patched-pin CPU regression at defaults.** Documented in-file:
   this section does NOT fail on the unpatched pin (masking above); it guards the patched pin.
2. `SET gpu_execution = false; SET disabled_optimizers = 'late_materialization,compressed_materialization';`
   — the bug-exercising CPU leg. **Q-S3 here is the single runtime CPU red sentinel** (on the
   unpatched pin it returned 2 rows incl. spurious `0,5,50,0`; its fact-probe pipeline depends on
   the inner `side` build, so the dim filter is populated before the scan — but this is
   schedule-dependent, not guaranteed; documented as such in-file). Q-up1/Q-up2/Q-S1/Q-S2 in this
   section are patched-pin regressions: on the unpatched pin they still returned correct rows due
   to the pipeline-timing race (the `fact→TOP_N` pipeline has no dependency on the dim build and
   scans before the min/max filter finalizes). Restore `disabled_optimizers` after.
3. `SET gpu_execution = false; SET disabled_optimizers = 'join_filter_pushdown';` — the design's
   "join-filter-pushdown-disabled execution" oracle cross-check (optimizer name at pinned
   `duckdb/src/common/enums/optimizer_type.cpp:40`); restore after.
4. `SET gpu_execution = true; SET enable_dynamic_filter_pushdown = false;` with the two masking
   optimizers disabled — GPU filters-off; restore after.
5. `SET gpu_execution = true; SET enable_dynamic_filter_pushdown = true;` with the two masking
   optimizers disabled — Phase 1 on; restore after.
6. Trailing comment block reserving a SIP-on section for C3 (`SET enable_dynamic_filter_sip = true`
   once it exists).

**Query set** (all `ORDER BY` on a unique key so explicit rows are deterministic):

Upstream sentinels, verbatim from `444f473a12` (kept as pin-upgrade sentinels; note their `WITH`
form is fine here because they are upstream-verbatim — Sirius variants use plain subqueries, §risk 3):
```sql
CREATE TABLE ordered_probe(flag BOOLEAN, ord INTEGER);
INSERT INTO ordered_probe VALUES (true, 1), (true, 2), (false, 3);
CREATE TABLE boolean_keys(flag_key BOOLEAN);
INSERT INTO boolean_keys VALUES (false);
-- Q-up1: WITH c AS (SELECT flag FROM ordered_probe ORDER BY ord OFFSET 2)
--        SELECT * FROM c INNER JOIN boolean_keys ON flag = flag_key;   → false, false
-- Q-up2: same with ORDER BY ord LIMIT 1                                 → empty
```

Sirius variants. Build sides carry a WHERE so `build_side_has_filter` is true and the Sirius wiring
gate at `src/planner/sirius_plan_comparison_join.cpp:426` passes. Sizing: **do not rely on the
"probe ≫ build" idiom** — `LIMIT 4` caps the selector-side cardinality estimate at ~4 rows, which
is comparable to a filtered 5-row dim, so the borrowed idiom from
`test/cpp/planner/test_distinct_hash_join_detection.cpp:141-153` does not apply here. Instead
shrink `dim` to 3 rows so the filtered build (~2 rows) is decisively smaller than the ~4-row
selector estimate, and assert build-side identity explicitly via `join_build_side_is_table(join,
"dim")` (RHS-binding check), not just subtree containment:
```sql
CREATE TABLE fact(k INTEGER, ord INTEGER);   -- 20 rows: ord = 1..20 unique, k = ord % 5
CREATE TABLE dim(dk INTEGER, tag VARCHAR);   -- 3 rows: (0,'keep'),(1,'drop'),(2,'keep')
CREATE TABLE side(ord INTEGER, s INTEGER);   -- 20 rows: ord = 1..20, s = ord * 10
```
- **Q-S1 (TOP-N over probe spine, LOGICAL_TOP_N):**
  `SELECT c.k, c.ord, d.dk FROM (SELECT k, ord FROM fact ORDER BY ord LIMIT 4) c JOIN dim d ON c.k = d.dk WHERE d.tag = 'keep' ORDER BY c.ord;`
  → expected exactly `2 2 2` (1 row). **Unpatched-pin behavior (measured):** correct at defaults
  AND with the two masking optimizers disabled (timing race) — a patched-pin regression, not a red
  sentinel.
- **Q-S2 (LIMIT/OFFSET, LOGICAL_LIMIT):**
  same shape with `(SELECT k, ord FROM fact ORDER BY ord OFFSET 16) c` → expected `2 17 2` and
  `0 20 0`. **Unpatched-pin behavior (measured):** correct, same masking — patched-pin regression.
- **Q-S3 (join-sourced selector input; inner join wholly inside the selector legitimately keeps its route):**
  ```sql
  SELECT c.k, c.ord, c.s, d.dk
  FROM (SELECT f.k, f.ord, s.s FROM fact f JOIN side s ON f.ord = s.ord
        WHERE s.s <= 60 ORDER BY f.ord LIMIT 4) c
  JOIN dim d ON c.k = d.dk WHERE d.tag = 'keep' ORDER BY c.ord;
  ```
  → expected exactly `2 2 20 2`. The inner `fact⋈side` (build = `side` filtered to 6 rows) is a
  legal Phase 1 producer to the `fact` scan (no selector between them); the outer join must not
  route below the TOP-N. **Unpatched-pin behavior (measured, optimizers-disabled leg):** 2 rows —
  spurious `0 5 50 0` from `k IN (0, 2) AND k>=0 AND k<=2` populated on the fact scan below the
  TOP_N (EXPLAIN ANALYZE-verified); the dependency on the inner build makes the wrongness
  observable, though still schedule-dependent.
  Use plain subqueries, not `WITH`, to avoid CTE-materialization ambiguity (materialized CTEs stop
  the walk and would mask the bug; design `:508`).

**Mandatory pre-landing step:** re-derive every expected-row prediction in this file (and §3.6)
**empirically** on the patched pin, per leg (CPU and GPU), before landing — the review demonstrated
that unexecuted predictions in this area are unreliable. The shrunken 3-row dim is expected to
leave all correct-result oracles unchanged (`2 2 2`; `2 17 2`+`0 20 0`; `2 2 20 2` — the removed
dk 3/4 rows were all `tag='drop'`), but this must be confirmed by execution, not derivation.

### 3.5 `test/cpp/planner/test_join_filter_pushdown_limit_plan.cpp` (new — the CI-enforced pin sentinel; THE deterministic red→green gate)
Fixture: clone of `distinct_hash_join_fixture`
(`test/cpp/planner/test_distinct_hash_join_detection.cpp:110-160`):
`SIRIUS_CONFIG_FILE=test/cpp/config/data/minimal.yaml` before `DuckDB db(nullptr)`,
`SIRIUS_DISABLE=1` after, tables from §3.4 created up front. Helpers split from
`generate_sirius_plan` (`:42-94`) into `build_optimized_logical_plan` + `to_sirius_plan`, with the
transaction owned by the caller as specified in §2. The harness disables **three** optimizers
(vs. the donor's two at `:50-51`):
```cpp
disabled.insert(OptimizerType::IN_CLAUSE);
disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
disabled.insert(OptimizerType::LATE_MATERIALIZATION);   // REQUIRED: without this the patched pin
    // rewrites Q-S1/Q-S2 into TOP_N(rowid) → SEMI join → full scan; the (patched) walk then
    // legitimately reaches the rowid-side LOGICAL_GET, probe_info is non-empty, and the
    // metadata assertion FAILS GREEN. Verified: with late-mat + compressed-mat disabled the
    // intended shape appears (TOP_N as probe child, dim as build); unpatched → probe_info
    // non-empty (red), patched → empty (green).
```
Do NOT disable `join_filter_pushdown`. Assertions run at **two levels**:

**(a) Logical-level DuckDB-metadata absence — never skipped, and the red→green gate proper**
(immune to the plan-gen `InternalException` WARN-skip precedent at `:168-171,80-84`): walk the
optimized logical plan; for each `duckdb::LogicalComparisonJoin` inspect `filter_pushdown` (pinned
member `duckdb/src/include/duckdb/planner/operator/logical_comparison_join.hpp:40`; `probe_info` at
`duckdb/src/include/duckdb/execution/operator/join/join_filter_pushdown.hpp:66`):
- Shape guards first, so drift fails loudly: `REQUIRE` on join count, `REQUIRE(join_build_side_is_table(outer, "dim"))`
  (RHS-binding check), and `subtree_contains(probe, TOP_N/LIMIT)` as a secondary check.
- Q-S1/Q-S2 outer join: `REQUIRE(!fp || fp->probe_info.empty());` — **fails on an unpatched pin,
  passes on the patched pin** (verified by execution with the three-optimizer disable set). The
  `!fp || empty()` form deliberately tolerates the patched pin's `compute_aggregates_anyway` path,
  which keeps `filter_pushdown` alive with empty `probe_info`.
- Q-S3: `REQUIRE(joins.size() == 2);` outer `probe_info` empty; inner
  `fp && !fp->probe_info.empty()`.

**(b) Sirius plan level** (skip-tolerant, same WARN pattern): construct
`sirius::planner::sirius_physical_plan_generator gen(*con->context)` in the test (router style,
`test/cpp/planner/test_dynamic_filter_router.cpp:69-78`), call `to_sirius_plan` inside the same
open transaction, then:
- outer join (identified as the pre-order hash join whose probe subtree contains
  `SiriusPhysicalOperatorType::TOP_N`/`LIMIT` via `subtree_contains`, with the logical-level
  build-side identity already asserted in (a); tree-walk modeled on `find_hash_join`,
  `test_distinct_hash_join_detection.cpp:97-108`):
  `REQUIRE_FALSE(hj->publishes_dynamic_filters())` (`src/include/op/sirius_physical_hash_join.hpp:166-170`)
  and `REQUIRE(!hj->filter_pushdown || hj->filter_pushdown->probe_info.empty())`
  (`src/include/op/sirius_physical_hash_join.hpp:100`).
- Q-S1/Q-S2: `REQUIRE(gen.dynamic_filter_channels.empty())`
  (`src/include/planner/sirius_physical_plan_generator.hpp:82-85`).
- Q-S3: inner join `REQUIRE(hj_inner->publishes_dynamic_filters())`;
  `REQUIRE(gen.dynamic_filter_channels.size() == 1)`. Wiring preconditions all hold:
  `build_side_has_filter` true (WHERE on `side`; gate at
  `src/planner/sirius_plan_comparison_join.cpp:426`), GPU+HOST spaces exist under minimal.yaml
  (gate at `:432-441`; targets built from `probe_info` at `:442-461`).
- Comment each assertion block: "keyed by plan position pending A1
  dynamic_filter_publication_plan_id" (design `:1018-1019`).

Named cases (tag `[dynamic_filter][limit_selector][isolated_context]`):
`"limit_selector - TOP-N blocks outer producer target (duckdb metadata)"`,
`"... (sirius wiring + channels)"`,
`"limit_selector - LIMIT/OFFSET blocks outer producer target (duckdb metadata)"`,
`"... (sirius wiring + channels)"`,
`"limit_selector - join inside selector input keeps its route"`,
`"limit_selector - upstream #22963 shapes produce no probe_info"` (Q-up1/Q-up2, logical level only;
these also need the late-mat disable — the rewrite masks upstream's own reproducers).

### 3.6 `test/cpp/integration/test_gpu_execution_join_filter_limit.cpp` (new — end-to-end explicit-rows regression, GPU)
Uses `GpuExecutionFixture` (`test/cpp/utils/gpu_execution_fixture.hpp:66-99`; on-disk ATTACH for
the native scan, `CHECKPOINT` after loading per header comment `:25-29`). Tag
`[integration][gpu_execution][dynamic_filter][limit_selector]` — `[integration]` binds the shared
env via the listener (`test/cpp/unittest.cpp:52-101`). **Every leg is bug-exercising and therefore
runs under a `disabled_optimizers_guard(con, "late_materialization,compressed_materialization")`**
— without it, late materialization rewrites Q-up/Q-S1/Q-S2 before Sirius sees them, so (a) the
Phase-1-on leg never exercises the filter-below-selector interaction (target becomes the rowid
scan, a legal route), and (b) the transparent path must plan a rowid-projecting scan + SEMI join,
which — if Sirius plan-gen rejects it — trips the zero-fallback delta assertion even on the patched
pin. The guard captures and restores the prior `disabled_optimizers` set (destructor-safe on
REQUIRE failure).

Per query shape (Q-up1, Q-up2, Q-S1, Q-S2, Q-S3), run the design's `:240` matrix against
**hardcoded expected rows** via `require_rows` (sorted-stringified compare reusing `collect_rows`,
`gpu_execution_fixture.hpp:129-142`):
1. `SET gpu_execution = false;` — patched CPU (regression, not a red sentinel — see §3.4).
2. `SET gpu_execution = true;` + `SET enable_dynamic_filter_pushdown = false;` — GPU filters-off,
   with `require_transparent_execution_delta(before, after, 1, 0, 1)` proving GPU ran with zero
   fallback (`test/cpp/utils/transparent_execution_test_utils.hpp:40-51`).
3. `SET enable_dynamic_filter_pushdown = true;` — Phase 1 on, same delta assertion.
4. SIP-on leg left as a comment for C3.

These legs are **patched-pin regressions**: on the unpatched pin, GPU runtime wrongness is racy —
transitive scan targets (a scan below TOP_N is transitive) "may observe no filter, a subset … or
the full set" (`docs/super-sirius/dynamic-filters.md:238`); only the immediate probe scan is
synchronized (`dynamic-filters.md:162,283`). Do not claim they fire on the unpatched pin.

**Mandatory pre-landing step:** empirically confirm zero-fallback (`1, 0, 1` deltas) for every
shape × leg on the patched pin before landing; if a leg falls back, restructure the query rather
than weakening the delta.

`pushdown_setting_guard` captures the current `enable_dynamic_filter_pushdown` value and restores
**the captured value** in its destructor — the setting mutates the process-global SiriusContext's
`operator_params` shared by every connection (verified: `src/sirius_extension.cpp:1604-1611` via
`get_operator_params` → registered `"sirius_state"`, `src/sirius_extension.cpp:1480-1488`; default
true at `src/include/sirius_config.hpp:102`), so leakage would poison later `[integration]` tests.
Never call `compare_gpu_vs_cpu` as the oracle (redundant with leg 1 on the patched pin, and an
unpatched CPU is not an oracle).

### 3.7 `CMakeLists.txt`
Add to `TEST_SOURCES` (`CMakeLists.txt:562` block, alphabetical-by-directory):
`test/cpp/integration/test_gpu_execution_join_filter_limit.cpp` next to the existing
`test/cpp/integration/test_gpu_execution_tpch.cpp` entry, and
`test/cpp/planner/test_join_filter_pushdown_limit_plan.cpp` next to
`test/cpp/planner/test_dynamic_filter_router.cpp`. Omission is a hard pre-commit failure
(`scripts/check_orphan_tests.py` hook, `.pre-commit-config.yaml:81-86`).

### 3.8 GitHub issue (file before the PR; PR body: "Fixes #NNN")
Title: *"Latent wrong-results bug: Phase 1 dynamic table-filter pushdown can apply build-derived
filters below LIMIT/TOP-N (upstream duckdb#22963)"*. Body — **honest exposure statement**:
- Root cause: pinned v1.5.4 `GetPushdownFilterTargets` treats `LOGICAL_LIMIT`/`LOGICAL_TOP_N` as
  transparent (pinned `duckdb/src/optimizer/join_filter_pushdown_optimizer.cpp:63-69`); Sirius
  consumes `probe_info` verbatim into producer targets
  (`src/planner/sirius_plan_comparison_join.cpp:442-461`) and applies filters post-decode at the
  scan (`src/op/scan/dynamic_filter_merge.cpp`), changing the row selector's input. Affects merged
  PR #794 independently of SIP (design `:231`).
- **Exposure is latent, not default-reproducible**: at default settings the bug is masked by
  (1) `late_materialization`, which rewrites LIMIT/TOP-N-over-scan so the filter target is the
  rowid-preserving base scan (a legal route); (2) `compressed_materialization`, whose compress
  projections defeat `PushdownJoinFilterExpression` — partly an artifact of small-int test columns,
  so NOT a safety guarantee for wide/real schemas; and (3) pipeline-timing races that let
  independent probe pipelines scan before the filter finalizes. It is reproducible
  **deterministically at the logical-metadata level** (non-empty `probe_info` on a join whose probe
  subtree contains a selector, with late materialization disabled) and **at runtime** in
  dependency-ordered shapes (Q-S3) with `late_materialization,compressed_materialization` disabled
  (schedule-dependent, EXPLAIN ANALYZE shows `k IN (0, 2) AND k>=0 AND k<=2` on the scan below the
  TOP_N; observed spurious row `0,5,50,0`).
- Still a correctness bug worth the backport: the masking mechanisms are optimizer heuristics and
  scheduling accidents, none of which Sirius controls or the design permits relying on.
- Reproducers: Q-S1..Q-S3 with per-leg optimizer settings and measured unpatched behavior as in
  §3.4 (expected-vs-actual only where actually wrong). Fix = pin backport of `444f473a12`. Labels:
  bug, correctness.

### 3.9 Follow-up note (small issue or in-PR comment; no code change in B1)
Pre-existing UB the new tests will tickle on the patched pin: for Q-S1/Q-S2 the
`compute_aggregates_anyway` path leaves `JoinFilterPushdownInfo::build_side_has_filter`
uninitialized (pinned `join_filter_pushdown.hpp:71` — no default initializer; only assigned when
`probe_info` is non-empty in the optimizer's `GenerateJoinFilters` tail), and Sirius reads it at
`src/planner/sirius_plan_comparison_join.cpp:426`. The B1 assertions stay robust either way (empty
`probe_info` → zero targets regardless of the read value), but record it: candidate fixes are a
`= false` default in a future backport branch commit or a defensive
`filter_pushdown->probe_info.empty()` short-circuit on the Sirius side (out of B1 scope).

## 4. TESTS (summary)

| File | Cases | GPU needed |
|---|---|---|
| `test/sql/join_filter_pushdown_limit.test` (new) | Q-up1/2 + Q-S1/2/3 × {CPU defaults (regression), CPU optimizers-disabled (Q-S3 = runtime red sentinel, schedule-dependent), CPU jfp-disabled oracle, GPU filters-off, Phase 1 on — the last two with masking optimizers disabled} | manual only; runs anywhere (GPU legs silently degrade to fallback if unsupported — acceptable, CI coverage is the Catch2 files). Smoke-run before landing with `SIRIUS_CONFIG_FILE` set and no stale `~/.sirius/sirius.yaml`. |
| `test/cpp/planner/test_join_filter_pushdown_limit_plan.cpp` | 6 cases in §3.5; the logical-metadata cases are the deterministic red→green gate | yes (SiriusContext bring-up + minimal.yaml spaces; all Catch2 CI is GPU — `test.yml:115,133-136`; no CPU job *runs* `sirius_unittest`, `check.yml:139,167` build only) |
| `test/cpp/integration/test_gpu_execution_join_filter_limit.cpp` | 5 shapes × 3 config legs (each under `disabled_optimizers_guard`), explicit rows + transparent-delta proof; patched-pin regressions | yes |
| Fork branch carries upstream `test/sql/join/inner/test_inner_join_filter_pushdown.test` | upstream regression, runnable manually: `(cd duckdb && cmake --build --preset release --target unittest)` then `build/release/test/unittest test/sql/join/inner/test_inner_join_filter_pushdown.test` from `duckdb/` | no |

## 5. GATE & ROLLBACK

**Merge gate (all required):**
1. **Red→green demonstration** (attach transcripts to PR), two tiers:
   - **Deterministic (the gate proper):** with the old gitlink (`08e34c447b`), the §3.5
     logical-metadata cases FAIL (outer-join `probe_info` non-empty for Q-S1/Q-S2 under the
     three-optimizer disable set); with `$NEW_SHA` they pass. This is what makes the sentinels
     real. (Empirically validated direction by the review: unpatched → non-empty, patched → empty.)
   - **Runtime (evidence, not a determinism requirement):** Q-S3 on the old gitlink, CPU, with
     `disabled_optimizers='late_materialization,compressed_materialization'`, observed wrong
     (spurious `0,5,50,0`); record the transcript and note the schedule dependence. Do NOT gate on
     Q-S1/Q-S2/Q-up runtime failures — measured correct on the unpatched pin at all settings tried.
2. **Mandatory pre-landing empirical steps:** (a) re-derive every expected-row prediction on the
   patched pin, per leg (§3.4/§3.6); (b) confirm zero-fallback `1,0,1` deltas for all integration
   legs; (c) smoke-run the sqllogic file (env note §3.4); (d) re-fetch upstream tags and re-verify
   `git tag --contains 444f473a12` is still empty of released tags (§1).
3. Verification checklist:
   - `git clone --recurse-submodules` of the PR branch into a scratch dir resolves `$NEW_SHA` via
     the fork URL (fork public, branch+tag pushed first).
   - `git -C duckdb log -1` shows the `-x` line `(cherry picked from commit 444f473a12...)`;
     `git -C duckdb merge-base --is-ancestor 08e34c447b HEAD` succeeds.
   - `pixi run make clean && pixi run make test` green (whole `sirius_unittest`);
     `pixi run pre-commit run -a` green (orphan hook, formatting).
   - Metadata-absence assertions green: outer `probe_info` empty, `gen.dynamic_filter_channels`
     empty for Q-S1/Q-S2, inner join of Q-S3 still wired.
   - CI `test-run` TPC-H snapshot (`test.yml:133-176`) unchanged — TPC-H LIMITs sit at plan roots
     above all joins, so the optimizer change is expected not to alter any TPC-H plan (note:
     late-materialization rewrites make this an expectation, not a proof); treat any
     perf/validation delta as a stop. This check runs at CI default INFO logging; no DEBUG/TRACE
     passes are involved in B1.
   - Optional smoke: `pixi run -e duckdb-python build-duckdb-python` (consumes the patched tree via
     `pixi.toml:115`).
4. **Flag defaults:** none added or changed (`enable_dynamic_filter_pushdown` stays default-true,
   `src/include/sirius_config.hpp:102`). This is a correctness fix, not an A/B.

**Rollback:** single `git revert` of the superproject commit restores the upstream URL and old
gitlink (`08e34c447b` remains fetchable from `duckdb/duckdb` as `v1.5.4`) and removes the tests
atomically — necessary, since the §3.5 metadata sentinels fail by design on the unpatched pin. Keep
the fork branch/tag immutable regardless. **Pin-advance retirement:** when a future release
contains `444f473a12`, bump the pin, restore `.gitmodules` to upstream, delete nothing from
Sirius's test tree — the sentinels re-verify every pin bump (design `:1096`).

**Post-merge developer note (PR description):** existing clones/worktrees must run
`git submodule sync duckdb && git submodule update --init --recursive`, or fetches of the new
gitlink fail against the old URL.

## 6. DEPENDENCIES & ORDERING

- **Within B1:** (1) fork creation (may need org-admin; the only external dependency) → (2)
  branch+tag push → (3) superproject PR (`.gitmodules` + gitlink + tests + issue). Never push the
  superproject before the fork branch is public.
- **Cross-track:** B1 depends on nothing else; A1-A4 and C1a/C1b/C2 may proceed in parallel. B1
  **blocks enablement** (not development) of C1c, C1e, C3 (design `:961-966`). C3 later appends the
  SIP-on leg to both the sqllogic file and the integration test; A1 later re-keys the plan
  assertions by `dynamic_filter_publication_plan_id`/`target_id` (design `:1018-1019`).

## 7. SIZE ESTIMATE

Superproject prod diff: ~5 lines (`.gitmodules` 2, gitlink 1, `CMakeLists.txt` 2). Fork diff: the
+33/−2 cherry-pick. Tests: sqllogic ~200 lines (extra optimizer-settings legs); planner Catch2
~330; integration Catch2 ~300; total ~830-880 test LOC. **One PR, do not split** — the gitlink bump
and its sentinels must land atomically (a gitlink-only PR would be an unverified pin move; a
tests-only PR fails on the unpatched pin).

## 8. RISKS (implementation-level) + MITIGATIONS

1. **Fork reachability breakage** (private fork, deleted branch, gitlink pushed first) breaks every
   `submodules: recursive` checkout (`test.yml:57-58`, `check.yml:32-33,107-109`) and all
   contributor clones. Mitigate: push order, branch protection, the `v1.5.4-sirius.1` tag, and the
   clean-clone simulation in the gate.
2. **`gh repo fork --org` permission** may be unavailable to the implementer. Mitigate: documented
   empty-repo + push fallback (§3.1); flag to the org owner early.
3. **Optimizer rewrites mask the bug** — this is not hypothetical: `late_materialization` and
   `compressed_materialization` mask ALL five shapes at defaults (review-measured), and a
   materialized CTE would stop DuckDB's walk regardless of the fix. Mitigate: three-optimizer
   disable set in the Catch2 harness, explicit `SET disabled_optimizers` legs in sqllogic and
   integration tests (with restore), plain subqueries in Q-S1..Q-S3 (upstream `WITH` queries kept
   only as verbatim sentinels). Residual: a future pin adding a new masking rewrite would surface
   as a green-side sentinel failure (shape guards fire), which is the intended loud failure mode.
4. **Join-order / plan-shape drift** silently changes which join carries `filter_pushdown`, and
   `subtree_contains(probe, LIMIT/TOP_N)` alone is NOT a sufficient guard — under the
   late-materialization rewrite the TOP_N sits inside the semi-join branch and the containment
   check still passes while the assertion target is wrong. Mitigate: `LIMIT 4` caps the selector
   estimate at ~4 rows, so shrink `dim` to 3 rows (filtered build ~2 rows ≪ 4) instead of relying
   on the 20-row-probe idiom; `REQUIRE` on join count; `join_build_side_is_table(outer, "dim")`
   RHS-binding assertion before any pushdown assertion, so drift fails loudly rather than asserting
   the wrong operator.
5. **Sirius plan-gen `InternalException` on internal table scans** would WARN-skip physical-level
   assertions (precedent `test_distinct_hash_join_detection.cpp:168-171`), leaving a sentinel hole.
   Mitigate: the logical-level `probe_info` assertions (§3.5a) never touch plan-gen and are the
   primary sentinel; physical-level checks are additive.
6. **Integration transparent-path fallback** (delta assertion `1,0,1` fails if any shape falls
   back). With the masking optimizers disabled per leg, the planned shapes use only GPU-supported
   operators (`LOGICAL_LIMIT`/`LOGICAL_TOP_N` handled at
   `src/planner/sirius_physical_plan_generator.cpp:182,192`; offset supported —
   `src/include/op/sirius_physical_limit.hpp:41,55`, `src/include/op/sirius_physical_top_n.hpp:43,50`).
   Note the defaults-path late-mat plan (rowid scan + SEMI join) was a real fallback risk — that is
   one reason the legs disable it. Mitigate: mandatory pre-landing empirical zero-fallback
   confirmation; if a leg still falls back, restructure the query rather than weakening the delta.
7. **Shared-state leakage from `SET enable_dynamic_filter_pushdown` / `SET disabled_optimizers`**
   into other `[integration]` tests (one process-global SiriusContext,
   `src/sirius_extension.cpp:1480-1488`; optimizer set is connection config mutated per leg).
   Mitigate: `pushdown_setting_guard` (capture-and-restore) and `disabled_optimizers_guard` RAII in
   every case, including failure paths.
8. **Schedule-dependence of the runtime red evidence:** Q-S3's unpatched wrongness relies on the
   probe pipeline depending on the inner build; a scheduler change could make even Q-S3 pass on an
   unpatched pin. Mitigate: the gate's deterministic tier is the metadata assertion; the runtime
   transcript is recorded evidence, not a repeatable gate condition.
9. **Wheel/version cosmetics:** duckdb-python and the extension report v1.5.4 while embedding the
   patched optimizer (dev SHA suffix may differ). Cosmetic; note in PR, keep `distribution.yml`
   labels at v1.5.4.

## Review resolution

- **Finding 1 (BLOCKER — red gate empirically false at defaults):** applied. §3.4 restructured: CPU-defaults section demoted to patched-pin regression with an in-file note; new optimizers-disabled section is the bug-exercising leg; Q-S3 there is the sole runtime CPU red sentinel, documented as schedule-dependent; all "Unpatched: N rows" predictions replaced with the review's measured behavior; gate item 1 rewritten with the deterministic burden on §3.5 metadata assertions; §3.8 issue text rewritten (see finding-10-adjacent honesty directive).
- **Finding 2 (BLOCKER — §3.5 sentinel fails green on patched pin without late-mat disable):** applied. Harness disables `OptimizerType::LATE_MATERIALIZATION` in addition to the donor's two, with an explanatory comment; noted the `!fp || empty()` form's tolerance of `compute_aggregates_anyway`; Risk 4 updated to say `subtree_contains` alone is insufficient.
- **Finding 3 (MAJOR — integration test at defaults never exercises the bug / risks patched-pin fallback):** applied. Every integration leg runs under a new `disabled_optimizers_guard('late_materialization,compressed_materialization')` with capture-and-restore; mandatory pre-landing empirical zero-fallback confirmation added (§3.6, gate item 2b).
- **Finding 4 (MAJOR — runtime wrongness racy on CPU and GPU):** applied. "Fires on CPU and GPU" claims removed; explicit-row legs labeled patched-pin regressions with the `dynamic-filters.md:238/:162/:283` transitive-target citations; Risk 8 added for Q-S3's schedule dependence.
- **Finding 5 (MAJOR — sizing rationale wrong, LIMIT caps probe estimate):** applied. `dim` shrunk to 3 rows (correct-result oracles verified unchanged by derivation — dropped dk 3/4 were both `tag='drop'` — with empirical re-derivation still mandated); new `join_build_side_is_table` RHS-binding assertion replaces reliance on subtree containment; the 20-row-probe idiom claim deleted.
- **Finding 6 (MINOR — no transparent-SQL sqllogic precedent; env-sensitive runner):** applied. §3.4 header documents `SIRIUS_CONFIG_FILE` / stale `~/.sirius/sirius.yaml` abort and mandates a pre-landing smoke run; "no gpu_buffer_init needed" downgraded to an assumption the smoke run must validate.
- **Finding 7 (MINOR — transaction ownership unspecified):** applied. §2 states caller owns one transaction spanning Parser→Optimizer→create_plan and the interleaved walk; helpers never BEGIN/COMMIT/ROLLBACK (mirrors `test_distinct_hash_join_detection.cpp:53-92`).
- **Finding 8 (MINOR — guard semantics):** applied. `pushdown_setting_guard` now captures and restores the prior value (not hardcoded true); no guard for `gpu_execution` (log-only callback `sirius_extension.cpp:1599-1602`, per-session option state — verified in code).
- **Finding 9 (MINOR — stale tag store):** applied. §1 and gate item 2d require re-fetching upstream tags and re-verifying `git tag --contains 444f473a12` before the claim enters the PR, with a stop condition if a released tag contains it.
- **Finding 10 (OBSERVATION — uninitialized `build_side_has_filter` on the compute_aggregates_anyway path):** applied as §3.9 follow-up note (verified: pinned `join_filter_pushdown.hpp:71` has no default initializer; Sirius reads it at `sirius_plan_comparison_join.cpp:426`); no B1 code change.
- **Review's corrected citations adopted:** `test.yml:57-58`, `check.yml:32-33,107-109` (+ build-only note `:139,:167`), `distribution.yml:33,:57`, `sirius_extension.cpp:1834-1848`, `publishes_dynamic_filters` `:166-170`, `sirius_physical_plan_generator.hpp:82-85`, `unittest.cpp:52-101`, `.pre-commit-config.yaml:81-86`, fixture `:110-160`.
