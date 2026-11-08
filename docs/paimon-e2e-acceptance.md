# Paimon E2E qualification: version-bound acceptance mapping

Owner: E2E qualification (#2213), 2026-11-08. This is the single acceptance/result mapping, not a new support contract. Requirements are [Sirius #2213](https://github.com/sirius-db/sirius/issues/2213), [#2181](https://github.com/sirius-db/sirius/issues/2181), [#2211](https://github.com/sirius-db/sirius/issues/2211), [#2212](https://github.com/sirius-db/sirius/issues/2212) and [duckdb-paimon #137](https://github.com/polardb/duckdb-paimon/issues/137). The coordinator owns the local roadmap and dependency index; changes to scope go there/public contracts, never through a test silently weakening a requirement.

All final E2E acceptance below is **NOT RUN**. Source audit and CPU harness checks qualify only their stated properties. No final candidate tuple, production SQL adapter binary, GPU lease, E2E CI execution or merged-main result exists in this packet.

## Candidate and evidence identity

| Input | Observed immutable revision / missing input | Qualification consequence |
| --- | --- | --- |
| Sirius integration baseline | `eb9c3a132d3dcd00d8a480852534f3a3b8cf490c` | Test preparation only; not a working final combination |
| DuckDB host gitlink (both host trees) | `d54cbbe3df17e8e5ce68312f483b4155836112aa` | Actual CLI/header/extension compatibility still needs build and binary hashes |
| Producer #150 public | `ad0a1b94776d064424a3673195593fd4203e645c` (Draft) | Does not qualify local revision |
| Producer #150 local candidate | `cb37bb1d26a1e3ba09fc13402cd426a4cf993efb` (owner coordination report) | Must obtain owner release/immutable inputs; new `paimon.predicate_pruning` contract |
| Native source component | `817b9bff` historical owner head; full SHA/status in raw snapshot | Not an integrated production source; dirty files excluded |
| Metadata R2 / SQL adapter | No delivered candidate pair | Owners supply immutable commits, APIs, profile and route evidence |
| Cleanup #2220 public | `f65f67316bdbfcaf8bc980801640f9ff4c568aa2` | Reuse only guarantees still needed after base audit; no automatic whole-diff inclusion |
| Native SDK | #462 merged as `914d1cfcbe8ffa32a947aaa26e02ee70989557d8` | This is not the selected SDK pin/artifact; planning drain is not complete resource admission |
| Final SDK/modules/Arrow/toolchain/GPU | Not selected | Need retrievable dependency bytes, actual loader maps and build receipts, GPU UUID/driver/CUDA/cuDF/RMM/config |
| Corpus and oracle | Checked-in `paimon_conformance` at integration baseline | Preserve manifest, file inventory, expectation and generator hashes; no regeneration required for initial case |

Workspace: `/Users/aaron/.codex/worktrees/paimon-e2e-qualification/sirius`, branch `codex/paimon-e2e-qualification`. Canonical raw packet: `/Users/aaron/workspace/sirius/.agent-channel/artifacts/paimon-e2e-qualification-20261108/`. `source-snapshot.json` records public observations, full source identities/dirty state and input hashes. Dirty owner trees are evidence only; never cherry-pick uncommitted changes or modify them. Integration commits must record owner input SHAs and any conflict resolutions separately.

Every executed stage retains: exact tuple; command/argv and exit/signal/timeout; selected cases and final report; raw stdout/stderr including partial output; SQL results; source/config/fixture/binary/SDK/module hashes; host/GPU identity; owner findings and dispositions. Setup failure means case NOT RUN and failed invocation, not empty success. Final merged-main validation is a separate run after merge SHAs are known.

## Acceptance matrix

IDs below index clauses in the linked public contracts. M1/M2 are delivery slices, not replacement support definitions. Every row is NOT RUN on the final tuple; rows include the owner for missing capability or product defects.

| ID / public clause | Positive example / reusable input | Required negative or counterexample | Evidence and missing capability / owner |
| --- | --- | --- | --- |
| Q01 / #2213 GPU route | Ordinary Paimon SELECT; same query's retained Paimon identity and actual GPU operator execution; fallback disabled | Forced CPU with identical correct rows; unrelated GPU query; stale/other scan evidence; setting-only proof must fail | Query-associated existing scan registry/counters/profile adapter; E2E owns checker, adapter owns signal correctness |
| Q02 / #2213 values & oracle | `append_a/b`, duplicates, NULL, values AND SQL types, same snapshot | Wrong multiplicity/type/value/snapshot; empty/missing/truncated result | Reuse literal expectations + recorded independent PyPaimon checks and inventory validation; new oracle only for new fixtures / E2E |
| Q03 / #2213 historical & empty | `empty_rows/total`, append A after B exists; empty-at-bind followed by commit | Empty metadata versus missing metadata; bound snapshot switched to latest | New live writer/statement orchestration; producer/R2/adapter |
| Q04 / #2213 multi-batch | Multiple files/partitions plus enforced small native batches | Drop/duplicate/reorder batch where ordering matters; zero-row batch versus EOF | Tiny existing corpus alone does not force multiple reader batches; source reports actual batch boundary/queue behavior / native owner |
| Q05 / #2213 projection/filter/types | `strings_projection/null`, `partition_filter`; reorder output; filter-only column; full SQL residual after optional pruning | Drop residual, wrong field ID/ordinal, incompatible Arrow layout, cap-overflow hints | R2 schema/field correspondence + adapter residual mapping; type support only after exact profile validation |
| Q06 / #2213 prepared/multi-scan | Prepare/execute/rebind; repeated execution; two scans with separate retained identities | Copy/replan loses identity; one scan incorrectly borrows another's snapshot; unsupported reconstruction explicitly refused | New same-session orchestration; producer/R2/adapter |
| Q07 / #2213 commits/schema/expiry | Commit after bind, schema change between executions, retained historical read | Delete bound snapshot: explicit error, never latest/empty; stale schema/field mismatch; unsupported branch/catalog refused | Disposable warehouse copy, deterministic barriers rather than sleeps; producer/native/adapter |
| Q08 / #2213 read/import failure | Known native status/throw, import failure after valid batch; accurate error stage | Crash containing expected text; wrong exit/type; partial results followed by failure never accepted as success | Reuse process status/log handling; new native/adapter fault hooks only after sole writer agreement |
| Q09 / #2211 + #2213 cancellation | Cancel preparation/read; stop publication, wait entered native work, drain before permitted replay | Premature pool/reader/root release; new work after stop; concurrent close/read; hang/partial timeout | Source/cleanup owners implement; E2E checks process status, lifecycle evidence and same-process subsequent query |
| Q10 / #2211 + #2213 resources | Refuse before unsafe allocation; bounded admitted batch queue and complete resource ledger | Oversized plan/splits, native scratch/export/staging/retirement; pool-only budget cannot certify all-native cap | Native resource contract unresolved; controlled fixtures do not authorize unbounded production enablement / native owner |
| Q11 / #2213 cleanup/liveness/replay | Success/error/cancel/refusal release roots/readers/pools/modules/queue/retirement in order; subsequent query live | Leaks/use-after-free, outstanding accepted tasks, replay before drain; fresh-process probe cannot prove same-session liveness | New same-session driver and source/cleanup evidence; permitted replay retains identity, otherwise explicit refusal / native+adapter+cleanup |
| Q12 / #137 co-load | Independently built actual extensions, both LOAD orders; CPU scan/catalog/prepared/schema/snapshot suite | Missing/old/wrong provider, ABI or host mismatch rejected before unsafe callback use | Historical nine-case dynamic bridge is design evidence only; current metadata protocol must replace old descriptor controls / R2+E2E |
| Q13 / #137 + #2213 delivery | Clean checkout/cold cache, retrievable SDK and format/filesystem modules, actual loader origins and receipt hashes | Missing module, shadow SDK, drifted artifact/gitlink/header, failed download; no unsafe guard bypass | Existing build receipt/inventory/downloader reusable, provider/native recipe pins need qualification; dependency packaging sole writer must be confirmed / build+native+E2E |
| Q14 / #2213 CI | Real entrypoint discovers intended cases, exact-head jobs executed with retained reports, failures propagate | Zero selection, missing/truncated report, setup fail, allowed failure/skipped job cannot become PASS | Main has no Paimon workflow/default-runner registration; separate cheap checker tests and leased GPU job / E2E+CI owner |
| Q15 / #2181 + #2213 M2 PK | `orders_pk` A–D update/delete/reinsert; `filter_no_old_row`; independent expected answers | Old row revived by non-key filter; multi-bucket/composite-key cross-file/batch histories from #95 | Existing single-bucket fixture is narrow seed; final-pin #95 investigation and expanded independent oracle required / native/provider+E2E |
| Q16 / #2213 final-main tuple | Repeat affected Q01–Q15 on actual merged producer/SDK/DuckDB/Sirius combination | Candidate PASS copied onto changed merge head/dependency bytes | Merge/readback SHAs, patch comparison and targeted revalidation; all owners / E2E |

## Existing evidence: permitted reuse

- `test/cpp/integration/data/paimon_conformance/{expectations,manifest}.json`: six real writer tables and 25 SQL cases. Hand-authored expected values/types are independent of the current native reader. Stored PyPaimon full-row checks are historical, input-bound corroboration, not a freshly executed oracle. Audit inventory and expectation consistency before and after reads.
- `run_conformance.py`: strict six-result SQL protocol, typed multiset comparison, liveness, structured subprocess exit/signal/timeout and retained partial logs. Reuse these invariants. Its normal modes are disabled/CPU; transparent smoke allows fallback and the disabled-fallback smoke expects planning rejection. Its green report does not establish Q01 and the rejection smoke must not be repurposed as successful GPU execution.
- `test_runner.py`, `test_orchestration.py` and other `test_*.py`: reusable CPU-only negative coverage for incomplete protocol, expected-text crashes, incorrect statuses, provenance drift and setup failure. Fake CLI results qualify harness decisions only.
- `build_extension.py`, `qualified_extension.py`, download helpers and `DEPENDENCIES.md`: reuse receipt/hash/host-pin validation. Current recipe pins provider `307fc34c04623fe18f0ac39125b6319ea0013c0f`, native `53f9c86d45aabb0a6f1a379271da07d7a9f27a3d`, DuckDB `d54cbbe3df17e8e5ce68312f483b4155836112aa` (matching this baseline's two host gitlinks); do not edit receipts or use nearby version strings to bypass actual host/artifact qualification. The older primary checkout recipe still names `88ecd0a`; it is not the integration baseline.
- Historical `.agent-channel/artifacts/paimon-137-qualification-20261016/` and `paimon-137-qualification-update.md`: actual old co-load/CPU/native checks, with their old pins and stated cached-build limits. No transfer of those PASS results to Q12/Q13 or current metadata ABI. The private scripts write to old hard-coded locations; inspect/adapt in isolation before any replay.
- Shared Arrow/component GPU and ordinary-table interception tests are component evidence. They do not prove Paimon SQL routing or an independent Paimon oracle.

## Smallest first E2E case

Reuse `append_basic` snapshot B (ID 2), not a staging DuckDB table. The first query is ordinary SQL:

```sql
SET gpu_execution = true;
SET enable_duckdb_fallback = false;
SELECT id, amount FROM paimon_scan('<copied-corpus>/warehouse/reference.db/append_basic', snapshot_from_id=2);
```

Prerequisites: actual compatible Sirius/provider extensions loaded; owner-supplied experimental opt-in if required; both planning and runtime fallback disabled under the pinned implementation; source profile admitted before native planning. Do not guess new settings. Assert exact types `BIGINT, DECIMAL(12,2)` and multiset `(1,10.00)` twice, `(2,NULL)`, `(3,5.00)`. Preserve existing fixture inventory and snapshot/schema hashes. After the tested statement execute a same-session liveness query.

The same process/query evidence must identify Paimon retained snapshot/schema/source and executed GPU scan/operator. A `gpu_execution=true` result, successful EXPLAIN or unrelated GPU counter is insufficient. Extract from existing registry/profile facilities before proposing new production observability. If those facilities cannot associate execution with the Paimon scan, Q01 stays NOT RUN and the gap goes to adapter owner.

Run a forced-CPU counterpart with `SET gpu_execution=false` and identical expected rows. Its data comparison should succeed while the GPU acceptance checker must reject it with an explicit route-evidence failure. Missing/wrong-query/stale source evidence, wrong rows/types, successful-looking partial logs plus nonzero exit, signal and timeout must also be rejected through the real E2E entrypoint. Checker regression success is distinct from intentional qualification rejection. An isolated reviewer must challenge this checker at its final immutable revision before it can certify GPU PASS.

The empty-table and historical A cases follow this first pair; multi-batch and live mutation/failure cases require the additional orchestration in the matrix. M1 is append-only; no M1 result closes Q15 or the full #2181 target.

## Execution and delivery sequence

1. Finish baseline audit/CPU harness regression and record missing APIs, without rewriting the corpus or common runner. E2E owns this mapping and new E2E-specific files. Confirm the unique writer before touching shared runner/build/CI facilities.
2. Owners deliver immutable R2/R3/R4, producer and necessary cleanup inputs. Integrate in this worktree, record SHAs/conflict resolutions, fix selected SDK/modules and qualified host. Product bugs return to their owner; do not patch their branch.
3. Reuse comparison/process machinery behind a distinct E2E entrypoint only after route evidence is concrete. Write real-entrypoint positive/negative tests first. Retain the CPU suite's intentionally different contract. Freeze source before independent checker/delivery/necessity/compatibility review.
4. Ask coordinator to schedule one serialized GPU lease for the exact tuple/host/expected occupancy. No current lease. Run the minimum pair first, then affected matrix slices; retain failed fixtures/reports before reruns. Baseline latency, first-batch latency, memory and transfer comparisons use the same data/tuple after correctness gates.
5. Publish a useful test PR to the personal fork against `main` after applicable local gates/isolated review. If real CI needs publication, identify a validation Draft and list pending evidence. Verify actual jobs/conditions/artifact revisions/discovery; do not infer execution from aggregate green. Keep public replies focused; full records stay in mailbox/artifacts.
6. After all required merges, freeze actual mainline tuple, reassess patch/dependency changes and rerun affected acceptance. Keep #2213/#137 open until their own requirements pass; #2181 additionally requires M2.

## Commands and current result boundary

Preparation command: `git submodule update --init --recursive` in the isolated tree (in progress at initial capture; build NOT RUN). Do not qualify builds while submodule gitlinks are incomplete or mismatched.

CPU harness command (isolated Python 3.12 Pixi environment on this Mac):

```sh
pixi run --manifest-path /Users/aaron/workspace/sirius/.agent-channel/artifacts/paimon-e2e-qualification-20261108/cpu-env/pixi.toml python -m unittest discover -s test/cpp/integration/data/paimon_conformance -p 'test_*.py' -v
```

Raw log: `cpu-harness.log` in the packet. Final process result and count belong in `stage-results.json`, not inferred from intermediate test lines. Current final native/GPU build/test commands are NOT RUN because the immutable owner tuple and route-evidence boundary are not delivered. Independent baseline review belongs in `independent-baseline-review.md`; it cannot approve a future checker. Final checker review/CI/merged-main qualification remain pending.
