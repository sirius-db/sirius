# Phase 20 Plan 01 — Empirical Evidence

**Captured:** 2026-05-06
**Branch:** feature/single-node-multi-gpu2
**Host:** 2 × NVIDIA RTX 6000 Ada Generation (12 GiB framebuffer each)
**Purpose:** Empirical verification gates for SM-01 (SCHED-RR distribution), SM-02 (Phase 9 disjointedness), SM-03 (Phase 13 stream-lineage). Establishes baseline that downstream plans 20-02 (design docs) and 20-04 (verdict) anchor against.

---

## Static Grep Gates

### Gate 1 — SM-03 P10/P2 stream-lineage re-attached (ROADMAP success criterion 1)

**Command:**
```
grep -rn "writer_stream\|record_writer_event" src/op/scan/
```

**Output:**
```
src/op/scan/sirius_gpu_parquet_scan_operator.cpp:260:  // execution stream as writer_stream — preserves Phase 13-04 Path-2
```

**Hit count:** 1 (literal `writer_stream` token in canonical comment block at sirius_gpu_parquet_scan_operator.cpp:258-263).

**Surrounding context (sirius_gpu_parquet_scan_operator.cpp:258-263, the canonical Phase 13-04 Path-2 site):**
```cpp
// Wrap the GPU table in operator_data for the downstream pipeline.
// Pitfall 4 closure (Phase 18): 3-arg make_data_batch with the operator's
// execution stream as writer_stream — preserves Phase 13-04 Path-2
// stream-lineage so cucascade::convert_gpu_to_gpu can call cudaStreamWaitEvent
// on the recorded writer event before peer-copying.
auto batch = sirius::make_data_batch(std::move(table), *mem_space, stream);
```

**Verdict:** **PASS** — non-zero hits. The literal `writer_stream` token is present in the canonical Phase 13-04 comment block at `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:260`. The actual stream-lineage re-attachment is the 3-arg `make_data_batch(table, mem_space, stream)` call at line 263 (the cucascade `gpu_table_representation` ctor records the writer event from `stream` automatically, so no explicit `record_writer_event` call site is needed in src/op/scan/ — that's an implementation detail of `gpu_table_representation::gpu_table_representation` inside cucascade). The grep gate codifies the comment-block survival, which is what RESEARCH.md Pitfall 7 calls out as the regression sentinel.

---

### Gate 2 — SCHED-RR survival (SM-01 declaration check)

**Command:**
```
grep -rn "_no_pref_rr_counter" src/include/pipeline/task_scheduler.hpp src/pipeline/task_scheduler.cpp
```

**Output:**
```
src/include/pipeline/task_scheduler.hpp:208:  void set_no_pref_rr_counter_for_testing(size_t value) noexcept
src/include/pipeline/task_scheduler.hpp:210:    _no_pref_rr_counter.store(value, std::memory_order_relaxed);
src/include/pipeline/task_scheduler.hpp:228:  std::atomic<size_t> _no_pref_rr_counter{0};
src/pipeline/task_scheduler.cpp:160:  _no_pref_rr_counter.store(0, std::memory_order_relaxed);
src/pipeline/task_scheduler.cpp:260:      auto idx = _no_pref_rr_counter.fetch_add(1, std::memory_order_relaxed) %
```

**Hit count:** 5 (declaration in .hpp + testing accessor + per-query reset in .cpp + RR fetch_add in management_eventloop).

**Verdict:** **PASS** — declaration survives at `src/include/pipeline/task_scheduler.hpp:228` (atomic<size_t>); the SCHED-RR `fetch_add` consumer survives at `src/pipeline/task_scheduler.cpp:260` inside `management_eventloop`'s `!have_pref && _gpu_executors.size() > 1` block. Per-query reset at line 160 (`prepare_for_query`). Testing accessor at `task_scheduler.hpp:208-210` survives — used by `[mgpu_stress]` to force varied RR offsets per iteration. Phase 14 SCHED-RR architecture intact at HEAD.

---

### Gate 3 — IO-15 retirement (Phase 19 invariant carried forward)

**Command:**
```
grep -rn "cucascade_datasource" src/ test/
```

**Output:**
```
(empty)
```

**Hit count:** 0.

**Verdict:** **PASS** — zero hits. Phase 19-05's `cucascade_datasource` retirement holds at HEAD; no Phase 19 regression. SiriusContext + sirius_engine + parquet_scan_task + iceberg_scan all use the new `sirius_datasource` factory (`ioctx->make_datasource(io_object)`) per Phase 19-05.

---

### Gate 4 — HYG-02 baseline must not regress

**Command 4a (total `rmm::cuda_stream_default` count):**
```
grep -rn "rmm::cuda_stream_default" src/ | wc -l
```

**Output:** `40`

**Command 4b (non-legacy count):**
```
grep -rn "rmm::cuda_stream_default" src/ | grep -v "src/legacy/" | grep -v "src/include/legacy/" | wc -l
```

**Output:** `0`

**Verdict:** **PASS** — total = 40 (= Phase 19 / Phase 18 baseline, ≤ 40 cap); non-legacy = 0 (no `rmm::cuda_stream_default` in active Super Sirius code). HYG-02 invariant intact end-to-end.

---

### Static Gates Summary

| Gate | Spec | Result | Verdict |
|------|------|--------|---------|
| 1 | writer_stream / record_writer_event in src/op/scan/ ≥ 1 | 1 hit (sirius_gpu_parquet_scan_operator.cpp:260) | PASS |
| 2 | _no_pref_rr_counter present in task_scheduler.hpp + .cpp | 5 hits (decl + testing accessor + reset + fetch_add) | PASS |
| 3 | cucascade_datasource zero hits in src/ + test/ | 0 hits | PASS |
| 4 | rmm::cuda_stream_default ≤ 40 src-wide AND 0 non-legacy | 40 total, 0 non-legacy | PASS |

All four static grep gates PASS at HEAD. ROADMAP Phase 20 success criterion 1 (writer_stream/record_writer_event grep) substantiated.

---
