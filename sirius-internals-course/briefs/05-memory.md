# Module 5: Memory & Multi-GPU

**File to write:** `modules/05-memory.html` — only a `<section class="module" id="module-5">…</section>` block (odd module → default background).

**AUDIENCE (course-wide override):** Senior systems engineer joining Sirius — no general CS/GPU explanations; tooltip Sirius-specific terms only. Sharp-colleague tone.

### Teaching Arc
- **Metaphor:** A hotel with an annex and a storage depot. You don't get a room by walking in — you *book* before travel (reservation before the task runs). When the main building fills, the night manager (downgrade executor) relocates idle long-stay guests to the annex (host memory), then the depot (disk). Your room number comes from your booking — never from asking whichever guest arrived first (the device contract).
- **Opening hook:** "GPU memory is the scarcest thing in this engine. Sirius's answer: nothing runs without a reservation, and pressure triggers relocation — not a crash."
- **Key insight:** reserve → run → spill-under-pressure → retry-on-OOM is a closed loop; and on multi-GPU, the task's *reservation* is the sole authority on which device you're on.
- **Why care:** The two ways new code dies in production here: holding memory the downgrade executor can't reach, and inferring the device from the wrong place.

### Content beats (4 screens)
1. Tiers & reservations: GPU (tier 0) → pinned host per NUMA (tier 1) → disk (tier 2), managed by `sirius_memory_reservation_manager` (cuCascade). The GPU executor's manager thread makes a reservation sized from the pipeline's memory *history* before dispatching a task; operators allocate within it; it's released at task end. No reservation → the task won't run.
2. Pressure: each memory space has a `downgrade_executor` — a monitor thread watching usage, a processing thread executing downgrade requests. Candidates: idle batches in repositories (tier 1 candidates), then even batches held inside *queued* tasks (via the inspectable task queue — PR #637). Show tier-flow diagram: GPU → host → disk (disk fallback added in #647).
3. OOM is a control-flow event, not a crash (snippet A, code↔English): allocations beyond the reservation throw `oom_reschedule_exception` carrying partial state + a resume index; the executor retries with backoff. Read the comment in the snippet out loud: retries were 10 × 5 ms and it was *too short at SF100* — now 100 × 50 ms. Real tuning, recorded in code.
4. Multi-GPU (#732): tasks are dispatched by data locality; round-robin only when nothing has a preference. The device contract (docs call it SCHED-RR, `docs/super-sirius/pipeline-execution.md`): `gpu_pipeline_task::execute` takes the memory space from its reservation and `prepare_for_processing` colocates *every* input batch onto it before any operator runs. War-story callout: #732's greedy push scheduling piled tasks into per-GPU queues beyond the downgrade executor's reach — #827 restored pull-signal backpressure; #996 turned "prefer this device" into "pin this device" for cuco-backed operators after cross-device SIGABRTs.

### Code Snippets (pre-extracted, use EXACTLY as-is)

Snippet A — File: src/pipeline/gpu_pipeline_executor.cpp (lines 330-343)
```cpp
          // SF100 scale, so 10 retries × 5ms backoff (50 ms total) was far
          // too short. With 100 retries × 50 ms backoff (~5 s) the probe
          // tasks get enough patience to clear the contention window while
          // still bailing out on truly wedged queries.
          static constexpr uint32_t MAX_OOM_RETRIES = 100;
          if (next_retry_count > MAX_OOM_RETRIES) {
            SIRIUS_LOG_ERROR(
              "GPU Pipeline Executor: task {} (original task {}) exceeded {} OOM retries at "
              "operator index {} — terminating query",
              gpu_task->get_task_id(),
              orig_task_id,
              MAX_OOM_RETRIES,
              oom.get_resume_operator_index());
```

Snippet B — File: src/pipeline/gpu_pipeline_task.cpp (lines 326-331, the OOM throw with resumable state)
```cpp
      throw oom_reschedule_exception(
        std::move(operator_input_output_data),
        i,
        "OOM at operator " + op.get_name() + " (index " + std::to_string(i) + ")");
```
(NOTE: snippet B is 4 lines starting at the `throw` — present it as "the moment a task gives up gracefully": it packages its intermediate data and the operator index to resume from.)

### Interactive Elements
- [x] **Code↔English translation** — snippet A (emphasize the tuning-history comment) and snippet B.
- [x] **Quiz** — 4 questions, style: spot-the-bug/debugging.
  1. Spot-the-bug: "A new operator picks its CUDA device with `batches[0]->get_memory_space()` before prepare_for_processing has run. Single-GPU tests pass. What happens on an 8-GPU box?" → batch 0 may still be host-resident or on another device; the authoritative space is the task's reservation — cross-device access → crash/corruption (#996 class).
  2. "Host tier is full and no disk tier is configured. What should the downgrade monitor do — and what bug did Sirius actually fix here?" → nothing can be freed; it must idle, not busy-spin full scans (#911).
  3. "Your operator's first run OOMs, retries, then succeeds using far more memory than the estimate. What mechanism makes the *next* query's estimate better?" → pipeline memory history records the failure peak; estimates keep the higher peak so retries reserve more.
  4. "Why does downgrade prefer batches in repositories over batches inside queued tasks?" → repo batches are idle (no locks, cheap to migrate); queued-task batches need the inspectable-queue path and coordination — it's the second-choice tier of candidates.
- [x] **Glossary tooltips** — reservation, memory space, tier, downgrade request, oom_reschedule_exception, memory history, SCHED-RR / device contract, pinned host memory, data locality, cuco (the GPU hash-table library), inspectable task queue.
- [x] **Other** — tier-flow diagram (hero visual: GPU→host→disk animated arrows); war-story callout (#732→#827 backpressure, #996 pinning).

### Reference Files to Read
- `/home/ubuntu/.claude/skills/codebase-to-course/references/interactive-elements.md` → "Code ↔ English Translation", "Multiple-Choice Quiz", "Callout Boxes", "Glossary Tooltips" (+ any simple diagram pattern)
- `/home/ubuntu/.claude/skills/codebase-to-course/references/design-system.md` → tokens, diagram styles
- `/home/ubuntu/.claude/skills/codebase-to-course/references/content-philosophy.md` → all (AUDIENCE override applies)
- `/home/ubuntu/.claude/skills/codebase-to-course/references/gotchas.md` → all

### Connections
- **Previous module:** "The Data Plane" — where batches live; this module is who pays for them and what happens when the bill is too high.
- **Next module:** "Design Decisions" — the why behind everything seen so far, with PR receipts. Hand off: "Every rule in this module was learned from a specific incident. Next: the decision history, so you inherit the scar tissue without the scars."
- **Tone/style notes:** teal accent; real type names in `code`; PR links https://github.com/sirius-db/sirius/pull/<N>.
