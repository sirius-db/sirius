# Module 7: The Streaming Frontier (#836 / #837)

**File to write:** `modules/07-streaming.html` — only a `<section class="module" id="module-7">…</section>` block (odd module → default background). This is the capstone — it speaks directly to the learner's own two issues.

**AUDIENCE (course-wide override):** Senior systems engineer joining Sirius — THIS IS THEIR OWN WORK: issues #836 (streaming source operator) and #837 (streaming sink operator) for the StarRocks compute-node integration. No general explanations; tooltip Sirius-specific terms. Sharp-colleague tone; this module can be more direct ("your operator", "your fix").

### Teaching Arc
- **Metaphor:** A mailroom with a motion-sensor light. The light (the completion check) only switches on when somebody walks OUT (a task completes). A courier who delivers *nothing* never trips the sensor — the room stays dark forever. The fix isn't a brighter bulb; it's a switch wired to the door itself (`close()` fires a wake). This is exactly the class of bug streaming operators introduce into an edge-triggered engine.
- **Opening hook:** "Everything in modules 1–6 assumed data comes from disk. Your job (#836/#837) is to teach the engine that data can come from — and leave to — another machine. The engine's loop barely changes. Its *assumptions* do."
- **Key insight:** The streaming source/sink reuse the exact hint-chain/completion machinery — the entire challenge is that a stream can produce **zero tasks**, and the completion machinery is only ever evaluated when a task completes.
- **Why care:** This module IS the learner's current sprint. It also generalizes: any operator whose input isn't the scan manager will face the same edge-trigger gaps.

### Content beats (5 screens)
1. Where the operators sit: replace the leaf `SCAN(orders)` of the Module-3 trace with input arriving over the network. Downstream (filter, aggregate, task loop) is untouched. Side-by-side card: GPU scan pulls from `split_connector` ⟷ streaming source pops handles from an `exchange_channel` a remote producer pushes into. The sink mirrors it at the top of a fragment: CONCAT-shape boundary operator (`is_source() && is_sink()`), NOT a RESULT_COLLECTOR clone — because backpressure must be a *task-creation condition* (full channel → no sink tasks → upstream port repo fills with idle, spillable batches → the engine throttles itself), never a blocked worker.
2. The channel contract (styled spec card, clearly labeled "planned API — from the design notes, not yet in tree"): bounded channel of `{batch_id, size_bytes}` handles; batches themselves are registered in a repository FIRST, then the handle is pushed; close-then-drain end-of-stream; engine side only ever uses try_pop/try_push. Why handles: Module 4's law — idle-in-repo = spillable; the channel must never own memory.
3. The hint table for a streaming source (3 cards): channel non-empty → `READY{this}` · open-but-empty → `WAITING_FOR_INPUT_DATA{nullptr}` · closed AND drained → `nullopt`; and `all_ports_empty()` → `channel->drained()`. Per-pull admission lives in `get_next_task_input_data()` because the task creator's loop doesn't re-poll the hint (Module 3, snippet C's loop).
4. THE WAR STORY (hero of the module — data flow animation + narrative): two hangs found while building this.
   - Bug 1, the empty stream: channel closes with zero batches → first poll returns `nullopt` → no task is ever created → nothing ever calls `update_pipeline_status()` → the pipeline never finishes → the query's future never resolves. Nothing crashed; nothing was wrong — nobody re-checked.
   - Bug 2, close-after-last-task: the last task completes while the channel is still open-but-empty → completion check runs, sees "not drained", doesn't finish → the close() arrives a moment later → no task remains to re-trigger the check → hang.
   - The fix (in task_creator.cpp): the drained-source-with-no-task case explicitly calls `update_pipeline_status()`, and `close()` fires a wake so the source gets re-polled. Frame with snippet A (the condition that needed a new trigger). Callout: "A scan gets its completion evaluation for free — there's always a task in flight to trip the sensor. A stream must install its own switch."
5. Ship checklist (closing card, ties the course together): channel carries handles, repo owns batches (M4) · barrier types on the wired ports must be streaming-compatible (M4) · admission in get_next_task_input_data, not just the hint (M3) · zero-task EOS paths must trigger status evaluation themselves (this module) · external consumers must not outlive drain_after_error's drain order (M2/M3) · device comes from the reservation when #838 partitioning arrives (M5). End the course with: standalone PRs first — operators + unit tests, no CN wiring; that's the agreed sequencing.

### Code Snippets (pre-extracted, use EXACTLY as-is)

Snippet A — File: src/pipeline/sirius_pipeline.cpp (lines 389-401, inside update_pipeline_status — reuse from Module 3 deliberately; frame as "read it again, now with streaming eyes")
```cpp
      if (limit_exhausted ||
          (first_node->is_source_pipeline_finished() && first_node->all_ports_empty()) ) {
        if (tasks_created.load() == tasks_completed.load()) {
          pipeline_finished.store(true);
          for (auto& op : get_operators()) {
            op.get().finalize_operator();
          }
          end_nvtx_range_if_finished();
          should_notify = true;
        }
      }
```
IMPORTANT: present exactly as in Module 3's snippet D (same file/lines); the duplicated study is intentional. English side now asks: "who calls this when a stream closes and no task is in flight?"

### Interactive Elements
- [x] **Data flow animation** — the Bug-1 hang, then the fix. Actors: Remote producer, exchange_channel, Streaming source, TaskCreator, Pipeline status, Query future. Steps: (1) producer connects, sends nothing, closes channel; (2) task creator polls source → nullopt; (3) highlight Pipeline-status node greyed "never evaluated"; (4) future node pulses "waiting forever…"; (5) FIX path: close() fires wake → source re-polled → drained-with-no-task branch calls update_pipeline_status → pipeline finished → future resolves.
- [x] **Code↔English translation** — snippet A with the streaming-eyes framing.
- [x] **Quiz** — 4 questions, style: debugging/design (these should feel like real review questions on the learner's PRs).
  1. "The streaming sink's channel is full and stays full for 30 s. What is the CORRECT observable behavior inside the engine?" → no new sink tasks get created; upstream port repos fill with idle (spillable) batches; workers stay unblocked. (Wrong answers: worker blocks on push; batches dropped; query fails.)
  2. "Why must the sink be a CONCAT-shape boundary operator instead of a RESULT_COLLECTOR-shape terminal sink?" → RESULT_COLLECTOR-shape has no tasks of its own → nothing to gate on a full channel → the only place to wait would be a worker thread.
  3. "A fragment's stream delivers 10 batches, all processed; close() arrives 5 ms after the last task completed. Pre-fix, what does the user see and why?" → hang: the last completion check saw open-not-drained; the close had no task left to re-trigger evaluation.
  4. "Your reviewer asks: 'why does the channel carry {batch_id, size} instead of shared_ptr<data_batch>?' Best answer?" → ownership stays with the repository so batches remain idle/spillable and the channel can never pin GPU memory; the id is resolved via pop_data_batch_by_id at task-creation time.
- [x] **Glossary tooltips** — exchange_channel, close-then-drain, EOS, boundary operator, admission control, backpressure, fragment (StarRocks sense), split_connector, edge-triggered vs level-triggered (one tooltip contrasting them).
- [x] **Other** — side-by-side scan-vs-stream card; hint table cards; ship checklist (closing hero card); callout ("install your own switch").

### Reference Files to Read
- `/home/ubuntu/.claude/skills/codebase-to-course/references/interactive-elements.md` → "Message Flow / Data Flow Animation", "Code ↔ English Translation", "Multiple-Choice Quiz", "Callout Boxes", "Glossary Tooltips"
- `/home/ubuntu/.claude/skills/codebase-to-course/references/design-system.md` → tokens
- `/home/ubuntu/.claude/skills/codebase-to-course/references/content-philosophy.md` → all (AUDIENCE override applies)
- `/home/ubuntu/.claude/skills/codebase-to-course/references/gotchas.md` → all

### Connections
- **Previous module:** "Design Decisions" — the two currents; this module is the newest ring on the tree.
- **Next module:** none — this closes the course. End warmly but concretely: link to `experimental/starrocks/docs/streaming-source-plan.md`, `streaming-sink-plan.md`, `discoveries.md`, and issues https://github.com/sirius-db/sirius/issues/836 and /837.
- **Tone/style notes:** teal accent; "your operator / your fix" voice is welcome here; DO NOT present the exchange_channel API as existing code — label planned surfaces clearly.
