# Module 3: Life of a Query

**File to write:** `modules/03-life-of-a-query.html` — only a `<section class="module" id="module-3">…</section>` block (odd module → default background).

**AUDIENCE (course-wide override):** Senior systems engineer joining Sirius — no general CS/GPU explanations; tooltip Sirius-specific terms only. Sharp-colleague tone. This is the course's centerpiece module — spend the effort here.

### Teaching Arc
- **Metaphor:** A kanban factory line. Stations never push work downstream; a station *pulls* when its input bin has parts (the hint chain). The andon board (pipeline status) only updates when a worker turns in a job card (`mark_task_completed`) — remember that detail; it becomes a bug story in Module 7.
- **Opening hook:** "Let's trace one query, end to end, with the real functions: `SELECT count(*) FROM orders WHERE amount > 100;`"
- **Key insight:** Execution is a pull-when-ready loop, and *completion is edge-triggered by task completion* — `update_pipeline_status()` only runs when something finishes.
- **Why care:** Every scheduling bug, hang, or premature-completion crash lives inside this loop. Understand it once, debug it forever.

### Content beats (5 screens)
1. Planning: DuckDB parses; Sirius converts and splits into TWO pipelines because the aggregate is a pipeline breaker (must see all input):
   `Pipeline A (streaming): SCAN(orders) → FILTER(amount>100) → [sink: partial COUNT state]`
   `Pipeline B (terminal): COUNT finalize → RESULT_COLLECTOR`
   B's input port has a FULL barrier (can't start until A completely finishes); A's internal hops are PIPELINE barriers (batch-by-batch).
2. Kickoff: `task_scheduler::start_query()` schedules the first scan (snippet A). One line of code starts everything; the returned future is what DuckDB's thread blocks on.
3. The heart — the hint protocol (snippet B + C): task_creator asks the operator `get_next_task_hint()`. Three possible answers, present as 3 cards: `READY{this}` = "build me a task" · `WAITING_FOR_INPUT_DATA{producer}` = "go ask upstream" · `nullopt` = "I'm done forever". Walk the base implementation (snippet C): FULL-barrier port with unfinished producer → WAITING pointed at that pipeline's source; all ports have data → READY.
4. A task runs: reserve GPU memory → `prepare_for_processing` (lock + colocate input batches) → `execute()` each operator → `publish_output()` calls the terminal `sink()`, pushing surviving rows into the next port's repository. Data moves by handle, not copy.
5. Completion drives everything (snippet D + E): `mark_task_completed()` → `update_pipeline_status()` asks "source exhausted AND all ports empty AND tasks_created == tasks_completed?" → finalize operators, notify downstream pipelines → B's FULL barrier satisfied → B runs, RESULT_COLLECTOR fulfills the future. Callout: "No timer, no poller. If no task completes, nobody ever re-checks." (Module 7 payoff.)

### Code Snippets (pre-extracted, use EXACTLY as-is)

Snippet A — File: src/pipeline/task_scheduler.cpp (lines 172-180)
```cpp
std::future<void> task_scheduler::start_query()
{
  std::scoped_lock lock(_query_mutex);
  const auto& scans = _query->get_scan_operators();

  _task_creator->schedule(scans.front());

  return _completion_handler->get_awaitable();
}
```

Snippet B — File: src/include/op/sirius_physical_operator.hpp (lines 49-56)
```cpp
enum class TaskCreationHint { WAITING_FOR_INPUT_DATA, READY };

enum class MemoryBarrierType { PIPELINE, PARTIAL, FULL };

struct task_creation_hint {
  TaskCreationHint hint{TaskCreationHint::WAITING_FOR_INPUT_DATA};
  sirius_physical_operator* producer{nullptr};
};
```

Snippet C — File: src/op/sirius_physical_operator.cpp (lines 275-289)
```cpp
std::optional<task_creation_hint> sirius_physical_operator::get_next_task_hint()
{
  if (ports.empty()) { return std::nullopt; }

  // look at the input ports and see if there are any unfinished hard barriers
  auto unfinished_barrier = std::find_if(_ports_list.begin(), _ports_list.end(), [](const auto& p) {
    return p->type == MemoryBarrierType::FULL && p->src_pipeline &&
           !p->src_pipeline->is_pipeline_finished();
  });

  if (unfinished_barrier != _ports_list.end()) {
    auto* producer = &((*unfinished_barrier)->src_pipeline->get_operators()[0].get());
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
  }
```

Snippet D — File: src/pipeline/sirius_pipeline.cpp (lines 389-401, inside update_pipeline_status)
```cpp
      if (limit_exhausted ||
          (first_node->is_source_pipeline_finished() && first_node->all_ports_empty())) {
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

Snippet E — File: src/pipeline/sirius_pipeline.cpp (lines 459-461, tail of mark_task_completed)
```cpp
  tasks_completed++;
  update_pipeline_status();
}
```

### Interactive Elements
- [x] **Data flow animation** (MANDATORY, hero of this module) — `.flow-animation` with `data-steps` JSON. Actors (nodes): DuckDB, Plan Generator, Scan, Filter, partial-COUNT sink, Repo (port), COUNT finalize, RESULT_COLLECTOR. Steps: (1) SQL arrives → plan split into A/B; (2) start_query schedules scan; (3) hint READY → task built; (4) task: scan batch → filter → partial count → handle pushed to repo; (5) loop repeats while splits remain (show 2-3 pulses); (6) scan exhausted + last task completes → Pipeline A finished → finalize; (7) FULL barrier satisfied → Pipeline B task; (8) result → future fulfilled → DuckDB answers.
- [x] **Code↔English translation** — two blocks: snippet A (kickoff) and snippet D (the completion condition — translate each clause: source exhausted / ports drained / in-flight accounting).
- [x] **Quiz** — 4 questions, style: tracing/debugging.
  1. "Why can't Pipeline B start as soon as the first filtered batch exists?" → FULL barrier: COUNT finalize needs the complete count state; partial results would be wrong.
  2. "The filter's hint returns WAITING_FOR_INPUT_DATA with producer=scan. What does the task creator do next?" → recurses: asks the scan for ITS hint (deepest producers run first).
  3. "A query hangs. Zero tasks in flight, pipeline never finishes. Which function had no reason to run?" → update_pipeline_status — it's only called from task completion (edge-triggered); with no completing task, the status is never re-evaluated.
  4. "tasks_created == tasks_completed is checked inside the finish condition. What bug does that prevent?" → declaring the pipeline finished while a task is still running (operators finalized under a live task → premature completion / use-after-free class of bugs).
- [x] **Glossary tooltips** — pipeline breaker, FULL/PARTIAL/PIPELINE barrier, hint chain, gpu_pipeline_task, split, RESULT_COLLECTOR, finalize, repository, port, edge-triggered completion.
- [x] **Other** — 3 hint-answer cards; "aha" callout on edge-triggered completion.

### Reference Files to Read
- `/home/ubuntu/.claude/skills/codebase-to-course/references/interactive-elements.md` → "Message Flow / Data Flow Animation", "Code ↔ English Translation", "Multiple-Choice Quiz", "Callout Boxes", "Glossary Tooltips"
- `/home/ubuntu/.claude/skills/codebase-to-course/references/design-system.md` → tokens, flow/diagram styles
- `/home/ubuntu/.claude/skills/codebase-to-course/references/content-philosophy.md` → all (AUDIENCE override applies)
- `/home/ubuntu/.claude/skills/codebase-to-course/references/gotchas.md` → all

### Connections
- **Previous module:** "Meet the Actors" — the components; this module runs them.
- **Next module:** "The Data Plane" — zooms into the repositories/ports/batches the data moved through. Hand off: "We hand-waved 'the batch goes to the repo'. Next: what a batch actually is, and why it never travels by function return."
- **Tone/style notes:** teal accent; real type names in `code`; keep the count(*) query consistent everywhere.
