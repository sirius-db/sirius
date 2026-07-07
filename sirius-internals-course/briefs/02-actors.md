# Module 2: Meet the Actors

**File to write:** `modules/02-actors.html` — only a `<section class="module module--alt" id="module-2">…</section>` block (even module → alternate background per design-system).

**AUDIENCE (course-wide override):** Senior systems engineer joining Sirius — do NOT explain general CS/GPU concepts; DO tooltip every Sirius-specific term. Sharp-colleague tone.

### Teaching Arc
- **Metaphor:** Mission control. One room, seven consoles, each with a single job: Flight Director (task_scheduler) owns the go/no-go; the trajectory team (task_creator) decides the next burn; each spacecraft has its own ops crew (per-GPU gpu_pipeline_executor); consumables/life-support (memory manager + downgrade executors) quietly keeps everyone alive; nothing talks directly — it goes through the loop.
- **Opening hook:** "Every Sirius bug report eventually reduces to: which of seven objects misbehaved? Learn their names and their threads and you can triage anything."
- **Key insight:** `SiriusContext` owns everything, and roles are strictly separated: *deciding* what runs next (task_creator) is a different object — and different threads — from *running* it (executors) and from *surviving memory pressure* (downgrade executors).
- **Why care:** "Where do I put this logic?" and "whose thread am I on?" are the two questions that prevent broken PRs.

### Content beats (4 screens)
1. The ownership tree: SiriusContext (a DuckDB ClientContextState) owns memory manager, data repository manager, task scheduler, downgrade executors (one per memory space), task creator, scan manager. Code↔English on snippet A below — the actual member list IS the component map.
2. Actor cards (7): sirius_engine + plan generator (builds pipelines at query start) · task_creator (hint chain → tasks; 2 threads) · task_scheduler (top-level orchestrator; management event loop bridging task requests to executors; renamed from pipeline_executor in #687) · gpu_pipeline_executor (one per GPU: 1 manager thread + 4 workers, each worker pinned with its own CUDA stream) · scan manager + scan executor (splits & ingest) · downgrade_executor (monitor + processing threads per memory space) · shared_data_repository_manager (where all inter-operator data lives).
3. The thread model screen: who blocks (the DuckDB query thread, on a future), who loops (scheduler management loop, per-GPU manager, task-creator manager, downgrade monitor). Use a simple annotated diagram, not prose.
4. Group chat (hero): the actors running `SELECT count(*) FROM orders WHERE amount > 100;`.

### Code Snippets (pre-extracted, use EXACTLY as-is)

File: src/include/sirius_context.hpp (lines 310-315)
```cpp
  std::shared_ptr<const sirius::telemetry::telemetry_context> telemetry_context_;
  std::unique_ptr<cucascade::shared_data_repository_manager> data_repository_manager_;
  std::unique_ptr<sirius::pipeline::task_scheduler> task_scheduler_;
  std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>> downgrade_executors_;
  std::unique_ptr<sirius::creator::task_creator> task_creator_;
  std::unique_ptr<sirius::scan_manager::sirius_scan_manager> scan_manager_;
```

File: src/include/sirius_config.hpp (lines 162 and 165, thread-pool defaults)
```cpp
  exec::thread_pool_config _task_creator_config{.num_threads        = 2,
```
```cpp
  exec::thread_pool_config _gpu_pipeline_executor_config{.num_threads        = 4,
```
(Present these two lines as one tiny block titled "the defaults, straight from sirius_config.hpp" — 2 task-creator threads, 4 GPU workers.)

### Interactive Elements
- [x] **Group chat animation** (MANDATORY, hero of this module) — container id `chat-actors`. Actors: DuckDB, Scheduler, TaskCreator, GPU-0, Downgrade. Flow for the count(*) query:
  1. DuckDB → all: "Prepared statement ready. Sirius, you're up — plan swapped in."
  2. Scheduler → TaskCreator: "start_query(). Please make a task for the SCAN."
  3. TaskCreator → GPU-0: "Scan says READY. Here's a gpu_pipeline_task for Pipeline A."
  4. GPU-0 → all: "Reserved 512 MB, ran scan→filter→partial count. Batch published to the repo."
  5. Downgrade → all: "GPU 0 at 92% — spilling two idle batches to host. Carry on."
  6. TaskCreator → GPU-0: "Scan drained, last task done → Pipeline A finished. Pipeline B is READY."
  7. GPU-0 → DuckDB: "COUNT finalized into RESULT_COLLECTOR. Future fulfilled — your rows, sir."
- [x] **Code↔English translation** — snippet A (the SiriusContext member list).
- [x] **Quiz** — 3 questions, style: debugging triage.
  1. "Queries stall with GPU idle and no tasks executing; nvidia-smi clean. First console to check?" → task_creator (hint chain deciding nothing is READY), not the executor.
  2. "You need per-query state shared by all operators. Where does it live?" → SiriusContext (owns everything with query lifetime).
  3. "A fix requires blocking inside a GPU worker until network data arrives. Why will reviewers reject it?" → workers are a fixed pool (4/GPU) running tasks to completion; blocking one starves the pipeline — the engine's answer to waiting is 'don't create the task yet' (foreshadow module 7).
- [x] **Glossary tooltips** — SiriusContext, ClientContextState, task_creator, task_scheduler, gpu_pipeline_executor, downgrade executor, memory space, scan manager, shared_data_repository_manager, hint chain.
- [x] **Other** — 7 actor cards with icons; thread-model diagram.

### Reference Files to Read
- `/home/ubuntu/.claude/skills/codebase-to-course/references/interactive-elements.md` → "Group Chat Animation", "Code ↔ English Translation", "Multiple-Choice Quiz", "Pattern Cards", "Glossary Tooltips"
- `/home/ubuntu/.claude/skills/codebase-to-course/references/design-system.md` → tokens + card patterns
- `/home/ubuntu/.claude/skills/codebase-to-course/references/content-philosophy.md` → all (with AUDIENCE override)
- `/home/ubuntu/.claude/skills/codebase-to-course/references/gotchas.md` → all

### Connections
- **Previous module:** "The Big Picture" — what Sirius is, interception, create_plan gate.
- **Next module:** "Life of a Query" — the same actors traced through one query end to end. Hand off with: "Now watch them run one query, step by step."
- **Tone/style notes:** teal accent; real type names in `code` font; PR links https://github.com/sirius-db/sirius/pull/<N>.
