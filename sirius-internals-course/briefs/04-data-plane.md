# Module 4: The Data Plane

**File to write:** `modules/04-data-plane.html` — only a `<section class="module module--alt" id="module-4">…</section>` block (even module → alternate background).

**AUDIENCE (course-wide override):** Senior systems engineer joining Sirius — no general CS/GPU explanations; tooltip Sirius-specific terms only. Sharp-colleague tone.

### Teaching Arc
- **Metaphor:** Freight docks with different unloading rules. Producers drop pallets (batches) at named docks (ports backed by repositories). Some docks release goods pallet-by-pallet as trucks arrive (PIPELINE), some in waves (PARTIAL), some are bonded warehouses that seal until the entire shipment has cleared customs (FULL). Crucially: pallets are tracked by manifest number — moving a pallet between buildings means handing over the manifest, not re-packing the goods.
- **Opening hook:** "In Sirius, no operator ever returns data to its caller. Ever. Here's why that one decision buys spilling, backpressure, and multi-GPU almost for free."
- **Key insight:** Repositories are first-class, inspectable objects — which is exactly what lets the downgrade executor *find* idle batches to spill, lets barriers define pipeline semantics, and lets handles move without copying buffers.
- **Why care:** Anyone adding an operator must decide its ports, barrier types, and batch ownership — get those wrong and you break spilling or correctness, not just performance.

### Content beats (4 screens)
1. The port (snippet A, code↔English): a barrier type + a repo pointer + the producing/consuming pipelines. Note the docstring: repo may be NULL for dependency-only ports (a scheduling edge with no data — a real design artifact from the scan-pipeline split, PR #620).
2. How output leaves an operator (snippet B, code↔English): the default `sink()` — every output batch is pushed to every downstream port. That's the entire "function return" replacement: ~7 lines.
3. Barriers screen: 3 dock cards (PIPELINE / PARTIAL / FULL) with their real uses: filter chains stream (PIPELINE); CONCAT-after-PARTITION consumes incrementally across pipeline boundaries (PARTIAL); hash-join build side seals until built (FULL). Then the drag-and-drop.
4. The batch lifecycle: `data_batch` is a handle; access ONLY via RAII locks — many `read_only` readers XOR one `mutable` holder; `idle` means nobody holds it. State-machine visual: idle ⇄ read_only, idle ⇄ mutable_locked. Callout: **idle + in a registered repository = spillable**. Batches you park in private queues or keep locked are invisible to the downgrade executor — that's how you cause OOMs (and why the streaming design in Module 7 keeps ownership in repositories and passes only `{batch_id, size}` handles through channels). History note: this tri-class design replaced a 4-state FSM (cuCascade PR #117, adopted in Sirius #689) — races became compile errors.

### Code Snippets (pre-extracted, use EXACTLY as-is)

Snippet A — File: src/include/op/sirius_physical_operator.hpp (lines 459-472)
```cpp
  struct port {
    MemoryBarrierType type;
    /// May be NULL for dependency-only ports that carry no data flow (e.g., "dependency").
    /// Null repos are treated as "empty, not data-gating" by the base-class port handling methods
    /// (get_next_task_hint, get_next_task_input_data, all_ports_empty, push_data_batch).
    ::cucascade::shared_data_repository* repo;
    duckdb::shared_ptr<pipeline::sirius_pipeline> src_pipeline;
    duckdb::shared_ptr<pipeline::sirius_pipeline> dest_pipeline;
    //! A UUID for a port on an operator at the beginning of a
    // pipeline. This port receives data from a prior pipeline,
    // forming an incoming edge from that pipeline.
    uuid::UUID source_port_uuid{uuid::now_v7()};
  };
```

Snippet B — File: src/op/sirius_physical_operator.cpp (lines 238-246)
```cpp
void sirius_physical_operator::sink(const operator_data& output_data, rmm::cuda_stream_view stream)
{
  auto& pipelineable_output = dynamic_cast<const pipelineable_operator_data&>(output_data);
  for (auto& batch : pipelineable_output.get_data_batches()) {
    for (auto& next_port_info : next_port_after_sink) {
      next_port_info.next_operator->push_data_batch(next_port_info.next_operator_port_name, batch);
    }
  }
}
```

### Interactive Elements
- [x] **Code↔English translation** — snippets A and B (two blocks).
- [x] **Drag-and-drop** — items: "Hash-join build side", "FILTER feeding a projection in the same pipeline", "CONCAT consuming partitions produced upstream", "Dependency-only scheduling edge (no data)". Targets: FULL, PIPELINE, PARTIAL, "port with repo == NULL".
- [x] **Quiz** — 3 questions, style: architecture/debugging.
  1. "Your new operator buffers output batches in a private std::deque of locked handles while waiting for a slow consumer. Under memory pressure the query OOMs instead of spilling. Why?" → the downgrade executor can only spill idle batches it finds in registered repositories; locked/private batches are unspillable.
  2. "You wire a streaming producer to its consumer with a FULL barrier. What breaks?" → nothing runs until the producer pipeline *completely finishes* — streaming becomes batch; for an unbounded stream, deadlock.
  3. "Two operators need the same batch concurrently — one reading, one wants to convert it to host memory. What happens?" → conversion needs the mutable lock; it must wait until all read_only locks release (locks serialize; idle is the only convertible state).
- [x] **Glossary tooltips** — port, repository, data_batch, idle/read_only/mutable_locked, downgrade, spill, dependency-only port, pipelineable_operator_data, handle, RAII lock (as used here).
- [x] **Other** — batch state-machine diagram (hero visual); 3 barrier dock cards; "aha" callout (idle-in-repo = spillable).

### Reference Files to Read
- `/home/ubuntu/.claude/skills/codebase-to-course/references/interactive-elements.md` → "Drag-and-Drop Quiz", "Code ↔ English Translation", "Multiple-Choice Quiz", "Callout Boxes", "Glossary Tooltips"
- `/home/ubuntu/.claude/skills/codebase-to-course/references/design-system.md` → tokens, card/diagram styles
- `/home/ubuntu/.claude/skills/codebase-to-course/references/content-philosophy.md` → all (AUDIENCE override applies)
- `/home/ubuntu/.claude/skills/codebase-to-course/references/gotchas.md` → all

### Connections
- **Previous module:** "Life of a Query" — the loop that moves batches; this module is what the batches move *through*.
- **Next module:** "Memory & Multi-GPU" — who pays for all these batches: reservations, tiers, spilling, and the device contract. Hand off: "Repositories decide where data *lives*. Next: who pays the memory bill."
- **Tone/style notes:** teal accent; count(*) query is the running example course-wide; PR links https://github.com/sirius-db/sirius/pull/<N> (cuCascade PRs → https://github.com/NVIDIA/cuCascade/pull/<N>).
