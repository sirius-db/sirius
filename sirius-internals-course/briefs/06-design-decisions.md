# Module 6: Design Decisions

**File to write:** `modules/06-design-decisions.html` — only a `<section class="module module--alt" id="module-6">…</section>` block (even module → alternate background).

**AUDIENCE (course-wide override):** Senior systems engineer joining Sirius — no general CS/GPU explanations; tooltip Sirius-specific terms only. Sharp-colleague tone.

### Teaching Arc
- **Metaphor:** Reading a tree's growth rings. Each ring is a season's pressure made permanent — drought years, growth spurts, a lightning scar. The engine's design is the same: every "weird" rule is a fossilized incident. Read the rings and you inherit the judgment without living the incidents.
- **Opening hook:** "Half of Sirius's design looks over-engineered until you learn which outage or rewrite produced it. Here are the receipts."
- **Key insight:** Two meta-trends explain almost every decision: (1) Sirius keeps *decoupling from DuckDB* (types → expressions → execution state → FFI toward a standalone libsirius), and (2) *cuCascade absorbs anything engine-agnostic* (memory, batches, repositories).
- **Why care:** New code that swims against these two currents gets rewritten. Knowing them = making PRs that merge.

### Content beats (4 screens)
1. The two currents (hero visual: two-lane diagram with milestones on each lane). Lane 1 "away from DuckDB": native type system #643 → sirius::ast #796/#847 → PIMPL retired #880 → Rust FFI #908 → engine context inside a StarRocks CN #960. Lane 2 "into cuCascade": extraction Dec 2025 → submodule #144 → partitioned repositories cc#26 → tri-class batches cc#117 → cudf-free core cc#150.
2. Decision cards (8 pattern cards; each = decision / why / PR receipt):
   - Extension, not standalone engine — free parser/optimizer/catalog; coupling now being paid back deliberately (inception 2024; #643, #880, #908).
   - cuDF over hand-written kernels — every operator got cheaper to build; legacy kernels deleted Jun 2025 (#6).
   - Total rewrite into task-based pipelines ("Super Sirius") — the old gpu_executor was single-threaded, no overlap, no tiering; built alongside, then cut over (#96, #198, #206).
   - Repositories + ports over function returns — spillability, barriers, inspectability (cc#26, #519, #689).
   - Plan-time wiring descriptors split from runtime materialization — engine-free planning, testable construction (#607, #770).
   - Transparent interception via optimizer extension — GPU support became a plan-generator decision, invisible to users, safe-by-default fallback (#518, #673).
   - Downgrade as an executor, not an allocator callback — spilling is scheduled work with its own threads and priorities (#97, #368, #579, #647).
   - Multi-GPU locality-first + reservation-device authority — and the #732→#827 backpressure lesson (#732, #827, #996).
3. Code↔English on snippet A — a *design scar in the wild*: the sanity check at the top of `mark_task_completed` exists because pipelines were once declared finished while tasks were still in flight (the query-end SEGFAULT saga, #766/#788/#804). The code now audits its own invariant and logs loudly.
4. Want the full history? Card linking to the interactive PR timeline (`../sirius-dev-timeline.html`, note: relative link — also say "at the repo root: sirius-dev-timeline.html") and `docs/super-sirius/` as the reference docs.

### Code Snippets (pre-extracted, use EXACTLY as-is)

Snippet A — File: src/pipeline/sirius_pipeline.cpp (lines 427-436, top of mark_task_completed)
```cpp
void sirius_pipeline::mark_task_completed()
{
  // Sanity check: a task is completing here, which means this pipeline was still
  // running work. If the pipeline was already marked finished, or any of its
  // operators were already finalized, then we declared the pipeline done too
  // early — i.e. a task was still in flight when we considered the pipeline
  // complete. That mismatch can make the whole query look done while work is
  // still running, so surface it loudly.
  const bool pipeline_was_finished = pipeline_finished.load();
  std::string finalized_ops;
```

### Interactive Elements
- [x] **Code↔English translation** — snippet A, framed as "archaeology: reading a bug's fossil".
- [x] **Quiz** — 3 questions, style: architecture decisions.
  1. "You're adding an operator that must hand result batches to an external process. Which two standing decisions dictate where those batches live and how they're referenced?" → repositories keep ownership (spillability) + handles/ids cross boundaries, never raw buffers.
  2. "You want to teach the planner a new DuckDB expression type. Which direction does the codebase want you to go?" → translate into sirius::ast (native AST is the lowering input since #847) — don't route raw DuckDB expressions deeper.
  3. "A reviewer asks why your pipeline-construction change takes a `pipeline_build_context` instead of `sirius_engine&`. What's the answer?" → #607/#770: plan-time construction must stay engine-free (descriptors now, materialization later) so planning can move earlier and stay testable.
- [x] **Glossary tooltips** — sirius::ast, PIMPL, FFI, libsirius, repository wiring descriptor, materialization (of wiring), pipeline_build_context, cuco, StarRocks CN (one-line: the compute-node embedding project).
- [x] **Other** — two-lane meta-trend diagram (hero); 8 decision pattern cards; link card to the PR timeline + docs.

### Reference Files to Read
- `/home/ubuntu/.claude/skills/codebase-to-course/references/interactive-elements.md` → "Pattern Cards", "Code ↔ English Translation", "Multiple-Choice Quiz", "Callout Boxes", "Glossary Tooltips"
- `/home/ubuntu/.claude/skills/codebase-to-course/references/design-system.md` → tokens, card styles
- `/home/ubuntu/.claude/skills/codebase-to-course/references/content-philosophy.md` → all (AUDIENCE override applies)
- `/home/ubuntu/.claude/skills/codebase-to-course/references/gotchas.md` → all

### Connections
- **Previous module:** "Memory & Multi-GPU" — the rules; this module is where the rules came from.
- **Next module:** "The Streaming Frontier" — the learner's own work (#836/#837) as the newest ring on the tree. Hand off: "The newest growth ring is being laid down right now — and you're the one drawing it."
- **Tone/style notes:** teal accent; PR links https://github.com/sirius-db/sirius/pull/<N>, cuCascade cc#N → https://github.com/NVIDIA/cuCascade/pull/<N>.
