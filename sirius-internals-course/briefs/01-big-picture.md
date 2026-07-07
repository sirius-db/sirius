# Module 1: The Big Picture

**File to write:** `modules/01-big-picture.html` — only a `<section class="module" id="module-1">…</section>` block. Even module → default background; this is module 1 (odd) → add class `module--alt` NO (odd modules use default; even use alt — follow design-system.md).

**AUDIENCE (course-wide override):** The learner is a senior systems engineer (GPU databases, C++, CUDA) joining the Sirius project — NOT a non-technical vibe coder. Do NOT explain SQL, GPUs, kernels, pointers, threads. DO explain every Sirius-specific concept and term. Glossary tooltips go on Sirius/project vocabulary (cuDF, RMM, cuCascade, optimizer extension, pipeline breaker, TPC-H is fine to skip) — not on general CS terms. Tone: sharp colleague giving you the real map, light humor OK, zero condescension.

### Teaching Arc
- **Metaphor:** A simultaneous interpreter in the booth. DuckDB speaks; Sirius translates the query to GPU execution live, mid-sentence. When it hits an idiom it can't translate, it hands the mic back — and the audience (the user) never notices the switch. That's transparent interception + silent CPU fallback.
- **Opening hook:** "You load one extension. Your SQL doesn't change. Supported queries silently run on the GPU. Where exactly does the swap happen?"
- **Key insight:** GPU support is a *plan-level* decision made in one place: `create_plan()` either succeeds (plan is wrapped and swapped) or throws (silent CPU fallback). Everything else follows from that.
- **Why care:** When a query unexpectedly runs on CPU, or you wonder "will my new operator be used?", you now know the single gate to check.

### Content beats (3-4 screens)
1. What Sirius is: GPU-native SQL engine shipped as a DuckDB extension, computing on cuDF/RMM with memory/data primitives from NVIDIA cuCascade. The one-sentence model (use verbatim): *"Sirius intercepts a SQL query, converts DuckDB's plan into its own physical operators, groups them into pipelines, and a task creator turns 'operator + available input' into GPU tasks that a scheduler runs. Data flows between operators through repositories (not function returns), and ports tell each operator where its input lives."*
2. The interception path: optimizer hook copies the optimized logical plan → `OnFinalizePrepare` runs `sirius_physical_plan_generator::create_plan()` → success wraps plan in `PhysicalSiriusExecution`; throw = silent CPU fallback. Code↔English on the snippet below.
3. A short history strip (4 eras, use era cards): 2024 hand-written CUDA "gpu_processing" era → Apr 2025 cuDF pivot (PR #6) → Dec 2025–Feb 2026 "Super Sirius" rewrite on cuCascade (#96 → #134 → #198 → first end-to-end query #206) → Apr 2026 transparent interception (#518). Legacy now quarantined under `src/legacy/` — never modify it.
4. The flow strip: SQL → Interception → Plan conversion → Operators → Pipelines & ports → Task creator → Scheduler, with repositories/memory tiers underneath. (Simple styled diagram — this module's hero visual.)

### Code Snippets (pre-extracted, use EXACTLY as-is)

File: src/transparent/sirius_optimizer_extension.cpp (lines 88-99)
```cpp
  // Copy the optimized plan. OnFinalizePrepare will attempt create_plan() on this
  // copy — that's the single source of truth for GPU support. If the plan contains
  // unsupported operators, create_plan() throws and we fall back to CPU.
  try {
    auto plan_copy = plan->Copy(context);
    ctx->set_captured_logical_plan(std::move(plan_copy));
  } catch (duckdb::NotImplementedException&) {
    // Plan not serializable — skip GPU.
  } catch (std::exception& e) {
    spdlog::debug("Transparent execution: failed to copy logical plan: {}", e.what());
  }
```

### Interactive Elements
- [x] **Code↔English translation** — the snippet above. English side: why the copy, why the comment calls create_plan the single source of truth, what each catch means for the user experience.
- [x] **Quiz** — 3 questions, style: scenario/architecture.
  1. "A teammate's query with a LATERAL join runs correct but slow, and nvidia-smi shows the GPU idle. What happened?" → create_plan() hit an unsupported operator, threw, silent CPU fallback (others: extension not loaded → would affect all queries; a crash → it returned fine).
  2. "You added a new GPU operator class but queries never use it. What's the one place that decides?" → sirius_physical_plan_generator::create_plan().
  3. "Why is the fallback *silent* by design?" → transparency contract: users write plain SQL; correctness always available on CPU; GPU is an optimization, not a requirement.
- [x] **Glossary tooltips** — cuDF, RMM, cuCascade, optimizer extension, OnFinalizePrepare, PhysicalSiriusExecution, gpu_processing (legacy), Super Sirius, transparent interception.
- [x] **Other** — era history cards (4 cards); the architecture flow strip (hero visual).

### Reference Files to Read
- `/home/ubuntu/.claude/skills/codebase-to-course/references/interactive-elements.md` → sections: "Code ↔ English Translation", "Multiple-Choice Quiz", "Callout Boxes", "Glossary Tooltips"
- `/home/ubuntu/.claude/skills/codebase-to-course/references/design-system.md` → skim tokens, module structure, screen layout
- `/home/ubuntu/.claude/skills/codebase-to-course/references/content-philosophy.md` → all (apply with the AUDIENCE override above)
- `/home/ubuntu/.claude/skills/codebase-to-course/references/gotchas.md` → all

### Connections
- **Previous module:** none — this opens the course. Open with the product, not the code.
- **Next module:** "Meet the Actors" — the seven runtime components that execute what this module described. End with: "Next: the seven objects that do all of this."
- **Tone/style notes:** Accent = teal. Course title: "Sirius Internals — Interactive Onboarding". Refer to components by their real type names in `code` font (task_creator, task_scheduler). PR numbers link to https://github.com/sirius-db/sirius/pull/<N>.
