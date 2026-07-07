# Sirius onboarding project — session handoff notes

> Working notes from the July 2026 sessions building Alexander's Sirius onboarding
> materials. Read this first when resuming work on the timeline, the onboarding doc,
> or the interactive course. Everything here was fact-checked against the working
> tree as of commit `03be7dd2` on branch `docs-tutorial` (2026-07-07).

## 1. Deliverables (all committed on `docs-tutorial`, pushed to `fork/docs-tutorial`)

Commits: `efb5655a` "phase 1" → `b9c40bc7` "docs phase 1" → `03be7dd2` "Refactor course
structure and content for Sirius Internals" (final state, incl. brief deletions and the
module-6 chip fix). No PR opened yet; `dev` is the PR target when the time comes.

| Artifact | Path | Status |
|---|---|---|
| Interactive PR timeline | `sirius-dev-timeline.html` (repo root) | Done, committed. 219 events, 8 swimlanes by architecture module, clickable PR links, fact-verified against git history. |
| Engine onboarding doc | `experimental/starrocks/docs/sirius-engine-onboarding.md` | Done, committed. Survived 4 Codex review rounds; §5 invariants checklist is the load-bearing part. Cross-linked from `onboarding.md`. |
| Interactive course | `sirius-internals-course/` | Done, committed. 8 modules, educative.io-style layout, light/dark themes, all review findings resolved. |
| Publish staging | `/tmp/sirius-timeline-pub/` | `index.html` landing + timeline + `course/` (not in git — ephemeral). Cloudflare tunnel was **killed**; relaunch mints a new URL. |

### Course structure (codebase-to-course skill conventions)
- `modules/01-big-picture.html` … `08-streaming.html` — section-only fragments (no `<script>`/`<style>`).
- `_base.html` — carries **all** customization (see §4). Never regenerate `styles.css`/`main.js` — they are stock skill assets; `main.js` positional nav-dot mapping means you must not reuse the `.nav-dot` class.
- `build.sh` assembles `index.html` (~300KB). After any module or `_base.html` edit: run `build.sh`, then re-copy `index.html` to `/tmp/sirius-timeline-pub/course/`.
- `briefs/ui-upgrade-spec.md` is the only remaining brief (stale module briefs were deleted because they contained errors that agents "verified" against).

Module map: 01 big picture (interception) · 02 actors + interactive directory-ownership tree · 03 life of a query + architecture explorer (click Hash Join → planner/operator/kernel/tests/bench) · 04 scan & IO subsystem (#675 #731 #740 #871 #997) · 05 pipelines/scheduling · 06 memory/downgrade · 07 design decisions · 08 streaming (#836/#837 hang bugs + fix).

## 2. Verified engine facts (safe to reuse; each survived adversarial review)

- **Interception**: optimizer extension + `OnFinalizePrepare` + a single gate in `create_plan()`; unsupported SQL silently falls back to DuckDB CPU.
- **Completion is edge-triggered**: `update_pipeline_status` is called only from `mark_task_completed` (`sirius_pipeline.cpp:461`) with parent cascade at `:352`.
- **RAII task accounting**: `gpu_pipeline_task` ctor/dtor at `gpu_pipeline_task.cpp:208/:231`; `cpu_source_task.cpp:59/:78` follows the same pattern (scan/source tasks DO participate — earlier claim to the contrary was wrong).
- **Task-creation lock rule** is documented in `sirius_pipeline.hpp:177-182`, used at `task_creator.cpp:253`. Quote the header; do not paraphrase into invented invariant names (a phantom "SCHED-RR contract" claim was a review finding).
- `MAX_OOM_RETRIES = 100` (`gpu_pipeline_executor.cpp:334`) — not 10.
- Base `can_create_more_tasks()` **throws** (`op.hpp:516`). `has_processed_all_tasks` **does not exist**.
- **task_creator hint chain**: READY / WAITING_FOR_INPUT_DATA / nullopt. Executor: 1 manager + 4 pinned workers per GPU.
- **gpu_ingestible**: only parquet and duckdb-native implement it (#871). S3 is a datasource (`src/io/s3/`), not a format.
- **Iceberg GPU path is entirely non-functional**: `iceberg_metadata_reader.cpp`/`iceberg_scan_task.cpp` excluded from CMake; no ICEBERG_SCAN in scan manager or task creator; delete reading stubbed with a warning (`sirius_engine.cpp:320`); planner still *accepts* `iceberg_scan`. Never present Iceberg as active.
- Streaming fix (#836/#837): drained-source-with-no-task case now explicitly calls `update_pipeline_status()`, and `close()` fires a wake; split_connector is close-then-drain.
- Stock `main.js:82-92` already binds all 4 arrow keys — do not add a duplicate keyboard handler.

## 3. Hard-won process rules

1. **Agent briefs must contain verified content.** Two shipped errors (`) ) {` misquote, wrong line ranges) originated in *my* briefs; agents "verified" against the brief, not the source. Verify snippets byte-exact against the working tree before writing them anywhere.
2. **Measure WCAG contrast programmatically** (node script) — never eyeball. `opacity: 0.5` on a label silently halves contrast.
3. Inline `style="color: white"` beats any selector — fix with the CSS-variable fallback pattern `color: var(--fill-ink, white)`.
4. Arrow-key shield: bubble-phase listener on `document.body` (capture phase starves focused controls).
5. `localStorage` writes need a startup probe + per-write try/catch (private-mode Safari throws in scroll callbacks).
6. Chip/regex injection with non-greedy `.*?` + DOTALL crosses panel boundaries when translation blocks are adjacent — this bug misplaced `rm-chip`s twice (modules 6 and 8). Prefer structural (per-panel) edits.
7. A verification grep matching inside the `<style>` block is a false positive — `.translation-label .rm-chip {` is a rule definition, not a chip.
8. The Codex stop-hook review gate is active in this setup: expect factual findings at session end and budget time to fix them.

## 4. `_base.html` customization inventory (do not lose on regeneration)

Teal accent (#2A7B9B family); responsive `--content-width: min(1120px, 94vw)`; translation grid `minmax(0, 1.2fr) minmax(0, 1fr)` stacking ≤1000px; `.repo-tree` interactive tree CSS; `.code-doc` JetBrains reader-mode CSS + `.doc-tag` chips + `.rm-chip`; `.course-toc` sidebar + `.toc-sub` + `.no-spy` fallback (shows all when IntersectionObserver missing); `.lesson-nav`; `.copy-btn` (excludes `.code-doc`, label made static in-flow with `padding-right: 84px`); `overflow-wrap: anywhere` on labels; dark theme block on `html[data-theme="dark"]` (`--color-bg: #262624`, `--color-text-secondary: #C9C6BC`, `--color-text-muted: #ABA79D`, `--fill-ink: #14211C` for dark ink on brightened fills, measured 5.25–8.42:1); pre-paint theme init script; augmentation script (theme toggle, scrollspy, copy buttons, sub-TOC, reading times, arrow shield, resume toast).

## 5. How to publish again

```bash
cd /tmp/sirius-timeline-pub && python3 -m http.server 8791 &
cloudflared tunnel --url http://127.0.0.1:8791   # prints a NEW trycloudflare.com URL
```
Refresh staging first if anything changed: rebuild course, copy `index.html` → `course/`, copy `sirius-dev-timeline.html`.

## 6. Open threads / next-session candidates

- **All deliverables are committed and pushed** (`fork/docs-tutorial`). Remaining git decisions: whether/when to open a PR against `dev`, and whether the timeline belongs at the repo root long-term.
- Optional: `git fetch` and re-mine the timeline to append early-July PRs (#1065 etc.).
- Alexander should visually review both pages in a real browser (this box has none) — especially dark theme and the module-6 reader-mode chip placement.
- His actual work: two PRs touching StarRocks + Sirius streaming (issues #836–#840); the course's Module 8 documents the two hang bugs and the fix — could evolve into PR description material.
- Possible future: wire the course/onboarding doc into `docs/super-sirius/` reading order once content is committed.

## 7. Last completed task (for continuity)

Final Codex finding: a lavender `rm-chip` sat on the light "Plain English" label in `modules/06-memory.html` (regex boundary-crossing bug). Fixed by moving it to the `gpu_pipeline_executor.cpp#L330-L343` code-panel label (line ~166), rebuilt, restaged. Post-fix verification shows 6 real chips, all on dark code panels; the 7th `rm-chip` string in `index.html` is the stylesheet rule. **Fully resolved — no pending fixes.**
