# Shared spec: interactive diagram language (UI/UX upgrade pass)

All diagram work in this pass follows one visual/interaction language so the course reads as
one system. Apply within existing constraints: **module files contain no `<script>` or
`<style>` tags; never touch styles.css or main.js.** Interactivity = native HTML
(`<details>`, anchors), CSS via inline styles + existing design-system classes, and the
existing `.term` tooltip machinery (main.js wires it).

## Zones (execution-domain shading)
Wrap diagram regions in zone containers with a small uppercase caption (11px, letter-spacing
0.06em, color var(--color-text-muted)):
- **CPU zone** — background `var(--color-bg-warm)`, 1px solid var(--color-border) border, radius 12px.
- **GPU zone** — background `var(--color-accent-light)`, 1px solid var(--color-accent-muted) border, radius 12px.
- **Storage / IO zone** — background transparent, 1px dashed var(--color-border) border, radius 12px.
Check design-system.md for the actual token names available (e.g., if `--color-bg-warm` or
`--color-border` differ, use the documented equivalents; verify against styles.css with grep).

## Nodes & arrows
- Node = rounded box (radius 8px, padding 8-12px, background var(--color-bg) or white-equivalent
  token, subtle shadow if a design-system class provides one). Node title in body font 600;
  a second line in 12px muted showing the owning path in `code` font.
- Arrows: use flex rows with `→` / `↓` glyphs in var(--color-text-muted) — never images.
- **Control plane vs data plane:** control arrows are plain `→`; data movement is a
  labeled chip (e.g. `⬤ batch handle`) in the accent color. Where relevant annotate:
  "data never travels by function return — it goes through repositories".

## Semantics badges (consistent across all modules)
- Blocking / pipeline-breaker operator: badge `⛔ breaker` (accent-colored border, not red).
- Streaming operator/edge: badge `≋ streams`.
- FULL barrier: a visible "gate" element between pipelines: `▮▮ FULL barrier — opens when
  producer pipeline finishes` .
- Sync point (CPU↔GPU or thread join): badge `⏚ sync` with a `.term` tooltip explaining what
  waits on what.

## Hover = ownership
Every diagram node carries a `.term` span (existing tooltip machinery) whose
`data-definition` states: what the node is, which thread runs it, which directory owns it.
This is the "hoverable execution pipeline showing CPU/GPU ownership" requirement.

## Expandable = `<details>`
Click-to-expand cards use native `<details class="explorer-card">`/`<summary>`:
- `<details>` inline style: `border:1px solid var(--color-border); border-radius:12px;
  padding:0; margin:10px 0; background:var(--color-bg);` (adapt token names to design-system).
- `<summary>` inline style: `cursor:pointer; padding:14px 18px; font-weight:600;
  list-style:none; display:flex; align-items:center; gap:10px;` and include a muted
  `▸ expand` affordance chip at the right (CSS can't rotate it without a stylesheet — a
  static `▸ details` chip is fine).
- Body: `padding:0 18px 14px;`.
`<details>`/`<summary>` are keyboard-accessible natively — do not replace them with divs.

## Clickable source paths
- Every file/dir reference in a diagram or explorer is a real link:
  files → `https://github.com/sirius-db/sirius/blob/dev/<path>`,
  directories → `https://github.com/sirius-db/sirius/tree/dev/<path>`.
- **VERIFY every path before linking** (ls/glob in /home/ubuntu/git/sirius-db/sirius). A
  broken deep link is worse than no link. If a path doesn't exist, name the concept without
  a link.
- Render paths in `code` font; the link carries a `↗` suffix.

## Explorer vertical-slice card (the "click Hash Join" pattern)
A `<details>` card whose summary is the component name + one-line role, and whose body is a
compact ladder (one row per layer, each row = layer label chip + file link + one-line note):
`planner node → physical operator → GPU work (cudf calls) → memory notes → tests → run command`.
Run commands come from CLAUDE.md, verbatim:
- one sqllogic file: `pixi run build/release/test/unittest --test-dir . test/sql/tpch-sirius.test`
- by Catch2 tag: `pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"`

## Sanity checklist before finishing any module edit
- No `<script>`, no `<style>`, single `<section>` root preserved, section id unchanged.
- All `data-steps` JSON still parses if touched.
- All links verified against the working tree.
- Tags balanced (open == close for div/details/summary).
