# Phase 2: Documentation - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Update BUILD_DEPLOY_TEST_GUIDE.md to reflect aarch64 as a supported platform. The build steps are identical on both architectures (Phase 1 makes them architecture-transparent), so the documentation changes are targeted edits to existing sections, not a rewrite.

</domain>

<decisions>
## Implementation Decisions

### Documentation Style
- **D-01:** Use inline mentions throughout existing sections — update platform references to include aarch64, add brief notes where aarch64 differs. No callout boxes, no separate appendix.
- **D-02:** Add a brief note in the Build Steps intro stating that all build steps are architecture-transparent (automatic detection, no user action needed).

### Hardware Guidance
- **D-03:** Document aarch64 GPU requirement as "NVIDIA SBSA-compliant GPU (Grace Hopper, Grace Blackwell, etc.)" — use the SBSA spec name for future-proofing, not specific product names.
- **D-04:** Explicitly note that Tegra/Jetson is NOT supported (different CUDA target directory).
- **D-05:** CUDA driver version requirement is the same for both architectures — no aarch64-specific driver note needed.

### Scope of Edits
- **D-06:** Targeted edits only — approximately 5-10 lines changed across 2-3 sections. No full rewrite of the guide.
- **D-07:** Sections to edit: (1) Software subsection in Build Environment — update platform line, (2) Hardware subsection — add SBSA GPU note for aarch64, (3) Build Steps intro — add arch-transparency note.
- **D-08:** Sections that need NO changes: Running the Cluster, Configuration Tuning, Performance Tests, Troubleshooting, Quick Start Checklist — the commands are identical on both platforms.

### Claude's Discretion
- Exact wording of the architecture-transparency note
- Whether to add "aarch64" to the Quick Start Checklist comment header (cosmetic)
- Formatting of SBSA GPU mention (parenthetical vs separate bullet)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Build guide (the file being edited)
- `doris/BUILD_DEPLOY_TEST_GUIDE.md` — Current build guide; Section 1 (Build Environment Requirements) is the primary edit target

### Project constraints
- `.planning/PROJECT.md` — Key Decisions table (sbsa-linux, platform-conditional sysroot)
- `.planning/REQUIREMENTS.md` — DOCS-01 acceptance criteria

### Phase 1 context (what changed in the build system)
- `.planning/phases/01-build-and-runtime/01-CONTEXT.md` — All build system changes that make aarch64 work (D-01 through D-09)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- BUILD_DEPLOY_TEST_GUIDE.md is well-structured with clear section hierarchy — edits fit naturally into existing sections
- pixi.toml already declares `platforms = ["linux-64", "linux-aarch64"]` — consistent with documenting both platforms

### Established Patterns
- Guide uses markdown tables for hardware requirements and CLI flags — aarch64 GPU info fits the existing table format
- Section headers follow a numbered hierarchy (1-7) — no new sections needed

### Integration Points
- Hardware section (line 18-20): Currently says "Linux x86_64" — primary edit point
- GPU requirement line (line 18): Lists x86_64 GPU families — needs aarch64 SBSA addition
- Build Steps intro (line 48): Mentions pixi managing dependencies — good place for arch-transparency note

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-documentation*
*Context gathered: 2026-04-06*
