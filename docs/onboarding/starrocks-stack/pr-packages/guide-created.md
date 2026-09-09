## Description

The StarRocks integration spans C++ streaming, Rust FFI, scanning, plan
translation, compute-node execution and GPU exchange memory. New contributors
need a readable map of these modules and the PRs that introduce their contracts.

This adds a plain-English README and 13 topic guides, grouped by feature and
module: project overview, C++ streaming, Rust FFI/DuckDB, Parquet scans/cache,
plan translation, compute node, staging area, exchange transport, memory/MIG,
benchmarks, PR landing, verified findings and a glossary. Each guide explains
its purpose, gives an example, names the relevant code and PRs, and states the
limits of the reviewed version. Staging ownership receives particular detail.

The optional interactive guide provides architecture and staging diagrams,
searchable PR/commit inventories, and a memory calculator. Its native fragment
links support Back/Forward, scroll reset and direct PR/package navigation. It
includes static content for readers without JavaScript and embeds the Markdown
research and topic guides; larger raw developer artifacts link to the repository.
The build uses only Python's standard library.

The source snapshot is bench/sf500-2-mig-gpus at 7610840c, compared with dev at
ea1c2783: 19 original draft heads, 59 non-merge commits and 21 integration merges.
All 40 commits outside those original drafts are accounted for. Ten source
commits were extracted into #1737, #1738 and #1739; the remaining 30 are organized
into 19 proposed core packages. Two mixed repairs have explicit file/function
splits. #1740 is this additional documentation work.

The documents distinguish merged foundations, current draft work, source-confirmed
findings, locally reproduced behavior, historical benchmarks and proposed work
in other branches. They do not present receive-credit or spilling candidates as
already implemented fixes. PR status was checked again on 9 September 2026:
the original 19 and the four review-created PRs were all still drafts.

## Validation

- Checked 147 local Markdown file/heading links and 72 immutable source links
  against Git objects and source line ranges; recorded the results.
- Rechecked the 23 PRs' status and heads and retained the API evidence.
- All applicable pre-commit checks pass for the documentation and build helpers.
- The browser checks cover four widths in light/dark mode and ten navigation
  scenarios, including a lone HTML file, disabled JavaScript and a sandboxed preview.
- The existing comparator probe reproduces the NaN false-MATCH and correctly
  rejects a wrong cold result followed by a correct warm result.

This is documentation and source review. Earlier translator/CN test attempts
stopped before execution because StarRocks Thrift submodule sources were missing.
No new GPU, NIXL or SF500 campaign is claimed. The source-confirmed lease leak
and InboundStore lifetime race remain unresolved engine findings; this PR does
not fix them or the comparator. Keep Draft for component-owner factual review.

## Checklist

- [x] Follow CONTRIBUTING.md and the self-contained fork-to-dev path.
- [x] Provide a linked README and plain-English module/feature documents.
- [x] Keep source version, PR status and validation limits explicit.
- [x] Validate links, formatting, the document build and navigation.
- [ ] Component owners review the technical map and proposed package boundaries.

## References

- Source branch: https://github.com/aocsa/sirius/tree/7610840c03f9086edfa072be72a0eb4c96e03d60
- Staging allocator: https://github.com/sirius-db/sirius/pull/1693
- Runbook: https://github.com/sirius-db/sirius/pull/1737
- Benchmark harness: https://github.com/sirius-db/sirius/pull/1738
- Concurrent PRPC: https://github.com/sirius-db/sirius/pull/1739
