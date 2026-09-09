# Sirius project guide

This is a plain-English guide to the StarRocks integration in Sirius. It explains
what each part does, why the changes exist, and how a query moves between them.
Start with the project overview, then read the topic that matches your work.

Sirius runs SQL operations on GPUs. In this integration, StarRocks plans the
distributed query, a Rust compute node translates and coordinates its pieces,
and the C++ engine executes them. The staging area provides temporary GPU memory
for moving results between compute nodes.

## Start here

1. [Project overview](topics/project-overview.md): follow one query through the system.
2. [Staging area](topics/staging-area.md): understand who owns the transferred bytes.
3. [PR map and landing order](topics/pr-map-and-landing.md): connect the features to their PRs.
4. [Verification and open issues](topics/verification-and-open-issues.md): see what was actually checked.

## Guides by topic, feature and module

| Topic | Features explained | Main module or layer |
| --- | --- | --- |
| [Project overview](topics/project-overview.md) | Query flow and component responsibilities | Whole integration |
| [C++ streaming](topics/cpp-streaming.md) | Streams, source and sink operators, completion, failures and row estimates | C++ execution and pipelines |
| [Rust FFI and DuckDB](topics/rust-ffi-duckdb.md) | Calling C++ from Rust, fragment lifetime, input declarations and transactions | Rust wrappers and C++ planning boundary |
| [Parquet scans and cache](topics/parquet-scans-and-cache.md) | Byte ranges, row groups, pinned files and safe subset reuse | Scans and cache |
| [Plan translation](topics/plan-translation.md) | Expressions, column order, partial aggregates, averages, joins and plan fusion | Rust plan translator |
| [Compute node](topics/compute-node.md) | Requests, scheduling, local results, cancellation and error reporting | Rust compute-node service |
| [Staging area](topics/staging-area.md) | Stable GPU memory, temporary leases and received-batch tickets | C++ allocator and FFI receive store |
| [Exchange transport](topics/exchange-transport.md) | Coordination messages, GPU transfers, connection warmup and retries | Rust PRPC and NIXL transport |
| [Memory and MIG](topics/memory-and-mig.md) | Pool versus arena, host and disk limits, GPU instance placement | Configuration and deployment |
| [Benchmarks and correctness](topics/benchmarks-and-correctness.md) | Query kit, reference answers, cold and warm runs, comparison limits | Shell scripts and Python tools |
| [PR map and landing order](topics/pr-map-and-landing.md) | Original stacks, new drafts, remaining packages and labels | Review workflow |
| [Verification and open issues](topics/verification-and-open-issues.md) | Evidence, reproduced failures and untested paths | Review and validation |
| [Glossary](topics/glossary.md) | Terms used in the guides | Shared vocabulary |

Each module guide explains the purpose, how the main path works, the relevant
features, known limits, and where to read the code. The explanations stand on
their own; the source links let you inspect the details.

## What version this describes

The reviewed source is
[`bench/sf500-2-mig-gpus` at `7610840c`](https://github.com/aocsa/sirius/tree/7610840c03f9086edfa072be72a0eb4c96e03d60)
in `/home/ubuntu/sirius-wt/s500-mig`. The comparison used upstream `dev` at
[`ea1c2783`](https://github.com/sirius-db/sirius/commit/ea1c2783191c0a5a2480665f8a18217dc9d9cba3).
This describes that reviewed version, not every future version of Sirius.

The branch contains 59 non-merge commits and 21 merge commits beyond that
baseline. Nineteen of the 59 are the heads of the original draft PRs. The other
40 are accounted for in the PR map. Some streaming foundations had already
landed in `dev`; the later integration work had not.

PR status was checked again on 9 September 2026. The original 19 PRs and the four
PRs created during this work were still drafts. The saved
[status record](research/markdown-pr-status.json) gives the exact check time and
heads. A draft being present in this branch does not mean it is merged upstream.

## How to read the evidence

- **Checked in code** means the behavior was traced to the reviewed source.
- **Reproduced** means a named local check was actually run and recorded.
- **Historical result** means a checked-in note reports an earlier experiment;
  this review did not rerun it.
- **Proposed** means work is planned or exists in another branch. It is not a
  verified capability of this branch.

Receive credits, some spilling changes, bounded transfer frames and the extra
configuration launcher described in other performance worktrees are separate
work. They must not be taught as already present here.

## Supporting material

The [interactive web guide](index.html) is an optional way to explore the review.
[Web guide details](WEB-GUIDE.md) explain its build and validation. The topic
documents above can be read directly in GitHub or any Markdown reader.

The detailed [review notes](research/pr-inventory.md) and
[prepared PR packages](topics/pr-map-and-landing.md#remaining-core-review-packages)
preserve the evidence and extraction details. The documentation is maintained in
[draft PR #1740](https://github.com/sirius-db/sirius/pull/1740).
