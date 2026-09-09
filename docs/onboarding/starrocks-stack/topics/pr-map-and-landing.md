# PR map and landing order

[Back to the guide](../README.md)

**Topic:** review workflow. **Features:** feature ownership and dependencies.
**Modules:** all modules in this integration. Source scope: `7610840c`, compared
with `dev` at `ea1c2783`. PR status was rechecked on 9 September 2026 and saved
in the [status record](../research/markdown-pr-status.json).

## What the status words mean

A pull request, or PR, proposes a change for review. A draft is still being
prepared; its presence in this branch does not mean it has reached `dev`.
A stacked PR depends on another PR below it. A proposed package in this guide is
a suggested review unit with named commits; it is not automatically an opened PR.

The 80 branch-only commits consist of 59 code/test/documentation commits and 21
merges. The original draft heads account for 19 of the 59. Ten more source commits
were extracted into #1737, #1738 and #1739. The remaining 30 unique source commits
are described by the 19 proposed core packages below. #1740 contains this new
documentation and is separate from that original source-commit count.

## Foundations already merged

These C++ streaming features were already merged upstream before the reviewed
integration work. Read [C++ streaming](cpp-streaming.md) for their behavior.

| PR | What it establishes |
| --- | --- |
| [#1094](https://github.com/sirius-db/sirius/pull/1094) | Initial streaming source and exchange channel |
| [#1320](https://github.com/sirius-db/sirius/pull/1320) | Repository-backed source, sender EOS and producer errors |
| [#1479](https://github.com/sirius-db/sirius/pull/1479) | Partitioned streaming sink |
| [#1480](https://github.com/sirius-db/sirius/pull/1480) | ID-addressed stream_session router |
| [#1481](https://github.com/sirius-db/sirius/pull/1481) | Fragment builder, runner and C++ FFI |

## The original 19 drafts

All 19 were still drafts at the status check. Each is present in the reviewed
branch. This table describes the feature, not a claim that its combined GPU path
has passed a new test run.

| Topic | PR | Main feature |
| --- | --- | --- |
| Engine & staging | [#1693](https://github.com/sirius-db/sirius/pull/1693) | Stable exchange staging arena |
| Engine & staging | [#1694](https://github.com/sirius-db/sirius/pull/1694) | Stream input cardinality |
| Scan | [#1696](https://github.com/sirius-db/sirius/pull/1696) | Byte-range ownership rule |
| FFI | [#1697](https://github.com/sirius-db/sirius/pull/1697) | Transaction-owned Substrait lowering |
| Engine & staging | [#1699](https://github.com/sirius-db/sirius/pull/1699) | No-progress query watchdog |
| Scan | [#1700](https://github.com/sirius-db/sirius/pull/1700) | Range-aware parquet ingest |
| FFI | [#1702](https://github.com/sirius-db/sirius/pull/1702) | Safe Rust Context / Fragment bindings |
| Translator | [#1704](https://github.com/sirius-db/sirius/pull/1704) | CLONE_EXPR and narrowed builtin casts |
| Compute node | [#1705](https://github.com/sirius-db/sirius/pull/1705) | FILES schema across all ranges |
| Compute node | [#1706](https://github.com/sirius-db/sirius/pull/1706) | Validated transport tunables |
| Compute node | [#1707](https://github.com/sirius-db/sirius/pull/1707) | Versioned exchange RPC patch |
| Translator | [#1708](https://github.com/sirius-db/sirius/pull/1708) | EXCHANGE_NODE as a stream read |
| Translator | [#1709](https://github.com/sirius-db/sirius/pull/1709) | Two-phase aggregate model |
| Translator | [#1710](https://github.com/sirius-db/sirius/pull/1710) | Materialized-slot wire order |
| Translator | [#1711](https://github.com/sirius-db/sirius/pull/1711) | Two-phase AVG expansion |
| Translator | [#1713](https://github.com/sirius-db/sirius/pull/1713) | Carry consumed common slots |
| Compute node | [#1714](https://github.com/sirius-db/sirius/pull/1714) | Per-GPU CN bring-up |
| Compute node | [#1715](https://github.com/sirius-db/sirius/pull/1715) | Failure propagation to polled results |
| Scan | [#1717](https://github.com/sirius-db/sirius/pull/1717) | Pinned file-subset serving |

The actual branch dependencies form these chains:

- **scan:** [#1696](https://github.com/sirius-db/sirius/pull/1696) → [#1700](https://github.com/sirius-db/sirius/pull/1700) → [#1717](https://github.com/sirius-db/sirius/pull/1717).
- **ffi:** [#1697](https://github.com/sirius-db/sirius/pull/1697) → [#1702](https://github.com/sirius-db/sirius/pull/1702).
- **translator:** [#1708](https://github.com/sirius-db/sirius/pull/1708) → [#1709](https://github.com/sirius-db/sirius/pull/1709) → [#1710](https://github.com/sirius-db/sirius/pull/1710) → [#1711](https://github.com/sirius-db/sirius/pull/1711) → [#1713](https://github.com/sirius-db/sirius/pull/1713).
- **compute-node:** [#1714](https://github.com/sirius-db/sirius/pull/1714) → [#1715](https://github.com/sirius-db/sirius/pull/1715).

Other relationships cross those chains. For example, packed GPU transport needs
both the staging allocator and the Rust/C++ interface. Review the allocator and
column-layout contracts before reviewing their transport consumers.

## Drafts created from the investigation

| PR | Topic | Source commits from the reviewed branch |
| --- | --- | --- |
| [#1737](https://github.com/sirius-db/sirius/pull/1737) | add the SF500 two-MIG memory and reproduction runbook | `7c18ce8b`, `7092f445`, `7610840c` |
| [#1738](https://github.com/sirius-db/sirius/pull/1738) | add TPC-H harness and GPU/MIG placement | `1e163623`, `41c31ec5`, `7c30489d`, `29df50ef`, `92b7eed4`, `59ab07c3` |
| [#1739](https://github.com/sirius-db/sirius/pull/1739) | serve concurrent PRPC requests on one connection | `5e71059c` |
| [#1740](https://github.com/sirius-db/sirius/pull/1740) | Plain-English and interactive onboarding documentation | New documentation work, outside the original 59-commit count |

These PRs remain drafts. Their published commits have different IDs because the
source changes were cherry-picked onto a clean base. The table lists the original
branch IDs so a reader can trace where the changes came from.

## Remaining core review packages

P01-P19 are proposals with prepared descriptions. P20 has already been opened as
#1739 and is shown for completeness. Follow a package link for exact file lists,
prerequisites and the tests needed before it is ready.

| Package | Feature | Source commits |
| --- | --- | --- |
| [P01](../pr-packages/p01.md) | carry byte ranges through Substrait into GPU scans | [`92e91adf`](https://github.com/aocsa/sirius/commit/92e91adf95d951568b060e1947c43e89baf5d7f3), [`d88265aa`](https://github.com/aocsa/sirius/commit/d88265aa3c73cfa3e1af7fe1658511ef5ca4ba3a) |
| [P02](../pr-packages/p02.md) | expose staging leases and packed exchange batches | [`b7452f67`](https://github.com/aocsa/sirius/commit/b7452f67c57279b0ad5c51a40a9f790477e6ba97) |
| [P03](../pr-packages/p03.md) | park output and rendezvous before receiver dispatch | [`85da8f6f`](https://github.com/aocsa/sirius/commit/85da8f6ff2ca0159984a2b4c3d5340a59642cb9b), [`37b48a46`](https://github.com/aocsa/sirius/commit/37b48a46afb2911f6cea3b3e6711416cdfc8e573) |
| [P04](../pr-packages/p04.md) | add the blocking peer PRPC client | [`86b3f9a6`](https://github.com/aocsa/sirius/commit/86b3f9a642af3ce6c5a36242ef0219f82563aa1d) |
| [P05](../pr-packages/p05.md) | register a NIXL arena and validate the link | [`36072ef8`](https://github.com/aocsa/sirius/commit/36072ef862e6c044bebab123d3ee2e204759afe6) |
| [P06](../pr-packages/p06.md) | transfer remote batches and warm peer sessions | [`9e547c89`](https://github.com/aocsa/sirius/commit/9e547c89a7ba1ee3e3fbf21a95167d0da38f9c8f), [`49d23e8a`](https://github.com/aocsa/sirius/commit/49d23e8acae38fe3d8cec7361c12d86dc809421d) |
| [P07](../pr-packages/p07.md) | stop futile OOM reschedules | [`9c002f97`](https://github.com/aocsa/sirius/commit/9c002f976665ff1f68d8c4b851adf9ee9fe257a6) |
| [P08](../pr-packages/p08.md) | retire failed and cancelled queries precisely | [`5b4ee255`](https://github.com/aocsa/sirius/commit/5b4ee255dbda657e6161db6822fcea9bd2637f6e), [`37360a1b`](https://github.com/aocsa/sirius/commit/37360a1b6b11741cbe7a672975c6fc11b91dc9c9), [`a0df9f43`](https://github.com/aocsa/sirius/commit/a0df9f4348cbaca4ba4475501089a677f9d8aca2) |
| [P09](../pr-packages/p09.md) | splice deferred leaf plans over exchanges | [`1289a33c`](https://github.com/aocsa/sirius/commit/1289a33c21a25aa1341c1f0c0e534dc532d6f0b2), [`f29f96d4`](https://github.com/aocsa/sirius/commit/f29f96d49dd312a21eeb06628cb29dfc13542874) |
| [P10](../pr-packages/p10.md) | fuse eligible local leaf senders | [`6f87c304`](https://github.com/aocsa/sirius/commit/6f87c30471cc5f5f17640da86613a6b04a8abf4c), [`1661eb5e`](https://github.com/aocsa/sirius/commit/1661eb5e4ac1cfe43733cfc1cfe497fba2984818), [`73ce2805`](https://github.com/aocsa/sirius/commit/73ce28054e5c7a6a3b4c5e985d459ea23cb221c2), [`281b13bc`](https://github.com/aocsa/sirius/commit/281b13bcb12321bac2927a8f4f996b710a463ec1) |
| [P11](../pr-packages/p11.md) | opt in to asynchronous sender-only dispatch | [`9ec9df31`](https://github.com/aocsa/sirius/commit/9ec9df314fa6091db47bf10e5e04715a5db0501f) |
| [P12](../pr-packages/p12.md) | lower RIGHT_SEMI_JOIN | [`441a03cf`](https://github.com/aocsa/sirius/commit/441a03cff9ebbaeb361f536eada1036dd4a3c29e) |
| [P13](../pr-packages/p13.md) | carry two-phase distinct-count state | [`72fd14af`](https://github.com/aocsa/sirius/commit/72fd14af0298fb93eb60561102d403d11d02c242) |
| [P14](../pr-packages/p14.md) | round floating decimal casts consistently | [`c39da5eb`](https://github.com/aocsa/sirius/commit/c39da5eb853f54755af81b420f50560200e4c848) |
| [P15](../pr-packages/p15.md) | round finalized decimal aggregates to scale | [`1e7d6020`](https://github.com/aocsa/sirius/commit/1e7d60209e393cc91198466884adbc1ced1148c9) |
| [P16](../pr-packages/p16.md) | copy received frames into pool-backed tickets | [`698312a1`](https://github.com/aocsa/sirius/commit/698312a111d3202f1dfda91e08a39b148e77fde1), [`b00334cb`](https://github.com/aocsa/sirius/commit/b00334cb55f257358aa94d34de1250235b0d7ab2) |
| [P17](../pr-packages/p17.md) | run shared CTE producers once with multicast sinks | [`50f2691d`](https://github.com/aocsa/sirius/commit/50f2691db947d0105b9eda333346f6ae677c909d), [`b00334cb`](https://github.com/aocsa/sirius/commit/b00334cb55f257358aa94d34de1250235b0d7ab2), [`98b49df9`](https://github.com/aocsa/sirius/commit/98b49df9bf312a5fac217087d69809aca5cbbf33) |
| [P18](../pr-packages/p18.md) | recognize dense-count joins through projections | [`4496bfe9`](https://github.com/aocsa/sirius/commit/4496bfe9c078576d94391785b7aeca51ceea5e1a), [`98b49df9`](https://github.com/aocsa/sirius/commit/98b49df9bf312a5fac217087d69809aca5cbbf33) |
| [P19](../pr-packages/p19.md) | report fragment failures to the FE coordinator | [`c91d0a84`](https://github.com/aocsa/sirius/commit/c91d0a84539a4f296b0733684c88fbf8381bdfc0) |
| [P20](../pr-packages/p20.md) / [#1739](https://github.com/sirius-db/sirius/pull/1739) | serve concurrent PRPC requests on one connection | [`5e71059c`](https://github.com/aocsa/sirius/commit/5e71059cf0817285dfe01dcc3f77c62f10055c1b) |

Two commits contain repairs for more than one feature. `b00334cb` has receive and
multicast changes, so its parts belong to P16 and P17. `98b49df9` has translator
and C++ planner changes, so its parts belong to P17 and P18. They are counted only
once in the total. Copying either whole commit into both PRs would duplicate work;
the package documents give the intended file/function split.

## A practical landing sequence

1. Review the independent roots: expression types, scan ownership, transaction
   scope, row estimates, schema inference, configuration and the RPC patch.
2. Finish the allocator, range-aware scan and Rust fragment interface contracts.
3. Follow the translator stack through exchange reads, aggregate states, column
   order and average expansion.
4. Finish CN bring-up, result failure propagation, common-expression slots and
   pinned file-subset serving.
5. Extract the next packages on their real prerequisites. Keep related repairs
   and tests with each feature, especially staging lifetime and retry handling.

The repository uses `gh-stack` for dependent stacks and `stacked/` branch names.
It lands the bottom PR through an individual merge-queue entry, then advances to
the next PR. Do not use "Enqueue stack" or `gh stack merge`. Self-contained changes
come from a contributor fork and target `dev`. These are repository instructions,
not a new workflow invented for this guide; see
[CONTRIBUTING.md](https://github.com/sirius-db/sirius/blob/ea1c2783191c0a5a2480665f8a18217dc9d9cba3/CONTRIBUTING.md#merging-a-stack).

## Labels help readers find the right module

The repository already has `starrocks`, `rust`, `IO`, `memory`, `multi-gpu`,
`benchmarking`, `config`, `perf`, `test` and `documentation` labels. The original
19 drafts were unlabeled at the check. The new drafts use existing labels.
Suggestions in the prepared package descriptions do not mean those labels have
already been applied on GitHub.

Before making a draft ready, resolve its important findings and run its relevant
tests on the actual extracted head. Successful checks on an earlier PR or a
related worktree are not proof for that new combination. Read
[verification and open issues](verification-and-open-issues.md) for the outstanding
staging and comparator findings.
