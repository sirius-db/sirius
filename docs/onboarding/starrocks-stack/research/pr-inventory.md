# Live PR inventory and bench coverage

Generated 2026-09-09 from GitHub REST and local Git. The comparison baseline is `upstream/dev` at `ea1c2783`; the examined tip is `bench/sf500-2-mig-gpus` at `7610840c`.

## Status at collection time

GitHub has **19 open PRs by `aocsa`**. They are exactly the 19 supplied PR numbers and every one is a Draft. None has a GitHub label. Their head check runs are successful; skipped CUDA jobs remain skipped. GitHub reports each Draft's mergeable state as `blocked`, so that value must not be read as a merge-conflict result.

| Area | Draft PRs | Dependency chain |
| --- | --- | --- |
| Engine / streaming | #1693 staging arena; #1694 stream cardinality; #1699 watchdog | each targets `dev` |
| Scan | #1696 ownership rule; #1700 ingestible range; #1717 pinned file subset | #1696 → #1700 → #1717 |
| FFI | #1697 transaction scope; #1702 Rust Fragment bindings | #1697 → #1702 |
| Translator | #1704 clone/builtin type fix; #1708 stream read; #1709 two-phase aggregate; #1710 wire order; #1711 avg expansion; #1713 carried slots | #1708 → #1709 → #1710 → #1711 → #1713; #1704 targets `dev` |
| Compute node | #1705 FILES schema; #1706 tunables; #1707 proto patch; #1714 cluster bring-up; #1715 result failure propagation | #1714 → #1715; the first three target `dev` |

The precise API metadata, diff sizes, heads, bases and issue/label context are in [pr-inventory.json](pr-inventory.json). The repository offers useful taxonomy labels (`starrocks`, `rust`, `IO`, `memory`, `multi-gpu`, `benchmarking`, `config`, `perf`, `test`), but the current PRs use none of them. The closest live issue context is #1635 (`enhancement`, `IO`, `starrocks`, `! - P2`) for range splitting and #1590 (`! - P2`) for non-blocking fragment/FFI work.

## Branch coverage

The branch is 80 commits above the live baseline: 21 merge commits and 59 non-merge commits. All 19 PR head commits are ancestors of the branch tip. `git cherry -v upstream/dev HEAD` marks every non-merge commit `+`; no commit is patch-equivalent to current `dev`.

Therefore, **40 non-merge commits are not represented by an open GitHub PR**. GitHub's commit-associated-pull-request endpoint returns no open PR for each. This is an intentionally strict result: integration merge commits are not counted as a feature PR, and a commit merely present in the integration branch is not called landed.

The full machine-readable mapping and dependency recommendations are in [commit-map.json](commit-map.json). It separates three states:

* **Draft:** all 19 GitHub PRs above; their commits are covered but not landed.
* **Integration-only:** the branch combines those drafts through merge commits; it is evidence of compatibility, not a landed state.
* **Uncovered:** 40 direct commits, grouped below.

## Concrete new-Draft candidates

These are the two isolated groups appropriate to open now. GitHub reports no existing open PR for their commits. Their paths do not overlap files changed by any of the 19 aocsa drafts.

| Proposed Draft | Exact commits | Changed surface | Suggested labels | Runtime note |
| --- | --- | --- | --- | --- |
| `chore(bench): add StarRocks TPC-H harness and GPU/MIG placement` | `1e163623`, `41c31ec5`, `7c30489d`, `29df50ef`, `92b7eed4`, `59ab07c3` | benchmark harness, all 22 query SQL, oracle/compare tools, `cluster8.sh`, CN distribution tool | `benchmarking`, `multi-gpu`, `starrocks`, `documentation` | The scripts exercise core exchange work that remains draft; state this clearly in the description. |
| `docs(bench): add SF500 two-MIG execution plan` | `7c18ce8b`, `7092f445`, `7610840c` | `plan-2mig/**` only | `documentation`, `benchmarking`, `multi-gpu`, `config` | Patch is documentation-only; reproducing it needs the harness and transport runtime. |

The first group contains 2,495 additions before later history, across the harness baseline and all-22 expansion; the two placement commits are a focused 1-file continuation. The second group contains 1,011 additions in its own unique directory. They should be opened from a personal fork as self-contained Drafts under the repository's contribution policy; no branch or PR was created by this investigation.

## Remaining uncovered work: landing proposals

Do not create one synthetic integration PR for the remaining core commits. Their dependencies cross the still-draft scan, FFI, translator and CN stacks. Preserve their review units after the prerequisite Drafts land:

| Proposal | Commits | Needs first |
| --- | --- | --- |
| Range crossing | `92e91adf`, `d88265aa` | scan #1696, #1700 and translator base |
| Packed staging FFI | `b7452f67` | #1693, #1694, #1697, #1702 |
| CN exchange transport foundation | `85da8f6f`, `37b48a46`, `86b3f9a6`, `36072ef8`, `9e547c89`, `49d23e8a` | #1707, #1714/#1715, staging FFI, exchange translation |
| Failure/cancellation | `9c002f97`, `5b4ee255`, `37360a1b` | CN foundation for the latter two |
| Deferred fusion | `1289a33c`, `6f87c304`, `a0df9f43`, `1661eb5e`, `f29f96d4`, `73ce2805`, `281b13bc`, `9ec9df31` | exchange translation and CN dispatch/parking |
| Translator/expression completion | `441a03cf`, `72fd14af`, `c39da5eb`, `1e7d6020` | translator base; decimal aggregate rounding follows decimal cast rounding |
| Multicast, receive-pool and repairs | `50f2691d`, `698312a1`, `c91d0a84`, `b00334cb`, `5e71059c`, `98b49df9` | CN foundation; receive-pool also needs packed FFI |
| Dense count join | `4496bfe9` | its integration correction `98b49df9` must be folded or sequenced first |

## Review and merge order

First merge the foundations represented by the current Drafts in this order: #1704, #1696, #1697, #1694, #1705, #1706, #1707, #1693, #1699, #1700, #1702, #1708, #1709, #1710, #1711, #1714, #1715, #1713, #1717. Then materialize the proposals in dependency order from `commit-map.json`, maintaining each CN sequence as a stack and merging bottom-up.

The staging arena is the central memory contract: #1693 allocates a stable CUDA region outside RMM with a coalescing free list for transport registration. The later packed-staging FFI and CN NIXL transport proposals consume that stable-address contract. It should be read before the transport, receive-pool, and migration materials.
