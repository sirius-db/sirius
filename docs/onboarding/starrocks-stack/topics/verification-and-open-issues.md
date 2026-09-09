# What was verified, and what remains open

[Back to the guide](../README.md)

**Topic:** evidence and limits. **Features:** validation and review findings.
**Modules:** staging, FFI, scanning, benchmarks and the documentation.
The code findings refer to the reviewed source at `7610840c`.

## Verified does not always mean tested on a GPU

This guide uses several kinds of evidence. They answer different questions.

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| Reading the exact source revision | Which paths, types, ownership rules and checks exist | That every runtime path is correct |
| Git history and GitHub metadata | Which commits belong to a branch or PR, and the PR's recorded status | That combining the PRs is ready to merge |
| A local reproduction | That a specific input produces a specific result | That unrelated cases also pass |
| A checked-in experiment note | What the earlier experiment reports | A fresh result from this review |
| A proposed implementation in another branch | A possible direction for later work | A verified fix in this branch |

Source links in the topic guides point to the fixed revision. The refreshed
[PR status record](../research/markdown-pr-status.json) gives the time of the
GitHub check. The original [PR snapshot](../research/github-pr-snapshot.json)
also preserves the earlier checks and file lists.

## Checks that were actually performed

The review accounted for all 59 non-merge commits and all 21 merge commits beyond
the comparison baseline. Every original draft head was found in the branch. The
PR map records the remaining source commits and the two repairs that need to be
split across feature packages.

The benchmark extraction passed shell syntax and Python compilation checks. A
temporary fixture confirmed that a wrong cold result causes failure even when a
later warm result matches. A second fixture showed that NaN incorrectly passes
against a finite reference. The
[reproduction script and results](../research/benchmark-validation.md) identify
the actual checks.

The web page was checked for layout and interaction in light and dark themes.
Its navigation tests cover history, deep links, copied HTML, disabled JavaScript
and a sandboxed preview. These tests check the document, not the SQL engine.

## Important findings

### P1: a failed transfer can leave memory allocated on the receiver

The sender first requests a staging lease from the receiving CN. If the GPU
WRITE fails before the sender announces the frame, the sender releases only its
own local lease. The receiving CN has no query-associated frame record through
which cancellation can find that remote lease. Repeated failures can consume the
receiver's arena.

This is a source-confirmed error path in
[nixl_transport.rs](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/nixl_transport.rs#L688).
It was not reproduced by injecting a GPU transfer failure. A fix must safely
stop any outstanding transfer, release the remote allocation exactly once, and
prove that its capacity can be reused. This belongs with [P06](../pr-packages/p06.md).

### P1: a received-batch copy can race with Context destruction

An `InboundStore` handle can be used from another thread and can outlive its
Context. Its `stage()` operation copies a raw GPU memory-space pointer while
holding a lock, then uses that pointer after releasing the lock. Context
destruction can happen between those steps. Checking the pointer again after
the copy does not protect the earlier use.

The race follows from the
[pointer snapshot and copy](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L435)
and [Context destruction](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L316).
It is independently confirmed in code, not a sanitizer reproduction. Destruction
must wait for active copies or retain the underlying owner until they finish.
This belongs with [P16](../pr-packages/p16.md).

### P1: NaN can be reported as a correct benchmark answer

The real comparator reported `MATCH` and exited successfully for a NaN result
against a `1.0` reference. This one was reproduced. The comparison needs an
explicit rule for non-finite numbers before
[#1738](https://github.com/sirius-db/sirius/pull/1738) is a reliable correctness
check. See [benchmarks and correctness](benchmarks-and-correctness.md).

### P2: incoming data can fill the GPU pool

Copying data out of the staging arena frees a transfer buffer, but the copied
data remains in the GPU pool. The receiver waits for its senders to finish, so
these copies can accumulate. The current receive path has no receive-credit
budget or connection to spillable reservations for those waiting tickets.

The [MIG checklist](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/plan-2mig/CHECKLIST-2mig.md#L21)
records this type of failure in earlier runs. This is a capacity limitation and
a production-readiness concern; it is not by itself a memory-safety defect.
Other branches contain possible admission and spilling work, which still needs
review and testing together. Read [staging area](staging-area.md).

### P2: a malformed byte range can overflow

The ownership helper computes `start + length` without checking for unsigned
overflow. The arithmetic example `start = 4`, `length = UINT64_MAX` wraps the end
to `3`, which can produce an empty selection instead of rejecting invalid input.
This arithmetic was checked, but normal frontend ranges were not shown to cause
it. The fix is input validation in the
[range path](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/op/scan/parquet_byte_range.cpp#L56),
with a test alongside the ordinary boundary cases.

## What was not completed by this review

Translator and pure-Rust CN tests were attempted, but stopped before test
execution because the StarRocks `gensrc/thrift` submodule sources were missing.
The review did not run a new C++ GPU, NIXL or SF500 benchmark campaign.

Existing tests and historical benchmark notes remain useful evidence, but they
are not new passing results for these extracted PRs. Likewise, the open findings
above were documented; this documentation work did not fix the engine or the
comparator. A new feature should be marked ready only after its actual code and
the relevant failure cases have been checked.
