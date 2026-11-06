# Tiled merge simplification qualification

Status: implementation checkpoints and the final suite validated by the developer;
workload comparison pending. This records the simplified coordinator, not the earlier recursive
recovery policy. The operator contract is documented in [Operators](operators.md).

## Implementation and reported validation

The developer ran builds and the requested `[tiled_merge]` selection after each checkpoint
and reported that they passed. The agent did not run builds or GPU tests. Hardware,
individual test counts and skips were not supplied; a pass does not establish that the
two-GPU case ran.

| Checkpoint | Change | Evidence |
|---|---|---|
| `86e8ea56` | Remove failure-driven tile, coalescing and copy subdivision | Developer reported build and tests passed |
| `5449aa71` | Immutable typed execution requests | Developer reported build and tests passed |
| `665f9af5` | Typed results and removal of overloaded work fields | Developer reported build and tests passed |
| `153dfe21` | Storage handoffs and retirement | Developer reported build and tests passed |
| `8369f1bf` | Stage-local queues, state and completion | Developer reported build and tests passed |

Formatting and commit hooks passed at these checkpoints. Those checks are separate from
compilation and GPU execution.

## Existing coverage and limits

- `test_tiled_merge.cpp` checks empty/single inputs, ordered ranges with reversed completion,
  normalization/coalescing, variable-width payload conservation, projection, concurrent
  claims, batch replacement and an injected prepublication retry.
- Its spill/restore and partial-publication failure cases use the real task and repository
  lifecycle. Failure cleanup is covered; this is not a dedicated cancellation qualification.
- Its allocation fixture injects failures beneath the reservation tracker. Preparation
  exhaustion covers normalization, coalescing, merge, sampling, splitter selection,
  extraction and publication. The coalescing and merge parameters were added during the
  final audit and passed in the developer-reported final suite run. The one-row owning-copy test checks compute
  exhaustion. These are injected failures, not measured physical GPU capacity limits.
- The retry case verifies successful replay and preserved rows. It does not independently
  demonstrate successful reservation growth for every operation.
- `test_gpu_sampled_merge.cpp` covers comparison semantics including nulls, NaNs, signed zero,
  mixed directions, hidden string keys and peer-heavy inputs. `test_physical_merge_sort.cpp`
  covers operator ordering and projection, and `test_merge_pipeline_fusion.cpp` exercises
  query results through pipeline integration. The developer reported that all selections below passed at `c902383f`.
- Dedicated operator coverage for multiple splitter pages and cancellation has not been
  established by this audit. Helper-level large inputs do not establish those lifecycle paths.

No new test cases were introduced. Four cases exercising removed recovery behavior were
deleted; retained failure assertions were updated, and the existing preparation-failure
parameterization was extended to cover the removed coalescing/merge recovery paths.

## Final suite validation

The developer reported that the build and this combined selection passed at `c902383f`:

```bash
pixi run build/release/test/cpp/sirius_unittest "[tiled_merge],[physical_merge_sort],[sort_comparison],[gpu_sampled_merge],[sampled_merge],[merge_fusion]"
```

GPU model/count and skipped-case details remain unspecified. Subsequent changes only
replace unconditional test stdout with Catch2 failure diagnostics; they have not been rebuilt.

## Workload comparison

No before/after timing, peak-memory, retry-count or copied-byte measurements have been
collected for this cleanup. No performance or allocation-bound claim follows from the
unit-test passes.

Use the same representative existing workload, input data, GPU, chunk target, capacity
and build configuration for both revisions. The implementation baseline is `3404ded9`
from the cleanup plan; verify its applicability to the workload before comparing.
Record both actual revisions and all configuration. Compare wall time and available peak
memory, retries and copied bytes, keeping unavailable metrics explicit. The workload
selection and baseline measurements are awaiting developer input.
