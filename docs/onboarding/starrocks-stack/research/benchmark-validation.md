# Benchmark comparator validation

Reviewed `experimental/starrocks/tools/compare.py` on
`bench/sf500-2-mig-gpus` at `7610840c`. The comparator was not modified.

## Reproduction

The review-owned fixture runner is
[benchmark-validation.py](/home/ubuntu/sirius-wt/s500-mig/docs/onboarding/starrocks-stack/research/benchmark-validation.py).
It creates temporary Oracle and Sirius output directories, invokes the product
comparator, and asserts its current output.

Executed from `experimental/starrocks`:

```text
pixi run python ../../docs/onboarding/starrocks-stack/research/benchmark-validation.py
baseline=PASS
nan_vs_finite=FALSE_MATCH
oracle_only_query=SUBSET_POLICY
empty_result_directory=CRASH
```

## Confirmed findings

### P1 — NaN can make a mismatching result pass

For an Oracle cell `1.0` and Sirius cell `nan`, `num()` accepts both as
floats. `d` becomes NaN; Python evaluates `NaN > TOL` as false, and
`max(0.0, NaN)` remains `0.0`. The comparator prints `MATCH` and exits zero
([compare.py:30](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/tools/compare.py#L30),
[compare.py:67](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/tools/compare.py#L67)).

This defeats the answer-correctness gate for non-finite numeric values. Reject
non-finite values before relative-error calculation, or compare matching
non-finite values under an explicit policy. The test should cover finite versus
NaN, finite versus infinity, and NaN versus NaN.

### Completeness limitation — no expected-query manifest

The query universe is constructed only from `sirius_out_dir/q*.out`
([compare.py:83](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/tools/compare.py#L83)).
With Oracle files `q01.tsv` and `q02.tsv`, but only `q01.r0.out`, the tool
reports `1/1 queries match` and returns zero. This is intentional for a subset
run: the benchmark README supports passing selected queries such as q01/q06
against a shared, full Oracle directory. It is therefore not an unconditional
comparator defect.

For an all-22 certification, the caller needs an explicit expected-query
manifest (for example the benchmark driver's selected `QUERIES`) and must fail
if any member has no run. Do not infer that contract from a shared Oracle
directory, which legitimately contains more queries than a subset run.

### P2 — an empty result directory crashes instead of failing clearly

When Sirius produces no `q*.out`, `summary` is empty and the width calculation
calls `max()` on an empty sequence ([compare.py:103](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/tools/compare.py#L103)).
The exit is non-zero, so this does not create a false pass, but it hides the
actionable cause. Emit a `NO-RESULT` summary and a concise non-zero failure.

## Staging finding validation

Independent code review of `staging.json` and `staging.md` yields these
verdicts:

| Claim | Verdict | Severity | Evidence |
|---|---|---:|---|
| A remote receiver lease leaks after a successful grant if NIXL WRITE fails before `transmit_packed`. | Confirmed code defect. | P1 | `send_fragment` obtains the receiver lease, then `write_and_wait` can fail; cleanup releases only the sender-local lease. The receiver learns leases only through `transmit_packed`, so cancellation has no source/ticket to release ([nixl_transport.rs:672](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/nixl_transport.rs#L672), [nixl_transport.rs:687](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/nixl_transport.rs#L687), [nixl_transport.rs:719](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/nixl_transport.rs#L719)). |
| `InboundStore::stage` can use the GPU memory-space pointer after Context teardown. | Confirmed use-after-free race. | P1 | Stage reads raw `gpu_space` under a mutex, drops the lock, then uses it for allocation and `make_data_batch`; Context close clears the pointer and context destruction follows while the exported handle retains only State ([sirius_ffi.cpp:316](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L316), [sirius_ffi.cpp:435](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L435)). The documented handle is intended to outlive Context and fail safely, which this interleaving violates ([sirius_ffi.hpp:180](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/sirius_ffi.hpp#L180)). |
| Copy-out ingress has no credit/reservation/spill admission and can exhaust the normal GPU pool before the EOS barrier releases the receiver. | Confirmed operational capacity limitation, not independently a memory-safety defect. | P2 / merge gate | `request_staging_lease` only leases the arena; the receive path stages each fresh frame into ordinary pool memory and waits for all EOS. This is a workload-dependent OOM risk that the non-ancestor credit/spill commits address. Keep it prominent as a production-readiness gate, but do not label it a P1 correctness bug without a bounded-contract promise. |

For the confirmed remote leak, add a lease identity and idempotent abort RPC that executes for every post-grant failure, including transfer timeout and ambiguous control-plane outcomes. For the lifetime race, close must synchronize with active `stage` calls or hold an owning guard for the memory-space lifetime; the existing second null check occurs after the unsafe use and is insufficient.
