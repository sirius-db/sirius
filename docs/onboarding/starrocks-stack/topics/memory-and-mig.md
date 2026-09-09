# Memory and MIG

[Back to the guide](../README.md)

**Classification:** topic: deployment and memory capacity; feature: one-compute-node (CN)-per-device placement and exchange sizing; modules: C++ engine allocator, Rust CN launcher, benchmark scripts; languages: C++, Rust, and shell.

**Scope and status snapshot:** [`7610840c`](https://github.com/aocsa/sirius/tree/7610840c03f9086edfa072be72a0eb4c96e03d60), `bench/sf500-2-mig-gpus`, reviewed on 2026-09-09. Related work: draft [PR #1693](https://github.com/sirius-db/sirius/pull/1693), draft [PR #1714](https://github.com/sirius-db/sirius/pull/1714), proposed [P05](../pr-packages/p05.md), and proposed [P16](../pr-packages/p16.md). Measurements are historical observations on one machine, not a general MIG guarantee.

## Four separate memory domains

The **GPU pool** is the engine's normal RMM (RAPIDS Memory Manager)/cuCascade-managed memory. It holds ordinary working batches and post-copy inbound tickets. The **staging arena** is a separate raw `cudaMalloc` slab for in-flight exchange payloads. It sits outside the pool budget and ordinary pool accounting. A **CUDA context** also consumes device memory. Therefore a CN's real device reservation is at least GPU pool + staging arena + context overhead, rather than the configured pool limit alone. The launcher states that contract [explicitly](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/benchmarks/cluster8.sh#L25-L46).

Host and DISK tiers are lower memory domains used by normal downgrade. They do not enlarge the arena. Here remote data becomes a pool-backed ticket before the EOS barrier permits receiver execution. No verified inbound-specific spill workflow, receive-credit admission, or copying-thread reservation binding exists. A larger host or disk tier alone therefore does not solve an inbound pool-allocation failure. The `DISK-TIER.md` helper is from another branch.

## What the recorded SF500/MIG result says

MIG (Multi-Instance GPU) splits a GPU into separately visible instances. The checked-in two-MIG checklist records two CNs using `GPU_DEVICES=0,1`, each with a 40 GiB pool, 4 GiB arena, and 40 GiB host layout. It recorded 16 of 22 TPC-H benchmark queries matching and the same six out-of-memory failures as the shared-GPU setup. It also recorded 45,530 MiB resident per 48 GiB MIG instance: pool + arena + about 474 MiB overhead. These are machine- and run-specific observations, not a sizing rule. The source is the [MIG checklist](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/plan-2mig/CHECKLIST-2mig.md#L10-L22).

The same note records q18 and q21 failing while staging a roughly 5.2e8-byte inbound frame into a full pool. That supports a P2 production-capacity finding: the EOS barrier can retain completed inputs faster than the receiver consumes them. It is not evidence of a P1 memory-safety defect. The separate P1 issues are the remote lease leak after a failed WRITE and the `InboundStore` teardown race, described in [Exchange staging area](staging-area.md).

## Device placement and MIG caveat

The launcher supports two inputs. `GPU_DEVICES=0,1` launches CNs with ordinal device flags; the recorded two-MIG experiment says this worked. `MIG_DEVICES=MIG-<uuid>,...` instead exports a MIG UUID through `CUDA_VISIBLE_DEVICES` and omits the ordinal flag. The launcher includes both paths [here](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/benchmarks/cluster8.sh#L48-L80).

The field note says UUID-only visibility fails in cuCascade/NVML (NVIDIA Management Library) with a device-count error, while ordinals 0 and 1 work. A CUDA ordinal is the process-visible numeric device identifier. `MIG_DEVICES` is launcher plumbing with a known failing route, not a validated MIG launch method. The one-box success of `GPU_DEVICES=0,1` is not a general MIG claim.

One CN should see one CUDA device. The NIXL transport registers the arena as device zero within that already-pinned process. A multi-device-visible process would need explicit device plumbing; it is not the contract exercised here.

## Practical review and operating guidance

Size the staging arena for simultaneously live transfer leases, including the packer's 8 MiB margin per export, then leave space for the GPU pool and CUDA context. Validation should measure simultaneously live arena bytes and check that all leases are returned after transfers finish and query state is retired. Test both successful and failed post-grant transfers before relying on sustained exchange traffic.

For MIG work, record the exact device mapping, pool, staging, host, batch sizing, query scale, and correctness result. Use `GPU_DEVICES=0,1` only as the recorded workaround on the reviewed box. Keep UUID mode marked unsupported until the engine's NVML/device-count path is fixed and validated.

## Related topics

- [Exchange staging area](staging-area.md) covers the allocator and two P1 findings.
- [Exchange transport](exchange-transport.md) covers peer registration and transfer lifecycle.
- [Back to the guide](../README.md)
