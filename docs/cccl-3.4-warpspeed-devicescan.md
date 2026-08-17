# [BUG] CCCL 3.4.0: warp-specialized (TMA) `cub::DeviceScan` on Blackwell — out-of-bounds `__shared__` read, silent result corruption under concurrency

> Status: workaround shipped in this repo (`CCCL_DISABLE_WARPSPEED_SCAN` defined globally in
> `CMakeLists.txt`); this document is the standalone analysis/report prepared for upstream
> (NVIDIA/cccl). Remove the workaround once a fixed CCCL ships.

## Summary

The warp-specialized (TMA) `DeviceScan` implementation introduced in CCCL 3.4 performs a
deterministic **out-of-bounds `__shared__` read** on Blackwell (verified on `sm_100` and
`sm_103a` builds, GB300 hardware). A bare `cub::DeviceScan::ExclusiveSum` over 6M `uint32`
reproduces it under `compute-sanitizer --tool memcheck` on the first invocation.

In quiet, single-kernel conditions the out-of-range read happens to return benign residue and
results are numerically correct — which makes the defect invisible to typical unit testing. In
a production workload with concurrent kernel diversity (a GPU database engine running 7
concurrent query streams), the same read returns foreign data and **scan outputs are silently
corrupted**. Because `DeviceScan` output is typically *geometry* (prefix-sum offsets), the
downstream blast radius is large: we observed corrupted string-offset arrays and batch-memcpy
tuple arrays causing warp out-of-range faults, `2^64 - N` byte allocation requests
(non-monotonic offset subtraction), and non-terminating consumer kernels. It cost us several
days of debugging in the wrong direction (memory-lifetime races) because the corruption always
surfaced in victim buffers far downstream of the scan.

Notably, CCCL already ships a warpspeed disable for `sm_120` (see cccl#8528); `sm_103`/`sm_100`
have no such guard.

## Environment

| | |
|---|---|
| CCCL | **3.4.0** (rapids-distributed; selected via `CCCL_DIR` due to rmm's CCCL >= 3.3 requirement) |
| Control | CTK-bundled CCCL **3.2.0**: memcheck-clean, results correct (same source, same box) |
| CUDA toolkit | 13.2 (V13.2.51) |
| Driver | 595.58.03 |
| GPU | NVIDIA GB300 (Blackwell), coherent unified memory; repro identical for `-arch=sm_100` and native `-arch=sm_103a` |
| OS | Linux 6.17 (aarch64) |

## Minimal reproducer (22 lines)

```cuda
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int main(int argc, char** argv) {
  size_t n = argc > 1 ? strtoull(argv[1], nullptr, 10) : 6000001;
  std::vector<unsigned> h(n, 3);
  unsigned *d_in, *d_out;
  cudaMalloc(&d_in, n * 4); cudaMalloc(&d_out, n * 4);
  cudaMemcpy(d_in, h.data(), n * 4, cudaMemcpyHostToDevice);
  void* tmp = nullptr; size_t tb = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, tb, d_in, d_out, (int)n, 0);
  cudaMalloc(&tmp, tb);
  cub::DeviceScan::ExclusiveSum(tmp, tb, d_in, d_out, (int)n, 0);
  cudaDeviceSynchronize();
  std::vector<unsigned> out(n);
  cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost);
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) if (out[i] != 3u * i) ++bad;
  printf("n=%zu temp=%zu wrong=%zu err=%s\n", n, tb, bad, cudaGetErrorString(cudaGetLastError()));
  return bad ? 1 : 0;
}
```

Build and run (point `-I` at the CCCL 3.4.0 checkout so it wins over the CTK headers):

```bash
nvcc -arch=sm_103a -I/path/to/cccl-3.4.0/include scan_repro.cu -o scan_repro
compute-sanitizer --tool memcheck ./scan_repro
```

## Observed

- **CCCL 3.4.0**: `compute-sanitizer` reports `Invalid __shared__ read of size 16` inside the
  warp-specialized scan kernel (large-shared-memory instance, ~137 KB class; faulting shared
  offsets observed at `0x21630` standalone and `0x7c00` in our embedded instance). With the
  kernel aborted by the sanitizer, **6,000,000 of 6,000,001 results are wrong**.
- Without the sanitizer, in a quiet process the results happen to be correct; under concurrent
  kernel activity in our production engine, scan outputs are corrupted (foreign bytes in
  prefix-sum output), with no error reported by CUDA.
- **CCCL 3.2.0** (only change): memcheck-clean, results correct.
- **CCCL 3.4.0 + `-DCCCL_DISABLE_WARPSPEED_SCAN`**: memcheck-clean, results correct — the
  classic decoupled-lookback path is unaffected.

## Expected

No out-of-bounds shared reads; scan results independent of co-resident kernel activity.

## Analysis / notes

- The regression is confined to the new warp-specialized (TMA) `DeviceScan` path added in
  CCCL 3.4 for Blackwell; the shared-memory footprint and kernel identity match that
  implementation.
- The OOB read is deterministic and arch-independent within Blackwell (`sm_100` and `sm_103a`
  builds both reproduce; not an arch-mismatch artifact).
- We did not pin down the exact channel by which foreign shared residue reaches the output
  under concurrency (a naive adversarial co-resident-kernel experiment over 100 reps did not
  force value flow) — but the OOB access itself is unambiguous under memcheck, and in our field
  workload every corruption event's garbage was scan-produced geometry, immune to every
  memory-lifetime fix we applied, and vanished entirely with `CCCL_DISABLE_WARPSPEED_SCAN`.
- Field validation of the workaround: with the disable defined globally (engine + libcudf
  rebuilt), our previously ~50–100%-failing concurrent workload passed a poison-on-free gate,
  an SF1 full-workload memcheck sweep (zero findings), and 8 consecutive clean full-scale
  concurrent runs with byte-identical results.

## Suggested action

Guard `sm_100`/`sm_103` off the warpspeed scan path (as already done for `sm_120` in
cccl#8528) until the OOB is fixed, and consider a memcheck pass over the warpspeed kernels'
shared indexing.

## Deployment note (sirius-specific)

The same CCCL 3.4.0 lives inside any **separately built libcudf** that the engine LD_PRELOADs
(e.g. `bench/sf1000-repro/build-libcudf.sh`). The CMake define here does not reach that build:
the custom libcudf must be rebuilt with `CCCL_DISABLE_WARPSPEED_SCAN` patched into its own CCCL
(we verified the identical `Invalid __shared__ read` inside the preloaded libcudf's
`sizes_to_offsets`/`exclusive_scan` instances before patching). Any stock (conda/pixi) libcudf
on Blackwell presumably carries the same warpspeed scans and must be memcheck-verified before
being trusted for concurrent workloads.
