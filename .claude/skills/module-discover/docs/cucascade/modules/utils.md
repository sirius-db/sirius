# Utils Module

**Status**: UNUSED
**Path**: `cucascade/include/cucascade/utils/` and `cucascade/include/cucascade/cuda_utils.hpp`

## Summary

Internal utility headers used within cuCascade's implementation. Provides atomic helpers for lock-free counters/trackers, a metaprogramming helper for `std::visit`, and CUDA error-checking macros.

## Key APIs

- `atomic_peak_tracker<T>` — Lock-free peak value tracker using CAS loop
- `atomic_bounded_counter<T>` — Atomic counter with bounded `try_add`/`try_sub` operations and `add_bounded`/`sub_bounded` clamping
- `overloaded<Ts...>` — Variadic struct for creating overloaded lambda sets (used with `std::visit`)
- `CUCASCADE_CUDA_TRY(call)` — CUDA runtime error checking macro (throws `rmm::cuda_error`)
- `CUCASCADE_CUDA_TRY_ALLOC(call)` — CUDA allocation error checking (throws `rmm::out_of_memory`)
- `CUCASCADE_FUNC_RANGE()` — NVTX range annotation for profiling

## Potential Relevance

The atomic utilities (`atomic_bounded_counter`, `atomic_peak_tracker`) are used internally by `reservation_aware_resource_adaptor` for allocation tracking. The CUDA macros could be useful if writing CUDA code that interacts with cuCascade's error handling conventions, but Sirius uses its own error handling patterns instead.
