# Error Handling

**Status**: USED
**Path**: `rmm/error.hpp`, `rmm/detail/error.hpp`
**Headers we include**: `<rmm/detail/error.hpp>`, `<rmm/error.hpp>` (via `<rmm/device_buffer.hpp>`)

## Summary

Exception types for RMM errors and CUDA error checking macros. Sirius catches `rmm::out_of_memory` in the pipeline OOM reschedule logic and uses `rmm::detail::error.hpp` for the `RMM_CUDA_TRY` macro in scan code.

## API Reference

### Exception Types

**Header**: `<rmm/error.hpp>`

```cpp
struct rmm::logic_error : public std::logic_error { ... };
struct rmm::cuda_error : public std::runtime_error { ... };
class  rmm::bad_alloc : public std::bad_alloc { ... };
class  rmm::out_of_memory : public bad_alloc { ... };
class  rmm::out_of_range : public std::out_of_range { ... };
```

**Our usage**:
- `test/cpp/pipeline/test_oom_reschedule.cpp:138` — `catch (const rmm::out_of_memory&)` in OOM reschedule tests

### Macros

**Header**: `<rmm/detail/error.hpp>`

| Macro | Description |
|-------|-------------|
| `RMM_EXPECTS(cond, msg [, exception])` | Throws if condition false (default: `logic_error`) |
| `RMM_FAIL(msg [, exception])` | Always throws |
| `RMM_CUDA_TRY(call [, exception])` | Checks CUDA return code, throws on error (default: `cuda_error`) |
| `RMM_CUDA_TRY_ALLOC(call [, bytes])` | Like CUDA_TRY but throws `bad_alloc`/`out_of_memory` |
| `RMM_ASSERT_CUDA_SUCCESS(call)` | Debug-only assert on CUDA errors |
| `RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(call)` | Like above but allows `cudaErrorCudartUnloading` |

**Our usage**:
- `src/op/scan/prefetched_data_source.cpp:19` — Includes `<rmm/detail/error.hpp>` for error macros
