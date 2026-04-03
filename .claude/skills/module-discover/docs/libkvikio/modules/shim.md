# Shim Layer

**Status**: UNUSED
**Path**: `kvikio/shim/cuda.hpp`, `kvikio/shim/cufile.hpp`

## Summary
Dynamic loading shims for CUDA and cuFile C APIs. Loads library symbols at runtime via `dlopen`/`dlsym`, enabling kvikio to work even when cuFile is not installed (graceful degradation). Provides singleton access to all required CUDA driver and cuFile functions.

## Key APIs
- `cudaAPI` — Singleton wrapping CUDA driver API functions (context, memory, streams)
- `cuFileAPI` — Singleton wrapping cuFile API functions (register, read, write, batch, async)
- `is_cuda_available()` — Check if CUDA library can be loaded
- `is_cufile_library_available()` — Check if cuFile library can be loaded
- `is_cufile_available()` — Check if cuFile is both loadable and functional
- `cufile_version()` — Get cuFile library version

## Potential Relevance
Not applicable to our use case. These are internal implementation details of kvikio's runtime library loading.
