# CUDA target

Use the same public registration and callback contract as the [cuDF target](cudf.md).
Generate the native scaffold with `--target cuda`; it contains an independently
written kernel and CPU reference. CUDA compilation builds the UDF package, not
Sirius, cuDF, or DuckDB.

Keep DuckDB registration in `udf.cpp`, compiled by the C++ compiler, and GPU
callbacks and kernels in `udf.cu`, compiled by NVCC. The private `gpu.hpp` shares
only Sirius's GPU callback types. DuckDB's concrete headers cannot be parsed by
the pinned NVCC toolchain; do not include them in CUDA translation units.

Choose ahead-of-time `.cu` compilation for arbitrary native code. For suitable
fixed-width elementwise transforms, the existing `cudf::transform_extended`
path can instead compile device source at runtime; a native registration adapter
is still necessary. Never imply the skill installs a general runtime compiler or
accepts untrusted CUDA safely.

Implement handling for empty batches, slices, NULLs, and the declared types.
Prefer cuDF factories for result allocation. Launch on the supplied stream,
allocate through the supplied RMM resource, and check launch failures. Keep
thread-local and device state out of immutable binder configuration. Account for
GPU architecture and the exact CUDA, cuDF, and DuckDB versions in the output's
build prerequisites.

Preserve original failure and arithmetic behavior; do not silently replace
exceptions with NULLs. Avoid host copies or per-row CPU work. Validate before
profiling, and measure both whole queries and kernels when deciding whether a
custom kernel improves a cuDF implementation.
