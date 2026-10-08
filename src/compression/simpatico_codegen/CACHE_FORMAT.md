# NVRTC cache identity format

A cached cubin is identified by two parts:

- the **epoch**, which identifies the code that turns a request into a cubin and is fixed
  for a build;
- the **request**, which identifies one NVRTC compilation within that build.

On disk, a cubin lives at

```
<cache dir>/<epoch>/nvrtc-<major>.<minor>/<request>.cubin
```

The in-process table keys on the request alone, because the epoch and NVRTC version are
fixed for a process. Hash collisions remain theoretically possible.

## Epoch

`cmake/jit_epoch.cmake` hashes the compiled objects of `simpatico_jitgen` and emits the
result as `codegen::jit::kJitEpoch` (`src/codegen/jit/jit_epoch.h`). The objects are:

- the encode and decode renderers;
- the NVRTC driver, which holds the compile options and the embedded project headers;
- the complete embedded CCCL header bundle;
- the request encoding below.

The hash covers the SHA-256 of each object's contents, in source order, and is truncated to
32 lowercase hex digits. Object paths are not part of it.

Any change that can alter the cubin for a given request alters these objects. That
includes renderer logic, inline header code, emitted CUDA text, embedded headers, NVRTC
options, and compiler version or flags. Such a change starts a new epoch automatically, so
there is no version to bump by hand. Changes elsewhere in simpatico (operators, plans,
launchers, tests) only decide *which* request is made, so they keep the epoch and the
cache stays warm. Comment-only edits normally keep it as well, unless debug info records
line numbers.

`simpatico_jitgen_link_check` links the jitgen objects without the rest of simpatico, and
with section garbage collection disabled. If jitgen code starts calling a non-inline
function defined elsewhere, that function could change cubins without changing the epoch,
so the build fails instead. Move such code into `simpatico_jitgen`, or make it header-only.

NVRTC itself is outside the objects: it is a shared library in Pixi builds and linked
statically into the final binary in vcpkg builds. Its version therefore forms the next
path level. A cubin from a different patch release of the same NVRTC version is still a
correct kernel for its source.

## Request encoding

All numbers are unsigned 64-bit integers in big-endian byte order. A string is its byte
length followed by that many bytes, without a terminator. The input is streamed into
XXH3-128 without concatenating the rendered source. Digests use xxHash's canonical
big-endian representation: 16 binary bytes in memory and 32 lowercase hexadecimal digits
in paths. Runtime hashing uses xxHash's header-only mode, with no shared-library
dependency.

A request record contains, in order:

1. The string `simpatico-request-v2` (record type and schema version).
2. Source, entry symbol, and program name, each encoded as a string.
3. The option count, followed by each effective NVRTC option as a string, in order
   (`codegen::jit::nvrtc_options`, the list `compile_plain_kernel` passes). Architecture is
   included through the actual `-arch=sm_...` option.

## Pruning

On its first disk-cache lookup, a process does the following:

1. Sets the modification time of its epoch directory, which records when the epoch was
   last used.
2. Removes legacy flat cubins (`<16 hex>_a<arch>_c<cudart>_d<driver>.cubin`).
3. Removes other epoch directories that are both older than one hour and not among the
   four most recently used, counting its own.

Entries that match neither pattern are left alone. `clear_jit_disk_cache()` removes every
epoch directory and legacy cubin. It is useful for timing cold compiles, but correctness
never requires it.

## Running host coverage

```sh
pixi run cmake -S src/compression/simpatico_codegen/tests/cache -B build/cache-host -G Ninja
pixi run cmake --build build/cache-host
pixi run ctest --test-dir build/cache-host -L cache_host --output-on-failure --no-tests=error
```

The Test workflow invokes this project on its CPU runners for both supported CUDA
environments according to the existing matrix. These tests do not load the CUDA driver. They
cover the request encoding and the epoch script: it is stable across reruns and object
relocation, and it changes when object contents, count, or boundaries change.
`test_jit_kernel_cache` covers the on-disk layout and pruning on a GPU.
