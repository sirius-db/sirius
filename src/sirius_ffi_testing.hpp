// Test-only seams of the FFI layer. Not part of the public FFI (include/sirius/ffi.hpp); the
// unit tests include this header directly to make otherwise racy interleavings deterministic.
#pragma once

#include <functional>

namespace sirius::ffi::testing {

/// Install a hook that runs once per execution right after the execution captured its
/// object-store snapshot (publication lock already released) and before the pre-bind guard
/// consumes that capture. An empty hook removes it. Default: no hook. Thread-safe to install;
/// the hook runs on the executing thread and must not call back into the Context that is
/// executing except through thread-safe members (`object_store_config`, `set_object_store`).
void set_capture_pause_hook(std::function<void()> hook);

}  // namespace sirius::ffi::testing
