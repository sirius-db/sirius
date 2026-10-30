// Test-only seams of the FFI layer. Not part of the public FFI (include/sirius/ffi.hpp); the
// unit tests include this header directly to make otherwise racy interleavings deterministic.
#pragma once

#include <functional>

namespace sirius::ffi::testing {

/// Install a hook that runs once per execution right after the execution captured its
/// object-store snapshot (publication lock already released) and before the pre-bind guard
/// consumes that capture. An empty hook removes it. Default: no hook. Thread-safe to install;
/// the hook runs on the executing thread and must not call back into the Context that is
/// executing except through thread-safe members (`object_store_config_for`, `put_object_store`, `drop_object_store`).
void set_capture_pause_hook(std::function<void()> hook);

/// Install a process-wide hook invoked before each in_transaction attempt, before conn_mutex is
/// acquired. This includes execute_substrait, Fragment::build and Fragment::run; run uses its
/// existing capture. The same installation and callback restrictions as the capture-pause hook
/// apply. An empty hook clears the installed slot.
void set_transaction_entry_hook(std::function<void()> hook);

}  // namespace sirius::ffi::testing
