/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade builtin_converters — ROCm stub.

#pragma once
#include "cucascade/data/representation_converter.hpp"

namespace cucascade {

/// Register built-in converters (stub: no-op).
inline void register_builtin_converters(representation_converter_registry& /*registry*/) {}

/// Convert GPU→GPU (stub: never called in Sirius, comment-only reference).
/// Declared but not defined — no linker reference exists.
// void convert_gpu_to_gpu(...);

}  // namespace cucascade
