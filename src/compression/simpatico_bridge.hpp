/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>

namespace sirius::compression {

/// Warm up the CUDA JIT context used by Simpatico's NVRTC-based codegen.
/// Must be called once before the first compress/decompress operation.
/// Safe to call multiple times (idempotent after the first call).
/// Hook this into extension load, after the CUDA primary context is established.
void initialize_simpatico_jit();

/// Return a unique temp file path for a DISK-tier spill in @p dir.
/// The path has the form "<dir>/<uuid>.hpln". The file is NOT created by this call.
std::string make_compressed_temp_path(const std::string& dir);

}  // namespace sirius::compression
