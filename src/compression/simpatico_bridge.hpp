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

namespace sirius::compression {

/**
 * @brief Call codegen::jit::ensure_cuda_context() to warm up the JIT CUDA context.
 *
 * Must be called after the CUDA device is selected and before any Simpatico
 * compression operation. Idempotent — safe to call multiple times. Throws
 * std::runtime_error if no GPU is available or the CUDA driver cannot be
 * initialised.
 */
void initialize_simpatico_jit();

}  // namespace sirius::compression
