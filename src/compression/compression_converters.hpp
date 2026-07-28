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

#include <cucascade/data/representation_converter.hpp>

namespace sirius {

/**
 * @brief Register Simpatico compression/decompression converters into @p registry.
 *
 * Registers:
 *   compressed_host_representation → gpu_table_representation
 *     (read the serialised .hpln file, select columns if projected, decompress)
 *
 * Called from converter_registry::initialize().
 */
void register_compression_converters(cucascade::representation_converter_registry& registry);

/// Column-parallelism degree for the decompress converters (process-global since
/// converters have no per-context config). Mirrors compression_config::column_threads.
void set_decompress_column_threads(int n) noexcept;
[[nodiscard]] int decompress_column_threads() noexcept;

}  // namespace sirius
