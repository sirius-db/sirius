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

#include <cstdint>

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

/// Compressed GPU-tier chunks this process has decoded on a GPU other than the one holding
/// them, by copying their compressed buffers across. Monotonic; for tests and diagnostics.
[[nodiscard]] std::uint64_t compressed_chunks_decoded_on_another_gpu() noexcept;

}  // namespace sirius
