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

#include <cstddef>

namespace cucascade {
class idata_representation;
}  // namespace cucascade

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

/**
 * @brief Peak device bytes needed to materialize @p data as a GPU table.
 *
 * For an uncompressed representation this is just its uncompressed footprint:
 * the H2D copy lands straight in the final buffers, so nothing else is live.
 *
 * A compressed representation decodes with its compressed payload resident on
 * the device *alongside* the table being built — read_compressed_table*() stages
 * every leaf buffer to device, and simpatico's decode holds that whole
 * compressed_table live while each column's output is allocated
 * (decompress_columns_parallel takes it by const&). Its peak is therefore both
 * footprints at once, and a reservation covering only the decompressed size lets
 * the converter allocate past it. That surfaces as an rmm::out_of_memory the
 * retry loop cannot clear, because the memory it needs was never reserved.
 */
[[nodiscard]] std::size_t estimated_materialization_bytes(
  const cucascade::idata_representation& data);

}  // namespace sirius
