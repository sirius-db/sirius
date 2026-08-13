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

#include <stdexcept>

#include <cucascade/data/representation_converter.hpp>

#include <cstddef>

namespace cucascade {
class idata_representation;
}  // namespace cucascade

namespace sirius {

/// Thrown when a compressed spill fails *after* it has taken ownership of the
/// batch's columns.
///
/// The distinction matters because the caller's normal response to a failed
/// compression is to spill the batch uncompressed instead. That is only valid
/// while the source is intact: once ownership has moved, the representation is
/// empty, and the uncompressed converter would happily produce a zero-column
/// batch whose emptiness only surfaces much later, as an out-of-range access
/// when something tries to materialize it. Callers must let this propagate.
class spill_source_consumed : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};


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
