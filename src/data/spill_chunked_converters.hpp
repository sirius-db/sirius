/*
 * Copyright 2025, Sirius Contributors.
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

#include <cucascade/cuda/stream.hpp>
#include <cucascade/data/representation_converter.hpp>

#include <cstddef>
#include <memory>

namespace cucascade {
class gpu_table_representation;
class host_data_representation;
}  // namespace cucascade

namespace sirius {
namespace spill {

/**
 * @brief Size a GPU table's payload for the Sirius GPU-to-HOST converter.
 *
 * @param source GPU table representation held against mutation by the caller
 * @param stream CUDA stream used to read metadata after the source's writer
 * @return Payload bytes before rounding to HOST blocks
 */
[[nodiscard]] std::size_t gpu_to_host_staging_bytes(
  const cucascade::gpu_table_representation& source, ::cuda::stream_ref stream);

/**
 * @brief Copy a GPU table into pinned HOST blocks from an existing reservation.
 *
 * @param source GPU table representation held against mutation by the caller
 * @param reservation Caller-owned reservation in the destination HOST space
 * @param stream CUDA stream used for the copy
 * @param chunk_bytes Bytes per copy submission. Zero submits all copies together.
 * @return The copied HOST table, ready to read
 */
[[nodiscard]] std::unique_ptr<cucascade::host_data_representation> copy_gpu_to_host(
  const cucascade::gpu_table_representation& source,
  cucascade::memory::reservation& reservation,
  ::cuda::stream_ref stream,
  std::size_t chunk_bytes);

/**
 * @brief Replace the builtin monolithic GPU->HOST spill converter with a chunked one.
 *
 * The builtin cucascade fast converter collects every column buffer's D2H copy and submits
 * them in one batched call followed by one blocking synchronize, so a multi-GB spill is a
 * single monolithic submission. The chunked replacement produces a byte-compatible
 * host_data_representation (same column_metadata layout, restored by the unchanged builtin
 * HOST->GPU converter) but submits the copies in ~chunk_bytes pieces as the column tree is
 * walked, so the DMA engine works on chunk k while chunk k+1 is still being collected.
 *
 * The HOST->GPU restore direction stays on the builtin converter: it reconstructs from the
 * self-describing column_metadata, so overriding it doubles the correctness surface for
 * little gain.
 *
 * @param registry    The converter registry to modify (the builtin pair must already be
 *                    registered and the registry must not yet be published).
 * @param chunk_bytes Copy submission granularity. 0 keeps the builtin converter.
 */
void register_chunked_spill_converters(cucascade::representation_converter_registry& registry,
                                       std::size_t chunk_bytes);

}  // namespace spill
}  // namespace sirius
