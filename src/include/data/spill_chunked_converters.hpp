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

#include <cucascade/data/representation_converter.hpp>

#include <cstddef>

namespace sirius {
namespace spill {

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
 * Idempotent; safe to call again (e.g. across context re-initializations).
 *
 * @param registry    The converter registry to modify (the builtin pair must already be
 *                    registered).
 * @param chunk_bytes Copy submission granularity. 0 keeps the builtin converter.
 */
void register_chunked_spill_converters(cucascade::representation_converter_registry& registry,
                                       std::size_t chunk_bytes);

}  // namespace spill
}  // namespace sirius
