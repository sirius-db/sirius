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

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <cstddef>
#include <optional>

namespace sirius::memory {

/**
 * @brief Bytes a new reservation on GPU @p space could still obtain, clamped at zero.
 *
 * cuCascade admits a reservation only under the space's reservation limit (`get_max_memory()`),
 * which can sit below its allocation capacity. `memory_space::get_available_memory()` measures
 * headroom against that larger capacity, so it over-states what a reservation can obtain by the
 * bytes already in use. The adaptor's charged counter covers live allocations and outstanding
 * reservations alike, so it is subtracted once.
 *
 * A snapshot, not a guarantee: other tasks can reserve or free memory immediately after.
 *
 * @return nullopt when the space's GPU resource is not the reservation-aware adaptor, so the
 *         caller can refuse to guess rather than read an unrelated number.
 */
[[nodiscard]] inline std::optional<std::size_t> gpu_reservable_bytes(
  const cucascade::memory::memory_space& space)
{
  auto const* adaptor = space.get_memory_resource_of<cucascade::memory::Tier::GPU>();
  if (adaptor == nullptr) { return std::nullopt; }
  auto const charged = adaptor->get_total_allocated_bytes();
  auto const limit   = space.get_max_memory();
  return limit > charged ? limit - charged : 0;
}

}  // namespace sirius::memory
