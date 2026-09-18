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

#include "cucascade/memory/common.hpp"
#include "cucascade/memory/memory_reservation_manager.hpp"
#include "telemetry-bridge/gen/quent.hpp"

#include <functional>
#include <optional>
#include <unordered_map>

namespace sirius::telemetry {

using memory_handle  = quent::Handle<quent::Memory>;
using channel_handle = quent::Handle<quent::Channel>;

struct channel_key {
  cucascade::memory::memory_space_id source;
  cucascade::memory::memory_space_id destination;

  bool operator==(const channel_key& other) const
  {
    return (this->source == other.source) && (this->destination == other.destination);
  }
};

struct channel_key_hash {
  std::size_t operator()(const channel_key& p) const noexcept
  {
    const std::size_t h1 = std::hash<cucascade::memory::memory_space_id>{}(p.source);
    const std::size_t h2 = std::hash<cucascade::memory::memory_space_id>{}(p.destination);

    // Combine the individual hashes using the golden ratio magic number
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

class memory_context {
 public:
  explicit memory_context(quent::Uuid engine_uuid,
                          const quent::Context& context,
                          const cucascade::memory::memory_reservation_manager* manager);
  ~memory_context() noexcept;

  std::optional<std::reference_wrapper<const memory_handle>> get_memory_handle(
    cucascade::memory::memory_space_id mem_space) const noexcept;

  std::optional<std::reference_wrapper<const channel_handle>> get_channel_handle(
    cucascade::memory::memory_space_id source,
    cucascade::memory::memory_space_id destination) const noexcept;

 private:
  std::unordered_map<cucascade::memory::memory_space_id, memory_handle> memory_handles_{};

  std::unordered_map<channel_key, channel_handle, channel_key_hash> channel_handles_;
};

}  // namespace sirius::telemetry
