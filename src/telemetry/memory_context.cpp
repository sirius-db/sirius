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

#include "telemetry/memory_context.hpp"

#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace {

std::string tier_to_string(cucascade::memory::Tier tier)
{
  switch (tier) {
    case cucascade::memory::Tier::GPU: return "gpu";
    case cucascade::memory::Tier::HOST: return "host";
    case cucascade::memory::Tier::DISK: return "disk";
    default: return "unknown";
  }
}

}  // namespace

namespace sirius::telemetry {

memory_context::memory_context(quent::Uuid engine_uuid,
                               const quent::Context& context,
                               const cucascade::memory::memory_reservation_manager* manager)
{
  if (manager == nullptr) { return; }

  for (const auto& mem_space : manager->get_all_memory_spaces()) {
    auto instance_name = mem_space->to_string();
    auto handle        = context.memory_observer()->handle();
    handle.declaration(quent::memory::Declaration{
      .instance_name   = instance_name,
      .parent_group_id = quent::engine::EngineId(engine_uuid),
      .bounds          = quent::records::MemoryBounds{.bytes = mem_space->get_max_memory()},
    });
    memory_handles_.emplace(mem_space->get_id(), std::move(handle));
  }

  for (const auto& [space_id_1, handle_1] : memory_handles_) {
    for (const auto& [space_id_2, handle_2] : memory_handles_) {
      if (space_id_1 == space_id_2) {
        continue;  // skip inserting a channel between the same space.
      }
      auto instance_name = std::format("{}-{}->{}-{}",
                                       tier_to_string(space_id_1.tier),
                                       space_id_1.device_id,
                                       tier_to_string(space_id_2.tier),
                                       space_id_2.device_id);
      auto handle        = context.channel_observer()->handle();
      handle.declaration(quent::channel::Declaration{
        .instance_name   = instance_name,
        .parent_group_id = quent::engine::EngineId(engine_uuid),
        .source_id       = handle_1.id(),
        .target_id       = handle_2.id(),
        .bounds = quent::records::ChannelBounds{.bytes = std::numeric_limits<uint64_t>::max()},
      });
      channel_handles_.emplace(channel_key{.source = space_id_1, .destination = space_id_2},
                               std::move(handle));
    }
  }
}

memory_context::~memory_context() noexcept = default;

std::optional<std::reference_wrapper<const memory_handle>> memory_context::get_memory_handle(
  cucascade::memory::memory_space_id mem_space) const noexcept
{
  if (auto it = memory_handles_.find(mem_space); it != memory_handles_.end()) { return it->second; }
  return std::nullopt;
}

std::optional<std::reference_wrapper<const channel_handle>> memory_context::get_channel_handle(
  cucascade::memory::memory_space_id source,
  cucascade::memory::memory_space_id destination) const noexcept
{
  const channel_key key{
    .source      = source,
    .destination = destination,
  };
  if (auto it = channel_handles_.find(key); it != channel_handles_.end()) { return it->second; }
  return std::nullopt;
}

}  // namespace sirius::telemetry
