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
#include <optional>

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

memory_context::memory_context(uuid::UUID engine_uuid,
                               const quent::Context& context,
                               const cucascade::memory::memory_reservation_manager* manager)
{
  if (manager == nullptr) { return; }

  for (const auto& mem_space : manager->get_all_memory_spaces()) {
    auto handle = quent::memory::create(context,
                                        {
                                          .instance_name   = mem_space->to_string(),
                                          .parent_group_id = engine_uuid,
                                        });
    handle->operating({
      .capacity_bytes = mem_space->get_max_memory(),
    });
    memory_handles_.insert({
      mem_space->get_id(),
      std::move(handle),
    });
  }

  for (const auto& [space_id_1, handle_1] : memory_handles_) {
    for (const auto& [space_id_2, handle_2] : memory_handles_) {
      if (space_id_1 == space_id_2) {
        continue;  // skip inserting a channel between the same space.
      }
      channel_handles_.insert({
        channel_key{.source = space_id_1, .destination = space_id_2},
        quent::channel::create(context,
                               {
                                 .instance_name   = std::format("{}-{}->{}-{}",
                                                              tier_to_string(space_id_1.tier),
                                                              space_id_1.device_id,
                                                              tier_to_string(space_id_2.tier),
                                                              space_id_2.device_id),
                                 .parent_group_id = engine_uuid,
                                 .source_id       = handle_1->uuid(),
                                 .target_id       = handle_2->uuid(),
                               }),
      });
    }
  }

  for (auto& [_, c_handle] : channel_handles_) {
    c_handle->operating({
      .capacity_bytes = std::numeric_limits<uint64_t>::max(),
    });
  }
}

memory_context::~memory_context()
{
  for (auto& [_, handle] : memory_handles_) {
    handle->finalizing();
    handle->exit();
  }

  for (auto& [_, handle] : channel_handles_) {
    handle->finalizing();
    handle->exit();
  }
}

std::optional<std::reference_wrapper<const quent::memory::MemoryHandle>>
memory_context::get_memory_handle(cucascade::memory::memory_space_id mem_space) const noexcept
{
  if (auto it = memory_handles_.find(mem_space); it != memory_handles_.end()) {
    return *(it->second);
  }
  return std::nullopt;
}

std::optional<std::reference_wrapper<const quent::channel::ChannelHandle>>
memory_context::get_channel_handle(cucascade::memory::memory_space_id source,
                                   cucascade::memory::memory_space_id destination) const noexcept
{
  const channel_key key{
    .source      = source,
    .destination = destination,
  };
  if (auto it = channel_handles_.find(key); it != channel_handles_.end()) { return *(it->second); }
  return std::nullopt;
}

}  // namespace sirius::telemetry
