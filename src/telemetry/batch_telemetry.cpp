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

#include "telemetry/batch_telemetry.hpp"

#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "telemetry/runtime_fsm_handle.hpp"
#include "telemetry/telemetry_context.hpp"

#include <cucascade/data/data_batch.hpp>

#include <array>
#include <atomic>
#include <format>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::telemetry {

namespace {

constexpr size_t kNumShards = 16;

constexpr std::string_view tier_name(cucascade::memory::Tier tier)
{
  switch (tier) {
    case cucascade::memory::Tier::GPU: return "GPU";
    case cucascade::memory::Tier::HOST: return "HOST";
    case cucascade::memory::Tier::DISK: return "DISK";
    default: return "UNKNOWN";
  }
}

constexpr std::array<cucascade::memory::Tier, 3> kTiers = {
  cucascade::memory::Tier::GPU,
  cucascade::memory::Tier::HOST,
  cucascade::memory::Tier::DISK,
};

/// Read tier + size; nullopt for batches with no data.
struct batch_snapshot {
  uint64_t batch_id;
  cucascade::memory::Tier tier;
  int32_t device_id;
  uint64_t bytes;
};

std::optional<batch_snapshot> snapshot(const std::shared_ptr<cucascade::data_batch>& batch)
{
  if (!batch) { return std::nullopt; }
  auto ro = batch->to_read_only();
  if (!ro.get_data()) { return std::nullopt; }
  const auto* space = ro.get_memory_space();
  return batch_snapshot{
    .batch_id  = ro.get_batch_id(),
    .tier      = ro.get_current_tier(),
    .device_id = space != nullptr ? space->get_id().device_id : 0,
    .bytes     = ro.get_data()->get_size_in_bytes(),
  };
}

std::optional<quent::refs::MemoryTierUsageRef> tier_usage(quent::Uuid resource_id, uint64_t bytes)
{
  if (resource_id == quent::nil_uuid()) { return std::nullopt; }
  return quent::refs::MemoryTierUsageRef{
    .target = quent::memory_tier::MemoryTierId(resource_id),
    .data   = quent::records::MemoryTierUsage{.bytes = bytes},
  };
}

std::optional<quent::port::PortId> optional_port_id(quent::Uuid id)
{
  if (id == quent::nil_uuid()) { return std::nullopt; }
  return quent::port::PortId(id);
}

}  // namespace

struct batch_telemetry_registry::impl {
  using placement_handle = runtime_fsm_handle<quent::BatchPlacement,
                                              quent::batch_placement_state::BatchQueued,
                                              quent::batch_placement_state::BatchPackaged,
                                              quent::batch_placement_state::BatchProcessing,
                                              quent::batch_placement_state::BatchConsumed>;

  struct placement {
    placement_handle handle;
    sirius::query_id_t query_id;
    quent::Uuid pipeline_uuid;
    quent::Uuid task_uuid;  // nil until packaged
    // Last seen tier/bytes, re-emitted verbatim by tier-agnostic transitions.
    quent::Uuid tier_resource_id;
    uint64_t bytes;
  };

  struct shard {
    std::mutex mutex;
    std::unordered_map<uint64_t, std::vector<placement>> placements;
  };

  struct port_info {
    sirius::query_id_t query_id;
    quent::Uuid pipeline_uuid;
    quent::Uuid port_uuid;
  };

  std::atomic<bool> enabled{false};

  // Immutable between install() and uninstall(); ordered by `enabled`.
  std::shared_ptr<const telemetry_context> context;
  // (tier, device) -> MemoryTier resource; HOST/DISK use device key 0.
  std::unordered_map<int64_t, quent::Uuid> tier_resources;

  static int64_t tier_key(cucascade::memory::Tier tier, int32_t device_id)
  {
    const int32_t device = tier == cucascade::memory::Tier::GPU ? device_id : 0;
    return (static_cast<int64_t>(tier) << 32) | static_cast<uint32_t>(device);
  }

  std::shared_mutex ports_mutex;
  std::unordered_map<const cucascade::shared_data_repository*, port_info> ports;

  std::array<shard, kNumShards> shards;

  shard& shard_of(uint64_t batch_id) { return shards[batch_id % kNumShards]; }

  quent::Uuid tier_resource_id(cucascade::memory::Tier tier, int32_t device_id) const
  {
    if (auto it = tier_resources.find(tier_key(tier, device_id)); it != tier_resources.end()) {
      return it->second;
    }
    // Unknown device: fall back to any resource of the tier.
    for (const auto& [key, id] : tier_resources) {
      if (static_cast<cucascade::memory::Tier>(key >> 32) == tier) { return id; }
    }
    return quent::nil_uuid();
  }

  /// Re-emit a placement's current state; the shard mutex must be held.
  void reemit_state(placement& p)
  {
    if (p.handle.transition<quent::batch_placement_state::BatchQueued>([&p](auto&& current) {
          return std::move(current).batch_queued(quent::batch_placement::BatchQueued{
            .tier = tier_usage(p.tier_resource_id, p.bytes),
          });
        })) {
      return;
    }
    if (p.handle.transition<quent::batch_placement_state::BatchPackaged>([&p](auto&& current) {
          return std::move(current).batch_packaged(quent::batch_placement::BatchPackaged{
            .task_uuid = p.task_uuid,
            .tier      = tier_usage(p.tier_resource_id, p.bytes),
          });
        })) {
      return;
    }
    if (!p.handle.transition<quent::batch_placement_state::BatchProcessing>([&p](auto&& current) {
          return std::move(current).batch_processing(quent::batch_placement::BatchProcessing{
            .task_uuid = p.task_uuid,
            .tier      = tier_usage(p.tier_resource_id, p.bytes),
          });
        })) {
      throw std::logic_error("invalid Quent BatchPlacement state during tier update");
    }
  }

  void consume(placement& p, batch_consumed_reason reason)
  {
    if (!p.handle.holds<quent::batch_placement_state::BatchConsumed>() &&
        !p.handle.transition<quent::batch_placement_state::BatchQueued,
                             quent::batch_placement_state::BatchPackaged,
                             quent::batch_placement_state::BatchProcessing>(
          [reason](auto&& current) {
            return std::move(current).batch_consumed(quent::batch_placement::BatchConsumed{
              .reason = std::string(to_string_view(reason)),
            });
          })) {
      throw std::logic_error("invalid Quent BatchPlacement state during consumption");
    }
  }
};

batch_telemetry_registry::batch_telemetry_registry() : impl_(std::make_unique<impl>()) {}
batch_telemetry_registry::~batch_telemetry_registry() = default;

batch_telemetry_registry& batch_telemetry_registry::instance()
{
  static batch_telemetry_registry registry;
  return registry;
}

void batch_telemetry_registry::install(
  std::shared_ptr<const telemetry_context> context,
  sirius::memory::sirius_memory_reservation_manager& memory_manager)
{
  if (!context) { return; }
  if (impl_->enabled.load(std::memory_order_acquire)) {
    SIRIUS_LOG_WARN("batch_telemetry_registry::install: already installed; ignoring.");
    return;
  }
  impl_->context = std::move(context);

  auto declare_tier =
    [&](
      cucascade::memory::Tier tier, int32_t device_id, std::string name, uint64_t capacity_bytes) {
      auto handle = impl_->context->context().memory_tier_observer()->handle();
      handle.declaration(quent::memory_tier::Declaration{
        .instance_name   = name,
        .parent_group_id = quent::engine::EngineId(impl_->context->engine_id()),
        .bounds          = quent::records::MemoryTierBounds{.bytes = capacity_bytes},
      });
      impl_->tier_resources[impl::tier_key(tier, device_id)] = handle.id().raw();
    };

  for (const auto* space :
       memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
    const auto device_id = space->get_id().device_id;
    declare_tier(cucascade::memory::Tier::GPU,
                 device_id,
                 std::format("GPU-{}", device_id),
                 space->get_max_memory());
  }
  for (auto tier : {cucascade::memory::Tier::HOST, cucascade::memory::Tier::DISK}) {
    uint64_t capacity_bytes = 0;
    for (const auto* space : memory_manager.get_memory_spaces_for_tier(tier)) {
      capacity_bytes += space->get_max_memory();
    }
    declare_tier(tier, 0, std::string(tier_name(tier)), capacity_bytes);
  }

  impl_->enabled.store(true, std::memory_order_release);
  SIRIUS_LOG_INFO("Batch telemetry installed ({} tier resources).", impl_->tier_resources.size());
}

void batch_telemetry_registry::uninstall()
{
  if (!impl_->enabled.exchange(false, std::memory_order_acq_rel)) { return; }

  for (auto& shard : impl_->shards) {
    std::lock_guard lock(shard.mutex);
    for (auto& [batch_id, placements] : shard.placements) {
      for (auto& p : placements) {
        impl_->consume(p, batch_consumed_reason::query_end);
      }
    }
    shard.placements.clear();
  }
  {
    std::unique_lock lock(impl_->ports_mutex);
    impl_->ports.clear();
  }
  impl_->tier_resources.clear();
  impl_->context.reset();
}

void batch_telemetry_registry::register_consumer_port(const cucascade::shared_data_repository* repo,
                                                      sirius::query_id_t query_id,
                                                      quent::Uuid pipeline_uuid,
                                                      quent::Uuid port_uuid)
{
  if (!impl_->enabled.load(std::memory_order_acquire) || repo == nullptr) { return; }
  std::unique_lock lock(impl_->ports_mutex);
  impl_->ports[repo] = {query_id, pipeline_uuid, port_uuid};
}

void batch_telemetry_registry::on_published(const std::shared_ptr<cucascade::data_batch>& batch,
                                            const cucascade::shared_data_repository* repo,
                                            const batch_origin origin)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }

  impl::port_info port;
  {
    std::shared_lock lock(impl_->ports_mutex);
    auto it = impl_->ports.find(repo);
    if (it == impl_->ports.end()) { return; }
    port = it->second;
  }

  auto snap = snapshot(batch);
  if (!snap) { return; }
  auto tier_resource_id = impl_->tier_resource_id(snap->tier, snap->device_id);

  auto& shard = impl_->shard_of(snap->batch_id);
  std::lock_guard lock(shard.mutex);
  auto registered = impl_->context->context().batch_placement_observer()->handle().batch_registered(
    quent::batch_placement::BatchRegistered{
      .instance_name = std::format("batch-{}", snap->batch_id),
      .batch_id      = snap->batch_id,
      .pipeline_uuid = quent::operator_::OperatorId(port.pipeline_uuid),
      .port_uuid     = optional_port_id(port.port_uuid),
      .origin        = std::string(to_string_view(origin)),
      .tier          = tier_usage(tier_resource_id, snap->bytes),
    });
  auto queued = std::move(registered)
                  .batch_queued(quent::batch_placement::BatchQueued{
                    .tier = tier_usage(tier_resource_id, snap->bytes),
                  });
  shard.placements[snap->batch_id].push_back(impl::placement{
    .handle           = impl::placement_handle{std::move(queued)},
    .query_id         = port.query_id,
    .pipeline_uuid    = port.pipeline_uuid,
    .task_uuid        = quent::nil_uuid(),
    .tier_resource_id = tier_resource_id,
    .bytes            = snap->bytes,
  });
}

void batch_telemetry_registry::on_packaged(const std::shared_ptr<cucascade::data_batch>& batch,
                                           sirius::query_id_t query_id,
                                           quent::Uuid consumer_pipeline_uuid,
                                           quent::Uuid task_uuid)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }
  auto snap = snapshot(batch);
  if (!snap) { return; }
  auto tier_resource_id = impl_->tier_resource_id(snap->tier, snap->device_id);

  auto& shard = impl_->shard_of(snap->batch_id);
  std::lock_guard lock(shard.mutex);
  auto& placements = shard.placements[snap->batch_id];

  // Prefer this consumer's queued placement; else a packaged one (re-claim).
  impl::placement* target = nullptr;
  for (auto& p : placements) {
    if (p.query_id == query_id && p.pipeline_uuid == consumer_pipeline_uuid &&
        p.handle.holds<quent::batch_placement_state::BatchQueued>()) {
      target = &p;
      break;
    }
  }
  if (target == nullptr) {
    for (auto& p : placements) {
      if (p.query_id == query_id && p.pipeline_uuid == consumer_pipeline_uuid) {
        target = &p;
        break;
      }
    }
  }

  if (target == nullptr) {
    // First sighting: register lazily, then package below.
    auto registered =
      impl_->context->context().batch_placement_observer()->handle().batch_registered(
        quent::batch_placement::BatchRegistered{
          .instance_name = std::format("batch-{}", snap->batch_id),
          .batch_id      = snap->batch_id,
          .pipeline_uuid = quent::operator_::OperatorId(consumer_pipeline_uuid),
          .port_uuid     = std::nullopt,
          .origin        = std::string(to_string_view(batch_origin::reschedule_intermediate)),
          .tier          = tier_usage(tier_resource_id, snap->bytes),
        });
    auto packaged = std::move(registered)
                      .batch_packaged(quent::batch_placement::BatchPackaged{
                        .task_uuid = task_uuid,
                        .tier      = tier_usage(tier_resource_id, snap->bytes),
                      });
    placements.push_back(impl::placement{
      .handle           = impl::placement_handle{std::move(packaged)},
      .query_id         = query_id,
      .pipeline_uuid    = consumer_pipeline_uuid,
      .task_uuid        = task_uuid,
      .tier_resource_id = tier_resource_id,
      .bytes            = snap->bytes,
    });
    return;
  }

  target->task_uuid        = task_uuid;
  target->tier_resource_id = tier_resource_id;
  target->bytes            = snap->bytes;
  if (!target->handle.transition<quent::batch_placement_state::BatchQueued,
                                 quent::batch_placement_state::BatchPackaged,
                                 quent::batch_placement_state::BatchProcessing>(
        [task_uuid, tier_resource_id, bytes = snap->bytes](auto&& current) {
          return std::move(current).batch_packaged(quent::batch_placement::BatchPackaged{
            .task_uuid = task_uuid,
            .tier      = tier_usage(tier_resource_id, bytes),
          });
        })) {
    throw std::logic_error("invalid Quent BatchPlacement transition to packaged");
  }
}

void batch_telemetry_registry::on_processing(const std::shared_ptr<cucascade::data_batch>& batch,
                                             quent::Uuid task_uuid)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }
  auto snap = snapshot(batch);
  if (!snap) { return; }
  auto tier_resource_id = impl_->tier_resource_id(snap->tier, snap->device_id);

  auto& shard = impl_->shard_of(snap->batch_id);
  std::lock_guard lock(shard.mutex);
  auto it = shard.placements.find(snap->batch_id);
  if (it == shard.placements.end()) { return; }
  for (auto& p : it->second) {
    if (p.task_uuid == task_uuid && p.handle.holds<quent::batch_placement_state::BatchPackaged>()) {
      p.tier_resource_id = tier_resource_id;
      p.bytes            = snap->bytes;
      static_cast<void>(p.handle.transition<quent::batch_placement_state::BatchPackaged>(
        [task_uuid, tier_resource_id, bytes = snap->bytes](auto&& current) {
          return std::move(current).batch_processing(quent::batch_placement::BatchProcessing{
            .task_uuid = task_uuid,
            .tier      = tier_usage(tier_resource_id, bytes),
          });
        }));
    }
  }
}

void batch_telemetry_registry::on_processing_by_id(uint64_t batch_id, quent::Uuid task_uuid)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }

  auto& shard = impl_->shard_of(batch_id);
  std::lock_guard lock(shard.mutex);
  auto it = shard.placements.find(batch_id);
  if (it == shard.placements.end()) { return; }
  for (auto& p : it->second) {
    if (p.task_uuid == task_uuid && p.handle.holds<quent::batch_placement_state::BatchPackaged>()) {
      static_cast<void>(p.handle.transition<quent::batch_placement_state::BatchPackaged>(
        [task_uuid, &p](auto&& current) {
          return std::move(current).batch_processing(quent::batch_placement::BatchProcessing{
            .task_uuid = task_uuid,
            .tier      = tier_usage(p.tier_resource_id, p.bytes),
          });
        }));
    }
  }
}

void batch_telemetry_registry::on_consumed(uint64_t batch_id, quent::Uuid task_uuid)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }

  auto& shard = impl_->shard_of(batch_id);
  std::lock_guard lock(shard.mutex);
  auto it = shard.placements.find(batch_id);
  if (it == shard.placements.end()) { return; }
  auto& placements = it->second;
  for (auto p = placements.begin(); p != placements.end();) {
    // Only the currently claiming task consumes; re-claims are left alone.
    if (p->task_uuid == task_uuid &&
        !p->handle.holds<quent::batch_placement_state::BatchQueued>()) {
      impl_->consume(*p,
                     p->handle.holds<quent::batch_placement_state::BatchProcessing>()
                       ? batch_consumed_reason::processed
                       : batch_consumed_reason::task_failed);
      p = placements.erase(p);
    } else {
      ++p;
    }
  }
  if (placements.empty()) { shard.placements.erase(it); }
}

void batch_telemetry_registry::on_tier_change(uint64_t batch_id,
                                              cucascade::memory::Tier tier,
                                              int32_t device_id,
                                              uint64_t bytes)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }
  auto tier_resource_id = impl_->tier_resource_id(tier, device_id);

  auto& shard = impl_->shard_of(batch_id);
  std::lock_guard lock(shard.mutex);
  auto it = shard.placements.find(batch_id);
  if (it == shard.placements.end()) { return; }
  for (auto& p : it->second) {
    if (p.tier_resource_id == tier_resource_id && p.bytes == bytes) { continue; }
    p.tier_resource_id = tier_resource_id;
    p.bytes            = bytes;
    impl_->reemit_state(p);
  }
}

quent::Uuid batch_telemetry_registry::tier_resource(cucascade::memory::Tier tier,
                                                    int32_t device_id) const
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return quent::nil_uuid(); }
  return impl_->tier_resource_id(tier, device_id);
}

void batch_telemetry_registry::on_query_end(sirius::query_id_t query_id)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }

  size_t drained = 0;
  for (auto& shard : impl_->shards) {
    std::lock_guard lock(shard.mutex);
    for (auto entry = shard.placements.begin(); entry != shard.placements.end();) {
      auto& placements = entry->second;
      for (auto placement = placements.begin(); placement != placements.end();) {
        if (placement->query_id != query_id) {
          ++placement;
          continue;
        }
        impl_->consume(*placement, batch_consumed_reason::query_end);
        placement = placements.erase(placement);
        ++drained;
      }
      if (placements.empty()) {
        entry = shard.placements.erase(entry);
      } else {
        ++entry;
      }
    }
  }
  {
    std::unique_lock lock(impl_->ports_mutex);
    std::erase_if(impl_->ports,
                  [query_id](const auto& entry) { return entry.second.query_id == query_id; });
  }
  if (drained > 0) {
    SIRIUS_LOG_DEBUG("Batch telemetry: drained {} placement(s) at query end.", drained);
  }
}

}  // namespace sirius::telemetry
