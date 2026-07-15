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
#include "telemetry-bridge/gen/batch.rs.h"
#include "telemetry-bridge/gen/memory_tier.rs.h"
#include "telemetry/telemetry_context.hpp"

#include <cucascade/data/data_batch.hpp>

#include <format>

#include <array>
#include <atomic>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
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

/// Read tier + size without assumptions about the batch's representation.
/// Returns nullopt for batches with no data (e.g. empty results).
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

}  // namespace

struct batch_telemetry_registry::impl {
  enum class placement_state { queued, packaged, processing };

  struct placement {
    rust::Box<quent::batch::BatchHandle> handle;
    uuid::UUID pipeline_uuid;
    uuid::UUID task_uuid;  // nil until packaged
    placement_state state;
    // Last seen tier/bytes, re-emitted verbatim by tier-agnostic transitions.
    uuid::UUID tier_resource_id;
    uint64_t bytes;
  };

  struct shard {
    std::mutex mutex;
    std::unordered_map<uint64_t, std::vector<placement>> placements;
  };

  struct port_info {
    uuid::UUID pipeline_uuid;
    uuid::UUID port_uuid;
  };

  std::atomic<bool> enabled{false};

  // Set at install() and immutable until uninstall(); the enabled flag
  // (checked on every entry point) orders access.
  std::shared_ptr<const telemetry_context> context;
  std::vector<rust::Box<quent::memory_tier::MemoryTierHandle>> tier_handles;
  // (tier, device) -> MemoryTier resource. GPU tiers are per-device
  // ("GPU-0", "GPU-1", ...); HOST/DISK are engine-wide (device key 0).
  std::unordered_map<int64_t, uuid::UUID> tier_resources;

  static int64_t tier_key(cucascade::memory::Tier tier, int32_t device_id)
  {
    const int32_t device = tier == cucascade::memory::Tier::GPU ? device_id : 0;
    return (static_cast<int64_t>(tier) << 32) | static_cast<uint32_t>(device);
  }

  std::shared_mutex ports_mutex;
  std::unordered_map<const cucascade::shared_data_repository*, port_info> ports;

  std::array<shard, kNumShards> shards;

  shard& shard_of(uint64_t batch_id) { return shards[batch_id % kNumShards]; }

  uuid::UUID tier_resource_id(cucascade::memory::Tier tier, int32_t device_id) const
  {
    if (auto it = tier_resources.find(tier_key(tier, device_id)); it != tier_resources.end()) {
      return it->second;
    }
    // Unknown device (e.g. a batch with no memory space): fall back to any
    // resource of the tier so the usage still lands on the right tier.
    for (const auto& [key, id] : tier_resources) {
      if (static_cast<cucascade::memory::Tier>(key >> 32) == tier) { return id; }
    }
    return uuid::new_nil();
  }

  /// Re-emit a placement's current state (used for tier changes and
  /// re-packaging); assumes the owning shard mutex is held.
  void reemit_state(placement& p)
  {
    switch (p.state) {
      case placement_state::queued:
        p.handle->batch_queued({
          .tier_resource_id    = p.tier_resource_id,
          .tier_capacity_bytes = p.bytes,
        });
        break;
      case placement_state::packaged:
        p.handle->batch_packaged({
          .instance_name       = "",
          .task_uuid           = p.task_uuid,
          .tier_resource_id    = p.tier_resource_id,
          .tier_capacity_bytes = p.bytes,
        });
        break;
      case placement_state::processing:
        p.handle->batch_processing({
          .instance_name       = "",
          .task_uuid           = p.task_uuid,
          .tier_resource_id    = p.tier_resource_id,
          .tier_capacity_bytes = p.bytes,
        });
        break;
    }
  }

  void consume(placement& p, std::string_view reason)
  {
    p.handle->batch_consumed({
      .instance_name = "",
      .reason        = std::string(reason),
    });
    p.handle->exit();
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

  auto declare_tier = [&](cucascade::memory::Tier tier,
                          int32_t device_id,
                          std::string name,
                          uint64_t capacity_bytes) {
    auto handle = quent::memory_tier::create(impl_->context->context(),
                                             {
                                               .instance_name   = std::move(name),
                                               .parent_group_id = impl_->context->engine_id(),
                                             });
    handle->operating({.capacity_bytes = capacity_bytes});
    impl_->tier_resources[impl::tier_key(tier, device_id)] = handle->uuid();
    impl_->tier_handles.push_back(std::move(handle));
  };

  // One tier resource per GPU device; HOST and DISK are engine-wide.
  for (const auto* space : memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
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
  SIRIUS_LOG_INFO("Batch telemetry installed ({} tier resources).",
                  impl_->tier_handles.size());
}

void batch_telemetry_registry::uninstall()
{
  if (!impl_->enabled.exchange(false, std::memory_order_acq_rel)) { return; }

  // Drain anything a query left behind (normally on_query_end already did).
  for (auto& shard : impl_->shards) {
    std::lock_guard lock(shard.mutex);
    for (auto& [batch_id, placements] : shard.placements) {
      for (auto& p : placements) {
        impl_->consume(p, "query_end");
      }
    }
    shard.placements.clear();
  }
  {
    std::unique_lock lock(impl_->ports_mutex);
    impl_->ports.clear();
  }
  for (auto& handle : impl_->tier_handles) {
    handle->finalizing();
    handle->exit();
  }
  impl_->tier_handles.clear();
  impl_->tier_resources.clear();
  impl_->context.reset();
}

void batch_telemetry_registry::register_consumer_port(
  const cucascade::shared_data_repository* repo, uuid::UUID pipeline_uuid, uuid::UUID port_uuid)
{
  if (!impl_->enabled.load(std::memory_order_acquire) || repo == nullptr) { return; }
  std::unique_lock lock(impl_->ports_mutex);
  impl_->ports[repo] = {pipeline_uuid, port_uuid};
}

void batch_telemetry_registry::on_published(const std::shared_ptr<cucascade::data_batch>& batch,
                                            const cucascade::shared_data_repository* repo,
                                            std::string_view origin)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }

  impl::port_info port;
  {
    std::shared_lock lock(impl_->ports_mutex);
    auto it = impl_->ports.find(repo);
    if (it == impl_->ports.end()) {
      // Repositories outside the plan's port model (if any) are not tracked.
      return;
    }
    port = it->second;
  }

  auto snap = snapshot(batch);
  if (!snap) { return; }
  auto tier_resource_id = impl_->tier_resource_id(snap->tier, snap->device_id);

  auto& shard = impl_->shard_of(snap->batch_id);
  std::lock_guard lock(shard.mutex);
  auto handle = quent::batch::create(impl_->context->context(),
                                     {
                                       .instance_name = std::format("batch-{}", snap->batch_id),
                                       .batch_id      = snap->batch_id,
                                       .pipeline_uuid = port.pipeline_uuid,
                                       .port_uuid     = port.port_uuid,
                                       .origin        = std::string(origin),
                                       .tier_resource_id    = tier_resource_id,
                                       .tier_capacity_bytes = snap->bytes,
                                     });
  handle->batch_queued({
    .tier_resource_id    = tier_resource_id,
    .tier_capacity_bytes = snap->bytes,
  });
  shard.placements[snap->batch_id].push_back(impl::placement{
    .handle           = std::move(handle),
    .pipeline_uuid    = port.pipeline_uuid,
    .task_uuid        = uuid::new_nil(),
    .state            = impl::placement_state::queued,
    .tier_resource_id = tier_resource_id,
    .bytes            = snap->bytes,
  });
}

void batch_telemetry_registry::on_packaged(const std::shared_ptr<cucascade::data_batch>& batch,
                                           uuid::UUID consumer_pipeline_uuid,
                                           uuid::UUID task_uuid)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }
  auto snap = snapshot(batch);
  if (!snap) { return; }
  auto tier_resource_id = impl_->tier_resource_id(snap->tier, snap->device_id);

  auto& shard = impl_->shard_of(snap->batch_id);
  std::lock_guard lock(shard.mutex);
  auto& placements = shard.placements[snap->batch_id];

  // Prefer the queued placement on this consumer; fall back to a previously
  // packaged one (re-claim after an OOM reschedule).
  impl::placement* target = nullptr;
  for (auto& p : placements) {
    if (p.pipeline_uuid == consumer_pipeline_uuid && p.state == impl::placement_state::queued) {
      target = &p;
      break;
    }
  }
  if (target == nullptr) {
    for (auto& p : placements) {
      if (p.pipeline_uuid == consumer_pipeline_uuid &&
          p.state != impl::placement_state::queued) {
        target = &p;
        break;
      }
    }
  }

  if (target == nullptr) {
    // First telemetry sighting of this batch: intermediates released by an
    // OOM reschedule (or otherwise outside the port model) enter directly as
    // packaged.
    auto handle = quent::batch::create(impl_->context->context(),
                                       {
                                         .instance_name = std::format("batch-{}", snap->batch_id),
                                         .batch_id      = snap->batch_id,
                                         .pipeline_uuid = consumer_pipeline_uuid,
                                         .port_uuid     = uuid::new_nil(),
                                         .origin        = "reschedule_intermediate",
                                         .tier_resource_id    = tier_resource_id,
                                         .tier_capacity_bytes = snap->bytes,
                                       });
    placements.push_back(impl::placement{
      .handle           = std::move(handle),
      .pipeline_uuid    = consumer_pipeline_uuid,
      .task_uuid        = uuid::new_nil(),
      .state            = impl::placement_state::queued,
      .tier_resource_id = tier_resource_id,
      .bytes            = snap->bytes,
    });
    target = &placements.back();
    // Note: entry state is registered; transition to packaged below.
  }

  target->task_uuid        = task_uuid;
  target->state            = impl::placement_state::packaged;
  target->tier_resource_id = tier_resource_id;
  target->bytes            = snap->bytes;
  target->handle->batch_packaged({
    .instance_name       = "",
    .task_uuid           = task_uuid,
    .tier_resource_id    = tier_resource_id,
    .tier_capacity_bytes = snap->bytes,
  });
}

void batch_telemetry_registry::on_processing(const std::shared_ptr<cucascade::data_batch>& batch,
                                             uuid::UUID task_uuid)
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
    if (p.task_uuid == task_uuid && p.state == impl::placement_state::packaged) {
      p.state            = impl::placement_state::processing;
      p.tier_resource_id = tier_resource_id;
      p.bytes            = snap->bytes;
      p.handle->batch_processing({
        .instance_name       = "",
        .task_uuid           = task_uuid,
        .tier_resource_id    = tier_resource_id,
        .tier_capacity_bytes = snap->bytes,
      });
    }
  }
}

void batch_telemetry_registry::on_consumed(uint64_t batch_id, uuid::UUID task_uuid)
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }

  auto& shard = impl_->shard_of(batch_id);
  std::lock_guard lock(shard.mutex);
  auto it = shard.placements.find(batch_id);
  if (it == shard.placements.end()) { return; }
  auto& placements = it->second;
  for (auto p = placements.begin(); p != placements.end();) {
    // Only the currently claiming task consumes; a placement re-claimed by a
    // rescheduled task carries that task's uuid and is left alone.
    if (p->task_uuid == task_uuid && p->state != impl::placement_state::queued) {
      impl_->consume(*p,
                     p->state == impl::placement_state::processing ? "processed" : "task_failed");
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

void batch_telemetry_registry::on_query_end()
{
  if (!impl_->enabled.load(std::memory_order_acquire)) { return; }

  size_t drained = 0;
  for (auto& shard : impl_->shards) {
    std::lock_guard lock(shard.mutex);
    for (auto& [batch_id, placements] : shard.placements) {
      for (auto& p : placements) {
        impl_->consume(p, "query_end");
        ++drained;
      }
    }
    shard.placements.clear();
  }
  {
    std::unique_lock lock(impl_->ports_mutex);
    impl_->ports.clear();
  }
  if (drained > 0) {
    SIRIUS_LOG_DEBUG("Batch telemetry: drained {} placement(s) at query end.", drained);
  }
}

}  // namespace sirius::telemetry
