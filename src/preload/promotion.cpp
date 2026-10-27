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

#include "preload/promotion.hpp"

#include "data/convertible_data.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "log/logging.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda/stream>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cuda/event.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <optional>
#include <typeinfo>
#include <vector>

namespace sirius::preload {

namespace {

using cucascade::memory::Tier;

/// Bytes of the INT64 copies of INT32 string offsets in @p columns, recursing through children.
[[nodiscard]] std::size_t widened_string_offsets_bytes(
  std::vector<cucascade::memory::column_metadata> const& columns) noexcept
{
  std::size_t total = 0;
  for (auto const& column : columns) {
    if (static_cast<cudf::type_id>(column.type_id) == cudf::type_id::STRING &&
        !column.children.empty() &&
        static_cast<cudf::type_id>(column.children.front().type_id) == cudf::type_id::INT32) {
      auto const rows = static_cast<std::size_t>(std::max(column.num_rows, 0));
      total           = sat_add(total, sat_mul(sizeof(std::int64_t), sat_add(rows, 1)));
    }
    total = sat_add(total, widened_string_offsets_bytes(column.children));
  }
  return total;
}

/// Consumed bytes of @p space (@ref consumed_bytes), counting every live reservation, are at or
/// below @p ceiling and the space is below its downgrade trigger.
[[nodiscard]] bool admissible(cucascade::memory::memory_space& space, std::size_t ceiling)
{
  auto const consumed = consumed_bytes(space.get_max_memory(), space.get_available_memory());
  return consumed <= ceiling && !space.should_downgrade_memory();
}

[[nodiscard]] bool gpu_held_off(promotion_gpu const& gpu, memory::memory_wait_board const& board)
{
  return held_off(board, gpu.space->get_device_id());
}

/// A GPU on @p host's NUMA node is held off or at its downgrade trigger.
[[nodiscard]] bool local_gpu_pressed(promotion_host const& host, promotion_env const& env)
{
  return std::any_of(host.local_gpus.begin(), host.local_gpus.end(), [&](std::size_t index) {
    if (index >= env.gpus.size() || env.gpus[index].space == nullptr) { return false; }
    auto const& gpu = env.gpus[index];
    return gpu_held_off(gpu, env.board) || gpu.space->should_downgrade_memory();
  });
}

[[nodiscard]] promotion_result outcome(promotion_outcome o) noexcept { return {o}; }

[[nodiscard]] promotion_result stale_at(cucascade::memory::memory_space const& current) noexcept
{
  promotion_result r{promotion_outcome::stale};
  r.observed = current.get_id();
  return r;
}

[[nodiscard]] promotion_result promoted(promotion_kind kind,
                                        std::size_t bytes,
                                        std::uint64_t conversion_seq) noexcept
{
  promotion_result r{promotion_outcome::promoted};
  r.bytes          = bytes;
  r.kind           = kind;
  r.conversion_seq = conversion_seq;
  return r;
}

/// A sequence number ordered after every downgrade record of this batch taken so far. Call it
/// while holding the batch's exclusive lock.
[[nodiscard]] std::uint64_t next_conversion_seq() noexcept
{
  return g_conversion_seq.fetch_add(1, std::memory_order_relaxed);
}

void run_test_hook(promotion_env const& env, promotion_job const& job)
{
  if (env.test_hook) { env.test_hook(job); }
}

/**
 * @brief Promote a HOST or DISK batch to the GPU target `env.gpus[job.t.index]`.
 *
 * Call only with the query's work lease held and no executor lock: the batch strong reference
 * must be released before the lease. The batch is locked only with @c try_to_mutable.
 *
 * @throws rmm::out_of_memory when the copy overruns the attached reservation. The source
 *         representation is left in place on any throw.
 */
promotion_result promote_to_gpu(promotion_job const& job, promotion_env const& env)
{
  if (job.t.index >= env.gpus.size()) { return outcome(promotion_outcome::failed); }
  auto const& gpu = env.gpus[job.t.index];
  if (gpu.space == nullptr || gpu.space->get_tier() != Tier::GPU || gpu.streams == nullptr) {
    return outcome(promotion_outcome::failed);
  }

  if (gpu_held_off(gpu, env.board)) { return outcome(promotion_outcome::held_off); }
  // Reserve before touching the batch, so a refused space costs no batch lock.
  auto reservation = gpu.space->make_reservation_or_null(std::max<std::size_t>(job.est_reserve, 1));
  if (!reservation) { return outcome(promotion_outcome::no_memory); }
  if (!admissible(*gpu.space, job.ceiling)) { return outcome(promotion_outcome::gated); }
  run_test_hook(env, job);

  auto batch = job.batch.lock();
  if (!batch) { return outcome(promotion_outcome::expired); }
  if (batch->get_state() != cucascade::batch_state::idle) {
    return outcome(promotion_outcome::contended);
  }
  auto mut = batch->try_to_mutable();
  if (!mut) { return outcome(promotion_outcome::contended); }
  auto* const current = mut->get_memory_space();
  auto* const data    = mut->get_data();
  if (current == nullptr || data == nullptr) { return outcome(promotion_outcome::expired); }
  // Never move a batch between GPUs: on any GPU, the consumer's own path handles it.
  if (current->get_tier() == Tier::GPU) { return outcome(promotion_outcome::already_resident); }
  if (current->get_id() != job.expected_source) { return stale_at(*current); }
  if (job.k == klass::B && batch->get_subscriber_count() > 0) { return stale_at(*current); }

  auto const kind = classify(*data);
  auto& registry  = sirius::converter_registry::get();
  if (kind == repr_kind::other) { return outcome(promotion_outcome::unsupported); }

  auto const need = sat_mul(with_slack(promotion_peak_bytes(*data)),
                            std::max<std::uint32_t>(job.oom_multiplier, 1));
  if (need > gpu.max_batch_bytes) { return outcome(promotion_outcome::too_big); }
  if (need > reservation->size() && !reservation->grow_by(need - reservation->size())) {
    return outcome(promotion_outcome::no_memory);
  }
  // Last checks before the copy: a task may have stalled, or memory filled, since the reserve.
  if (gpu_held_off(gpu, env.board)) { return outcome(promotion_outcome::held_off); }
  if (!admissible(*gpu.space, job.ceiling)) { return outcome(promotion_outcome::gated); }

  bool const from_disk = current->get_tier() == Tier::DISK;
  sirius::nvtx_scoped_range const range{from_disk ? "sirius::preload::promote_d2g"
                                                  : "sirius::preload::promote_h2g"};
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{gpu.space->get_device_id()}};
  auto stream = gpu.streams->acquire_stream(
    cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
  ::cuda::stream_ref const s{stream.get()};
  if (cudaEvent_t const writer = data->get_writer_event(); writer != nullptr) {
    cucascade::cuda::cuda_event_view{writer}.wait(s);
  }

  // GPU reservations are tracked per allocating thread (or stream): attach ours, so every
  // allocation of the copy draws from it, and an overrun fails rather than exceeding it.
  auto* const allocator = reservation->get_memory_resource_of<Tier::GPU>();
  if (allocator == nullptr ||
      !allocator->attach_reservation_to_tracker(
        s,
        std::move(reservation),
        std::make_unique<cucascade::memory::fail_reservation_limit_policy>())) {
    return outcome(promotion_outcome::failed);
  }
  absl::Cleanup const detach = [allocator, s] { allocator->reset_stream_reservation(s); };

  mut->convert_to<cucascade::gpu_table_representation>(registry, gpu.space, s);
  return promoted(
    from_disk ? promotion_kind::d2g : promotion_kind::h2g, need, next_conversion_seq());
}

/**
 * @brief Stage a DISK batch into the HOST target `env.hosts[job.t.index]`.
 *
 * Same preconditions as @ref promote_to_gpu.
 *
 * @throws rmm::out_of_memory when the host pool cannot supply the blocks. The source
 *         representation is left in place on any throw.
 */
promotion_result stage_to_host(promotion_job const& job, promotion_env const& env)
{
  if (job.t.index >= env.hosts.size()) { return outcome(promotion_outcome::failed); }
  auto const& host = env.hosts[job.t.index];
  if (host.space == nullptr || host.space->get_tier() != Tier::HOST) {
    return outcome(promotion_outcome::failed);
  }

  if (local_gpu_pressed(host, env)) { return outcome(promotion_outcome::held_off); }
  auto reservation =
    host.space->make_reservation_or_null(std::max<std::size_t>(job.est_reserve, 1));
  if (!reservation) { return outcome(promotion_outcome::no_memory); }
  if (!admissible(*host.space, job.ceiling)) { return outcome(promotion_outcome::gated); }
  run_test_hook(env, job);

  auto batch = job.batch.lock();
  if (!batch) { return outcome(promotion_outcome::expired); }
  if (batch->get_state() != cucascade::batch_state::idle) {
    return outcome(promotion_outcome::contended);
  }
  auto mut = batch->try_to_mutable();
  if (!mut) { return outcome(promotion_outcome::contended); }
  auto* const current = mut->get_memory_space();
  auto* const data    = mut->get_data();
  if (current == nullptr || data == nullptr) { return outcome(promotion_outcome::expired); }
  if (current->get_tier() != Tier::DISK) { return outcome(promotion_outcome::already_resident); }
  if (current->get_id() != job.expected_source) { return stale_at(*current); }
  if (job.k == klass::B && batch->get_subscriber_count() > 0) { return stale_at(*current); }

  auto const kind = classify(*data);
  if (kind != repr_kind::disk_data) { return outcome(promotion_outcome::unsupported); }
  // A table read draws its host blocks from the reservation; the 4 KiB-aligned DISK size is an
  // upper bound on them, so the ceiling check may slightly over-count.
  auto const need = data->get_size_in_bytes();
  if (need > reservation->size() && !reservation->grow_by(need - reservation->size())) {
    return outcome(promotion_outcome::no_memory);
  }
  if (local_gpu_pressed(host, env)) { return outcome(promotion_outcome::held_off); }
  if (!admissible(*host.space, job.ceiling)) { return outcome(promotion_outcome::gated); }

  std::optional<cucascade::memory::borrowed_stream> borrowed;
  ::cuda::stream_ref s{cudaStreamPerThread};
  if (host.ordering_streams != nullptr) {
    borrowed.emplace(host.ordering_streams->acquire_stream(
      cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW));
    s = borrowed->get();
  }

  auto& registry = sirius::converter_registry::get();
  sirius::nvtx_scoped_range const range{"sirius::preload::promote_d2h"};
  mut->convert_to<cucascade::host_data_representation>(registry, *reservation, s);
  return promoted(promotion_kind::d2h, need, next_conversion_seq());
}

}  // namespace

repr_kind classify(cucascade::idata_representation const& data) noexcept
{
  auto const& type = typeid(data);
  if (type == typeid(cucascade::host_data_representation)) { return repr_kind::host_data; }
  if (type == typeid(cucascade::disk_data_representation)) { return repr_kind::disk_data; }
  return repr_kind::other;
}

std::size_t promotion_peak_bytes(cucascade::idata_representation const& data)
{
  auto const base = sirius::peak_materialization_bytes(&data);
  switch (classify(data)) {
    case repr_kind::host_data: {
      auto const& table =
        static_cast<cucascade::host_data_representation const&>(data).get_host_table();
      return table ? sat_add(base, widened_string_offsets_bytes(table->columns)) : base;
    }
    case repr_kind::disk_data:
      return sat_add(
        base,
        widened_string_offsets_bytes(
          static_cast<cucascade::disk_data_representation const&>(data).get_disk_table().columns));
    case repr_kind::other: break;
  }
  return base;
}

promotion_result promote_one(promotion_job const& job, promotion_env const& env) noexcept
{
  try {
    // The lease keeps the query's repositories, and so the batch's consumers, alive for the
    // whole attempt. Declared first, it is released last.
    auto const lease = env.lifecycle.try_acquire_work(job.query);
    if (!lease) { return outcome(promotion_outcome::retired); }
    return job.t.tier == Tier::GPU    ? promote_to_gpu(job, env)
           : job.t.tier == Tier::HOST ? stage_to_host(job, env)
                                      : outcome(promotion_outcome::failed);
  } catch (rmm::out_of_memory const&) {
    // On a GPU this is a low estimate, local to the candidate. A HOST pool that is out of
    // blocks has no estimate to grow: refuse the space, as a failed reservation does.
    return outcome(job.t.tier == Tier::GPU ? promotion_outcome::oom : promotion_outcome::no_memory);
  } catch (std::exception const& e) {
    // The executor warns once per query; the cause of every failure is logged here.
    log::log_noexcept([&] {
      SIRIUS_LOG_DEBUG("[preload] promotion of batch {} failed: {}", job.batch_id, e.what());
    });
    return outcome(promotion_outcome::failed);
  } catch (...) {
    return outcome(promotion_outcome::failed);
  }
}

}  // namespace sirius::preload
