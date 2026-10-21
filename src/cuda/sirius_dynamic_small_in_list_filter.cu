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

// sirius
#include <log/logging.hpp>
#include <op/dynamic_filter/dynamic_filter_device.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_reservation.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_space.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_transfer.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

// cccl
#include <cuda/dynamic_filter_fused_probe.cuh>
#include <thrust/copy.h>

// cucascade
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

// cuda
#include <cuda_runtime_api.h>

// standard library
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op {

//===----------------------------------------------------------------------===//
// Per-device needle storage (PIMPL)
//===----------------------------------------------------------------------===//

/// @brief Per-device raw snapshots of the build keys at the key rep. Mirrors
/// sirius_dynamic_in_list_filter's replica store, but a device_buffer of raw bytes needs no cuco
/// set / typed variant: the outer class's _domain.rep decodes the bytes.
struct sirius_dynamic_small_in_list_filter::needle_store {
  /// @brief One device-local needle buffer. Frees on its owning device (an rmm::device_buffer
  /// frees on the current device, so teardown must restore that device first — mirrors
  /// set_replica).
  struct needle_replica {
    int device_id = -1;
    rmm::device_buffer needles;

    needle_replica(int device_id, rmm::device_buffer needles)
      : device_id{device_id}, needles{std::move(needles)}
    {
    }

    needle_replica(needle_replica const&)            = delete;
    needle_replica& operator=(needle_replica const&) = delete;
    needle_replica(needle_replica&&)                 = delete;
    needle_replica& operator=(needle_replica&&)      = delete;

    ~needle_replica() noexcept
    {
      if (device_id < 0 || needles.is_empty()) { return; }
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      needles = rmm::device_buffer{};
    }
  };

  int source_device    = -1;
  std::size_t num_keys = 0;  ///< Keys in every replica; also the filter's size().
  std::vector<std::unique_ptr<needle_replica>> replicas;

  [[nodiscard]] needle_replica const* find(int device_id) const noexcept
  {
    auto const it =
      std::find_if(replicas.begin(), replicas.end(), [device_id](auto const& replica) {
        return replica->device_id == device_id;
      });
    return it == replicas.end() ? nullptr : it->get();
  }

  /// The `Store` interface of detail::compute_membership_mask.
  template <class Fn>
  void with_lookup(membership_key_domain const& domain, int device_id, Fn&& fn) const
  {
    auto const* replica = find(device_id);
    if (!replica) { return; }
    detail::dispatch_key_rep(domain.rep, [&](auto tag) {
      using key_type = decltype(tag);
      fn(detail::needle_lookup<key_type>{static_cast<key_type const*>(replica->needles.data()),
                                         static_cast<int>(num_keys)});
    });
  }
};

//===----------------------------------------------------------------------===//
// sirius_dynamic_small_in_list_filter
//===----------------------------------------------------------------------===//

bool sirius_dynamic_small_in_list_filter::supports(cudf::column_view const& keys) noexcept
{
  // The size gate counts the keys that will actually be stored: null build slots are compacted
  // out at construction, so a nullable column qualifies on its valid rows.
  auto const num_keys = static_cast<std::size_t>(keys.size() - keys.null_count());
  return num_keys >= 1 && num_keys <= k_max_keys && membership_key_supported(keys.type());
}

sirius_dynamic_small_in_list_filter::sirius_dynamic_small_in_list_filter(
  cudf::column_view const& keys, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (!supports(keys)) {
    throw std::invalid_argument(
      "[sirius_dynamic_small_in_list_filter] unsupported key column (1..k_max_keys valid keys of "
      "a membership_key_supported type required).");
  }
  // Classifies, checks the DECIMAL128 fit, compacts null build keys out, and names the source
  // device; `build.compacted` stays alive until the needle copy is queued on `stream`.
  auto const build =
    prepare_membership_build("[sirius_dynamic_small_in_list_filter]", keys, stream, mr);
  _domain               = build.domain;
  _store                = std::make_unique<needle_store>();
  _store->source_device = build.source_device;
  _store->num_keys      = static_cast<std::size_t>(build.keys.size());

  // Needles are stored at the rep so one kernel per (adapter, rep) serves every build carrier;
  // a build carrier other than the rep converts per element on the way in.
  auto const bytes = _store->num_keys * membership_rep_bytes(_domain.rep);
  rmm::device_buffer needles{bytes, stream, mr};
  bool const copied = detail::dispatch_key_rep(_domain.rep, [&](auto key_tag) {
    using key_type = decltype(key_tag);
    return detail::with_build_key_iterator<key_type>(
      _domain, build.keys, stream, mr, [&](auto first, auto last) {
        thrust::copy(
          rmm::exec_policy_nosync(stream, mr), first, last, static_cast<key_type*>(needles.data()));
      });
  });
  if (!copied) {
    throw std::logic_error(
      "[sirius_dynamic_small_in_list_filter] build carrier does not fit its rep.");
  }
  _store->replicas.push_back(
    std::make_unique<needle_store::needle_replica>(_store->source_device, std::move(needles)));
}

sirius_dynamic_small_in_list_filter::~sirius_dynamic_small_in_list_filter() = default;

std::unique_ptr<cudf::column> sirius_dynamic_small_in_list_filter::compute_mask(
  cudf::column_view const& probe,
  std::uint32_t const* prior_mask_words,
  int device_id,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  return compute_mask_impl(probe, prior_mask_words, nullptr, device_id, stream, mr);
}

std::unique_ptr<cudf::column> sirius_dynamic_small_in_list_filter::compute_mask_if(
  cudf::column_view const& probe,
  cudf::column_view const& prior_keep,
  int device_id,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  return compute_mask_impl(
    probe, nullptr, detail::checked_prior_keep(probe, prior_keep), device_id, stream, mr);
}

std::unique_ptr<cudf::column> sirius_dynamic_small_in_list_filter::compute_mask_impl(
  cudf::column_view const& probe,
  std::uint32_t const* prior_mask_words,
  bool const* prior_keep,
  int device_id,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  return detail::compute_membership_mask(
    _store.get(), _domain, probe, prior_mask_words, prior_keep, device_id, stream, mr);
}

bool sirius_dynamic_small_in_list_filter::prepare_fused_probe(cudf::column_view const& probe,
                                                              int device_id,
                                                              detail::fused_probe& out) const
{
  return detail::prepare_fused_membership_probe(_store.get(), _domain, probe, device_id, out);
}

void sirius_dynamic_small_in_list_filter::replicate_to_devices(
  std::span<dynamic_filter_replica_space const> spaces)
{
  if (!_store || _store->replicas.empty()) { return; }
  auto const* source = _store->find(_store->source_device);
  if (!source) { return; }
  auto const bytes = source->needles.size();
  if (bytes == 0) { return; }  // empty build side: nothing to replicate.

  auto const source_target = std::find_if(spaces.begin(), spaces.end(), [this](auto const& target) {
    return target.get_gpu_space().get_device_id() == _store->source_device;
  });
  if (source_target == spaces.end()) {
    SIRIUS_LOG_WARN(
      "[sirius_dynamic_small_in_list_filter] source GPU {} has no replica memory space; remote "
      "GPUs will skip this optional filter.",
      _store->source_device);
    return;
  }
  auto const& source_space = source_target->get_gpu_space();

  // Retain every destination and pooled stream while direct peer copies are submitted. Waiting
  // only after this loop lets different destination GPUs transfer concurrently.
  std::vector<std::pair<std::unique_ptr<needle_store::needle_replica>, ::cuda::stream_ref>> pending;
  pending.reserve(spaces.size());
  _store->replicas.reserve(_store->replicas.size() + spaces.size());
  for (auto const& target : spaces) {
    auto const& target_space = target.get_gpu_space();
    auto const device_id     = target_space.get_device_id();
    if (device_id == _store->source_device || _store->find(device_id)) { continue; }
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      auto const stream = target_space.acquire_stream();

      auto reservation = detail::scoped_replica_reservation::try_acquire(
        target, detail::tracked_replica_allocation_bytes(bytes), stream);
      if (!reservation) {
        SIRIUS_LOG_WARN(
          "[sirius_dynamic_small_in_list_filter] replica GPU {} -> GPU {} skipped: destination "
          "reservation for {} bytes unavailable.",
          _store->source_device,
          device_id,
          bytes);
        continue;
      }

      auto replica = std::make_unique<needle_store::needle_replica>(
        device_id, rmm::device_buffer{bytes, stream, reservation->allocator()});
      detail::enqueue_replica_copy(replica->needles.data(),
                                   rmm::cuda_device_id{device_id},
                                   source->needles.data(),
                                   source_space,
                                   bytes,
                                   stream,
                                   target.get_host_staging_space());
      pending.emplace_back(std::move(replica), stream);
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN(
        "[sirius_dynamic_small_in_list_filter] replica GPU {} -> GPU {} unavailable: {}. That GPU "
        "will skip this optional filter.",
        _store->source_device,
        device_id,
        e.what());
      continue;
    }
    SIRIUS_LOG_DEBUG(
      "[sirius_dynamic_small_in_list_filter] queued {}-byte replica GPU {} -> GPU {}.",
      bytes,
      _store->source_device,
      device_id);
  }

  for (auto& [replica, stream] : pending) {
    auto const device_id = replica->device_id;
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      stream.sync();
      _store->replicas.push_back(std::move(replica));
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN(
        "[sirius_dynamic_small_in_list_filter] replica GPU {} -> GPU {} unavailable: {}. That GPU "
        "will skip this optional filter.",
        _store->source_device,
        device_id,
        e.what());
    }
  }
}

bool sirius_dynamic_small_in_list_filter::is_available_on_device(int device_id) const noexcept
{
  return _store && _store->find(detail::resolve_dynamic_filter_device_id(device_id)) != nullptr;
}

std::size_t sirius_dynamic_small_in_list_filter::replica_count() const noexcept
{
  return _store ? _store->replicas.size() : 0;
}

std::size_t sirius_dynamic_small_in_list_filter::size() const noexcept
{
  return _store ? _store->num_keys : 0;
}

}  // namespace sirius::op
