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
#include "memory/runtime_stream_pool.hpp"

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
#include <cuda/dynamic_filter_probe.cuh>
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

namespace {

/// @brief Brute-force needle scan, the filter-specific half of detail::membership_probe_functor:
/// a converted key is a member iff it equals any of the m needles. For the small m this filter
/// gates on (<= k_max_keys), a compare-all linear scan beats a hash probe and reserves no sentinel
/// value. String needles are 64-bit fingerprints compared as such (one code path, no byte
/// compare), so the scan is exact for integers and no-false-negatives for strings.
template <class KeyT>
struct needle_lookup {
  KeyT const* __restrict__ needles;
  int m;
  __device__ __forceinline__ bool operator()(KeyT x) const noexcept
  {
    bool hit = false;
    for (int j = 0; j < m; ++j) {
      hit |= (x == needles[j]);
    }
    return hit;
  }
};

}  // namespace

namespace sirius::op {

//===----------------------------------------------------------------------===//
// Per-device needle storage (PIMPL)
//===----------------------------------------------------------------------===//

/// @brief Per-device raw snapshots of the build keys at the key rep. Mirrors
/// sirius_dynamic_in_list_filter's replica store, but a device_buffer of raw bytes needs no cuco
/// set / typed variant: the outer class's _domain.rep / _num_keys decode the bytes in compute_mask.
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

  int source_device = -1;
  std::vector<std::unique_ptr<needle_replica>> replicas;

  [[nodiscard]] needle_replica const* find(int device_id) const noexcept
  {
    auto const it =
      std::find_if(replicas.begin(), replicas.end(), [device_id](auto const& replica) {
        return replica->device_id == device_id;
      });
    return it == replicas.end() ? nullptr : it->get();
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
  _domain   = build.domain;
  _num_keys = static_cast<std::size_t>(build.keys.size());

  _store                = std::make_unique<needle_store>();
  _store->source_device = build.source_device;

  // Needles are stored at the rep so one kernel per (adapter, rep) serves every build carrier;
  // a build carrier other than the rep converts per element on the way in.
  auto const bytes = _num_keys * membership_rep_bytes(_domain.rep);
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
  auto const* replica =
    _store ? _store->find(detail::resolve_dynamic_filter_device_id(device_id)) : nullptr;
  if (!replica) { return nullptr; }

  auto const m = static_cast<int>(_num_keys);
  return detail::dispatch_key_rep(_domain.rep, [&](auto key_tag) {
    using key_type      = decltype(key_tag);
    auto const* needles = static_cast<key_type const*>(replica->needles.data());
    return detail::run_membership_probe<key_type>(
      _domain, probe, prior_mask_words, stream, mr, needle_lookup<key_type>{needles, m});
  });
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
  std::vector<
    std::pair<std::unique_ptr<needle_store::needle_replica>, cucascade::memory::borrowed_stream>>
    pending;
  pending.reserve(spaces.size());
  _store->replicas.reserve(_store->replicas.size() + spaces.size());
  for (auto const& target : spaces) {
    auto const& target_space = target.get_gpu_space();
    auto const device_id     = target_space.get_device_id();
    if (device_id == _store->source_device || _store->find(device_id)) { continue; }
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      auto stream_lease = sirius::memory::runtime_stream_pool::acquire(target_space);
      auto const stream = stream_lease.get();

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
      pending.emplace_back(std::move(replica), std::move(stream_lease));
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

  for (auto& [replica, stream_lease] : pending) {
    auto const stream    = stream_lease.get();
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

}  // namespace sirius::op
