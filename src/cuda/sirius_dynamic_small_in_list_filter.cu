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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

// cccl
#include <cub/device/device_for.cuh>

// cucascade
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

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

/// @brief Per-row brute-force membership scan: out[idx] == true iff probe[idx] equals any of the m
/// needles. For the small m this filter gates on (<= k_max_keys), a compare-all linear scan beats a
/// hash probe and reserves no sentinel value.
template <class KeyT>
struct small_in_list_scan {
  KeyT const* __restrict__ probe;
  KeyT const* __restrict__ needles;
  int m;
  bool* __restrict__ out;

  __device__ __forceinline__ void operator()(cudf::size_type idx) const noexcept
  {
    auto const x = probe[idx];
    bool hit     = false;
    for (int j = 0; j < m; ++j) {
      hit |= (x == needles[j]);
    }
    out[idx] = hit;
  }
};

}  // namespace

namespace sirius::op {

//===----------------------------------------------------------------------===//
// Per-device needle storage (PIMPL)
//===----------------------------------------------------------------------===//

/// @brief Per-device raw snapshots of the build keys. Mirrors sirius_dynamic_in_list_filter's
/// replica store, but a device_buffer of raw bytes needs no cuco set / typed variant: the outer
/// class's _key_type / _num_keys decode the bytes in compute_mask.
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
  auto const num_keys = static_cast<std::size_t>(keys.size());
  auto const id       = keys.type().id();
  return num_keys >= 1 && num_keys <= k_max_keys &&
         (id == cudf::type_id::INT32 || id == cudf::type_id::INT64) && keys.null_count() == 0;
}

sirius_dynamic_small_in_list_filter::sirius_dynamic_small_in_list_filter(
  cudf::column_view const& keys, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  : _key_type(keys.type()), _num_keys(static_cast<std::size_t>(keys.size()))
{
  if (!supports(keys)) {
    throw std::invalid_argument(
      "[sirius_dynamic_small_in_list_filter] unsupported key column (1..k_max_keys INT32/INT64 "
      "keys with no nulls required).");
  }

  _store = std::make_unique<needle_store>();
  if (cudaGetDevice(&_store->source_device) != cudaSuccess) {
    throw std::runtime_error(
      "[sirius_dynamic_small_in_list_filter] failed to identify source device.");
  }

  auto const bytes = _num_keys * static_cast<std::size_t>(cudf::size_of(_key_type));
  void const* src =
    (_key_type.id() == cudf::type_id::INT32 ? static_cast<void const*>(keys.data<std::int32_t>())
                                            : static_cast<void const*>(keys.data<std::int64_t>()));
  _store->replicas.push_back(std::make_unique<needle_store::needle_replica>(
    _store->source_device, rmm::device_buffer{src, bytes, stream, mr}));
}

sirius_dynamic_small_in_list_filter::~sirius_dynamic_small_in_list_filter() = default;

std::unique_ptr<cudf::column> sirius_dynamic_small_in_list_filter::compute_mask(
  cudf::column_view const& probe,
  int device_id,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  if (probe.type() != _key_type) { return nullptr; }
  auto const* replica =
    _store ? _store->find(detail::resolve_dynamic_filter_device_id(device_id)) : nullptr;
  if (!replica) { return nullptr; }

  auto const n = probe.size();
  auto out     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* const outp = out->mutable_view().data<bool>();
  auto const m     = static_cast<int>(_num_keys);

  switch (_key_type.id()) {
    case cudf::type_id::INT32: {
      auto const* needles = static_cast<std::int32_t const*>(replica->needles.data());
      CUCASCADE_CUDA_TRY(cub::DeviceFor::Bulk(
        n,
        small_in_list_scan<std::int32_t>{probe.data<std::int32_t>(), needles, m, outp},
        stream.value()));
      break;
    }
    case cudf::type_id::INT64: {
      auto const* needles = static_cast<std::int64_t const*>(replica->needles.data());
      CUCASCADE_CUDA_TRY(cub::DeviceFor::Bulk(
        n,
        small_in_list_scan<std::int64_t>{probe.data<std::int64_t>(), needles, m, outp},
        stream.value()));
      break;
    }
    default: return nullptr;
  }

  if (probe.nullable() && probe.null_count() > 0) {
    out->set_null_mask(cudf::copy_bitmask(probe, stream, mr), probe.null_count());
  }
  return out;
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
  std::vector<std::pair<std::unique_ptr<needle_store::needle_replica>, rmm::cuda_stream_view>>
    pending;
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

      rmm::device_buffer destination{bytes, stream, reservation->allocator()};
      detail::enqueue_replica_copy(destination.data(),
                                   rmm::cuda_device_id{device_id},
                                   source->needles.data(),
                                   source_space,
                                   bytes,
                                   stream,
                                   target.get_host_staging_space());
      pending.emplace_back(
        std::make_unique<needle_store::needle_replica>(device_id, std::move(destination)), stream);
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
      stream.synchronize();
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
