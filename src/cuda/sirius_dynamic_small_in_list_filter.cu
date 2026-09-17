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
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

// cccl
#include <cub/device/device_for.cuh>
#include <cuda/dynamic_filter_probe.cuh>
#include <thrust/copy.h>

// cucascade
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
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
/// hash probe and reserves no sentinel value. The adapter converts probe values into the needle
/// domain per element; one the domain cannot represent is a definite non-member, and so is a null
/// probe row (the needles hold no nulls and the join never matches them). String needles are
/// 64-bit fingerprints compared as such (one code path, no byte compare), so the scan is exact for
/// integers and no-false-negatives for strings. Rows the prior keep-mask killed skip the scan.
template <class Adapter, class KeyT>
struct small_in_list_scan {
  Adapter adapt;
  KeyT const* __restrict__ needles;
  int m;
  bool* __restrict__ out;
  std::uint32_t const* __restrict__ prior_words;  // packed 1 bit/row, or null
  sirius::op::detail::probe_validity valid;

  __device__ __forceinline__ void operator()(cudf::size_type idx) const noexcept
  {
    if (!sirius::op::detail::prior_mask_keeps(prior_words, idx) || !valid(idx)) {
      out[idx] = false;
      return;
    }
    KeyT x;
    if (!adapt(idx, x)) {
      out[idx] = false;
      return;
    }
    bool hit = false;
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
  cudf::column_view const& keys, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const domain = classify_membership_key(keys.type());
  if (!domain.has_value() || !supports(keys)) {
    throw std::invalid_argument(
      "[sirius_dynamic_small_in_list_filter] unsupported key column (1..k_max_keys valid keys of "
      "a membership_key_supported type required).");
  }
  // A DECIMAL128 build whose unscaled values exceed the int64 rep cannot be stored exactly.
  if (!membership_build_fits_rep(keys, stream, mr)) {
    throw std::invalid_argument(
      "[sirius_dynamic_small_in_list_filter] build keys do not fit the key rep (DECIMAL128 values "
      "outside int64).");
  }
  _domain = *domain;

  // Null build keys match nothing under the join's null_equality::UNEQUAL, so they are dropped
  // exactly rather than copied. The compacted storage stays alive until the needle copy is queued
  // on `stream`; its stream-ordered free then follows the copy.
  std::unique_ptr<cudf::table> compacted;
  cudf::column_view build_keys = keys;
  if (keys.null_count() > 0) {
    compacted  = cudf::drop_nulls(cudf::table_view{{keys}}, {0}, stream, mr);
    build_keys = compacted->view().column(0);
  }
  _num_keys = static_cast<std::size_t>(build_keys.size());

  _store = std::make_unique<needle_store>();
  if (cudaGetDevice(&_store->source_device) != cudaSuccess) {
    throw std::runtime_error(
      "[sirius_dynamic_small_in_list_filter] failed to identify source device.");
  }

  // Needles are stored at the rep so one kernel per (adapter, rep) serves every build carrier;
  // a build carrier other than the rep converts per element on the way in.
  auto const bytes = _num_keys * membership_rep_bytes(_domain.rep);
  rmm::device_buffer needles{bytes, stream, mr};
  bool const copied = detail::dispatch_key_rep(_domain.rep, [&](auto key_tag) {
    using key_type = decltype(key_tag);
OURS
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
  int device_id,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return compute_mask(probe, /*prior_mask_words=*/nullptr, device_id, stream, mr);
}

std::unique_ptr<cudf::column> sirius_dynamic_small_in_list_filter::compute_mask(
  cudf::column_view const& probe,
  std::uint32_t const* prior_mask_words,
  int device_id,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  // A pinned chunk may store this key narrowed while the filter was published at the native
  // carrier; the kernel converts per element rather than materializing a widened copy.
  auto const* replica =
    _store ? _store->find(detail::resolve_dynamic_filter_device_id(device_id)) : nullptr;
  if (!replica) { return nullptr; }

  std::unique_ptr<cudf::column> out;
  auto const n          = probe.size();
  auto const m          = static_cast<int>(_num_keys);
  auto const dispatched = detail::dispatch_key_rep(_domain.rep, [&](auto key_tag) {
    using key_type      = decltype(key_tag);
    auto const* needles = static_cast<key_type const*>(replica->needles.data());
    return detail::dispatch_probe_adapter<key_type>(_domain, probe, stream, [&](auto adapter) {
      out = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::BOOL8}, n, cudf::mask_state::UNALLOCATED, stream, mr);
      auto* const outp = out->mutable_view().data<bool>();
      CUCASCADE_CUDA_TRY(cub::DeviceFor::Bulk(
        n,
        small_in_list_scan<decltype(adapter), key_type>{
          adapter, needles, m, outp, prior_mask_words, detail::probe_validity_of(probe)},
        stream.value()));
    });
  });
  if (!dispatched || !out) { return nullptr; }
  // Null probe rows were written as `false` in-kernel: the mask is non-nullable by construction.
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
