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
#include <op/dynamic_filter/dynamic_filter_replica_transfer.hpp>
#include <op/dynamic_filter/prepared_build_keys.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

// cudf
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>

// cccl
#include <cuco/extent.cuh>
#include <cuco/operator.hpp>
#include <cuco/static_set.cuh>
#include <cuco/storage.cuh>
#include <cuda/dynamic_filter_fused_probe.cuh>
#include <cuda/sirius_rmm_cuco_allocator.cuh>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/stream>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_device.hpp>

// standard library
#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op {

namespace {
// Baseline load factor: capacity = 2 × keys.
constexpr std::size_t kCapacityFactor = 2;
constexpr double kLoadFactor          = 1.0 / kCapacityFactor;

// Largest capacity multiplier whose table still fits `budget`, else the baseline. Growth only
// makes an already-chosen set sparser: estimated_set_bytes (and so IN-list vs Bloom selection)
// is always evaluated at the baseline factor.
constexpr std::size_t kGrowthCandidates[] = {4, 3};
static_assert(kGrowthCandidates[0] >= kGrowthCandidates[1] &&
                kGrowthCandidates[0] >= kCapacityFactor,
              "in_list_storage_upper_bound sizes the set at the first growth candidate");

[[nodiscard]] std::size_t capacity_factor_for(std::size_t num_keys, std::size_t key_bytes) noexcept
{
  int l2     = 0;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device) != cudaSuccess || l2 <= 0) {
    return kCapacityFactor;
  }
  auto const budget = static_cast<std::size_t>(l2) / 2;
  for (auto const factor : kGrowthCandidates) {
    // num_keys * key_bytes is bounded by the build column's own size, so this cannot overflow.
    if (num_keys * key_bytes * factor <= budget) { return factor; }
  }
  return kCapacityFactor;
}

// Minimum set capacity.
constexpr std::size_t kMinCapacity = 8;

/// The capacity the constructor requests for @p num_keys keys at @p factor slots per key.
[[nodiscard]] std::size_t requested_capacity(std::size_t num_keys, std::size_t factor) noexcept
{
  auto const capacity = num_keys > std::numeric_limits<std::size_t>::max() / factor
                          ? std::numeric_limits<std::size_t>::max()
                          : factor * num_keys;
  return std::max(capacity, kMinCapacity);
}

using detail::set_alloc;
using detail::set_type;

template <class KeyT>
using set_owner = std::unique_ptr<set_type<KeyT>>;

// A live replica owns exactly one typed set (one alternative per membership_key_rep); its active
// owner becomes null only during teardown.
using set_storage = std::variant<set_owner<std::int32_t>,
                                 set_owner<std::int64_t>,
                                 set_owner<std::uint32_t>,
                                 set_owner<std::uint64_t>>;

template <class KeyT>
set_owner<KeyT> make_set(std::size_t capacity,
                         rmm::device_async_resource_ref mr,
                         cuda::stream_ref stream)
{
  return set_owner<KeyT>(
    new set_type<KeyT>{cuco::extent<std::size_t>{capacity},
                       cuco::empty_key<KeyT>{detail::set_sentinel<KeyT>::value},
                       {},
                       {},
                       {},
                       {},
                       set_alloc<KeyT>{mr},
                       stream});
}

template <class KeyT>
set_owner<KeyT> build_set(cudf::column_view const& prepared,
                          std::size_t capacity,
                          rmm::device_async_resource_ref mr,
                          cuda::stream_ref stream)
{
  auto set = make_set<KeyT>(capacity, mr, stream);
  if (prepared.size() > 0) {
    auto const* first = prepared.data<KeyT>();
    set->insert_async(first, first + prepared.size(), stream);
  }
  return set;
}

}  // namespace

struct set_replica {
  int device_id = -1;
  set_storage set;

  template <class KeyT>
  set_replica(int device_id, set_owner<KeyT> owner)
    : device_id{device_id}, set{std::in_place_type<set_owner<KeyT>>, std::move(owner)}
  {
  }

  ~set_replica() noexcept
  {
    if (device_id < 0) { return; }
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
    std::visit([](auto& owner) { owner.reset(); }, set);
  }

  [[nodiscard]] bool has_set() const noexcept
  {
    return std::visit([](auto const& owner) { return owner != nullptr; }, set);
  }
};

struct sirius_dynamic_in_list_filter::set_impl {
  int source_device = -1;
  std::vector<std::unique_ptr<set_replica>> replicas;

  [[nodiscard]] set_replica const* find(int device_id) const noexcept
  {
    auto const it =
      std::find_if(replicas.begin(), replicas.end(), [device_id](auto const& replica) {
        return replica->device_id == device_id;
      });
    return it == replicas.end() ? nullptr : it->get();
  }

  /// @brief The `Store` interface of detail::compute_membership_mask.
  template <class Fn>
  void with_lookup(membership_key_domain const&, int device_id, Fn&& fn) const
  {
    auto const* replica = find(device_id);
    if (!replica || !replica->has_set()) { return; }
    std::visit(
      [&](auto const& set) {
        auto ref = set->ref(cuco::contains);
        fn(detail::set_lookup<decltype(ref)>{ref});
      },
      replica->set);
  }
};

sirius_dynamic_in_list_filter::sirius_dynamic_in_list_filter(cudf::column_view const& keys,
                                                             ::cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr)
{
  // The prepared column frees stream-ordered after the insert is queued on `stream`.
  auto const prepared = prepare_build_column("[sirius_dynamic_in_list_filter]", keys, stream, mr);
  construct(prepared.domain, prepared.keys->view(), stream, mr);
}

sirius_dynamic_in_list_filter::sirius_dynamic_in_list_filter(membership_key_domain const& domain,
                                                             cudf::column_view const& prepared,
                                                             ::cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr)
{
  detail::require_prepared("[sirius_dynamic_in_list_filter]", domain, prepared);
  construct(domain, prepared, stream, mr);
}

void sirius_dynamic_in_list_filter::construct(membership_key_domain const& domain,
                                              cudf::column_view const& prepared,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  _domain   = domain;
  _num_keys = static_cast<std::size_t>(prepared.size());

  cuda::stream_ref const s{stream.get()};
  auto const rep_bytes = membership_rep_bytes(_domain.rep);
  auto const factor    = capacity_factor_for(_num_keys, rep_bytes);
  auto const capacity  = requested_capacity(_num_keys, factor);
  _set                 = std::make_unique<set_impl>();
  _set->source_device  = detail::current_device("[sirius_dynamic_in_list_filter]");
  auto source          = detail::dispatch_key_rep(_domain.rep, [&](auto key_tag) {
    using key_type = decltype(key_tag);
    return std::make_unique<set_replica>(_set->source_device,
                                         build_set<key_type>(prepared, capacity, mr, s));
  });
  SIRIUS_LOG_DEBUG(
    "[sirius_dynamic_in_list_filter] built set: {} keys, bucket_size={}, capacity_factor={}, "
    "capacity={} slots ({} bytes).",
    _num_keys,
    detail::kBucketSize,
    factor,
    capacity,
    capacity * rep_bytes);
  _set->replicas.push_back(std::move(source));
}

bool sirius_dynamic_in_list_filter::supports(cudf::column_view const& keys) noexcept
{
  // Nullable keys are accepted: null build slots are compacted out at construction.
  return supports(keys.type(), static_cast<std::size_t>(keys.size() - keys.null_count()));
}

bool sirius_dynamic_in_list_filter::supports(cudf::data_type key_type,
                                             std::size_t /*valid_rows*/) noexcept
{
  return membership_key_supported(key_type);
}

sirius_dynamic_in_list_filter::~sirius_dynamic_in_list_filter() = default;

bool sirius_dynamic_in_list_filter::has_persistent_set() const noexcept
{
  return _set && std::any_of(_set->replicas.begin(), _set->replicas.end(), [](auto const& replica) {
           return replica->has_set();
         });
}

void sirius_dynamic_in_list_filter::replicate_to_devices(
  std::span<dynamic_filter_replica_space const> spaces)
{
  if (!_set || _set->replicas.empty()) { return; }
  auto const* source = _set->find(_set->source_device);
  if (!source) { return; }
  auto const source_target = std::find_if(spaces.begin(), spaces.end(), [this](auto const& target) {
    return target.get_gpu_space().get_device_id() == _set->source_device;
  });
  if (source_target == spaces.end()) {
    SIRIUS_LOG_WARN(
      "[sirius_dynamic_in_list_filter] source GPU {} has no replica memory space; remote GPUs "
      "will skip this optional filter.",
      _set->source_device);
    return;
  }
  auto const& source_space = source_target->get_gpu_space();

  // Retain every destination and pooled stream while direct peer copies are submitted. Waiting
  // only after this loop lets different destination GPUs transfer concurrently.
  std::vector<std::pair<std::unique_ptr<set_replica>, sirius::memory::runtime_stream_pool::lease>>
    pending;
  pending.reserve(spaces.size());
  _set->replicas.reserve(_set->replicas.size() + spaces.size());
  detail::replica_fanout_failures failures;
  for (auto const& target : spaces) {
    auto const& target_space = target.get_gpu_space();
    auto const device_id     = target_space.get_device_id();
    if (device_id == _set->source_device || _set->find(device_id)) { continue; }
    std::size_t bytes = 0;
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      auto stream_lease = sirius::memory::runtime_stream_pool::acquire(target_space);
      auto const stream = stream_lease.get();

      auto replica = std::visit(
        [&](auto const& source_set) {
          if (!source_set) {
            throw std::logic_error(
              "[sirius_dynamic_in_list_filter] source replica has no persistent set.");
          }
          using owner_type = std::decay_t<decltype(source_set)>;
          using key_type   = typename owner_type::element_type::key_type;

          auto const capacity = source_set->capacity();
          bytes               = capacity * sizeof(key_type);
          // Cover cuco's allocator-side alignment slot without depending on its private layout.
          auto const reservation_bytes =
            detail::tracked_replica_allocation_bytes(bytes) + rmm::CUDA_ALLOCATION_ALIGNMENT;
          auto reservation =
            detail::scoped_replica_reservation::try_acquire(target, reservation_bytes, stream);
          if (!reservation) { return std::unique_ptr<set_replica>{}; }

          auto destination_set =
            make_set<key_type>(capacity, reservation->allocator(), cuda::stream_ref{stream.get()});
          if (destination_set->capacity() != capacity) {
            throw std::logic_error("destination static_set capacity changed during replication");
          }
          auto result       = std::make_unique<set_replica>(device_id, std::move(destination_set));
          auto& destination = *std::get<set_owner<key_type>>(result->set);
          detail::enqueue_replica_copy(destination.data(),
                                       rmm::cuda_device_id{device_id},
                                       source_set->data(),
                                       source_space,
                                       bytes,
                                       stream,
                                       target.get_host_staging_space());
          return result;
        },
        source->set);
      if (!replica) {
        SIRIUS_LOG_WARN(
          "[sirius_dynamic_in_list_filter] replica GPU {} -> GPU {} skipped: destination "
          "reservation for {} bytes unavailable.",
          _set->source_device,
          device_id,
          bytes);
        continue;
      }
      pending.emplace_back(std::move(replica), std::move(stream_lease));
    } catch (...) {
      if (failures.skip_if_recoverable(
            "[sirius_dynamic_in_list_filter]", _set->source_device, device_id)) {
        continue;
      }
      // Earlier destinations still own asynchronous copies; drain them before propagating.
      break;
    }
    SIRIUS_LOG_DEBUG("[sirius_dynamic_in_list_filter] queued {}-byte replica GPU {} -> GPU {}.",
                     bytes,
                     _set->source_device,
                     device_id);
  }

  for (auto& [replica, stream_lease] : pending) {
    auto const stream    = stream_lease.get();
    auto const device_id = replica->device_id;
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      CUCASCADE_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    } catch (...) {
      failures.record_fatal("[sirius_dynamic_in_list_filter]", _set->source_device, device_id);
    }
  }
  failures.rethrow_if_fatal();
  for (auto& [replica, stream_lease] : pending) {
    _set->replicas.push_back(std::move(replica));
  }
}

bool sirius_dynamic_in_list_filter::is_available_on_device(int device_id) const noexcept
{
  return _set && _set->find(detail::resolve_dynamic_filter_device_id(device_id)) != nullptr;
}

std::size_t sirius_dynamic_in_list_filter::replica_count() const noexcept
{
  return _set ? _set->replicas.size() : 0;
}

std::unique_ptr<cudf::column> sirius_dynamic_in_list_filter::compute_mask(
  cudf::column_view const& probe,
  std::uint32_t const* prior_mask_words,
  int device_id,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  return compute_mask_impl(probe, prior_mask_words, nullptr, device_id, stream, mr);
}

std::unique_ptr<cudf::column> sirius_dynamic_in_list_filter::compute_mask_if(
  cudf::column_view const& probe,
  cudf::column_view const& prior_keep,
  int device_id,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  return compute_mask_impl(
    probe, nullptr, detail::checked_prior_keep(probe, prior_keep), device_id, stream, mr);
}

std::unique_ptr<cudf::column> sirius_dynamic_in_list_filter::compute_mask_impl(
  cudf::column_view const& probe,
  std::uint32_t const* prior_mask_words,
  bool const* prior_keep,
  int device_id,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  return detail::compute_membership_mask(_set.get(),
                                         _domain,
                                         probe,
                                         prior_mask_words,
                                         prior_keep,
                                         device_id,
                                         stream,
                                         mr,
                                         preserves_nulls());
}

bool sirius_dynamic_in_list_filter::prepare_fused_probe(cudf::column_view const& probe,
                                                        int device_id,
                                                        detail::fused_probe& out) const
{
  return detail::prepare_fused_membership_probe(
    _set.get(), _domain, probe, device_id, out, preserves_nulls());
}

std::size_t sirius_dynamic_in_list_filter::size() const noexcept { return _num_keys; }

std::size_t sirius_dynamic_in_list_filter::set_slots(std::size_t num_keys,
                                                     std::size_t capacity_factor) noexcept
{
  // The rounding cuco applies when the set is constructed with this capacity.
  return static_cast<std::size_t>(
    cuco::make_valid_extent<static_cast<std::int32_t>(detail::kCgSize), detail::kBucketSize>(
      requested_capacity(num_keys, capacity_factor))
      .value());
}

std::size_t sirius_dynamic_in_list_filter::storage_bytes() const noexcept
{
  if (!_set) { return 0; }
  auto const* source = _set->find(_set->source_device);
  if (source == nullptr || !source->has_set()) { return 0; }
  return std::visit(
    [](auto const& set) {
      using key_type = typename std::decay_t<decltype(*set)>::key_type;
      return static_cast<std::size_t>(set->capacity()) * sizeof(key_type);
    },
    source->set);
}

std::size_t in_list_storage_upper_bound(std::size_t num_keys, membership_key_rep rep) noexcept
{
  auto constexpr maximum = std::numeric_limits<std::size_t>::max();
  auto const rep_bytes   = membership_rep_bytes(rep);
  auto const slots       = sirius_dynamic_in_list_filter::set_slots(num_keys, kGrowthCandidates[0]);
  auto const set_bytes   = slots > maximum / rep_bytes ? maximum : slots * rep_bytes;
  auto const needles     = num_keys > maximum / rep_bytes ? maximum : num_keys * rep_bytes;
  auto const charge      = [](std::size_t bytes) {
    return bytes > maximum - (rmm::CUDA_ALLOCATION_ALIGNMENT - 1)
                  ? maximum
                  : detail::tracked_replica_allocation_bytes(bytes);
  };
  auto const set_charge    = charge(set_bytes);
  auto const needle_charge = charge(needles);
  return set_charge > maximum - needle_charge ? maximum : set_charge + needle_charge;
}

// Reports the *baseline* (kCapacityFactor) footprint — a lower bound on the real allocation
// rather than an exact size, keeping the IN-list/Bloom choice independent of probe-side tuning.
std::size_t sirius_dynamic_in_list_filter::estimated_set_bytes(std::size_t num_keys,
                                                               cudf::data_type key_type) noexcept
{
  // Slots hold the key *rep*, not the build carrier: an INT8 build column still fills 4-byte
  // slots. Unclassifiable types fall back to their own width (or 8 bytes when variable-width) so
  // the estimate stays meaningful for callers that consult it before the supports() gate.
  auto const domain      = classify_membership_key(key_type);
  std::size_t const slot = domain.has_value() ? membership_rep_bytes(domain->rep)
                           : cudf::is_fixed_width(key_type)
                             ? static_cast<std::size_t>(cudf::size_of(key_type))
                             : sizeof(std::int64_t);
  return static_cast<std::size_t>(static_cast<double>(num_keys) * static_cast<double>(slot) /
                                  kLoadFactor);
}

}  // namespace sirius::op
