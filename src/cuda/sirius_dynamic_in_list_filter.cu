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
#include <op/dynamic_filter/dynamic_filter_replica_transfer.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>

// cccl
#include <cub/device/device_for.cuh>
#include <cuco/operator.hpp>
#include <cuco/static_set.cuh>
#include <cuco/storage.cuh>
#include <cuda/sirius_rmm_cuco_allocator.cuh>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/stream_ref>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_device.hpp>

// standard library
#include <algorithm>
#include <cstdint>
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

// Threads per key probe.
constexpr std::size_t kCgSize = 1;
static_assert(kCgSize == 1, "cuco::static_set requires kCgSize==1 for device_for bulk iteration");

// Keys per bucket. Sized for a double-hashed probe, where each step is a random fetch and a
// wider bucket retires several dependent fetches in one; the right value depends on the probing
// scheme, not on the key type.
constexpr std::int32_t kBucketSize = 4;

// Largest capacity multiplier whose table still fits `budget`, else the baseline. Growth only
// makes an already-chosen set sparser: estimated_set_bytes (and so IN-list vs Bloom selection)
// is always evaluated at the baseline factor.
constexpr std::size_t kGrowthCandidates[] = {4, 3};

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

template <class KeyT>
using set_alloc = sirius::rmm_cuco_allocator<KeyT>;

template <class KeyT>
using set_type = cuco::static_set<KeyT,
                                  cuco::extent<std::size_t>,
                                  cuda::thread_scope_device,
                                  cuda::std::equal_to<KeyT>,
                                  cuco::double_hashing<kCgSize, cuco::default_hash_function<KeyT>>,
                                  set_alloc<KeyT>,
                                  cuco::storage<kBucketSize>>;

template <class KeyT>
using set_owner = std::unique_ptr<set_type<KeyT>>;

// A live replica owns exactly one typed set; its active owner becomes null only during teardown.
using set_storage = std::variant<set_owner<std::int32_t>, set_owner<std::int64_t>>;

template <class KeyT>
set_owner<KeyT> make_set(std::size_t capacity,
                         rmm::device_async_resource_ref mr,
                         cuda::stream_ref stream)
{
  return set_owner<KeyT>(
    new set_type<KeyT>{cuco::extent<std::size_t>{capacity},
                       cuco::empty_key<KeyT>{cuda::std::numeric_limits<KeyT>::min()},
                       {},
                       {},
                       {},
                       {},
                       set_alloc<KeyT>{mr},
                       stream});
}

template <class KeyT>
set_owner<KeyT> build_set(cudf::column_view const& keys,
                          std::size_t capacity,
                          rmm::device_async_resource_ref mr,
                          cuda::stream_ref stream)
{
  auto set = make_set<KeyT>(capacity, mr, stream);
  if (keys.size() > 0) {
    auto const* d = keys.data<KeyT>();
    set->insert_async(d, d + keys.size(), stream);
  }
  return set;
}

template <class KeyT>
struct equals_sentinel {
  __device__ __forceinline__ bool operator()(KeyT const& k) const noexcept
  {
    return k == cuda::std::numeric_limits<KeyT>::min();
  }
};

template <class KeyT, class SetRef>
struct contains_or_sentinel {
  KeyT const* probe;
  bool* out;
  SetRef set;
  __device__ __forceinline__ void operator()(cudf::size_type idx) const noexcept
  {
    auto const& key = probe[idx];
    out[idx]        = set.contains(key) || equals_sentinel<KeyT>{}(key);
  }
};

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
};

sirius_dynamic_in_list_filter::sirius_dynamic_in_list_filter(cudf::column_view const& keys,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
  : _key_type(keys.type()), _num_keys(static_cast<std::size_t>(keys.size()))
{
  if (!supports(keys)) {
    throw std::invalid_argument(
      "[sirius_dynamic_in_list_filter] unsupported key column (INT32/INT64, no nulls required).");
  }

  cuda::stream_ref const s{stream.value()};
  auto const factor =
    capacity_factor_for(_num_keys, static_cast<std::size_t>(cudf::size_of(_key_type)));
  auto const capacity = std::max<std::size_t>(factor * _num_keys, kMinCapacity);
  _set                = std::make_unique<set_impl>();
  if (cudaGetDevice(&_set->source_device) != cudaSuccess) {
    throw std::runtime_error("[sirius_dynamic_in_list_filter] failed to identify source device.");
  }
  std::unique_ptr<set_replica> source;
  switch (_key_type.id()) {
    case cudf::type_id::INT32:
      source = std::make_unique<set_replica>(_set->source_device,
                                             build_set<std::int32_t>(keys, capacity, mr, s));
      break;
    case cudf::type_id::INT64:
      source = std::make_unique<set_replica>(_set->source_device,
                                             build_set<std::int64_t>(keys, capacity, mr, s));
      break;
    default:
      throw std::logic_error(
        "[sirius_dynamic_in_list_filter] supported key type changed during construction.");
  }
  SIRIUS_LOG_DEBUG(
    "[sirius_dynamic_in_list_filter] built set: {} keys, bucket_size={}, capacity_factor={}, "
    "capacity={} slots ({} bytes).",
    _num_keys,
    kBucketSize,
    factor,
    capacity,
    capacity * static_cast<std::size_t>(cudf::size_of(_key_type)));
  _set->replicas.push_back(std::move(source));
}

bool sirius_dynamic_in_list_filter::supports(cudf::column_view const& keys) noexcept
{
  auto const id = keys.type().id();
  return (id == cudf::type_id::INT32 || id == cudf::type_id::INT64) && keys.null_count() == 0;
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
  std::vector<std::pair<std::unique_ptr<set_replica>, rmm::cuda_stream_view>> pending;
  pending.reserve(spaces.size());
  _set->replicas.reserve(_set->replicas.size() + spaces.size());
  for (auto const& target : spaces) {
    auto const& target_space = target.get_gpu_space();
    auto const device_id     = target_space.get_device_id();
    if (device_id == _set->source_device || _set->find(device_id)) { continue; }
    std::size_t bytes = 0;
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      auto const stream = target_space.acquire_stream();

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

          auto destination_set = make_set<key_type>(
            capacity, reservation->allocator(), cuda::stream_ref{stream.value()});
          if (destination_set->capacity() != capacity) {
            throw std::runtime_error("destination static_set capacity changed during replication");
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
      pending.emplace_back(std::move(replica), stream);
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN(
        "[sirius_dynamic_in_list_filter] replica GPU {} -> GPU {} unavailable: {}. "
        "That GPU will skip this optional filter.",
        _set->source_device,
        device_id,
        e.what());
      continue;
    }
    SIRIUS_LOG_DEBUG("[sirius_dynamic_in_list_filter] queued {}-byte replica GPU {} -> GPU {}.",
                     bytes,
                     _set->source_device,
                     device_id);
  }

  for (auto& [replica, stream] : pending) {
    auto const device_id = replica->device_id;
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      stream.synchronize();
      _set->replicas.push_back(std::move(replica));
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN(
        "[sirius_dynamic_in_list_filter] replica GPU {} -> GPU {} unavailable: {}. "
        "That GPU will skip this optional filter.",
        _set->source_device,
        device_id,
        e.what());
    }
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
  int device_id,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  if (probe.type() != _key_type) { return nullptr; }
  auto const* replica =
    _set ? _set->find(detail::resolve_dynamic_filter_device_id(device_id)) : nullptr;
  if (!replica) { return nullptr; }

  auto const n = probe.size();
  auto out     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* const outp = out->mutable_view().data<bool>();

  std::visit(
    [&](auto const& set) {
      using owner_type = std::decay_t<decltype(set)>;
      using key_type   = typename owner_type::element_type::key_type;
      auto const* d    = probe.data<key_type>();
      auto ref         = set->ref(cuco::contains);
      cub::DeviceFor::Bulk(
        n, contains_or_sentinel<key_type, decltype(ref)>{d, outp, ref}, stream.value());
    },
    replica->set);
  if (probe.nullable() && probe.null_count() > 0) {
    out->set_null_mask(cudf::copy_bitmask(probe, stream, mr), probe.null_count());
  }
  return out;
}

std::size_t sirius_dynamic_in_list_filter::size() const noexcept { return _num_keys; }

// Reports the *baseline* (kCapacityFactor) footprint — a lower bound on the real allocation
// rather than an exact size, keeping the IN-list/Bloom choice independent of probe-side tuning.
std::size_t sirius_dynamic_in_list_filter::estimated_set_bytes(std::size_t num_keys,
                                                               cudf::data_type key_type) noexcept
{
  std::size_t const slot =
    cudf::is_fixed_width(key_type)
      ? static_cast<std::size_t>(cudf::size_of(key_type))
      : sizeof(std::int64_t);  // variable-width keys hash to ~8B slots [currently unreachable]
  return static_cast<std::size_t>(static_cast<double>(num_keys) * static_cast<double>(slot) /
                                  kLoadFactor);
}

}  // namespace sirius::op
