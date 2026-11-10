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

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/accumulation_partial.cuh>
#include <cuda/std/limits>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <helper/numeric_narrowing.hpp>
#include <memory/size_arithmetic.hpp>
#include <op/dynamic_filter/detail/bounds_accumulator.hpp>
#include <op/dynamic_filter/detail/restore_scalar.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <sirius/exception.hpp>
#include <telemetry/nvtx.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace sirius::op::detail {

namespace {

/// Bytes of one slot: the batch's minimum, then its maximum, each in 16 bytes.
constexpr std::size_t k_bound_bytes = 16;
constexpr std::size_t k_slot_bytes  = 2 * k_bound_bytes;

using wide = __int128_t;  // Holds every bound storage value: signed, unsigned up to 64 bits, BOOL8

/**
 * @brief Invokes @p fn with a value of the integer storage type of @p type's bounds.
 *
 * @throw accumulation_invariant_error for a type without numeric bounds
 */
template <class Fn>
decltype(auto) dispatch_bound_storage(cudf::data_type type, Fn&& fn)
{
  switch (type.id()) {
    case cudf::type_id::INT8: return fn(std::int8_t{});
    case cudf::type_id::INT16: return fn(std::int16_t{});
    case cudf::type_id::INT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32: return fn(std::int32_t{});
    case cudf::type_id::INT64:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DECIMAL64: return fn(std::int64_t{});
    case cudf::type_id::UINT8: return fn(std::uint8_t{});
    case cudf::type_id::UINT16: return fn(std::uint16_t{});
    case cudf::type_id::UINT32: return fn(std::uint32_t{});
    case cudf::type_id::UINT64: return fn(std::uint64_t{});
    case cudf::type_id::BOOL8: return fn(bool{});
    case cudf::type_id::DECIMAL128: return fn(__int128_t{});
    default:
      throw accumulation_invariant_error{"[bounds_accumulator] a type without numeric bounds"};
  }
}

/// A valid row's value, or @p identity for a null row, so a reduction skips the nulls.
template <class T>
struct masked_bound {
  T const* values;
  cudf::bitmask_type const* mask;  // Null when the column has no nulls
  cudf::size_type offset;
  T identity;

  __device__ T operator()(cudf::size_type row) const
  {
    return mask == nullptr || cudf::bit_is_set(mask, offset + row) ? values[row] : identity;
  }
};

/// Reduces @p column's valid values to their minimum at @p out and maximum at `out + 16` bytes.
template <class T>
void enqueue_bounds(cudf::column_view const& column,
                    std::byte* out,
                    ::cuda::stream_ref stream,
                    rmm::device_async_resource_ref scratch)
{
  auto const* mask  = column.null_count() == 0 ? nullptr : column.null_mask();
  auto const rows   = column.size();
  auto const values = column.data<T>();
  auto const lowest = thrust::make_transform_iterator(
    thrust::counting_iterator<cudf::size_type>{0},
    masked_bound<T>{values, mask, column.offset(), ::cuda::std::numeric_limits<T>::max()});
  auto const highest = thrust::make_transform_iterator(
    thrust::counting_iterator<cudf::size_type>{0},
    masked_bound<T>{values, mask, column.offset(), ::cuda::std::numeric_limits<T>::lowest()});
  auto* const min_out   = reinterpret_cast<T*>(out);
  auto* const max_out   = reinterpret_cast<T*>(out + k_bound_bytes);
  std::size_t min_bytes = 0;
  std::size_t max_bytes = 0;
  check_cuda(cub::DeviceReduce::Min(nullptr, min_bytes, lowest, min_out, rows, stream.get()),
             "cub::DeviceReduce::Min(size)");
  check_cuda(cub::DeviceReduce::Max(nullptr, max_bytes, highest, max_out, rows, stream.get()),
             "cub::DeviceReduce::Max(size)");
  rmm::device_buffer temporary{std::max(min_bytes, max_bytes), stream, scratch};
  check_cuda(cudaGetLastError(), "pending CUDA error before bounds reduction");
  check_cuda(
    cub::DeviceReduce::Min(temporary.data(), min_bytes, lowest, min_out, rows, stream.get()),
    "cub::DeviceReduce::Min",
    true);
  check_cuda(
    cub::DeviceReduce::Max(temporary.data(), max_bytes, highest, max_out, rows, stream.get()),
    "cub::DeviceReduce::Max",
    true);
}

/// The storage value of @p type held in the low bytes of @p bytes, widened.
[[nodiscard]] wide decode(cudf::data_type type, std::byte const* bytes)
{
  return dispatch_bound_storage(type, [&](auto tag) {
    decltype(tag) value{};
    std::memcpy(&value, bytes, sizeof(value));
    return static_cast<wide>(value);
  });
}

/// Builds a valid scalar of a fixed-width type from its integer storage value, without a host wait.
struct scalar_from_storage {
  template <class T>
  std::unique_ptr<cudf::scalar> operator()(cudf::data_type type,
                                           wide value,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr) const
  {
    if constexpr (cudf::is_fixed_point<T>()) {
      return std::make_unique<cudf::fixed_point_scalar<T>>(
        static_cast<typename T::rep>(value), numeric::scale_type{type.scale()}, true, stream, mr);
    } else if constexpr (cudf::is_timestamp<T>()) {
      return std::make_unique<cudf::timestamp_scalar<T>>(
        T{typename T::duration{static_cast<typename T::rep>(value)}}, true, stream, mr);
    } else if constexpr (std::is_integral_v<T>) {
      return std::make_unique<cudf::numeric_scalar<T>>(static_cast<T>(value), true, stream, mr);
    } else {
      throw accumulation_invariant_error{"[bounds_accumulator] a type without numeric bounds"};
    }
  }
};

/// The device address of a fixed-width scalar's value.
struct scalar_storage {
  template <class T>
  void const* operator()(cudf::scalar const& bound) const
  {
    if constexpr (cudf::is_fixed_point<T>()) {
      return static_cast<cudf::fixed_point_scalar<T> const&>(bound).data();
    } else if constexpr (cudf::is_fixed_width<T>()) {
      return static_cast<cudf::detail::fixed_width_scalar<T> const&>(bound).data();
    } else {
      throw accumulation_invariant_error{"[bounds_accumulator] a type without numeric bounds"};
    }
  }
};

/// A valid scalar of @p type whose storage holds @p value.
[[nodiscard]] std::unique_ptr<cudf::scalar> make_bound(cudf::data_type type,
                                                       wide value,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(type, scalar_from_storage{}, type, value, stream, mr);
}

/// Byte-wise unsigned order: cuDF's ordering of strings.
[[nodiscard]] bool string_less(std::string const& left, std::string const& right)
{
  return std::lexicographical_compare(
    left.begin(), left.end(), right.begin(), right.end(), [](char a, char b) {
      return static_cast<unsigned char>(a) < static_cast<unsigned char>(b);
    });
}

}  // namespace

struct bounds_accumulator::impl {
  /**
   * @brief What a batch's slot of one key holds, known on the host.
   */
  struct slot {
    cudf::data_type carrier{cudf::type_id::EMPTY};  ///< The batch's column type
    int device = -1;                                ///< The GPU whose slot array holds the bounds
    bool valid = false;                             ///< Whether the batch had a valid row
  };

  impl(std::vector<key> bound, std::size_t batch_count)
    : keys{std::move(bound)},
      batches{batch_count},
      slots(keys.size(), std::vector<slot>(batch_count)),
      strings(keys.size(),
              std::vector<std::optional<std::pair<std::string, std::string>>>(batch_count))
  {
  }

  [[nodiscard]] bool is_string(std::size_t index) const noexcept
  {
    return keys[index].recorded.id() == cudf::type_id::STRING;
  }

  std::vector<key> keys;
  std::size_t batches;
  std::vector<std::vector<slot>> slots;  // [key][batch]; each batch writes only its own entries
  std::mutex leaf_mutex;  // Guards `strings`; a leaf, never held across CUDA work or a host wait.
  std::vector<std::vector<std::optional<std::pair<std::string, std::string>>>>
    strings;  // [key][batch]
  std::atomic<bool> leak_storage{false};
  std::vector<std::unique_ptr<accumulation_partial>> partials;  // Destroyed before leak_storage.
};

bounds_accumulator::bounds_accumulator(std::unique_ptr<impl> state) noexcept
  : _impl{std::move(state)}
{
}

bounds_accumulator::~bounds_accumulator()                                        = default;
bounds_accumulator::bounds_accumulator(bounds_accumulator&&) noexcept            = default;
bounds_accumulator& bounds_accumulator::operator=(bounds_accumulator&&) noexcept = default;

std::optional<bounds_accumulator> bounds_accumulator::try_create(
  std::vector<key> keys, std::size_t batches, std::span<dynamic_filter_replica_space const> targets)
{
  std::vector<std::size_t> array_bytes;
  array_bytes.reserve(keys.size());
  for (auto const& bound : keys) {
    if (!sirius_dynamic_zone_map_filter::supports(bound.recorded)) {
      throw accumulation_invariant_error{"[bounds_accumulator] a key type without bounds"};
    }
    if (batches > std::numeric_limits<std::size_t>::max() / k_slot_bytes) {
      throw accumulation_invariant_error{"[bounds_accumulator] too many batches"};
    }
    // STRING bounds live on the host.
    array_bytes.push_back(bound.recorded.id() == cudf::type_id::STRING ? 0
                                                                       : batches * k_slot_bytes);
  }
  auto state    = std::make_unique<impl>(std::move(keys), batches);
  auto partials = lease_partials(targets, array_bytes, /*zero_fill=*/false, state->leak_storage);
  if (!partials) { return std::nullopt; }
  state->partials = std::move(*partials);
  return bounds_accumulator{std::move(state)};
}

std::size_t bounds_accumulator::contribution_scratch_bytes(std::size_t rows)
{
  auto const items = static_cast<cudf::size_type>(
    std::min<std::size_t>(rows, std::numeric_limits<cudf::size_type>::max()));
  std::size_t widest = 0;
  for (auto const id : {cudf::type_id::INT8,
                        cudf::type_id::INT16,
                        cudf::type_id::INT32,
                        cudf::type_id::INT64,
                        cudf::type_id::BOOL8,
                        cudf::type_id::DECIMAL128}) {
    dispatch_bound_storage(cudf::data_type{id}, [&](auto tag) {
      using T          = decltype(tag);
      auto const input = thrust::make_transform_iterator(
        thrust::counting_iterator<cudf::size_type>{0}, masked_bound<T>{nullptr, nullptr, 0, T{}});
      std::size_t min_bytes = 0;
      std::size_t max_bytes = 0;
      check_cuda(cub::DeviceReduce::Min(nullptr, min_bytes, input, static_cast<T*>(nullptr), items),
                 "cub::DeviceReduce::Min(size)");
      check_cuda(cub::DeviceReduce::Max(nullptr, max_bytes, input, static_cast<T*>(nullptr), items),
                 "cub::DeviceReduce::Max(size)");
      widest = std::max({widest, min_bytes, max_bytes});
    });
  }
  return tracked_replica_allocation_bytes(widest);
}

std::size_t bounds_accumulator::finish_scratch_bytes(std::size_t narrowed_carriers) noexcept
{
  if (narrowed_carriers == 0) { return 0; }
  // Each held pair is two scalars of a value and a validity flag. A restore in progress holds the
  // pair's restored minimum, the narrow scalar, `restore_scalar`'s one-row column and cast result,
  // and the restored scalar: eight allocations. Every allocation holds at most one bound.
  auto const allocations = memory::saturating_add(memory::saturating_mul(narrowed_carriers, 4), 8);
  return memory::saturating_mul(allocations, tracked_replica_allocation_bytes(k_bound_bytes));
}

bool bounds_accumulator::enqueue_add(std::size_t batch_index,
                                     cudf::table_view const& input,
                                     rmm::cuda_device_id device,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref scratch)
{
  nvtx_scoped_range range{"dynfilter::bounds::contribute"};
  check_cuda(cudaGetLastError(), "pending CUDA error before bounds contribution");
  auto* target = find_partial(_impl->partials, device.value());
  if (target == nullptr) { return false; }
  if (batch_index >= _impl->batches) {
    throw accumulation_invariant_error{"[bounds_accumulator] batch index out of range"};
  }
  int stream_device = -1;
  check_cuda(cudaStreamGetDevice(stream.get(), &stream_device), "cudaStreamGetDevice");
  if (stream_device != device.value()) {
    throw accumulation_invariant_error{
      "[bounds_accumulator::enqueue_add] the contribution stream belongs to another GPU"};
  }
  auto const& keys = _impl->keys;
  rmm::cuda_set_device_raii guard{device};
  std::array const joined{stream};
  try {
    join_on_failure(joined, [&] {
      target->enqueue_contribution(stream, [&] {
        for (std::size_t index = 0; index < keys.size(); ++index) {
          auto const column = input.column(keys[index].ordinal);
          if (_impl->is_string(index) || column.size() == column.null_count()) { continue; }
          auto* const out =
            static_cast<std::byte*>(target->arrays[index]->data()) + batch_index * k_slot_bytes;
          dispatch_bound_storage(column.type(), [&](auto tag) {
            enqueue_bounds<decltype(tag)>(column, out, stream, scratch);
          });
          _impl->slots[index][batch_index] = {column.type(), device.value(), true};
        }
      });
    });
  } catch (unjoined_gpu_work const&) {
    _impl->leak_storage.store(true);
    throw;
  }
  // STRING bounds: the one host wait a contribution may make, outside the leaf mutex.
  for (std::size_t index = 0; index < keys.size(); ++index) {
    if (!_impl->is_string(index)) { continue; }
    auto const column = input.column(keys[index].ordinal);
    if (column.type().id() != cudf::type_id::STRING) {
      throw accumulation_invariant_error{
        "[bounds_accumulator] a STRING key's column is not STRING"};
    }
    if (column.size() == column.null_count()) { continue; }
    // The results are read straight back to the host, so they live in pinned host memory and take
    // nothing from the task's scratch.
    auto const reduce = [&](std::unique_ptr<cudf::reduce_aggregation> aggregation) {
      auto const bound = cudf::reduce(
        column, *aggregation, column.type(), stream, cudf::get_pinned_memory_resource());
      return static_cast<cudf::string_scalar const&>(*bound).to_string(stream);
    };
    auto minimum = reduce(cudf::make_min_aggregation<cudf::reduce_aggregation>());
    auto maximum = reduce(cudf::make_max_aggregation<cudf::reduce_aggregation>());
    std::scoped_lock lock(_impl->leaf_mutex);
    _impl->strings[index][batch_index].emplace(std::move(minimum), std::move(maximum));
  }
  return true;
}

std::vector<std::optional<bounds_accumulator::key_bounds>> bounds_accumulator::finish(
  ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range range{"dynfilter::bounds::finish"};
  auto const& keys     = _impl->keys;
  auto const& partials = _impl->partials;

  // Every GPU's slots in one pinned buffer: [partial][key] arrays, back to back.
  std::vector<std::vector<std::size_t>> slot_offsets(partials.size(),
                                                     std::vector<std::size_t>(keys.size(), 0));
  std::size_t slot_bytes = 0;
  for (std::size_t partial = 0; partial < partials.size(); ++partial) {
    for (std::size_t index = 0; index < keys.size(); ++index) {
      slot_offsets[partial][index] = slot_bytes;
      if (partials[partial]->arrays[index]) {
        slot_bytes += partials[partial]->arrays[index]->size();
      }
    }
  }
  auto host_slots = cudf::detail::make_pinned_vector_async<std::byte>(slot_bytes, stream);
  // Each GPU's copies follow its contributions on its partial's stream; `stream` waits for them
  // all.
  for (std::size_t partial = 0; partial < partials.size(); ++partial) {
    auto const& source = *partials[partial];
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{source.device_id}};
    for (std::size_t index = 0; index < keys.size(); ++index) {
      if (!source.arrays[index]) { continue; }
      check_cuda(cudaMemcpyAsync(host_slots.data() + slot_offsets[partial][index],
                                 source.arrays[index]->data(),
                                 source.arrays[index]->size(),
                                 cudaMemcpyDeviceToHost,
                                 source.stream->value()),
                 "cudaMemcpyAsync(bound slots)");
    }
    cuda_event copied;
    check_cuda(cudaEventRecord(copied.get(), source.stream->value()),
               "cudaEventRecord(bound slots)");
    check_cuda(cudaStreamWaitEvent(stream.get(), copied.get(), 0),
               "cudaStreamWaitEvent(bound slots)");
  }
  stream.sync();
  auto const partial_of = [&](int device) {
    return static_cast<std::size_t>(
      std::ranges::find(partials, device, [](auto const& entry) { return entry->device_id; }) -
      partials.begin());
  };

  // Per key: candidates already at the recorded type, and carriers whose pair must be restored.
  struct restored_pair {
    std::size_t key;
    std::unique_ptr<cudf::scalar> low;
    std::unique_ptr<cudf::scalar> high;
  };
  std::vector<std::vector<key_bounds::storage_bounds>> candidates(keys.size());
  std::vector<restored_pair> restoring;
  std::vector<std::optional<key_bounds>> merged(keys.size());
  for (std::size_t index = 0; index < keys.size(); ++index) {
    auto const recorded = keys[index].recorded;
    if (_impl->is_string(index)) {
      std::optional<key_bounds::string_bounds> best;
      for (auto const& batch : _impl->strings[index]) {
        if (!batch) { continue; }
        if (!best) {
          best = batch;
          continue;
        }
        if (string_less(batch->first, best->first)) { best->first = batch->first; }
        if (string_less(best->second, batch->second)) { best->second = batch->second; }
      }
      if (best) { merged[index] = key_bounds{recorded, std::move(*best), true}; }
      continue;
    }
    // Per carrier, the minimum and maximum by integer storage.
    std::vector<std::pair<cudf::data_type, key_bounds::storage_bounds>> groups;
    for (auto const& slot : _impl->slots[index]) {
      if (!slot.valid) { continue; }
      auto const* bytes =
        host_slots.data() + slot_offsets[partial_of(slot.device)][index] +
        static_cast<std::size_t>(&slot - _impl->slots[index].data()) * k_slot_bytes;
      auto const low  = decode(slot.carrier, bytes);
      auto const high = decode(slot.carrier, bytes + k_bound_bytes);
      auto group = std::ranges::find(groups, slot.carrier, &decltype(groups)::value_type::first);
      if (group == groups.end()) {
        groups.push_back({slot.carrier, {low, high}});
      } else {
        group->second.first  = std::min(group->second.first, low);
        group->second.second = std::max(group->second.second, high);
      }
    }
    for (auto const& [carrier, pair] : groups) {
      if (carrier == recorded) {
        candidates[index].push_back(pair);
        continue;
      }
      // One restore at a time: each narrow scalar is released before the next is made.
      auto const restore = [&](wide value) {
        auto const narrow = make_bound(carrier, value, stream, mr);
        return restore_scalar(*narrow, recorded, stream, mr);
      };
      restoring.push_back({index, restore(pair.first), restore(pair.second)});
    }
  }

  // One read of every restored value, after which the restored scalars are released.
  if (!restoring.empty()) {
    auto restored_values = cudf::detail::make_pinned_vector_async<std::byte>(
      2 * restoring.size() * k_bound_bytes, stream);
    for (std::size_t pair = 0; pair < restoring.size(); ++pair) {
      auto const& restored = restoring[pair];
      for (auto const& [bound, at] : {std::pair{restored.low.get(), 2 * pair},
                                      std::pair{restored.high.get(), 2 * pair + 1}}) {
        check_cuda(cudaMemcpyAsync(restored_values.data() + at * k_bound_bytes,
                                   cudf::type_dispatcher(bound->type(), scalar_storage{}, *bound),
                                   cudf::size_of(bound->type()),
                                   cudaMemcpyDeviceToHost,
                                   stream.get()),
                   "cudaMemcpyAsync(restored bound)");
      }
    }
    stream.sync();
    for (std::size_t pair = 0; pair < restoring.size(); ++pair) {
      auto const recorded = keys[restoring[pair].key].recorded;
      auto const* low     = restored_values.data() + 2 * pair * k_bound_bytes;
      candidates[restoring[pair].key].emplace_back(decode(recorded, low),
                                                   decode(recorded, low + k_bound_bytes));
    }
    restoring.clear();
  }

  for (std::size_t index = 0; index < keys.size(); ++index) {
    if (candidates[index].empty()) { continue; }
    auto best = candidates[index].front();
    for (auto const& [low, high] : candidates[index]) {
      best.first  = std::min(best.first, low);
      best.second = std::max(best.second, high);
    }
    auto const recorded = keys[index].recorded;
    bool fits           = true;
    if (recorded.id() == cudf::type_id::DECIMAL128) {
      // The `column_values_fit` rule against DECIMAL64 at the key's scale, applied to the bounds.
      auto const scale = recorded.scale();
      fits             = scale <= 0 && scale >= -std::numeric_limits<std::uint8_t>::max() &&
             sirius::numeric_range_fits(
               cudf::data_type{cudf::type_id::DECIMAL64, scale},
               sirius::decimal_range(best.first, best.second, static_cast<std::uint8_t>(-scale)));
    }
    merged[index] = key_bounds{recorded, best, fits};
  }
  return merged;
}

std::size_t bounds_accumulator::fixed_width_scalar_bytes(cudf::data_type recorded) noexcept
{
  return 2 * (tracked_replica_allocation_bytes(cudf::size_of(recorded)) +
              tracked_replica_allocation_bytes(sizeof(bool)));
}

std::size_t bounds_accumulator::key_bounds::device_bytes() const noexcept
{
  if (auto const* strings = std::get_if<string_bounds>(&values)) {
    // A string scalar takes its characters (none for an empty string) and its validity flag.
    auto const text = [](std::string const& bound) {
      return tracked_replica_allocation_bytes(std::max<std::size_t>(bound.size(), 1)) +
             tracked_replica_allocation_bytes(sizeof(bool));
    };
    return text(strings->first) + text(strings->second);
  }
  return fixed_width_scalar_bytes(type);
}

std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>>
bounds_accumulator::key_bounds::to_scalars(::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr) const
{
  if (auto const* strings = std::get_if<string_bounds>(&values)) {
    return {std::make_unique<cudf::string_scalar>(strings->first, true, stream, mr),
            std::make_unique<cudf::string_scalar>(strings->second, true, stream, mr)};
  }
  auto const& [low, high] = std::get<storage_bounds>(values);
  return {make_bound(type, low, stream, mr), make_bound(type, high, stream, mr)};
}

bool bounds_accumulator::releasable_here() const noexcept
{
  if (!_impl) { return true; }
  try {
    return std::ranges::all_of(_impl->partials,
                               [](auto const& entry) { return entry->releasable_here(); });
  } catch (...) {
    return false;
  }
}

}  // namespace sirius::op::detail
