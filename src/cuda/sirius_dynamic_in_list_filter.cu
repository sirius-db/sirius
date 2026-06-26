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

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/exec_policy.hpp>

#include <cuco/static_set.cuh>
#include <cuda/sirius_rmm_cuco_allocator.cuh>
#include <cuda/std/functional>
#include <cuda/stream_ref>

#include <op/sirius_dynamic_filter.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace sirius::op {

namespace {
// Match estimated_set_bytes' 0.5 load factor: capacity = 2 × keys.
constexpr std::size_t kCapacityFactor = 2;
constexpr double kLoadFactor          = 1.0 / kCapacityFactor;

// Threads per key probe.
constexpr std::size_t kCgSize = 1;

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
                                  set_alloc<KeyT>>;

template <class KeyT>
std::unique_ptr<set_type<KeyT>> build_set(cudf::column_view const& keys,
                                          std::size_t capacity,
                                          rmm::device_async_resource_ref mr,
                                          cuda::stream_ref stream)
{
  std::unique_ptr<set_type<KeyT>> set(
    new set_type<KeyT>{cuco::extent<std::size_t>{capacity},
                       cuco::empty_key<KeyT>{std::numeric_limits<KeyT>::min()},
                       {},
                       {},
                       {},
                       {},
                       set_alloc<KeyT>{mr},
                       stream});
  if (keys.size() > 0) {
    auto const* d = keys.data<KeyT>();
    set->insert_async(d, d + keys.size(), stream);
  }
  return set;
}
}  // namespace

struct sirius_dynamic_in_list_filter::set_impl {
  std::unique_ptr<set_type<std::int32_t>> s32;
  std::unique_ptr<set_type<std::int64_t>> s64;
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
  auto const capacity = std::max<std::size_t>(kCapacityFactor * _num_keys, kMinCapacity);
  _set                = std::make_unique<set_impl>();
  switch (_key_type.id()) {
    case cudf::type_id::INT32: _set->s32 = build_set<std::int32_t>(keys, capacity, mr, s); break;
    case cudf::type_id::INT64: _set->s64 = build_set<std::int64_t>(keys, capacity, mr, s); break;
    default: break;  // unreachable: supports() gates the type
  }
}

bool sirius_dynamic_in_list_filter::supports(cudf::column_view const& keys) noexcept
{
  auto const id = keys.type().id();
  return (id == cudf::type_id::INT32 || id == cudf::type_id::INT64) && keys.null_count() == 0;
}

sirius_dynamic_in_list_filter::~sirius_dynamic_in_list_filter() = default;

bool sirius_dynamic_in_list_filter::has_persistent_set() const noexcept
{
  return _set && (_set->s32 || _set->s64);
}

std::unique_ptr<cudf::column> sirius_dynamic_in_list_filter::compute_mask(
  cudf::column_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  if (probe.type() != _key_type) { return nullptr; }

  auto const n = probe.size();
  auto out     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cuda::stream_ref const s{stream.value()};
  auto* const outp = out->mutable_view().data<bool>();

  // probe.type() == _key_type, so the populated set matches the probe width.
  if (_set->s32) {
    auto const* d = probe.data<std::int32_t>();
    _set->s32->contains_async(d, d + n, outp, s);
  } else {
    auto const* d = probe.data<std::int64_t>();
    _set->s64->contains_async(d, d + n, outp, s);
  }
  if (probe.nullable() && probe.null_count() > 0) {
    out->set_null_mask(cudf::copy_bitmask(probe, stream, mr), probe.null_count());
  }
  return out;
}

std::size_t sirius_dynamic_in_list_filter::size() const noexcept { return _num_keys; }

std::size_t sirius_dynamic_in_list_filter::estimated_set_bytes(std::size_t num_keys,
                                                               cudf::data_type key_type) noexcept
{
  std::size_t const slot = cudf::is_fixed_width(key_type)
                             ? static_cast<std::size_t>(cudf::size_of(key_type))
                             : sizeof(std::int64_t);  // variable-width keys hash to ~8B slots
  return static_cast<std::size_t>(static_cast<double>(num_keys) * static_cast<double>(slot) /
                                  kLoadFactor);
}

}  // namespace sirius::op
