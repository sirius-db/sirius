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
// Sentinel marking empty slots; may never legitimately be inserted. TPC-H-style surrogate keys do
// not use INT64_MIN; supporting arbitrary INT64 domains would require an alternate sentinel path.
constexpr std::int64_t kEmptySentinel = std::numeric_limits<std::int64_t>::min();

// Match estimated_set_bytes' 0.5 load factor: capacity = 2 × keys.
constexpr std::size_t kCapacityFactor = 2;
constexpr double kLoadFactor          = 1.0 / kCapacityFactor;

// The number of threads to use per key probe.
constexpr std::size_t kCgSize = 1;

// Minimum capacity for the set.
constexpr std::size_t kMinCapacity = 8;

// TODO: template on the key type to support other types.
using set_alloc = sirius::rmm_cuco_allocator<std::int64_t>;
using set_type =
  cuco::static_set<std::int64_t,
                   cuco::extent<std::size_t>,
                   cuda::thread_scope_device,
                   cuda::std::equal_to<std::int64_t>,
                   cuco::double_hashing<kCgSize, cuco::default_hash_function<std::int64_t>>,
                   set_alloc>;
}  // namespace

struct sirius_dynamic_in_list_filter::set_impl {
  set_type set;

  set_impl(std::size_t capacity, rmm::device_async_resource_ref mr, cuda::stream_ref stream)
    : set{cuco::extent<std::size_t>{capacity},
          cuco::empty_key<std::int64_t>{kEmptySentinel},
          {},
          {},
          {},
          {},
          set_alloc{mr},
          stream}
  {
  }
};

sirius_dynamic_in_list_filter::sirius_dynamic_in_list_filter(cudf::column_view const& keys,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
  : _key_type(keys.type()), _num_keys(static_cast<std::size_t>(keys.size()))
{
  if (!supports(keys)) {
    throw std::invalid_argument(
      "[sirius_dynamic_in_list_filter] unsupported key column (INT64 with no nulls required).");
  }

  cuda::stream_ref const s{stream.value()};
  _set = std::make_unique<set_impl>(
    std::max<std::size_t>(kCapacityFactor * _num_keys, kMinCapacity), mr, s);
  if (_num_keys > 0) {
    auto const* d = keys.data<std::int64_t>();
    _set->set.insert_async(d, d + _num_keys, s);
  }
}

bool sirius_dynamic_in_list_filter::supports(cudf::column_view const& keys) noexcept
{ return keys.type().id() == cudf::type_id::INT64 && keys.null_count() == 0; }

sirius_dynamic_in_list_filter::~sirius_dynamic_in_list_filter() = default;

bool sirius_dynamic_in_list_filter::has_persistent_set() const noexcept
{ return static_cast<bool>(_set); }

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
  auto const* d = probe.data<std::int64_t>();
  _set->set.contains_async(d, d + n, out->mutable_view().data<bool>(), s);
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
