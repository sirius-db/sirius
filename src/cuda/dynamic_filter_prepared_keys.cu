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
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/dynamic_filter_probe.cuh>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <op/dynamic_filter/prepared_build_keys.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sirius::op {

namespace {

/// Writes row `row` to the same position when it fits; the input has no nulls.
template <class KeyT, class Iter>
struct bounded_dense_write {
  Iter keys;
  KeyT* out;
  cudf::size_type capacity;
  std::uint32_t* overflow;

  __device__ void operator()(cudf::size_type row) const
  {
    if (row < capacity) {
      out[row] = keys[row];
    } else if (overflow != nullptr) {
      atomicOr(overflow, 1u);
    }
  }
};

/// Writes valid row `row` to its rank among the valid rows when that rank fits.
template <class KeyT, class Iter>
struct bounded_compacting_write {
  Iter keys;
  cudf::bitmask_type const* mask;
  cudf::size_type offset;
  cudf::size_type const* ranks;
  KeyT* out;
  cudf::size_type capacity;
  std::uint32_t* overflow;

  __device__ void operator()(cudf::size_type row) const
  {
    if (!cudf::bit_is_set(mask, offset + row)) { return; }
    auto const rank = ranks[row];
    if (rank < capacity) {
      out[rank] = keys[row];
    } else if (overflow != nullptr) {
      atomicOr(overflow, 1u);
    }
  }
};

/// 1 for a valid row, 0 for a null one.
struct valid_flag {
  cudf::bitmask_type const* mask;
  cudf::size_type offset;

  __device__ cudf::size_type operator()(cudf::size_type row) const
  {
    return cudf::bit_is_set(mask, offset + row) ? 1 : 0;
  }
};

template <class KeyT, class Iter>
void compact_bounded(Iter keys,
                     cudf::column_view const& column,
                     KeyT* out,
                     cudf::size_type capacity,
                     std::uint32_t* overflow,
                     ::cuda::stream_ref stream,
                     rmm::device_async_resource_ref scratch)
{
  auto const rows = column.size();
  if (rows == 0) { return; }
  auto const policy = rmm::exec_policy_nosync(stream, scratch);
  auto const first  = thrust::counting_iterator<cudf::size_type>(0);
  if (column.null_count() == 0) {
    thrust::for_each_n(
      policy, first, rows, bounded_dense_write<KeyT, Iter>{keys, out, capacity, overflow});
    return;
  }
  // A valid row's destination is the number of valid rows before it.
  rmm::device_uvector<cudf::size_type> ranks(rows, stream, scratch);
  auto const flags =
    thrust::make_transform_iterator(first, valid_flag{column.null_mask(), column.offset()});
  thrust::exclusive_scan(policy, flags, flags + rows, ranks.begin());
  thrust::for_each_n(
    policy,
    first,
    rows,
    bounded_compacting_write<KeyT, Iter>{
      keys, column.null_mask(), column.offset(), ranks.data(), out, capacity, overflow});
}

}  // namespace

void prepare_build_keys(membership_key_domain const& domain,
                        cudf::column_view const& keys,
                        cudf::mutable_column_view out,
                        std::uint32_t* overflow,
                        ::cuda::stream_ref stream,
                        rmm::device_async_resource_ref scratch)
{
  if (out.type() != membership_rep_type(domain) || out.nullable()) {
    throw std::invalid_argument(
      "[prepare_build_keys] the output must be a non-nullable column of the domain's rep type");
  }
  if (domain.family == membership_key_family::string_hash) {
    if (keys.type().id() != cudf::type_id::STRING) {
      throw std::logic_error(
        "[prepare_build_keys] the build column is not a carrier of its domain");
    }
    detail::with_string_fingerprint_iterator(keys, stream, [&](auto fingerprints) {
      compact_bounded<std::uint64_t>(
        fingerprints, keys, out.data<std::uint64_t>(), out.size(), overflow, stream, scratch);
    });
    return;
  }
  bool const prepared = detail::dispatch_key_rep(domain.rep, [&](auto key_tag) {
    using key_type = decltype(key_tag);
    return detail::with_build_key_iterator<key_type>(
      domain, keys, stream, scratch, [&](auto begin, auto) {
        compact_bounded<key_type>(
          begin, keys, out.data<key_type>(), out.size(), overflow, stream, scratch);
      });
  });
  if (!prepared) {
    throw std::logic_error("[prepare_build_keys] the build column is not a carrier of its domain");
  }
}

prepared_build_column prepare_build_column(std::string_view filter_name,
                                           cudf::column_view const& keys,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const prefix = std::string{filter_name};
  auto const domain = classify_membership_key(keys.type());
  if (!domain.has_value()) {
    throw std::invalid_argument(prefix + " unsupported key type (see membership_key_supported).");
  }
  // A DECIMAL128 build whose unscaled values exceed the int64 rep cannot be stored exactly.
  if (!membership_build_fits_rep(keys, stream, mr)) {
    throw std::invalid_argument(
      prefix + " build keys do not fit the key rep (DECIMAL128 values outside int64).");
  }
  // Null build keys match nothing under the join's null_equality::UNEQUAL, so they are left out
  // exactly; the valid-row count is known on the host, so nothing is read back.
  prepared_build_column prepared{
    .domain = *domain,
    .keys   = cudf::make_fixed_width_column(membership_rep_type(*domain),
                                          keys.size() - keys.null_count(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr)};
  prepare_build_keys(*domain, keys, prepared.keys->mutable_view(), nullptr, stream, mr);
  return prepared;
}

}  // namespace sirius::op
