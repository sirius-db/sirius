/*
 * Copyright 2025, Sirius Contributors.
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

#include "op/right_join_build_matches.hpp"

#include <cudf/join/join.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace sirius::op {

namespace {

struct mark_row {
  std::uint8_t* matched;
  cudf::size_type num_rows;
  __device__ void operator()(cudf::size_type row) const
  {
    if (row >= 0 && row < num_rows) { matched[row] = 1; }
  }
};

struct is_unmatched {
  std::uint8_t const* matched;
  __device__ bool operator()(cudf::size_type row) const { return matched[row] == 0; }
};

struct count_pair {
  unsigned long long* counts;
  cudf::size_type num_rows;
  cudf::size_type const* probe_rows;
  cudf::size_type const* build_rows;
  cudf::bitmask_type const* counted_mask;
  cudf::size_type counted_offset;
  __device__ void operator()(std::size_t i) const
  {
    auto const build = build_rows[i];
    auto const probe = probe_rows[i];
    if (build < 0 || build >= num_rows || probe == cudf::JoinNoMatch) { return; }
    if (counted_mask != nullptr && !cudf::bit_is_set(counted_mask, probe + counted_offset)) {
      return;
    }
    atomicAdd(counts + build, 1ULL);
  }
};

struct fill_zero {
  std::int64_t fill;
  __device__ std::int64_t operator()(std::int64_t count) const { return count == 0 ? fill : count; }
};

}  // namespace

void count_build_matches(rmm::device_uvector<std::int64_t>& counts,
                         rmm::device_uvector<cudf::size_type> const& probe_rows,
                         rmm::device_uvector<cudf::size_type> const& build_rows,
                         cudf::bitmask_type const* counted_mask,
                         cudf::size_type counted_offset,
                         ::cuda::stream_ref stream)
{
  static_assert(sizeof(std::int64_t) == sizeof(unsigned long long));
  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::counting_iterator<std::size_t>{0},
                     build_rows.size(),
                     count_pair{reinterpret_cast<unsigned long long*>(counts.data()),
                                static_cast<cudf::size_type>(counts.size()),
                                probe_rows.data(),
                                build_rows.data(),
                                counted_mask,
                                counted_offset});
}

void fill_unmatched_counts(rmm::device_uvector<std::int64_t>& counts,
                           std::int64_t unmatched_fill,
                           ::cuda::stream_ref stream)
{
  if (unmatched_fill == 0) { return; }
  thrust::transform(rmm::exec_policy_nosync(stream),
                    counts.begin(),
                    counts.end(),
                    counts.begin(),
                    fill_zero{unmatched_fill});
}

void mark_build_matches(rmm::device_uvector<std::uint8_t>& matched,
                        rmm::device_uvector<cudf::size_type> const& build_rows,
                        ::cuda::stream_ref stream)
{
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   build_rows.begin(),
                   build_rows.end(),
                   mark_row{matched.data(), static_cast<cudf::size_type>(matched.size())});
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> unmatched_build_rows(
  rmm::device_uvector<std::uint8_t> const* matched,
  cudf::size_type num_rows,
  ::cuda::stream_ref stream)
{
  auto rows = std::make_unique<rmm::device_uvector<cudf::size_type>>(num_rows, stream);
  if (matched == nullptr) {
    thrust::sequence(rmm::exec_policy_nosync(stream), rows->begin(), rows->end());
    return rows;
  }
  auto const end = thrust::copy_if(rmm::exec_policy(stream),
                                   thrust::counting_iterator<cudf::size_type>{0},
                                   thrust::counting_iterator<cudf::size_type>{num_rows},
                                   rows->begin(),
                                   is_unmatched{matched->data()});
  rows->resize(static_cast<std::size_t>(end - rows->begin()), stream);
  return rows;
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> no_match_rows(std::size_t count,
                                                                     ::cuda::stream_ref stream)
{
  auto rows = std::make_unique<rmm::device_uvector<cudf::size_type>>(count, stream);
  thrust::fill(rmm::exec_policy_nosync(stream), rows->begin(), rows->end(), cudf::JoinNoMatch);
  return rows;
}

}  // namespace sirius::op
