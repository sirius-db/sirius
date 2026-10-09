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

#include "op/aggregate/aggregate_op_util.hpp"

#include "cudf/cudf_utils.hpp"
#include "duckdb/common/assert.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/node.hpp"
#include "sirius/exception.hpp"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius {
namespace op {

namespace {

// Single place that builds the "Unsupported aggregate function: <name>" diagnostic so the
// message (and the aggregate_id -> name lookup) is not repeated at every rejection site.
[[noreturn]] void throw_unsupported_aggregate(sirius::aggregate_id fid,
                                              std::string_view detail = {})
{
  auto const name = sirius::to_duckdb_aggregate_name(fid);
  throw std::runtime_error(detail.empty()
                             ? std::format("Unsupported aggregate function: {}", name)
                             : std::format("Unsupported aggregate function: {} {}", name, detail));
}

}  // namespace

std::optional<cudf::aggregation::Kind> to_cudf_aggregation_kind(sirius::aggregate_id id)
{
  switch (id) {
    case sirius::aggregate_id::sum:
    case sirius::aggregate_id::sum_no_overflow: return cudf::aggregation::Kind::SUM;
    case sirius::aggregate_id::count: return cudf::aggregation::Kind::COUNT_VALID;
    case sirius::aggregate_id::count_star: return cudf::aggregation::Kind::COUNT_ALL;
    case sirius::aggregate_id::min: return cudf::aggregation::Kind::MIN;
    case sirius::aggregate_id::max: return cudf::aggregation::Kind::MAX;
    case sirius::aggregate_id::avg:
    case sirius::aggregate_id::first: return std::nullopt;
  }
  return std::nullopt;
}

CudfAggregateDefinitions convert_duckdb_aggregates_to_cudf(
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& groups_p,
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& expressions)
{
  CudfAggregateDefinitions result;

  // 1. Extract group_idx from groups_p
  for (const auto& group : groups_p) {
    auto const& ref =
      sirius::ast::require_reference(group.get(), "convert_duckdb_aggregates_to_cudf group");
    result.group_idx.push_back(static_cast<int>(ref.column_index));
  }

  // 2. Extract aggregates (cudf::aggregation::Kind) from expressions
  for (const auto& aggregate : expressions) {
    auto const& aggr = sirius::ast::require_aggregate(
      aggregate.get(), "convert_duckdb_aggregates_to_cudf aggregate");
    auto const fid       = aggr.function();
    auto const& children = aggr.arguments();

    // Handle AVG specially: it expands into SUM + COUNT_VALID
    if (fid == sirius::aggregate_id::avg) {
      D_ASSERT(children.size() == 1);
      D_ASSERT(children[0]->is_reference());
      auto col_idx = static_cast<int>(children[0]->as_reference().column_index);

      size_t sum_position = result.cudf_aggregates.size();
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::SUM);
      result.cudf_aggregate_idx.push_back(col_idx);
      result.cudf_aggregate_struct_col_indices.push_back({});
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::COUNT_VALID);
      result.cudf_aggregate_idx.push_back(col_idx);
      result.cudf_aggregate_struct_col_indices.push_back({});
      result.aggregate_slots.push_back(
        AggregateSlot{true, false, sum_position, sirius::get_cudf_type(aggr.return_type())});
      result.has_avg = true;
      continue;
    }

    // Handle COUNT(DISTINCT col) and COUNT(DISTINCT (col1, col2, ...)):
    // Use COLLECT_SET locally; merge via MERGE_SETS; then count list elements.
    // For multi-column, a struct column is synthesized from the component columns.
    if (aggr.distinct() && fid == sirius::aggregate_id::count) {
      D_ASSERT(children.size() == 1);
      auto const& child = *children[0];
      size_t position   = result.cudf_aggregates.size();
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::COLLECT_SET);

      if (child.is_reference()) {
        // Single-column case: COUNT(DISTINCT col)
        result.cudf_aggregate_idx.push_back(static_cast<int>(child.as_reference().column_index));
        result.cudf_aggregate_struct_col_indices.push_back({});
      } else {
        // Multi-column case: COUNT(DISTINCT (col1, col2, ...)) — child is a struct_pack expression
        D_ASSERT(child.is_function_call());
        auto const& func_expr = child.as_function_call();
        std::vector<int> struct_indices;
        for (auto const& arg : func_expr.arguments()) {
          D_ASSERT(arg->is_reference());
          struct_indices.push_back(static_cast<int>(arg->as_reference().column_index));
        }
        D_ASSERT(!struct_indices.empty());
        result.cudf_aggregate_idx.push_back(-1);  // sentinel: struct column, see gpu_aggregate_impl
        result.cudf_aggregate_struct_col_indices.push_back(std::move(struct_indices));
      }

      result.aggregate_slots.push_back(AggregateSlot{false, true, position});
      result.has_count_distinct = true;
      continue;
    }

    auto const agg_kind = to_cudf_aggregation_kind(fid);
    if (!agg_kind) { throw_unsupported_aggregate(fid); }
    size_t current_position = result.cudf_aggregates.size();
    result.cudf_aggregates.push_back(*agg_kind);

    // 3. Extract aggregate_idx from the children of the aggregate expression
    if (children.empty()) {
      // COUNT(*) has no children - use 0 as a placeholder (will be handled by COUNT_ALL)
      if (fid == sirius::aggregate_id::count_star) {
        result.cudf_aggregate_idx.push_back(0);
      } else {
        throw_unsupported_aggregate(fid, "with no children");
      }
    } else {
      if (children.size() == 1) {
        // Extract the column index from the first child (most aggregates have one child)
        D_ASSERT(children[0]->is_reference());
        result.cudf_aggregate_idx.push_back(
          static_cast<int>(children[0]->as_reference().column_index));
      } else {
        throw_unsupported_aggregate(fid, "with " + std::to_string(children.size()) + " children");
      }
    }
    result.cudf_aggregate_struct_col_indices.push_back({});
    result.aggregate_slots.push_back(AggregateSlot{false, false, current_position});
  }

  return result;
}

namespace {

/// A DECIMAL32 or DECIMAL64 scalar's unscaled value: its device address and size in bytes, both
/// derived from the scalar's own type so they cannot disagree. Anything else is a caller bug.
struct scalar_rep {
  void const* data;
  size_t bytes;
  cudf::type_id id;
};

scalar_rep scalar_rep_of(cudf::scalar& s)
{
  switch (s.type().id()) {
    case cudf::type_id::DECIMAL32:
      return {static_cast<cudf::fixed_point_scalar<numeric::decimal32>&>(s).data(),
              sizeof(int32_t),
              cudf::type_id::DECIMAL32};
    case cudf::type_id::DECIMAL64:
      return {static_cast<cudf::fixed_point_scalar<numeric::decimal64>&>(s).data(),
              sizeof(int64_t),
              cudf::type_id::DECIMAL64};
    default:
      throw sirius::internal_exception("scalar_rep_of: expected a DECIMAL32 or DECIMAL64 scalar");
  }
}

/// A small pinned host buffer for a device-to-host readback. It comes from cuDF's pinned resource,
/// which Sirius backs with a slab pool for allocations this small (cuDF's default pool is used when
/// no Sirius context is installed). Allocation and release are ordered on @p stream.
class pinned_host_buffer {
 public:
  pinned_host_buffer(size_t bytes, ::cuda::stream_ref stream)
    : mr_(cudf::get_pinned_memory_resource()),
      stream_(stream),
      bytes_(bytes),
      data_(mr_.allocate(stream, bytes, alignof(int64_t)))
  {
  }
  ~pinned_host_buffer() { mr_.deallocate(stream_, data_, bytes_, alignof(int64_t)); }
  pinned_host_buffer(pinned_host_buffer const&)            = delete;
  pinned_host_buffer& operator=(pinned_host_buffer const&) = delete;

  int64_t* data() const { return static_cast<int64_t*>(data_); }

 private:
  rmm::host_device_async_resource_ref mr_;
  ::cuda::stream_ref stream_;
  size_t bytes_;
  void* data_;
};

/// |v| as an unsigned 128-bit value. The caller holds v in 128 bits already widened from a 32 or
/// 64-bit decimal, so negating the most negative 32/64-bit value (-2^31, -2^63) cannot overflow.
__uint128_t magnitude(__int128_t v) { return static_cast<__uint128_t>(v < 0 ? -v : v); }

}  // namespace

std::optional<cudf::data_type> widened_decimal_sum_type(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::DECIMAL32: return cudf::data_type(cudf::type_id::DECIMAL64, type.scale());
    case cudf::type_id::DECIMAL64: return cudf::data_type(cudf::type_id::DECIMAL128, type.scale());
    default: return std::nullopt;
  }
}

std::unordered_set<int> decimal_sums_needing_widening(cudf::table_view const& table,
                                                      std::vector<int> const& candidates,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  for (int col_id : candidates) {
    if (!widened_decimal_sum_type(table.column(col_id).type())) {
      throw sirius::internal_exception(
        "decimal_sums_needing_widening: column {} is not DECIMAL32 or DECIMAL64", col_id);
    }
  }
  std::unordered_set<int> widen;
  auto const num_rows = static_cast<__uint128_t>(table.num_rows());
  if (num_rows == 0 || candidates.empty()) { return widen; }

  // Per candidate, one 3 x int64 slot {min, max, is_valid}. The minmax scalars are copied into
  // the slots on the stream, so the host waits for the device and copies back exactly once.
  constexpr size_t slot_words = 3;
  size_t const slot_bytes     = slot_words * sizeof(int64_t);
  rmm::device_buffer slots(candidates.size() * slot_bytes, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(slots.data(), 0, slots.size(), stream.get()));

  auto copy_to_slot = [&](size_t slot, size_t word, void const* src, size_t bytes) {
    auto* dst = static_cast<char*>(slots.data()) + slot * slot_bytes + word * sizeof(int64_t);
    CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream.get()));
  };
  // The scalars own the device memory read by the copies, so they live until the host copy ends.
  std::vector<std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>>> extremes;
  std::vector<cudf::type_id> rep_ids;
  extremes.reserve(candidates.size());
  rep_ids.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    auto [lo, hi]     = cudf::minmax(table.column(candidates[i]), stream, mr);
    auto const lo_rep = scalar_rep_of(*lo);
    auto const hi_rep = scalar_rep_of(*hi);
    copy_to_slot(i, 0, lo_rep.data, lo_rep.bytes);
    copy_to_slot(i, 1, hi_rep.data, hi_rep.bytes);
    copy_to_slot(i, 2, lo->validity_data(), sizeof(bool));
    rep_ids.push_back(lo_rep.id);
    extremes.emplace_back(std::move(lo), std::move(hi));
  }

  pinned_host_buffer host(slots.size(), stream);
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(host.data(), slots.data(), slots.size(), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();

  for (size_t i = 0; i < candidates.size(); ++i) {
    auto const type_id = rep_ids[i];
    auto const* slot   = host.data() + i * slot_words;
    bool is_valid      = false;
    std::memcpy(&is_valid, slot + 2, sizeof(bool));
    if (!is_valid) { continue; }  // no valid value: nothing to overflow
    // Widen to 128 bits before taking magnitudes (see magnitude()).
    auto const read = [&](int64_t const* word) -> __int128_t {
      if (type_id == cudf::type_id::DECIMAL32) {
        int32_t v;
        std::memcpy(&v, word, sizeof(v));
        return v;
      }
      int64_t v;
      std::memcpy(&v, word, sizeof(v));
      return v;
    };
    auto const max_abs = std::max(magnitude(read(slot)), magnitude(read(slot + 1)));
    auto const limit   = type_id == cudf::type_id::DECIMAL32
                           ? static_cast<__uint128_t>(std::numeric_limits<int32_t>::max())
                           : static_cast<__uint128_t>(std::numeric_limits<int64_t>::max());
    if (num_rows * max_abs > limit) { widen.insert(candidates[i]); }
  }
  return widen;
}

}  // namespace op
}  // namespace sirius
