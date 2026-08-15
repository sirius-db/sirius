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

#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <op/dynamic_filter/top_n_group_key_producer.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace sirius::op {

namespace {

/// One key column's host image of the leading rows of the sorted distinct-key table: the raw
/// element bytes and, for a nullable column, the validity words covering those rows.
struct host_key_column {
  std::vector<std::byte> elements;           ///< Row-major element bytes of the leading rows
  std::vector<cudf::bitmask_type> validity;  ///< Empty when the column cannot hold nulls
  cudf::size_type first_bit = 0;             ///< Bit index of row 0 within @c validity
  std::size_t element_width = 0;             ///< Bytes per element of the key's storage type
  bool extractable          = false;         ///< False: every row's component is recorded null
};

/// Physical width of an admitted ORDER BY key's storage type, or zero when it is not admitted.
std::size_t admitted_element_width(cudf::data_type type) noexcept
{
  switch (type.id()) {
    case cudf::type_id::INT8: return sizeof(std::int8_t);
    case cudf::type_id::INT16: return sizeof(std::int16_t);
    case cudf::type_id::INT32: return sizeof(std::int32_t);
    case cudf::type_id::INT64: return sizeof(std::int64_t);
    case cudf::type_id::TIMESTAMP_DAYS: return sizeof(std::int32_t);
    // Fixed-point keys are their scaled integer; `decimal128` is outside the allowlist.
    case cudf::type_id::DECIMAL32: return sizeof(std::int32_t);
    case cudf::type_id::DECIMAL64: return sizeof(std::int64_t);
    default: return 0;
  }
}

/// Reinterpret one element's bytes as the exact host scalar of @p type.
exact_host_scalar read_element(std::byte const* element, cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT8: {
      std::int8_t value{};
      std::memcpy(&value, element, sizeof(value));
      return exact_host_scalar{value, type};
    }
    case cudf::type_id::INT16: {
      std::int16_t value{};
      std::memcpy(&value, element, sizeof(value));
      return exact_host_scalar{value, type};
    }
    case cudf::type_id::INT64:
    case cudf::type_id::DECIMAL64: {
      std::int64_t value{};
      std::memcpy(&value, element, sizeof(value));
      return exact_host_scalar{value, type};
    }
    default: {
      // INT32, TIMESTAMP_DAYS (whose day count is an int32 -- the same representation the Top-N
      // row producer records for a DATE key), and DECIMAL32 (its scaled integer).
      // `admitted_element_width` reaches this function for no other type.
      std::int32_t value{};
      std::memcpy(&value, element, sizeof(value));
      return exact_host_scalar{value, type};
    }
  }
}

/// Whether row @p row of @p column is non-null, read from the host copy of its validity words.
bool host_row_is_valid(host_key_column const& column, cudf::size_type row) noexcept
{
  if (column.validity.empty()) { return true; }
  auto const bit  = column.first_bit + row;
  auto const word = column.validity[static_cast<std::size_t>(cudf::word_index(bit) -
                                                             cudf::word_index(column.first_bit))];
  return (word & (cudf::bitmask_type{1} << cudf::intra_word_index(bit))) != 0;
}

/**
 * @brief Issue the device-to-host copies for the leading @p rows of one sorted key column
 *
 * The copies are stream-ordered and not awaited here; the caller synchronizes once for every
 * column, which is the completion the witness handoff requires before the values are read.
 */
host_key_column stage_key_column(cudf::column_view const& column,
                                 cudf::data_type storage_type,
                                 cudf::size_type rows,
                                 rmm::cuda_stream_view stream)
{
  host_key_column staged;
  staged.element_width = admitted_element_width(storage_type);
  // An unadmitted tail type has no exact host representation; recording it as a null component is
  // what the planner's key admission documents. Uniformly disengaged components compare equal, so
  // such keys merge in the witness set -- which only ever loosens the boundary.
  // Full `data_type` equality, not just the id: a fixed-point column's scale is part of its
  // identity, and comparing scaled integers across different scales is silently wrong. For the
  // integer and date keys both sides carry scale 0, so this is the same test as before for them.
  if (staged.element_width == 0 || column.type() != storage_type) { return staged; }
  staged.extractable = true;

  staged.elements.resize(static_cast<std::size_t>(rows) * staged.element_width);
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(staged.elements.data(),
                    column.head<std::byte>() + static_cast<std::ptrdiff_t>(column.offset()) *
                                                 static_cast<std::ptrdiff_t>(staged.element_width),
                    staged.elements.size(),
                    cudaMemcpyDeviceToHost,
                    stream.value()));

  if (column.nullable()) {
    staged.first_bit      = column.offset();
    auto const first_word = cudf::word_index(staged.first_bit);
    auto const last_word  = cudf::word_index(staged.first_bit + rows - 1);
    staged.validity.resize(static_cast<std::size_t>(last_word - first_word) + 1);
    CUDF_CUDA_TRY(cudaMemcpyAsync(staged.validity.data(),
                                  column.null_mask() + first_word,
                                  staged.validity.size() * sizeof(cudf::bitmask_type),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
  }
  return staged;
}

/**
 * @brief The batch's K best distinct ORDER-BY key tuples, exact on the host
 *
 * Distinct first, then sort: the sort is bounded by the batch's distinct key count rather than its
 * row count. `cudf::sort` applies the same direction and null placement the coordinator compares
 * with, so "best" here is the comparison the boundary itself uses.
 *
 * The `cudf::distinct` call is a **pruning optimization only**. The K-distinct proof rests solely
 * on the coordinator's union-by-value, which deduplicates again across every offer; dropping this
 * call would only make the witness set fill more slowly. It may be changed or removed without
 * revisiting the proof.
 */
std::vector<exact_host_key_tuple> best_distinct_keys(cudf::table_view const& batch,
                                                     std::span<cudf::size_type const> key_columns,
                                                     std::span<top_n_key_semantics const> keys,
                                                     std::size_t k,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  std::vector<cudf::column_view> key_views;
  key_views.reserve(key_columns.size());
  for (auto const column : key_columns) {
    key_views.push_back(batch.column(column));
  }
  cudf::table_view const key_table{key_views};

  std::vector<cudf::size_type> every_column(key_views.size());
  std::iota(every_column.begin(), every_column.end(), 0);
  auto const distinct_keys = cudf::distinct(key_table,
                                            every_column,
                                            cudf::duplicate_keep_option::KEEP_ANY,
                                            cudf::null_equality::EQUAL,
                                            cudf::nan_equality::ALL_EQUAL,
                                            stream,
                                            mr);

  std::vector<cudf::order> orders;
  std::vector<cudf::null_order> null_orders;
  orders.reserve(keys.size());
  null_orders.reserve(keys.size());
  for (auto const& key : keys) {
    orders.push_back(key.order);
    null_orders.push_back(key.null_order);
  }
  auto const sorted = cudf::sort(distinct_keys->view(), orders, null_orders, stream, mr);

  auto const rows = static_cast<cudf::size_type>(
    std::min<std::size_t>(k, static_cast<std::size_t>(sorted->num_rows())));
  if (rows <= 0) { return {}; }

  std::vector<host_key_column> staged;
  staged.reserve(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    staged.push_back(stage_key_column(
      sorted->view().column(static_cast<cudf::size_type>(i)), keys[i].storage_type, rows, stream));
  }
  stream.synchronize();  // observed completion: every host value below is final

  std::vector<exact_host_key_tuple> tuples;
  tuples.reserve(static_cast<std::size_t>(rows));
  for (cudf::size_type row = 0; row < rows; ++row) {
    std::vector<std::optional<exact_host_scalar>> components;
    components.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
      auto const& column = staged[i];
      if (!column.extractable || !host_row_is_valid(column, row)) {
        components.emplace_back();
        continue;
      }
      components.push_back(
        read_element(column.elements.data() + static_cast<std::size_t>(row) * column.element_width,
                     keys[i].storage_type));
    }
    tuples.emplace_back(std::move(components));
  }
  return tuples;
}

}  // namespace

top_n_group_key_producer::top_n_group_key_producer(
  std::shared_ptr<top_n_threshold_coordinator> coordinator,
  std::vector<cudf::size_type> key_columns,
  double gate_keep_threshold)
  : _coordinator(std::move(coordinator)),
    _key_columns(std::move(key_columns)),
    _prefilter_gate(gate_keep_threshold)
{
}

detail::boundary_filter_result top_n_group_key_producer::prefilter(
  cudf::table_view const& batch, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (batch.num_rows() == 0) { return {}; }
  // Decline paths return the all-pass form {nullptr, batch rows}, honoring
  // boundary_filter_result's "rows_kept always valid" invariant: a caller computing a keep ratio
  // from a declined batch must see 1.0, never 0.
  auto const observed_updates = _coordinator->boundary_update_count();
  if (!_prefilter_gate.applicable(observed_updates)) {
    return {.filtered = nullptr, .rows_kept = batch.num_rows()};
  }
  auto const boundary = _coordinator->tightest_boundary();
  if (!boundary) { return {.filtered = nullptr, .rows_kept = batch.num_rows()}; }

  // Inclusive at every key count -- see the class documentation. The degraded form compares only
  // key zero when a tail type is outside the allowlist, exactly as the row producer's prefilter
  // does.
  auto const params = detail::make_boundary_filter_params(
    *boundary,
    _coordinator->keys(),
    _coordinator->lex_admitted() ? _coordinator->keys().size() : std::size_t{1},
    /*strict=*/false);
  auto result = detail::apply_boundary_filter(batch, _key_columns, params, stream, mr);

  auto const rows_in  = static_cast<std::size_t>(batch.num_rows());
  auto const rows_out = static_cast<std::size_t>(result.rows_kept);
  _coordinator->record_prefilter(rows_in, rows_out);
  _prefilter_gate.record_keep_ratio(rows_in, rows_out, observed_updates);
  if (!_prefilter_gate.applicable(observed_updates)) { _coordinator->record_prefilter_disabled(); }
  return result;
}

void top_n_group_key_producer::witness(cudf::table_view const& batch,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  if (batch.num_rows() == 0) { return; }
  auto keys =
    best_distinct_keys(batch, _key_columns, _coordinator->keys(), _coordinator->k(), stream, mr);
  if (keys.empty()) { return; }
  // The offer's result needs no handling here: accumulation, publication, and rejection are the
  // coordinator's business and are recorded in its counters.
  (void)_coordinator->offer(top_n_distinct_key_witness{.best_keys = std::move(keys)});
}

}  // namespace sirius::op
