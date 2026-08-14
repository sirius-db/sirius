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

/**
 * @file test_sirius_dynamic_range_filters.cpp
 * @brief Class-level tests for the Top-N boundary filters: the RANGE parameter mapping, reader
 *        AST lowering against the design doc's derivation table, compaction parity against an
 *        independent host reference, all-or-nothing replication, capability dispatch, and
 *        immutability.
 *
 * Both filters are checked against host references rather than against each other, and those
 * references restate every ordering rule -- direction, strictness, and null placement -- instead
 * of calling the production helpers, so a shared sign or direction error cannot hide. The one
 * derivation production owns alone, `detail::nulls_first_in_output`, is additionally pinned
 * against the oracle's independent restatement. The AST path is validated by evaluating the
 * lowered tree with `cudf::compute_column`; production never evaluates it that way (it belongs to
 * the parquet reader checkpoint), but evaluation is the robust oracle for shape plus semantics.
 */

#include "operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/dynamic_filter/top_n_boundary_filter.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

using sirius::op::dynamic_filter_null_policy;
using sirius::op::exact_host_key_tuple;
using sirius::op::exact_host_scalar;
using sirius::op::lex_component_semantics;
using sirius::op::range_bound_side;
using sirius::op::sirius_ast_lowerable;
using sirius::op::sirius_compaction_applicable;
using sirius::op::sirius_device_replicable;
using sirius::op::sirius_dynamic_filter;
using sirius::op::sirius_dynamic_filter_kind;
using sirius::op::sirius_dynamic_lex_range_filter;
using sirius::op::sirius_dynamic_range_filter;
using sirius::op::sirius_multi_column_ast_lowerable;
using sirius::op::top_n_key_semantics;

namespace {

using host_value  = std::optional<std::int64_t>;
using host_column = std::vector<host_value>;

constexpr auto k_int32 = cudf::data_type{cudf::type_id::INT32};
constexpr auto k_int64 = cudf::data_type{cudf::type_id::INT64};

exact_host_scalar int32_bound(std::int32_t v) { return exact_host_scalar{v, k_int32}; }

/// A boundary holding @p rep at @p type, with the variant alternative matching the type's width.
/// For a decimal @p type the rep at the type's scale *is* its scaled integer.
exact_host_scalar typed_bound(std::int64_t rep, cudf::data_type type)
{
  if (type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::DECIMAL32) {
    return exact_host_scalar{static_cast<std::int32_t>(rep), type};
  }
  return exact_host_scalar{rep, type};
}

top_n_key_semantics typed_key(cudf::data_type type, cudf::order order, cudf::null_order null_order)
{
  return {.storage_type = type, .order = order, .null_order = null_order};
}

top_n_key_semantics int32_key(cudf::order order, cudf::null_order null_order)
{
  return typed_key(k_int32, order, null_order);
}

/// One nullable column of @p type (INT32, INT64, DECIMAL32, or DECIMAL64); `std::nullopt` rows
/// are null. Decimal reps come straight from the host ints -- a fixed-point value at the column's
/// scale *is* its scaled integer -- so one host representation serves every admitted width.
std::unique_ptr<cudf::column> make_column(host_column const& values,
                                          cudf::data_type type,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const size = static_cast<cudf::size_type>(values.size());
  auto null_mask  = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  auto* mask_ptr  = static_cast<cudf::bitmask_type*>(null_mask.data());
  cudf::size_type null_count = 0;
  for (cudf::size_type i = 0; i < size; ++i) {
    if (!values[static_cast<std::size_t>(i)]) {
      cudf::set_null_mask(mask_ptr, i, i + 1, false, stream);
      ++null_count;
    }
  }
  auto const fixed_point =
    type.id() == cudf::type_id::DECIMAL32 || type.id() == cudf::type_id::DECIMAL64;
  auto column =
    fixed_point
      ? cudf::make_fixed_point_column(type, size, std::move(null_mask), null_count, stream, mr)
      : cudf::make_numeric_column(type, size, std::move(null_mask), null_count, stream, mr);
  auto const copy_reps = [&](auto rep_tag) {
    using rep_type = decltype(rep_tag);
    std::vector<rep_type> host(values.size(), 0);
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (values[i]) { host[i] = static_cast<rep_type>(*values[i]); }
    }
    if (!host.empty()) {
      cudaMemcpy(column->mutable_view().head<rep_type>(),
                 host.data(),
                 host.size() * sizeof(rep_type),
                 cudaMemcpyHostToDevice);
    }
  };
  if (type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::DECIMAL32) {
    copy_reps(std::int32_t{});
  } else {
    copy_reps(std::int64_t{});
  }
  return column;
}

/// One nullable INT32 column; `std::nullopt` rows are null.
std::unique_ptr<cudf::column> make_int32_column(host_column const& values,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  return make_column(values, k_int32, stream, mr);
}

/// Evaluate a lowered fragment and report which rows survive `apply_boolean_mask` semantics: a
/// row is kept only when the predicate is valid and true.
std::vector<bool> evaluate_keep(cudf::table_view const& table,
                                cudf::ast::expression const& root,
                                rmm::cuda_stream_view stream)
{
  auto mask = cudf::compute_column(table, root, stream, cudf::get_current_device_resource_ref());
  REQUIRE(mask->type().id() == cudf::type_id::BOOL8);
  auto const size = static_cast<std::size_t>(mask->size());
  std::vector<std::uint8_t> values(size);
  if (size > 0) {
    cudaMemcpyAsync(values.data(),
                    mask->view().data<bool>(),
                    size * sizeof(bool),
                    cudaMemcpyDeviceToHost,
                    stream.value());
  }
  std::vector<cudf::bitmask_type> validity;
  auto const nullable = mask->nullable() && mask->null_count() > 0;
  if (nullable) {
    validity.resize(cudf::num_bitmask_words(mask->size()));
    cudaMemcpyAsync(validity.data(),
                    mask->view().null_mask(),
                    validity.size() * sizeof(cudf::bitmask_type),
                    cudaMemcpyDeviceToHost,
                    stream.value());
  }
  stream.synchronize();

  std::vector<bool> keep(size);
  for (std::size_t i = 0; i < size; ++i) {
    bool const valid = !nullable || ((validity[i / 32] >> (i % 32)) & 1U) != 0;
    keep[i]          = valid && values[i] != 0;
  }
  return keep;
}

/// Host reference for the SQL meaning of a one-sided RANGE predicate.
bool range_keeps(host_value value,
                 std::int64_t bound,
                 range_bound_side side,
                 bool inclusive,
                 dynamic_filter_null_policy null_policy)
{
  if (!value) { return null_policy == dynamic_filter_null_policy::ADMIT; }
  if (side == range_bound_side::LOWER) { return inclusive ? *value >= bound : *value > bound; }
  return inclusive ? *value <= bound : *value < bound;
}

/// Independent restatement of "this key's nulls sort before every value in the output order".
/// Deliberately not `detail::nulls_first_in_output`: sharing that helper with the implementation
/// would let an inverted derivation flip the oracle and the code together and stay green.
bool oracle_nulls_first(top_n_key_semantics const& key)
{
  // cuDF places nulls per null_order in the ascending frame and reverses the whole verdict for a
  // descending key, so BEFORE means "nulls first" only while ascending.
  if (key.order == cudf::order::ASCENDING) { return key.null_order == cudf::null_order::BEFORE; }
  return key.null_order == cudf::null_order::AFTER;
}

/// Host reference for lexicographic boundary semantics: walk components, better keeps, worse
/// drops, ties fall through; a full-tuple tie survives only the inclusive form. Null placement
/// follows each key's output order.
bool lex_keeps(std::vector<host_column> const& columns,
               exact_host_key_tuple const& boundary,
               std::vector<lex_component_semantics> const& components,
               bool inclusive,
               std::size_t row)
{
  for (std::size_t i = 0; i < components.size(); ++i) {
    auto const& value      = columns[i][row];
    auto const& bound      = boundary.component(i);
    auto const nulls_first = oracle_nulls_first(components[i].key);
    if (!value || !bound) {
      if (!value && !bound) { continue; }
      return !value.has_value() == nulls_first;
    }
    auto const bound_value =
      std::visit([](auto v) { return static_cast<std::int64_t>(v); }, bound->value());
    if (*value == bound_value) { continue; }
    bool const less = *value < bound_value;
    return components[i].key.order == cudf::order::DESCENDING ? !less : less;
  }
  return inclusive;
}

/// @param key_types Per-key-column storage types; empty means all-INT32. The row-id witness
/// column appended below stays INT32 regardless -- only key columns matter to the kernel.
std::vector<bool> compaction_survivors(sirius_compaction_applicable const& filter,
                                       std::vector<host_column> const& columns,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr,
                                       std::vector<cudf::data_type> const& key_types = {})
{
  REQUIRE((key_types.empty() || key_types.size() == columns.size()));
  std::vector<std::unique_ptr<cudf::column>> owned;
  std::vector<cudf::column_view> views;
  std::vector<cudf::size_type> key_columns;
  for (std::size_t i = 0; i < columns.size(); ++i) {
    owned.push_back(
      make_column(columns[i], key_types.empty() ? k_int32 : key_types[i], stream, mr));
    views.push_back(owned.back()->view());
    key_columns.push_back(static_cast<cudf::size_type>(i));
  }
  auto const rows = static_cast<cudf::size_type>(columns.empty() ? 0 : columns[0].size());
  host_column row_ids(static_cast<std::size_t>(rows));
  for (cudf::size_type i = 0; i < rows; ++i) {
    row_ids[static_cast<std::size_t>(i)] = i;
  }
  owned.push_back(make_int32_column(row_ids, stream, mr));
  views.push_back(owned.back()->view());

  auto const batch  = cudf::table_view{views};
  auto const result = filter.apply_compact(batch, key_columns, -1, stream, mr);

  std::vector<bool> keep(static_cast<std::size_t>(rows), false);
  if (result.rows_kept == rows) {
    REQUIRE(result.filtered == nullptr);  // all-pass fast path
    std::fill(keep.begin(), keep.end(), true);
    return keep;
  }
  REQUIRE(result.filtered != nullptr);
  auto const survivors = sirius::test::operator_utils::copy_column_to_host<std::int32_t>(
    result.filtered->view().column(batch.num_columns() - 1));
  REQUIRE(survivors.size() == static_cast<std::size_t>(result.rows_kept));
  for (auto const row : survivors) {
    keep[static_cast<std::size_t>(row)] = true;
  }
  return keep;
}

bool require_two_gpus()
{
  int count      = 0;
  auto const err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count < 2) {
    WARN("Top-N boundary-filter replication test requires at least two visible GPUs; skipping");
    return false;
  }
  return true;
}

template <typename MemoryManager>
std::vector<sirius::op::dynamic_filter_replica_space> get_replica_spaces(
  MemoryManager& memory_manager)
{
  auto const gpu_spaces  = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto const host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(gpu_spaces.size() == 2);
  REQUIRE_FALSE(host_spaces.empty());

  std::vector<sirius::op::dynamic_filter_replica_space> result;
  result.reserve(gpu_spaces.size());
  for (auto const* gpu_space_view : gpu_spaces) {
    auto* gpu_space = memory_manager.get_memory_space(cucascade::memory::Tier::GPU,
                                                      gpu_space_view->get_device_id());
    REQUIRE(gpu_space != nullptr);
    auto const local_host =
      std::find_if(host_spaces.begin(), host_spaces.end(), [gpu_space](auto const* host_space) {
        return host_space->get_device_id() == gpu_space->get_device_id();
      });
    auto const* host_space = local_host == host_spaces.end() ? host_spaces.front() : *local_host;
    result.emplace_back(*gpu_space, *host_space);
  }
  std::sort(result.begin(), result.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.get_gpu_space().get_device_id() < rhs.get_gpu_space().get_device_id();
  });
  return result;
}

}  // namespace

//===----------------------------------------------------------------------===//
// R5: the RANGE -> launch-parameter mapping, checked before anything builds on it
//===----------------------------------------------------------------------===//

TEST_CASE("the output-order null-placement derivation is what the oracle independently states",
          "[dynamic_filter][top_n]")
{
  // Guards the one derivation the oracle would otherwise have to share with production: if
  // nulls_first_in_output were "simplified" to `null_order == BEFORE`, descending keys would
  // silently mis-place nulls in both reader pruning and compaction. Asserting the production
  // helper against the independent restatement makes that failure loud here.
  for (auto const order : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    for (auto const null_order : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
      auto const key = int32_key(order, null_order);
      CAPTURE(order == cudf::order::DESCENDING, null_order == cudf::null_order::BEFORE);
      REQUIRE(sirius::op::detail::nulls_first_in_output(key) == oracle_nulls_first(key));
    }
  }
  // And the derivation is not the identity on null_order: descending inverts it.
  REQUIRE(oracle_nulls_first(int32_key(cudf::order::ASCENDING, cudf::null_order::BEFORE)));
  REQUIRE_FALSE(oracle_nulls_first(int32_key(cudf::order::DESCENDING, cudf::null_order::BEFORE)));
}

TEST_CASE("RANGE parameter mapping matches the kernel's keep semantics", "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  auto const side      = GENERATE(range_bound_side::LOWER, range_bound_side::UPPER);
  bool const inclusive = GENERATE(false, true);
  auto const policy =
    GENERATE(dynamic_filter_null_policy::ADMIT, dynamic_filter_null_policy::REJECT);
  CAPTURE(side == range_bound_side::LOWER, inclusive, policy == dynamic_filter_null_policy::ADMIT);

  constexpr std::int32_t k_bound = 40;
  host_column const values{39, 40, 41, std::nullopt, 0, 100};

  // The mapping's own claims, spelled out independently of the kernel.
  auto const params = sirius_dynamic_range_filter::make_boundary_filter_params(
    int32_bound(k_bound), side, inclusive, policy);
  REQUIRE(params.count == 1);
  REQUIRE(params.strict == !inclusive);
  REQUIRE(params.components[0].engaged);
  REQUIRE(params.components[0].value == k_bound);
  REQUIRE(params.components[0].width == 4);
  // A LOWER bound keeps rows above it, so "better" must mean "larger" -- the descending frame.
  REQUIRE(params.components[0].descending == (side == range_bound_side::LOWER));
  // ADMIT keeps null probes, so nulls must order better than every value.
  REQUIRE(params.components[0].nulls_first == (policy == dynamic_filter_null_policy::ADMIT));

  // And the end-to-end verdict through the kernel matches the SQL predicate row for row.
  sirius_dynamic_range_filter const filter{int32_bound(k_bound), side, inclusive, policy};
  auto const actual = compaction_survivors(filter, {values}, stream, mr);
  std::vector<bool> expected(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    expected[i] = range_keeps(values[i], k_bound, side, inclusive, policy);
  }
  REQUIRE(actual == expected);
}

//===----------------------------------------------------------------------===//
// Reader AST lowering vs the derivation table
//===----------------------------------------------------------------------===//

TEST_CASE("RANGE lowers to the documented comparison under both null policies",
          "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  auto const side      = GENERATE(range_bound_side::LOWER, range_bound_side::UPPER);
  bool const inclusive = GENERATE(false, true);
  auto const policy =
    GENERATE(dynamic_filter_null_policy::ADMIT, dynamic_filter_null_policy::REJECT);
  CAPTURE(side == range_bound_side::LOWER, inclusive, policy == dynamic_filter_null_policy::ADMIT);

  constexpr std::int32_t k_bound = 40;
  host_column const values{39, 40, 41, std::nullopt, 0, 100};
  auto column      = make_int32_column(values, stream, mr);
  auto const table = cudf::table_view{{column->view()}};

  sirius_dynamic_range_filter const filter{int32_bound(k_bound), side, inclusive, policy};
  cudf::ast::tree tree;
  auto const& column_ref = tree.emplace<cudf::ast::column_reference>(0);
  auto const& root       = filter.to_ast(tree, column_ref);

  auto const actual = evaluate_keep(table, root, stream);
  std::vector<bool> expected(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    expected[i] = range_keeps(values[i], k_bound, side, inclusive, policy);
  }
  REQUIRE(actual == expected);
}

TEST_CASE("RANGE decimal null policies match the SQL reference", "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  auto const type      = GENERATE(cudf::data_type{cudf::type_id::DECIMAL32, -2},
                             cudf::data_type{cudf::type_id::DECIMAL64, -3});
  auto const side      = GENERATE(range_bound_side::LOWER, range_bound_side::UPPER);
  bool const inclusive = GENERATE(false, true);
  auto const policy =
    GENERATE(dynamic_filter_null_policy::ADMIT, dynamic_filter_null_policy::REJECT);
  CAPTURE(type.id() == cudf::type_id::DECIMAL64,
          side == range_bound_side::LOWER,
          inclusive,
          policy == dynamic_filter_null_policy::ADMIT);

  // At equal storage type (which apply_compact now enforces) decimal reps compare exactly as
  // their scaled integers, so the INT32 oracle serves unchanged: the reference takes reps.
  constexpr std::int64_t k_bound_rep = 4000;
  host_column const reps{3999, 4000, 4001, std::nullopt, 0, 10000};

  sirius_dynamic_range_filter const filter{typed_bound(k_bound_rep, type), side, inclusive, policy};
  std::vector<bool> expected(reps.size());
  for (std::size_t i = 0; i < reps.size(); ++i) {
    expected[i] = range_keeps(reps[i], k_bound_rep, side, inclusive, policy);
  }

  // The fused compaction pass, null policy included (ADMIT keeps null reps, REJECT drops them).
  REQUIRE(compaction_survivors(filter, {reps}, stream, mr, {type}) == expected);

  // The reader AST path over the same rows: the ADMIT `IS_NULL OR cmp` wrap and the REJECT bare
  // comparison's null-verdict drop, now with a fixed-point literal.
  auto column      = make_column(reps, type, stream, mr);
  auto const table = cudf::table_view{{column->view()}};
  cudf::ast::tree tree;
  auto const& column_ref = tree.emplace<cudf::ast::column_reference>(0);
  auto const& root       = filter.to_ast(tree, column_ref);
  REQUIRE(evaluate_keep(table, root, stream) == expected);
}

TEST_CASE("LEX_RANGE lowering matches the derivation table across directions and null orders",
          "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  auto const head_order         = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  auto const tail_order         = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  auto const tail_nulls         = GENERATE(cudf::null_order::BEFORE, cudf::null_order::AFTER);
  bool const inclusive          = GENERATE(false, true);
  bool const null_tail_boundary = GENERATE(false, true);
  CAPTURE(head_order == cudf::order::DESCENDING,
          tail_order == cudf::order::DESCENDING,
          tail_nulls == cudf::null_order::BEFORE,
          inclusive,
          null_tail_boundary);

  std::vector<lex_component_semantics> const components{
    {.consumer_ordinal = 0, .key = int32_key(head_order, cudf::null_order::AFTER)},
    {.consumer_ordinal = 1, .key = int32_key(tail_order, tail_nulls)}};
  std::vector<std::optional<exact_host_scalar>> boundary_components{
    int32_bound(5), null_tail_boundary ? std::optional<exact_host_scalar>{} : int32_bound(7)};
  exact_host_key_tuple const boundary{boundary_components};

  // Rows spanning both sides of the head, head ties decided by the tail, and nulls on both keys.
  std::vector<host_column> const columns{{4, 6, 5, 5, 5, 5, std::nullopt, 5},
                                         {9, 0, 6, 7, 8, std::nullopt, 3, std::nullopt}};

  sirius_dynamic_lex_range_filter const filter{boundary, components, inclusive};

  std::vector<std::unique_ptr<cudf::column>> owned;
  std::vector<cudf::column_view> views;
  for (auto const& column : columns) {
    owned.push_back(make_int32_column(column, stream, mr));
    views.push_back(owned.back()->view());
  }
  auto const table = cudf::table_view{views};

  cudf::ast::tree tree;
  auto const resolver = [&tree](std::size_t ordinal) -> cudf::ast::expression const& {
    return tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(ordinal));
  };
  auto const& root  = filter.to_ast(tree, resolver);
  auto const actual = evaluate_keep(table, root, stream);

  std::vector<bool> expected(columns[0].size());
  for (std::size_t row = 0; row < columns[0].size(); ++row) {
    expected[row] = lex_keeps(columns, boundary, components, inclusive, row);
  }
  REQUIRE(actual == expected);

  // The same filter's compaction path must agree with its own AST lowering, row for row.
  REQUIRE(compaction_survivors(filter, columns, stream, mr) == expected);
}

TEST_CASE("LEX decimal null derivations match the reference", "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  // Mixed widths are the sharper marshalling case: a DECIMAL32 head beside a DECIMAL64 tail means
  // a per-component width error cannot cancel out across the tuple.
  auto const k_dec32_s2 = cudf::data_type{cudf::type_id::DECIMAL32, -2};
  auto const k_dec64_s2 = cudf::data_type{cudf::type_id::DECIMAL64, -2};

  auto const head_order         = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  auto const tail_order         = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  auto const tail_nulls         = GENERATE(cudf::null_order::BEFORE, cudf::null_order::AFTER);
  bool const inclusive          = GENERATE(false, true);
  bool const null_tail_boundary = GENERATE(false, true);
  CAPTURE(head_order == cudf::order::DESCENDING,
          tail_order == cudf::order::DESCENDING,
          tail_nulls == cudf::null_order::BEFORE,
          inclusive,
          null_tail_boundary);

  std::vector<lex_component_semantics> const components{
    {.consumer_ordinal = 0, .key = typed_key(k_dec32_s2, head_order, cudf::null_order::AFTER)},
    {.consumer_ordinal = 1, .key = typed_key(k_dec64_s2, tail_order, tail_nulls)}};
  // The null tail boundary variant runs the disengaged-component terms (`IS NULL` /
  // `NOT IS_NULL` / dropped disjunct) on decimal columns.
  std::vector<std::optional<exact_host_scalar>> boundary_components{
    typed_bound(5, k_dec32_s2),
    null_tail_boundary ? std::optional<exact_host_scalar>{} : typed_bound(7, k_dec64_s2)};
  exact_host_key_tuple const boundary{boundary_components};

  // Reps spanning both sides of the head, head ties decided by the tail, and nulls on both keys.
  std::vector<host_column> const columns{{4, 6, 5, 5, 5, 5, std::nullopt, 5},
                                         {9, 0, 6, 7, 8, std::nullopt, 3, std::nullopt}};
  std::vector<cudf::data_type> const key_types{k_dec32_s2, k_dec64_s2};

  sirius_dynamic_lex_range_filter const filter{boundary, components, inclusive};

  std::vector<bool> expected(columns[0].size());
  for (std::size_t row = 0; row < columns[0].size(); ++row) {
    expected[row] = lex_keeps(columns, boundary, components, inclusive, row);
  }

  // Both production paths against the one reference: the reader AST lowering...
  std::vector<std::unique_ptr<cudf::column>> owned;
  std::vector<cudf::column_view> views;
  for (std::size_t i = 0; i < columns.size(); ++i) {
    owned.push_back(make_column(columns[i], key_types[i], stream, mr));
    views.push_back(owned.back()->view());
  }
  cudf::ast::tree tree;
  auto const resolver = [&tree](std::size_t ordinal) -> cudf::ast::expression const& {
    return tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(ordinal));
  };
  auto const& root = filter.to_ast(tree, resolver);
  REQUIRE(evaluate_keep(cudf::table_view{views}, root, stream) == expected);

  // ...and the fused compaction pass.
  REQUIRE(compaction_survivors(filter, columns, stream, mr, key_types) == expected);
}

//===----------------------------------------------------------------------===//
// Compaction parity and edge shapes
//===----------------------------------------------------------------------===//

TEST_CASE("boundary filter compaction handles empty, all-pass, and all-fail batches",
          "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  sirius_dynamic_range_filter const filter{int32_bound(50),
                                           range_bound_side::UPPER,
                                           /*inclusive=*/true,
                                           dynamic_filter_null_policy::REJECT};

  SECTION("empty batch")
  {
    auto const keep = compaction_survivors(filter, {host_column{}}, stream, mr);
    REQUIRE(keep.empty());
  }
  SECTION("all rows pass")
  {
    auto const keep = compaction_survivors(filter, {host_column{1, 2, 3}}, stream, mr);
    REQUIRE(keep == std::vector<bool>{true, true, true});
  }
  SECTION("no row passes")
  {
    auto const keep = compaction_survivors(filter, {host_column{60, 70, 80}}, stream, mr);
    REQUIRE(keep == std::vector<bool>{false, false, false});
  }
}

TEST_CASE("a single-component LEX boundary agrees with the equivalent RANGE",
          "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  // LEX requires two components, so pin the tail to a constant that never decides: the head's
  // strict ascending verdict must equal an UPPER strict RANGE on the same bound.
  std::vector<lex_component_semantics> const components{
    {.consumer_ordinal = 0, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)},
    {.consumer_ordinal = 1, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)}};
  std::vector<std::optional<exact_host_scalar>> boundary_components{int32_bound(40),
                                                                    int32_bound(0)};
  std::vector<host_column> const columns{{39, 40, 41, 0, 100}, {5, 5, 5, 5, 5}};

  sirius_dynamic_lex_range_filter const lex{exact_host_key_tuple{boundary_components},
                                            components,
                                            /*inclusive=*/false};
  sirius_dynamic_range_filter const range{int32_bound(40),
                                          range_bound_side::UPPER,
                                          /*inclusive=*/false,
                                          dynamic_filter_null_policy::REJECT};

  auto const lex_keep   = compaction_survivors(lex, columns, stream, mr);
  auto const range_keep = compaction_survivors(range, {columns[0]}, stream, mr);
  REQUIRE(lex_keep == range_keep);
  REQUIRE(lex_keep == std::vector<bool>{true, false, false, true, false});
}

TEST_CASE("a mismatched consumer column type makes compaction all-pass", "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  // The consumer leg of the type contract: the kernel's widths are the producer's, so a batch
  // whose key columns differ in width or scale must pass through unfiltered rather than be
  // compared. Each section chooses values a missing guard would deterministically drop, so this
  // direct test is itself the guard's trigger.
  auto const require_all_pass = [&](sirius_compaction_applicable const& filter,
                                    std::vector<host_column> const& columns,
                                    std::vector<cudf::data_type> const& types) {
    std::vector<std::unique_ptr<cudf::column>> owned;
    std::vector<cudf::column_view> views;
    std::vector<cudf::size_type> key_columns;
    for (std::size_t i = 0; i < columns.size(); ++i) {
      owned.push_back(make_column(columns[i], types[i], stream, mr));
      views.push_back(owned.back()->view());
      key_columns.push_back(static_cast<cudf::size_type>(i));
    }
    auto const batch  = cudf::table_view{views};
    auto const result = filter.apply_compact(batch, key_columns, -1, stream, mr);
    REQUIRE(result.filtered == nullptr);
    REQUIRE(result.rows_kept == batch.num_rows());
  };

  SECTION("RANGE: an INT64 column against an INT32 boundary")
  {
    // All values sit strictly above the positive strict LOWER bound. Without the guard the
    // kernel reads the INT64 column at width 4: element 1's load lands on element 0's high half
    // (zero, little-endian), fails the bound, and drops a row -- so `filtered` going non-null is
    // the deterministic symptom.
    sirius_dynamic_range_filter const filter{int32_bound(10),
                                             range_bound_side::LOWER,
                                             /*inclusive=*/false,
                                             dynamic_filter_null_policy::REJECT};
    require_all_pass(filter, {host_column{50, 60, 70}}, {k_int64});
  }

  SECTION("LEX: a mismatched tail column poisons the whole batch")
  {
    // Head correct (INT32), tail INT64 against an INT32 component: one bad component degrades
    // the whole batch to all-pass, never partial application. Head values tie the boundary so
    // every verdict falls to the tail, whose true values all order worse than its bound -- a
    // missing guard therefore drops rows deterministically.
    std::vector<lex_component_semantics> const components{
      {.consumer_ordinal = 0, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)},
      {.consumer_ordinal = 1, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)}};
    std::vector<std::optional<exact_host_scalar>> boundary_components{int32_bound(5),
                                                                      int32_bound(10)};
    sirius_dynamic_lex_range_filter const filter{exact_host_key_tuple{boundary_components},
                                                 components,
                                                 /*inclusive=*/false};
    require_all_pass(filter, {host_column{5, 5, 5}, host_column{50, 60, 70}}, {k_int32, k_int64});
  }

  SECTION("DECIMAL32: ids agree, scales differ")
  {
    // The exact Stage-7 N1 case: same type id and width, different scale -- reachable because
    // cuDF derives a decimal's width from the parquet physical type. `cudf::data_type` equality
    // includes the scale, so plain `!=` covers it. The reps all order worse than the raw bound,
    // so a missing guard's raw comparison drops every row.
    auto const k_dec32_s2 = cudf::data_type{cudf::type_id::DECIMAL32, -2};
    auto const k_dec32_s4 = cudf::data_type{cudf::type_id::DECIMAL32, -4};
    sirius_dynamic_range_filter const filter{typed_bound(4000, k_dec32_s2),
                                             range_bound_side::LOWER,
                                             /*inclusive=*/false,
                                             dynamic_filter_null_policy::REJECT};
    require_all_pass(filter, {host_column{100, 200, 300}}, {k_dec32_s4});
  }

  SECTION("RANGE: an empty key-column set passes through")
  {
    // The merge site always supplies the channel column, so an empty set is reachable only
    // through the public apply_compact seam -- which is exactly where the contract must hold:
    // nothing to compare means nothing to drop. Without the short-circuit, front() on an empty
    // span has no defined behavior.
    sirius_dynamic_range_filter const filter{int32_bound(10),
                                             range_bound_side::LOWER,
                                             /*inclusive=*/false,
                                             dynamic_filter_null_policy::REJECT};
    auto const column = make_column(host_column{1, 2, 3}, k_int32, stream, mr);
    std::vector<cudf::column_view> const views{column->view()};
    auto const batch  = cudf::table_view{views};
    auto const result = filter.apply_compact(batch, {}, -1, stream, mr);
    REQUIRE(result.filtered == nullptr);
    REQUIRE(result.rows_kept == batch.num_rows());
  }

  SECTION("LEX: an arity mismatch passes through")
  {
    // Two components but one supplied key column of the correct type, so only the arity clause
    // can refuse. Without it the kernel would read a second key column that was never supplied.
    std::vector<lex_component_semantics> const components{
      {.consumer_ordinal = 0, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)},
      {.consumer_ordinal = 1, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)}};
    std::vector<std::optional<exact_host_scalar>> boundary_components{int32_bound(5),
                                                                      int32_bound(10)};
    sirius_dynamic_lex_range_filter const filter{exact_host_key_tuple{boundary_components},
                                                 components,
                                                 /*inclusive=*/false};
    auto const head = make_column(host_column{1, 2, 3}, k_int32, stream, mr);
    auto const tail = make_column(host_column{1, 2, 3}, k_int32, stream, mr);
    std::vector<cudf::column_view> const views{head->view(), tail->view()};
    auto const batch = cudf::table_view{views};
    std::vector<cudf::size_type> const key_columns{0};
    auto const result = filter.apply_compact(batch, key_columns, -1, stream, mr);
    REQUIRE(result.filtered == nullptr);
    REQUIRE(result.rows_kept == batch.num_rows());
  }
}

//===----------------------------------------------------------------------===//
// Construction contracts, capability dispatch, immutability
//===----------------------------------------------------------------------===//

TEST_CASE("boundary filters reject inadmissible construction", "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  REQUIRE(memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0));

  auto const admitted = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER);
  auto const unadmitted =
    top_n_key_semantics{.storage_type = cudf::data_type{cudf::type_id::FLOAT64},
                        .order        = cudf::order::ASCENDING,
                        .null_order   = cudf::null_order::AFTER};

  SECTION("RANGE rejects a type outside the allowlist")
  {
    REQUIRE_THROWS_AS(sirius_dynamic_range_filter(
                        exact_host_scalar{std::int64_t{1}, cudf::data_type{cudf::type_id::FLOAT64}},
                        range_bound_side::LOWER,
                        true,
                        dynamic_filter_null_policy::ADMIT),
                      std::invalid_argument);
  }
  SECTION("LEX requires at least two components")
  {
    std::vector<std::optional<exact_host_scalar>> single{int32_bound(1)};
    REQUIRE_THROWS_AS(sirius_dynamic_lex_range_filter(exact_host_key_tuple{single},
                                                      std::vector<lex_component_semantics>{
                                                        {.consumer_ordinal = 0, .key = admitted}},
                                                      false),
                      std::invalid_argument);
  }
  SECTION("LEX requires an engaged head component")
  {
    std::vector<std::optional<exact_host_scalar>> head_null{std::nullopt, int32_bound(2)};
    REQUIRE_THROWS_AS(sirius_dynamic_lex_range_filter(exact_host_key_tuple{head_null},
                                                      std::vector<lex_component_semantics>{
                                                        {.consumer_ordinal = 0, .key = admitted},
                                                        {.consumer_ordinal = 1, .key = admitted}},
                                                      false),
                      std::invalid_argument);
  }
  SECTION("LEX rejects a component type outside the allowlist")
  {
    std::vector<std::optional<exact_host_scalar>> pair{int32_bound(1), int32_bound(2)};
    REQUIRE_THROWS_AS(sirius_dynamic_lex_range_filter(exact_host_key_tuple{pair},
                                                      std::vector<lex_component_semantics>{
                                                        {.consumer_ordinal = 0, .key = admitted},
                                                        {.consumer_ordinal = 1, .key = unadmitted}},
                                                      false),
                      std::invalid_argument);
  }
}

TEST_CASE("boundary filters expose exactly their intended capabilities", "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  REQUIRE(memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0));

  std::vector<lex_component_semantics> const components{
    {.consumer_ordinal = 3, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)},
    {.consumer_ordinal = 7, .key = int32_key(cudf::order::DESCENDING, cudf::null_order::BEFORE)}};
  std::vector<std::optional<exact_host_scalar>> boundary_components{int32_bound(1), int32_bound(2)};

  sirius_dynamic_range_filter range{
    int32_bound(1), range_bound_side::LOWER, true, dynamic_filter_null_policy::ADMIT};
  sirius_dynamic_lex_range_filter lex{exact_host_key_tuple{boundary_components}, components, true};

  sirius_dynamic_filter* range_base = &range;
  sirius_dynamic_filter* lex_base   = &lex;

  REQUIRE(range_base->kind() == sirius_dynamic_filter_kind::RANGE);
  REQUIRE(lex_base->kind() == sirius_dynamic_filter_kind::LEX_RANGE);

  // RANGE is single-column AST-lowerable; LEX is multi-column. Neither acquires the other's.
  REQUIRE(dynamic_cast<sirius_ast_lowerable*>(range_base) != nullptr);
  REQUIRE(dynamic_cast<sirius_multi_column_ast_lowerable*>(range_base) == nullptr);
  REQUIRE(dynamic_cast<sirius_multi_column_ast_lowerable*>(lex_base) != nullptr);
  REQUIRE(dynamic_cast<sirius_ast_lowerable*>(lex_base) == nullptr);

  // Both compact on device and both replicate.
  REQUIRE(dynamic_cast<sirius_compaction_applicable*>(range_base) != nullptr);
  REQUIRE(dynamic_cast<sirius_compaction_applicable*>(lex_base) != nullptr);
  REQUIRE(dynamic_cast<sirius_device_replicable*>(range_base) != nullptr);
  REQUIRE(dynamic_cast<sirius_device_replicable*>(lex_base) != nullptr);

  // Neither is mask-applicable: consumers must dispatch on compaction, not compute_mask.
  REQUIRE(dynamic_cast<sirius::op::sirius_mask_applicable*>(range_base) == nullptr);
  REQUIRE(dynamic_cast<sirius::op::sirius_mask_applicable*>(lex_base) == nullptr);

  // The existing join filters must not acquire the new capability.
  std::vector<sirius::op::zone_map_entry> zones;
  zones.push_back(
    {std::make_unique<cudf::numeric_scalar<std::int32_t>>(
       0, true, cudf::get_default_stream(), cudf::get_current_device_resource_ref()),
     std::make_unique<cudf::numeric_scalar<std::int32_t>>(
       9, true, cudf::get_default_stream(), cudf::get_current_device_resource_ref())});
  sirius::op::sirius_dynamic_zone_map_filter zone_map{std::move(zones)};
  sirius_dynamic_filter* zone_base = &zone_map;
  REQUIRE(dynamic_cast<sirius_compaction_applicable*>(zone_base) == nullptr);
  REQUIRE(dynamic_cast<sirius_multi_column_ast_lowerable*>(zone_base) == nullptr);

  REQUIRE(lex.referenced_ordinals().size() == 2);
  REQUIRE(lex.referenced_ordinals()[0] == 3);  // primary first
  REQUIRE(lex.referenced_ordinals()[1] == 7);
}

TEST_CASE("boundary filter state is immutable across lowering and application",
          "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  sirius_dynamic_range_filter const range{int32_bound(42),
                                          range_bound_side::UPPER,
                                          /*inclusive=*/true,
                                          dynamic_filter_null_policy::ADMIT};
  REQUIRE(std::get<std::int32_t>(range.bound().value()) == 42);
  REQUIRE(range.side() == range_bound_side::UPPER);
  REQUIRE(range.inclusive());
  REQUIRE(range.null_policy() == dynamic_filter_null_policy::ADMIT);

  (void)compaction_survivors(range, {host_column{1, 50}}, stream, mr);
  {
    cudf::ast::tree tree;
    auto const& column_ref = tree.emplace<cudf::ast::column_reference>(0);
    (void)range.to_ast(tree, column_ref);
  }

  REQUIRE(std::get<std::int32_t>(range.bound().value()) == 42);
  REQUIRE(range.side() == range_bound_side::UPPER);
  REQUIRE(range.inclusive());
  REQUIRE(range.null_policy() == dynamic_filter_null_policy::ADMIT);

  std::vector<lex_component_semantics> const components{
    {.consumer_ordinal = 0, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)},
    {.consumer_ordinal = 1, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)}};
  std::vector<std::optional<exact_host_scalar>> boundary_components{int32_bound(4), std::nullopt};
  sirius_dynamic_lex_range_filter const lex{exact_host_key_tuple{boundary_components},
                                            components,
                                            /*inclusive=*/false};
  REQUIRE(lex.boundary().size() == 2);
  REQUIRE(std::get<std::int32_t>(lex.boundary().component(0)->value()) == 4);
  REQUIRE_FALSE(lex.boundary().component(1).has_value());
  REQUIRE_FALSE(lex.inclusive());

  (void)compaction_survivors(lex, {host_column{1, 9}, host_column{1, 1}}, stream, mr);
  REQUIRE(lex.boundary().size() == 2);
  REQUIRE(std::get<std::int32_t>(lex.boundary().component(0)->value()) == 4);
  REQUIRE_FALSE(lex.boundary().component(1).has_value());
  REQUIRE(lex.components().size() == 2);
  REQUIRE_FALSE(lex.inclusive());
}

//===----------------------------------------------------------------------===//
// Replication
//===----------------------------------------------------------------------===//

TEST_CASE("boundary filter replication is all-or-nothing under a denied target reservation",
          "[dynamic_filter][top_n][mgpu][replica][reservation]")
{
  if (!require_two_gpus()) { return; }

  auto memory_manager      = sirius::test::operator_utils::initialize_memory_manager(2);
  auto replica_spaces      = get_replica_spaces(*memory_manager);
  auto& target_space       = replica_spaces.back().get_gpu_space();
  auto const target_device = target_space.get_device_id();
  auto* target_mr =
    target_space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(target_mr != nullptr);

  auto const allocated_before = target_mr->get_total_allocated_bytes();
  REQUIRE(allocated_before < target_space.get_max_memory());
  auto reservation_pressure =
    target_space.make_reservation_or_null(target_space.get_max_memory() - allocated_before);
  REQUIRE(reservation_pressure != nullptr);
  REQUIRE(target_space.make_reservation_or_null(rmm::CUDA_ALLOCATION_ALIGNMENT) == nullptr);

  rmm::cuda_set_device_raii const source_device{rmm::cuda_device_id{0}};
  std::vector<lex_component_semantics> const components{
    {.consumer_ordinal = 0, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)},
    {.consumer_ordinal = 1, .key = int32_key(cudf::order::DESCENDING, cudf::null_order::BEFORE)}};
  std::vector<std::optional<exact_host_scalar>> boundary_components{int32_bound(1), int32_bound(2)};

  auto require_all_or_nothing = [&](auto& filter) {
    REQUIRE(filter.is_available_on_device(0));
    REQUIRE_FALSE(filter.is_available_on_device(target_device));
    auto const target_allocated = target_mr->get_total_allocated_bytes();

    REQUIRE_THROWS_AS(filter.replicate_to_devices(replica_spaces), std::runtime_error);

    // Nothing installed: the denied target gained no replica and no allocation, and the source
    // keeps exactly the availability it had before the attempt.
    REQUIRE_FALSE(filter.is_available_on_device(target_device));
    REQUIRE(filter.is_available_on_device(0));
    REQUIRE(target_mr->get_total_allocated_bytes() == target_allocated);
  };

  SECTION("RANGE")
  {
    sirius_dynamic_range_filter filter{
      int32_bound(7), range_bound_side::LOWER, true, dynamic_filter_null_policy::REJECT};
    require_all_or_nothing(filter);
  }
  SECTION("LEX_RANGE")
  {
    sirius_dynamic_lex_range_filter filter{exact_host_key_tuple{boundary_components},
                                           components,
                                           /*inclusive=*/false};
    require_all_or_nothing(filter);
  }
}

TEST_CASE("boundary filter replicas serve the AST path on every planned device",
          "[dynamic_filter][top_n][mgpu][replica]")
{
  if (!require_two_gpus()) { return; }

  auto memory_manager      = sirius::test::operator_utils::initialize_memory_manager(2);
  auto replica_spaces      = get_replica_spaces(*memory_manager);
  auto const target_device = replica_spaces.back().get_gpu_space().get_device_id();

  rmm::cuda_set_device_raii const source_device{rmm::cuda_device_id{0}};
  // A null tail component owns no scalar on any device, so replication must still succeed.
  std::vector<lex_component_semantics> const components{
    {.consumer_ordinal = 0, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)},
    {.consumer_ordinal = 1, .key = int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)}};
  std::vector<std::optional<exact_host_scalar>> boundary_components{int32_bound(5), std::nullopt};
  sirius_dynamic_lex_range_filter lex{exact_host_key_tuple{boundary_components}, components, false};

  REQUIRE(lex.is_available_on_device(0));
  REQUIRE_FALSE(lex.is_available_on_device(target_device));
  lex.replicate_to_devices(replica_spaces);
  REQUIRE(lex.is_available_on_device(0));
  REQUIRE(lex.is_available_on_device(target_device));

  // The remote replica lowers a usable tree on its own device.
  {
    rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{target_device}};
    auto const stream = cudf::get_default_stream();
    auto const mr     = replica_spaces.back().get_gpu_space().get_default_allocator();
    std::vector<host_column> const columns{{4, 6, 5}, {1, 1, 1}};
    std::vector<std::unique_ptr<cudf::column>> owned;
    std::vector<cudf::column_view> views;
    for (auto const& column : columns) {
      owned.push_back(make_int32_column(column, stream, mr));
      views.push_back(owned.back()->view());
    }
    cudf::ast::tree tree;
    auto const resolver = [&tree](std::size_t ordinal) -> cudf::ast::expression const& {
      return tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(ordinal));
    };
    auto const& root = lex.to_ast(tree, resolver, target_device);
    auto const keep  = evaluate_keep(cudf::table_view{views}, root, stream);
    std::vector<bool> expected(columns[0].size());
    for (std::size_t row = 0; row < columns[0].size(); ++row) {
      expected[row] =
        lex_keeps(columns, exact_host_key_tuple{boundary_components}, components, false, row);
    }
    REQUIRE(keep == expected);
  }

  rmm::cuda_set_device_raii const teardown_device{rmm::cuda_device_id{0}};
}
