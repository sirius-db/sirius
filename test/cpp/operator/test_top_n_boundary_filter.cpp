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

// Parity harness for the fused boundary-filter kernel: every case is checked against an
// independent host reference of the per-row three-way lexicographic compare, so the kernel and
// the host reference must agree row-for-row across types, directions, null placements, and
// strictness.

#include "operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <catch.hpp>
#include <op/dynamic_filter/top_n_boundary_filter.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace cucascade;
using sirius::op::detail::apply_boundary_filter;
using sirius::op::detail::boundary_filter_params;

namespace {

using host_value  = std::optional<__int128_t>;
using host_column = std::vector<host_value>;

// 10^19: one above std::int64_t's range, so truncation to int64 wraps it negative.
constexpr __int128_t k_ten_pow_19 = __int128_t{10} * 1'000'000'000'000'000'000LL;
constexpr __int128_t k_two_pow_64 = __int128_t{1} << 64;

struct component_spec {
  cudf::data_type type;
  bool descending;
  bool nulls_first;  // placement in the final output order
  host_value bound;  // disengaged models a null boundary component
};

/// Independent host reference of the kernel's contract: walk components, better keeps, worse
/// drops, ties fall through; a full-tuple tie keeps exactly when inclusive.
bool host_keep(std::vector<host_column> const& columns,
               std::vector<component_spec> const& components,
               bool strict,
               std::size_t row)
{
  for (std::size_t i = 0; i < components.size(); ++i) {
    auto const& spec  = components[i];
    auto const& value = columns[i][row];
    if (!value || !spec.bound) {
      if (!value && !spec.bound) { continue; }
      return !value.has_value() == spec.nulls_first;
    }
    if (*value == *spec.bound) { continue; }
    bool const less = *value < *spec.bound;
    return spec.descending ? !less : less;
  }
  return !strict;
}

template <typename T>
void fill_column_data(cudf::mutable_column_view view, host_column const& values)
{
  std::vector<T> host(values.size(), T{});
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i]) { host[i] = static_cast<T>(*values[i]); }
  }
  if (!host.empty()) {
    cudaMemcpy(view.data<T>(), host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
  }
}

std::unique_ptr<cudf::column> make_key_column(host_column const& values,
                                              cudf::data_type dtype,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto const size = static_cast<cudf::size_type>(values.size());
  auto null_mask  = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  auto* mask_ptr  = static_cast<cudf::bitmask_type*>(null_mask.data());
  cudf::size_type null_count = 0;
  for (cudf::size_type i = 0; i < size; ++i) {
    if (!values[i]) {
      cudf::set_null_mask(mask_ptr, i, i + 1, false, stream);
      ++null_count;
    }
  }

  auto column =
    cudf::is_fixed_point(dtype)
      ? cudf::make_fixed_point_column(dtype, size, std::move(null_mask), null_count, stream, mr)
    : dtype.id() == cudf::type_id::TIMESTAMP_DAYS
      ? cudf::make_timestamp_column(dtype, size, std::move(null_mask), null_count, stream, mr)
      : cudf::make_numeric_column(dtype, size, std::move(null_mask), null_count, stream, mr);
  switch (dtype.id()) {
    case cudf::type_id::INT8: fill_column_data<std::int8_t>(column->mutable_view(), values); break;
    case cudf::type_id::INT16:
      fill_column_data<std::int16_t>(column->mutable_view(), values);
      break;
    case cudf::type_id::INT64:
    case cudf::type_id::DECIMAL64:
      fill_column_data<std::int64_t>(column->mutable_view(), values);
      break;
    case cudf::type_id::DECIMAL128:
      fill_column_data<__int128_t>(column->mutable_view(), values);
      break;
    default: fill_column_data<std::int32_t>(column->mutable_view(), values); break;
  }
  return column;
}

std::uint8_t width_of(cudf::data_type dtype)
{
  switch (dtype.id()) {
    case cudf::type_id::INT8: return 1;
    case cudf::type_id::INT16: return 2;
    case cudf::type_id::INT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32: return 4;
    case cudf::type_id::INT64:
    case cudf::type_id::DECIMAL64: return 8;
    case cudf::type_id::DECIMAL128: return 16;
    default:
      // A silent width-4 marshal for an unknown type is exactly the bug class under test.
      throw std::logic_error("width_of: unmapped test key type");
  }
}

/// Run the kernel over the case's key columns plus a trailing row-id column and require exact
/// agreement with the host reference (including the all-pass null-result contract).
void check_parity(cucascade::memory::memory_space& space,
                  std::vector<component_spec> const& components,
                  std::vector<host_column> const& columns,
                  bool strict)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = space.get_default_allocator();
  auto const rows   = static_cast<cudf::size_type>(columns.empty() ? 0 : columns[0].size());

  std::vector<std::unique_ptr<cudf::column>> owned;
  std::vector<cudf::column_view> views;
  std::vector<cudf::size_type> key_columns;
  for (std::size_t i = 0; i < components.size(); ++i) {
    owned.push_back(make_key_column(columns[i], components[i].type, stream, mr));
    views.push_back(owned.back()->view());
    key_columns.push_back(static_cast<cudf::size_type>(i));
  }
  host_column row_ids(static_cast<std::size_t>(rows));
  for (cudf::size_type i = 0; i < rows; ++i) {
    row_ids[static_cast<std::size_t>(i)] = i;
  }
  owned.push_back(make_key_column(row_ids, cudf::data_type{cudf::type_id::INT32}, stream, mr));
  views.push_back(owned.back()->view());
  auto const batch = cudf::table_view{views};

  boundary_filter_params params{};
  params.count  = static_cast<std::uint32_t>(components.size());
  params.strict = strict;
  for (std::size_t i = 0; i < components.size(); ++i) {
    auto& component       = params.components[i];
    component.engaged     = components[i].bound.has_value();
    component.value       = components[i].bound.value_or(0);
    component.descending  = components[i].descending;
    component.nulls_first = components[i].nulls_first;
    component.width       = width_of(components[i].type);
  }

  std::vector<std::int32_t> expected;
  for (cudf::size_type row = 0; row < rows; ++row) {
    if (host_keep(columns, components, strict, static_cast<std::size_t>(row))) {
      expected.push_back(row);
    }
  }

  auto const result = apply_boundary_filter(batch, key_columns, params, stream, mr);
  REQUIRE(result.rows_kept == static_cast<cudf::size_type>(expected.size()));
  if (result.rows_kept == rows) {
    // All-pass (vacuously for an empty batch): the caller forwards the input unchanged.
    REQUIRE(result.filtered == nullptr);
    return;
  }
  REQUIRE(result.filtered != nullptr);
  REQUIRE(result.filtered->num_columns() == batch.num_columns());
  auto const survivors = sirius::test::operator_utils::copy_column_to_host<std::int32_t>(
    result.filtered->view().column(batch.num_columns() - 1));
  REQUIRE(survivors == expected);
}

}  // namespace

TEST_CASE("boundary filter kernel parity: single-component sweep", "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto const dtype =
    std::array{cudf::data_type{cudf::type_id::INT8},
               cudf::data_type{cudf::type_id::INT16},
               cudf::data_type{cudf::type_id::INT32},
               cudf::data_type{cudf::type_id::INT64},
               cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}}[GENERATE(0, 1, 2, 3, 4)];
  bool const descending  = GENERATE(false, true);
  bool const nulls_first = GENERATE(false, true);
  bool const strict      = GENERATE(false, true);
  CAPTURE(static_cast<int>(dtype.id()), descending, nulls_first, strict);

  // Ties, nulls, both strict sides of the boundary, and negatives; values fit every tested width.
  // The negatives are the sign-extension discriminator: a load that zero-extends any width turns
  // them into large positives and inverts their verdict against the bound.
  host_column const values{40, std::nullopt, 38, 42, -40, std::nullopt, -39, 41, 40};
  check_parity(*space,
               {{.type = dtype, .descending = descending, .nulls_first = nulls_first, .bound = 40}},
               {values},
               strict);
}

TEST_CASE("boundary filter kernel parity: multi-component cascade with null tails",
          "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  bool const strict = GENERATE(false, true);
  CAPTURE(strict);

  // Component 1's boundary is disengaged (a null tail), so rows tying component 0 are decided by
  // null-vs-value placement there, and full-tuple ties (rows with a null component 1) fall to
  // component 2.
  std::vector<component_spec> const components{{.type       = cudf::data_type{cudf::type_id::INT32},
                                                .descending = false,
                                                .nulls_first = false,
                                                .bound       = 5},
                                               {.type       = cudf::data_type{cudf::type_id::INT64},
                                                .descending = true,
                                                .nulls_first = true,
                                                .bound       = std::nullopt},
                                               {.type       = cudf::data_type{cudf::type_id::INT16},
                                                .descending = false,
                                                .nulls_first = true,
                                                .bound       = 7}};
  std::vector<host_column> const columns{
    {4, 6, 5, 5, 5, 5, 5, std::nullopt},
    {9, std::nullopt, 3, std::nullopt, std::nullopt, std::nullopt, 3, 1},
    {0, 0, 9, 6, 7, 8, std::nullopt, 2}};
  check_parity(*space, components, columns, strict);
}

TEST_CASE("boundary marshalling emits width 16 and the exact widened DECIMAL128 value",
          "[dynamic_filter][top_n]")
{
  // The widened {1,2,4,8,16} width gate's trigger set is now empty by exhaustion: every cuDF
  // fixed-width type maps to one of those widths. The gate is retained as defence for the next
  // widening -- the same explicitly-untestable-guard status as the publication sync's race window
  // (top_n_threshold_coordinator.cpp, publish_revision). What is testable is the positive form:
  // width 16 marshals and the value round-trips a rep above int64's range exactly, so a widening
  // that truncated anywhere between the variant and the launch parameters fails here.
  auto const k_dec128        = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  constexpr __int128_t k_rep = k_ten_pow_19 + 7;
  std::vector<sirius::op::top_n_key_semantics> const wide{{.storage_type = k_dec128,
                                                           .order        = cudf::order::ASCENDING,
                                                           .null_order = cudf::null_order::AFTER}};
  sirius::op::exact_host_key_tuple const boundary{{sirius::op::exact_host_scalar{k_rep, k_dec128}}};
  REQUIRE(cudf::size_of(wide.front().storage_type) == 16);
  auto const wide_params =
    sirius::op::detail::make_boundary_filter_params(boundary, wide, 1, /*strict=*/true);
  REQUIRE(wide_params.components[0].width == 16);
  REQUIRE(wide_params.components[0].value == k_rep);

  // The DECIMAL64 control, mirroring the width-16 assertions one band down.
  std::vector<sirius::op::top_n_key_semantics> const narrow{
    {.storage_type = cudf::data_type{cudf::type_id::DECIMAL64, -2},
     .order        = cudf::order::ASCENDING,
     .null_order   = cudf::null_order::AFTER}};
  sirius::op::exact_host_key_tuple const ok{{sirius::op::exact_host_scalar{
    std::int64_t{1234}, cudf::data_type{cudf::type_id::DECIMAL64, -2}}}};
  auto const params = sirius::op::detail::make_boundary_filter_params(ok, narrow, 1, true);
  REQUIRE(params.components[0].width == 8);
  REQUIRE(params.components[0].value == 1234);
}

TEST_CASE("boundary filter kernel parity: edge shapes", "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  SECTION("empty batch")
  {
    check_parity(*space,
                 {{.type        = cudf::data_type{cudf::type_id::INT32},
                   .descending  = false,
                   .nulls_first = false,
                   .bound       = 1}},
                 {host_column{}},
                 true);
  }
  SECTION("all rows pass: null result forwards the batch")
  {
    check_parity(*space,
                 {{.type        = cudf::data_type{cudf::type_id::INT32},
                   .descending  = false,
                   .nulls_first = false,
                   .bound       = 100}},
                 {host_column{1, 2, 3, 4}},
                 true);
  }
  SECTION("no row passes: empty table, all columns preserved")
  {
    check_parity(*space,
                 {{.type        = cudf::data_type{cudf::type_id::INT32},
                   .descending  = false,
                   .nulls_first = false,
                   .bound       = 0}},
                 {host_column{1, 2, 3, 4}},
                 true);
  }
  SECTION("full-tuple ties at the component limit: dropped when strict, kept when inclusive")
  {
    // Eight tying components -- the inclusive form is what a >k_max_components boundary
    // degrades to, and it must keep every prefix-tie.
    std::vector<component_spec> components;
    std::vector<host_column> columns;
    for (int i = 0; i < 8; ++i) {
      components.push_back({.type        = cudf::data_type{cudf::type_id::INT32},
                            .descending  = false,
                            .nulls_first = false,
                            .bound       = 1});
      columns.push_back(host_column{1, 1, 1});
    }
    check_parity(*space, components, columns, /*strict=*/true);
    check_parity(*space, components, columns, /*strict=*/false);
  }
}

TEST_CASE("boundary filter parity at width 16", "[dynamic_filter][top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto const k_dec128 = cudf::data_type{cudf::type_id::DECIMAL128, -4};
  // Each value makes int64 truncation *invert* an ordering, not just perturb it: +/-(1<<64) have
  // a zero low word, and +/-10^19 wrap across int64's range and change sign.
  host_column const wide_values{
    0, 1, -1, k_two_pow_64, -k_two_pow_64, k_ten_pow_19, -k_ten_pow_19, std::nullopt};

  SECTION("single-component sweep against the host reference")
  {
    bool const descending  = GENERATE(false, true);
    bool const nulls_first = GENERATE(false, true);
    bool const strict      = GENERATE(false, true);
    // Bounds sit at a truncation-inverting value, at a narrow value the wide rows straddle, and
    // at an exact-tie value, so every fixture row lands on each side of some bound.
    auto const bound = std::array<__int128_t, 3>{1, k_two_pow_64, -k_ten_pow_19}[GENERATE(0, 1, 2)];
    CAPTURE(descending, nulls_first, strict, static_cast<double>(bound));
    check_parity(
      *space,
      {{.type = k_dec128, .descending = descending, .nulls_first = nulls_first, .bound = bound}},
      {wide_values},
      strict);
  }

  SECTION("multi-key: DECIMAL128 head with an INT32 tail")
  {
    // Head ties on the wide boundary value fall through to the narrow tail, and vice versa in
    // the inverted case below -- a per-component width error cannot cancel out across the tuple.
    std::vector<component_spec> const components{
      {.type = k_dec128, .descending = false, .nulls_first = false, .bound = k_two_pow_64},
      {.type        = cudf::data_type{cudf::type_id::INT32},
       .descending  = true,
       .nulls_first = true,
       .bound       = 7}};
    std::vector<host_column> const columns{
      {k_two_pow_64, k_two_pow_64, k_two_pow_64, -k_two_pow_64, k_ten_pow_19, std::nullopt},
      {9, 7, 3, 0, 0, 5}};
    check_parity(*space, components, columns, /*strict=*/true);
    check_parity(*space, components, columns, /*strict=*/false);
  }

  SECTION("multi-key: INT32 head with a DECIMAL128 tail")
  {
    std::vector<component_spec> const components{
      {.type        = cudf::data_type{cudf::type_id::INT32},
       .descending  = false,
       .nulls_first = false,
       .bound       = 5},
      {.type = k_dec128, .descending = false, .nulls_first = false, .bound = -k_ten_pow_19}};
    std::vector<host_column> const columns{
      {4, 6, 5, 5, 5, 5}, {0, 0, -k_two_pow_64, -k_ten_pow_19, k_ten_pow_19, std::nullopt}};
    check_parity(*space, components, columns, /*strict=*/true);
    check_parity(*space, components, columns, /*strict=*/false);
  }

  SECTION("all-pass fast path with a boundary worse than every row")
  {
    check_parity(*space,
                 {{.type        = k_dec128,
                   .descending  = false,
                   .nulls_first = false,
                   .bound       = k_ten_pow_19 + k_two_pow_64}},
                 {host_column{0, -k_two_pow_64, k_two_pow_64, k_ten_pow_19}},
                 true);
  }
}

TEST_CASE("boundary filter refuses a misaligned 16-byte key column", "[dynamic_filter][top_n]")
{
  // The kernel's width-16 path is a single natural load, adopting cuDF's fixed-point alignment
  // contract; this is the host-side gate's triggering test. rmm allocations are 256-byte aligned,
  // so a buffer offset by 8 bytes is misaligned for __int128_t while still device-valid.
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();
  auto const mr     = space->get_default_allocator();

  constexpr cudf::size_type k_rows = 4;
  rmm::device_buffer backing{
    (static_cast<std::size_t>(k_rows) + 1) * sizeof(__int128_t), stream, mr};
  auto const* misaligned_base = static_cast<std::byte const*>(backing.data()) + 8;
  cudf::column_view const misaligned{
    cudf::data_type{cudf::type_id::DECIMAL128, -2}, k_rows, misaligned_base, nullptr, 0};
  auto const batch = cudf::table_view{{misaligned}};

  boundary_filter_params params{};
  params.count                 = 1;
  params.strict                = true;
  params.components[0].engaged = true;
  params.components[0].value   = 0;
  params.components[0].width   = 16;
  std::vector<cudf::size_type> const key_columns{0};
  REQUIRE_THROWS_AS(apply_boundary_filter(batch, key_columns, params, stream, mr),
                    std::invalid_argument);
}
