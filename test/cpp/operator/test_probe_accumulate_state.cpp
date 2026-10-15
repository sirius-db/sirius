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

#include "operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <catch.hpp>
#include <op/aggregate/probe_accumulate_impl.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;

namespace {

auto stream() { return cudf::get_default_stream(); }
auto mr() { return cudf::get_current_device_resource_ref(); }

// nullopt marks a NULL row.
template <typename T>
std::unique_ptr<cudf::column> make_column(std::vector<std::optional<T>> const& values,
                                          cudf::data_type type)
{
  auto const n = static_cast<cudf::size_type>(values.size());
  auto col     = cudf::make_numeric_column(type, n, cudf::mask_state::UNALLOCATED, stream(), mr());
  std::vector<T> data(values.size());
  // cuDF requires the mask buffer padded to its allocation size, not just the used words.
  std::vector<cudf::bitmask_type> words(
    cudf::bitmask_allocation_size_bytes(n) / sizeof(cudf::bitmask_type), ~cudf::bitmask_type{0});
  cudf::size_type nulls = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i]) {
      data[i] = *values[i];
    } else {
      data[i] = T{};
      words[i / 32] &= ~(cudf::bitmask_type{1} << (i % 32));
      ++nulls;
    }
  }
  cudaMemcpy(col->mutable_view().template data<T>(),
             data.data(),
             sizeof(T) * data.size(),
             cudaMemcpyHostToDevice);
  if (nulls > 0) {
    col->set_null_mask(
      rmm::device_buffer(words.data(), words.size() * sizeof(words[0]), stream(), mr()), nulls);
  }
  return col;
}

rmm::device_uvector<cudf::size_type> to_device(std::vector<cudf::size_type> const& host)
{
  rmm::device_uvector<cudf::size_type> out(host.size(), stream(), mr());
  if (!host.empty()) {
    cudaMemcpy(
      out.data(), host.data(), host.size() * sizeof(cudf::size_type), cudaMemcpyHostToDevice);
  }
  return out;
}

struct reference_slot {
  std::optional<int32_t> min, max;
  int64_t valid = 0;
  int64_t hits  = 0;
};

constexpr auto i32 = cudf::data_type{cudf::type_id::INT32};

}  // namespace

TEST_CASE("probe accumulate keeps MIN, MAX and COUNT per matched build row", "[probe_accumulate]")
{
  // Build keys 100..103; slot s holds key 100 + s.
  auto build_key =
    make_column<int64_t>({100, 101, 102, 103}, cudf::data_type{cudf::type_id::INT64});
  // Probe values and a matching-pair list. Slot 3 never matches; slot 2's only input is NULL.
  auto values = make_column<int32_t>({5, 9, std::nullopt, 7, 2, 11}, i32);
  cudf::table_view probe({values->view()});
  auto probe_rows = to_device({0, 1, 2, 3, 4, 5});
  auto build_rows = to_device({0, 0, 2, 1, 1, 0});

  probe_accumulate_state state({{accumulate_kind::MIN, 0, i32},
                                {accumulate_kind::MAX, 0, i32},
                                {accumulate_kind::COUNT, 0, i32},
                                {accumulate_kind::COUNT_STAR, 0, i32}});
  state.initialize(4, stream(), mr());
  state.accumulate(probe, probe_rows, build_rows, stream());
  auto out = state.emit(build_key->view(), stream(), mr());
  stream().synchronize();

  REQUIRE(out->num_columns() == 5);
  // Slots 0, 1, 2 matched; slot 3 did not.
  REQUIRE(out->num_rows() == 3);
  auto const view = out->view();
  REQUIRE(copy_column_to_host<int64_t>(view.column(0)) == std::vector<int64_t>{100, 101, 102});
  // MIN / MAX: slot 2 saw only a NULL input, so both are NULL there.
  auto const min_v     = copy_column_to_host<int32_t>(view.column(1));
  auto const max_v     = copy_column_to_host<int32_t>(view.column(2));
  auto const min_valid = copy_validity_to_host(view.column(1));
  auto const max_valid = copy_validity_to_host(view.column(2));
  REQUIRE(min_valid == std::vector<bool>{true, true, false});
  REQUIRE(max_valid == std::vector<bool>{true, true, false});
  REQUIRE(min_v[0] == 5);
  REQUIRE(max_v[0] == 11);
  REQUIRE(min_v[1] == 2);
  REQUIRE(max_v[1] == 7);
  // COUNT skips NULL inputs; COUNT_STAR does not.
  REQUIRE(copy_column_to_host<int64_t>(view.column(3)) == std::vector<int64_t>{3, 2, 0});
  REQUIRE(copy_column_to_host<int64_t>(view.column(4)) == std::vector<int64_t>{3, 2, 1});
  REQUIRE(view.column(3).type().id() == cudf::type_id::INT64);
}

TEST_CASE("probe accumulate emits nothing when no probe row matched", "[probe_accumulate]")
{
  auto build_key = make_column<int64_t>({1, 2, 3}, cudf::data_type{cudf::type_id::INT64});
  auto values    = make_column<int32_t>({1}, i32);
  cudf::table_view probe({values->view()});
  auto none = to_device({});

  probe_accumulate_state state(
    {{accumulate_kind::MIN, 0, i32}, {accumulate_kind::COUNT_STAR, 0, i32}});
  state.initialize(3, stream(), mr());
  state.accumulate(probe, none, none, stream());
  auto out = state.emit(build_key->view(), stream(), mr());
  stream().synchronize();
  REQUIRE(out->num_rows() == 0);
  REQUIRE(out->num_columns() == 3);
}

TEST_CASE("probe accumulate folds several probe batches into one set of slots",
          "[probe_accumulate]")
{
  auto build_key = make_column<int64_t>({7, 8}, cudf::data_type{cudf::type_id::INT64});
  probe_accumulate_state state({{accumulate_kind::MAX, 0, cudf::data_type{cudf::type_id::INT64}},
                                {accumulate_kind::COUNT_STAR, 0, i32}});
  state.initialize(2, stream(), mr());
  for (std::int64_t batch = 0; batch < 3; ++batch) {
    auto values =
      make_column<int64_t>({batch * 10, batch * 10 + 1}, cudf::data_type{cudf::type_id::INT64});
    cudf::table_view probe({values->view()});
    // Row 0 of each batch lands on slot 1, row 1 on slot 0.
    state.accumulate(probe, to_device({0, 1}), to_device({1, 0}), stream());
  }
  auto out = state.emit(build_key->view(), stream(), mr());
  stream().synchronize();
  REQUIRE(out->num_rows() == 2);
  REQUIRE(copy_column_to_host<int64_t>(out->view().column(1)) == std::vector<int64_t>{21, 20});
  REQUIRE(copy_column_to_host<int64_t>(out->view().column(2)) == std::vector<int64_t>{3, 3});
}

TEST_CASE("probe accumulate matches a host reference on random data", "[probe_accumulate]")
{
  constexpr int num_build = 5000;
  constexpr int num_pairs = 200000;
  std::mt19937 rng(42);
  std::vector<std::optional<int32_t>> host_values(num_pairs);
  std::vector<cudf::size_type> probe_rows(num_pairs);
  std::vector<cudf::size_type> build_rows(num_pairs);
  std::vector<reference_slot> reference(num_build);
  std::vector<std::optional<int64_t>> keys(num_build);
  for (int s = 0; s < num_build; ++s) {
    keys[s] = 1000 + s;
  }
  for (int p = 0; p < num_pairs; ++p) {
    probe_rows[p] = p;
    // Skewed target so some slots are hot and many stay unmatched.
    build_rows[p] = static_cast<cudf::size_type>((rng() % num_build) * (rng() % 4 == 0 ? 1 : 0) +
                                                 (rng() % 3 == 0 ? rng() % 64 : rng() % num_build));
    build_rows[p] %= num_build;
    if (rng() % 8 != 0) {
      host_values[p] = static_cast<int32_t>(static_cast<int32_t>(rng()) % 1000000 - 500000);
    }
    auto& slot = reference[build_rows[p]];
    ++slot.hits;
    if (host_values[p]) {
      ++slot.valid;
      slot.min = slot.min ? std::min(*slot.min, *host_values[p]) : *host_values[p];
      slot.max = slot.max ? std::max(*slot.max, *host_values[p]) : *host_values[p];
    }
  }
  auto build_key = make_column<int64_t>(keys, cudf::data_type{cudf::type_id::INT64});
  auto values    = make_column<int32_t>(host_values, i32);
  cudf::table_view probe({values->view()});

  probe_accumulate_state state({{accumulate_kind::MIN, 0, i32},
                                {accumulate_kind::MAX, 0, i32},
                                {accumulate_kind::COUNT, 0, i32},
                                {accumulate_kind::COUNT_STAR, 0, i32}});
  state.initialize(num_build, stream(), mr());
  state.accumulate(probe, to_device(probe_rows), to_device(build_rows), stream());
  auto out = state.emit(build_key->view(), stream(), mr());
  stream().synchronize();

  std::vector<int> matched;
  for (int s = 0; s < num_build; ++s) {
    if (reference[s].hits > 0) { matched.push_back(s); }
  }
  REQUIRE(static_cast<std::size_t>(out->num_rows()) == matched.size());
  auto const view      = out->view();
  auto const out_keys  = copy_column_to_host<int64_t>(view.column(0));
  auto const out_min   = copy_column_to_host<int32_t>(view.column(1));
  auto const out_max   = copy_column_to_host<int32_t>(view.column(2));
  auto const min_valid = copy_validity_to_host(view.column(1));
  auto const out_count = copy_column_to_host<int64_t>(view.column(3));
  auto const out_star  = copy_column_to_host<int64_t>(view.column(4));
  for (std::size_t i = 0; i < matched.size(); ++i) {
    auto const& ref = reference[matched[i]];
    REQUIRE(out_keys[i] == 1000 + matched[i]);
    REQUIRE(min_valid[i] == ref.min.has_value());
    if (ref.min) {
      REQUIRE(out_min[i] == *ref.min);
      REQUIRE(out_max[i] == *ref.max);
    }
    REQUIRE(out_count[i] == ref.valid);
    REQUIRE(out_star[i] == ref.hits);
  }
}

TEST_CASE("probe accumulate rejects types without a device atomic", "[probe_accumulate]")
{
  REQUIRE(probe_accumulate_supports_type(i32));
  REQUIRE(probe_accumulate_supports_type(cudf::data_type{cudf::type_id::INT64}));
  REQUIRE(probe_accumulate_supports_type(cudf::data_type{cudf::type_id::DECIMAL64, -2}));
  REQUIRE(probe_accumulate_supports_type(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}));
  REQUIRE_FALSE(probe_accumulate_supports_type(cudf::data_type{cudf::type_id::FLOAT64}));
  REQUIRE_FALSE(probe_accumulate_supports_type(cudf::data_type{cudf::type_id::INT16}));
  REQUIRE_FALSE(probe_accumulate_supports_type(cudf::data_type{cudf::type_id::STRING}));
  REQUIRE_THROWS(
    probe_accumulate_state({{accumulate_kind::MIN, 0, cudf::data_type{cudf::type_id::FLOAT32}}}));
}
