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

/*
 * Tests `accumulated_bloom_builder` over every membership key family against the Bloom column
 * constructor: a key accumulated from any split of its build, at any carrier, sets exactly the bits
 * the column constructor sets for the whole build, publishes them under the construction type's
 * domain, and contributes without waiting on the host.
 */

#include "operator/dynamic_filter_accumulation_test_utils.hpp"
#include "operator/dynamic_filter_column_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <op/dynamic_filter/bloom_sizing.hpp>
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace op  = sirius::op;
namespace acc = sirius::test::accumulation;
using builder = op::detail::accumulated_bloom_builder;
using sirius::test::columns::families;
using sirius::test::columns::family;
using sirius::test::columns::k_rows;
using sirius::test::columns::make_column;
using sirius::test::columns::string_family;
using sirius::test::columns::valid_rows;
using sirius::test::columns::values;

/// The bits the Bloom column constructor sets for @p keys on GPU 0.
std::vector<std::byte> column_bits(cudf::column_view const& keys, ::cuda::stream_ref stream)
{
  op::sirius_dynamic_bloom_filter const bloom{
    keys, stream, cudf::get_current_device_resource_ref()};
  auto const contents = builder::inspect_replica(bloom, rmm::cuda_device_id{0}, stream);
  REQUIRE(contents);
  return contents->bytes;
}

builder make_builder(acc::fixture const& fixture, std::vector<builder::key> keys)
{
  auto result = builder::try_create(std::move(keys), fixture.spaces);
  REQUIRE(result);
  return std::move(*result);
}

/// Contributes @p column as a one-column table on GPU 0's task stream and retires it.
void contribute(builder& accumulated,
                acc::fixture const& fixture,
                std::vector<cudf::column_view> const& columns)
{
  REQUIRE(accumulated.enqueue_add(
    cudf::table_view{columns}, rmm::cuda_device_id{0}, fixture.task_stream(0)));
  fixture.task_stream(0).sync();
}

}  // namespace

TEST_CASE("an accumulated key sets the column constructor's bits for any split of its build",
          "[dynamic_filter][construction][bloom]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  for (auto const& tested : families()) {
    for (std::size_t splits : {1, 2, 5}) {
      CAPTURE(tested.name, splits);
      auto const build  = tested.carrier(stream);
      auto const oracle = tested.oracle(stream);
      auto const blocks = op::bloom_blocks_for(valid_rows(k_rows));
      auto accumulated  = make_builder(fixture, {{0, tested.construction, blocks}});

      // Uneven slices of the build, an empty one among them; then an all-null batch.
      std::vector<cudf::size_type> bounds{0};
      if (splits == 2) { bounds.push_back(213); }
      if (splits == 5) { bounds.insert(bounds.end(), {1, 1, 250, 251}); }
      bounds.push_back(static_cast<cudf::size_type>(k_rows));
      for (std::size_t index = 0; index + 1 < bounds.size(); ++index) {
        auto const slice = cudf::slice(build->view(), {bounds[index], bounds[index + 1]}).front();
        contribute(accumulated, fixture, {slice});
      }
      auto const all_null = cudf::slice(build->view(), {2, 3}).front();  // Row 2 is null.
      contribute(accumulated, fixture, {all_null});

      auto const partial = accumulated.inspect_partial(0, rmm::cuda_device_id{0});
      REQUIRE(partial);
      CHECK(partial->bytes == column_bits(oracle->view(), stream));
    }
  }
}

TEST_CASE("an accumulated key publishes under its construction type's domain",
          "[dynamic_filter][construction][bloom]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  for (auto const& tested : families()) {
    CAPTURE(tested.name);
    auto const build  = tested.carrier(stream);
    auto const oracle = tested.oracle(stream);
    auto accumulated =
      make_builder(fixture, {{0, tested.construction, op::bloom_blocks_for(valid_rows(k_rows))}});
    contribute(accumulated, fixture, {build->view()});
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
    auto const published = accumulated.finish(rmm::cuda_device_id{0},
                                              fixture.publication_stream(0),
                                              fixture.gpu(0).get_default_allocator());
    REQUIRE(published);
    REQUIRE(published->filters.size() == 1);
    auto const& filter = *published->filters.front();
    CHECK(filter.domain() == *op::classify_membership_key(tested.construction));
    auto const replica = builder::inspect_replica(filter, rmm::cuda_device_id{0});
    REQUIRE(replica);
    CHECK(replica->bytes == column_bits(oracle->view(), stream));
  }
}

TEST_CASE("keys of different sizes and carriers accumulate side by side",
          "[dynamic_filter][construction][bloom]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  using cudf::type_id;
  // One key arrives at a different carrier in each batch; the oracle is the build at its type.
  auto const int16_rows =
    make_column(cudf::data_type{type_id::INT16}, values<std::int16_t>(100), stream);
  auto const int32_rows =
    make_column(cudf::data_type{type_id::INT32}, values<std::int32_t>(300), stream);
  auto const int64_rows =
    make_column(cudf::data_type{type_id::INT64}, values<std::int64_t>(200), stream);
  auto const strings = string_family().carrier(stream);

  std::vector<builder::key> const keys{
    {0, cudf::data_type{type_id::INT64}, op::bloom_blocks_for(k_rows)},
    {1, cudf::data_type{type_id::STRING}, op::bloom_blocks_for(valid_rows(k_rows))}};
  auto accumulated = make_builder(fixture, keys);
  contribute(
    accumulated, fixture, {int16_rows->view(), cudf::slice(strings->view(), {0, 100}).front()});
  contribute(
    accumulated, fixture, {int32_rows->view(), cudf::slice(strings->view(), {100, 400}).front()});
  contribute(accumulated,
             fixture,
             {int64_rows->view(),
              cudf::slice(strings->view(), {400, static_cast<cudf::size_type>(k_rows)}).front()});

  std::vector<std::unique_ptr<cudf::column>> widened;
  for (auto const* part : {int16_rows.get(), int32_rows.get(), int64_rows.get()}) {
    widened.push_back(cudf::cast(part->view(), cudf::data_type{type_id::INT64}, stream));
  }
  auto const integers = cudf::concatenate(
    std::vector<cudf::column_view>{widened[0]->view(), widened[1]->view(), widened[2]->view()},
    stream);

  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto const published = accumulated.finish(
    rmm::cuda_device_id{0}, fixture.publication_stream(0), fixture.gpu(0).get_default_allocator());
  REQUIRE(published);
  REQUIRE(published->filters.size() == 2);
  auto const first  = builder::inspect_replica(*published->filters[0], rmm::cuda_device_id{0});
  auto const second = builder::inspect_replica(*published->filters[1], rmm::cuda_device_id{0});
  REQUIRE(first);
  REQUIRE(second);
  CHECK(first->bytes.size() != second->bytes.size());
  CHECK(first->bytes == column_bits(integers->view(), stream));
  CHECK(second->bytes == column_bits(strings->view(), stream));
}

TEST_CASE("an accumulated contribution never waits on the host",
          "[dynamic_filter][construction][bloom]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  for (auto const& tested : families()) {
    CAPTURE(tested.name);
    auto const build = tested.carrier(stream);
    auto accumulated =
      make_builder(fixture, {{0, tested.construction, op::bloom_blocks_for(valid_rows(k_rows))}});
    // An ungated contribution first: a kernel's first, lazy load would wait behind the gate.
    contribute(accumulated, fixture, {build->view()});

    acc::stream_gate gate{stream};
    REQUIRE(
      accumulated.enqueue_add(cudf::table_view{{build->view()}}, rmm::cuda_device_id{0}, stream));
    cudaEvent_t after{};
    REQUIRE(cudaEventCreateWithFlags(&after, cudaEventDisableTiming) == cudaSuccess);
    REQUIRE(cudaEventRecord(after, stream.get()) == cudaSuccess);
    CHECK(cudaEventQuery(after) == cudaErrorNotReady);
    gate.open();
    REQUIRE(cudaEventSynchronize(after) == cudaSuccess);
    REQUIRE(cudaEventDestroy(after) == cudaSuccess);
  }
}

TEST_CASE("a contribution whose column is not a carrier of its key fails before inserting",
          "[dynamic_filter][construction][bloom]")
{
  acc::fixture fixture;
  auto const stream = fixture.task_stream(0);
  auto const doubles =
    make_column(cudf::data_type{cudf::type_id::FLOAT64}, std::vector<double>{1.0, 2.0}, stream);
  auto accumulated = make_builder(fixture, {{0, cudf::data_type{cudf::type_id::INT64}, 1}});
  CHECK_THROWS_AS(
    accumulated.enqueue_add(cudf::table_view{{doubles->view()}}, rmm::cuda_device_id{0}, stream),
    op::detail::accumulation_invariant_error);
  CHECK_THROWS_AS(
    builder::try_create({{0, cudf::data_type{cudf::type_id::FLOAT64}, 1}}, fixture.spaces),
    op::detail::accumulation_invariant_error);
  CHECK_THROWS_AS(
    builder::try_create({{0, cudf::data_type{cudf::type_id::INT64}, 0}}, fixture.spaces),
    op::detail::accumulation_invariant_error);
}
