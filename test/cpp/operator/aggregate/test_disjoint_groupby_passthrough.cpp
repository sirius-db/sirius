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

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "utils/data_utils.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/bound_result_modifier.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <helper/type_conversions.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::op;

namespace {

using namespace sirius::test::operator_utils;
using I64Traits = gpu_type_traits<int64_t>;
using sirius::test::vector_to_cudf_column;

template <typename T>
struct passthrough_key_traits : gpu_type_traits<T> {};

template <>
struct passthrough_key_traits<uint64_t> {
  using type                               = uint64_t;
  static constexpr cudf::type_id cudf_type = cudf::type_id::UINT64;
  static constexpr bool is_decimal         = false;
  static constexpr bool is_string          = false;
  static constexpr bool is_ts              = false;
};

template <typename KeyTraits>
std::unique_ptr<cudf::table> make_partial_table(const std::vector<typename KeyTraits::type>& keys,
                                                const std::vector<int64_t>& values,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(vector_to_cudf_column<KeyTraits>(keys, stream, mr));
  columns.push_back(vector_to_cudf_column<I64Traits>(values, stream, mr));
  return std::make_unique<cudf::table>(std::move(columns));
}

template <typename KeyTraits>
std::shared_ptr<data_batch> make_partial_batch(const std::vector<typename KeyTraits::type>& keys,
                                               const std::vector<int64_t>& values,
                                               memory_space& space,
                                               rmm::cuda_stream_view stream)
{
  return sirius::make_data_batch(
    make_partial_table<KeyTraits>(keys, values, stream, get_resource_ref(space)),
    space,
    stream,
    sirius::telemetry::batch_telemetry_info{});
}

std::shared_ptr<data_batch> make_nullable_partial_batch(const std::vector<int64_t>& keys,
                                                        const std::vector<bool>& valid,
                                                        const std::vector<int64_t>& values,
                                                        memory_space& space,
                                                        rmm::cuda_stream_view stream)
{
  auto mr         = get_resource_ref(space);
  auto key_column = vector_to_cudf_column<I64Traits>(keys, stream, mr);
  auto mask       = cudf::create_null_mask(
    static_cast<cudf::size_type>(keys.size()), cudf::mask_state::ALL_VALID, stream, mr);
  auto* mask_data            = static_cast<cudf::bitmask_type*>(mask.data());
  cudf::size_type null_count = 0;
  for (cudf::size_type index = 0; index < static_cast<cudf::size_type>(valid.size()); ++index) {
    if (!valid[static_cast<std::size_t>(index)]) {
      cudf::set_null_mask(mask_data, index, index + 1, false, stream);
      ++null_count;
    }
  }
  key_column->set_null_mask(std::move(mask), null_count);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(key_column));
  columns.push_back(vector_to_cudf_column<I64Traits>(values, stream, mr));
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                 space,
                                 stream,
                                 sirius::telemetry::batch_telemetry_info{});
}

void attach_input_repository(sirius_physical_grouped_aggregate_merge& merge,
                             cucascade::shared_data_repository& repository)
{
  auto input_port           = std::make_unique<sirius_physical_operator::port>();
  input_port->type          = MemoryBarrierType::PIPELINE;
  input_port->repo          = &repository;
  input_port->src_pipeline  = nullptr;
  input_port->dest_pipeline = nullptr;
  merge.add_port("input", std::move(input_port));
}

constexpr std::string_view passthrough_log_prefix = "disjoint_groupby_passthrough outcome=";

void require_passthrough_outcome(const sirius::test::scoped_recording_log_sink& logs,
                                 std::string_view outcome,
                                 std::size_t input_batches)
{
  auto const records = logs.records();
  auto const record  = std::find_if(records.begin(), records.end(), [](auto const& candidate) {
    return candidate.message.starts_with(passthrough_log_prefix);
  });
  auto const outcome_count =
    std::count_if(records.begin(), records.end(), [](auto const& candidate) {
      return candidate.message.starts_with(passthrough_log_prefix);
    });
  REQUIRE(outcome_count == 1);
  REQUIRE(record != records.end());
  REQUIRE(record->level == sirius::log::level::debug);
  REQUIRE(record->message == std::string{passthrough_log_prefix} + std::string{outcome} +
                               " input_batches=" + std::to_string(input_batches));
}

void require_no_passthrough_outcome(const sirius::test::scoped_recording_log_sink& logs)
{
  auto const records = logs.records();
  REQUIRE(std::none_of(records.begin(), records.end(), [](auto const& record) {
    return record.message.starts_with(passthrough_log_prefix);
  }));
}

std::unique_ptr<operator_data> execute_with_outcome(
  sirius_physical_grouped_aggregate_merge& merge,
  const std::vector<std::shared_ptr<data_batch>>& inputs,
  rmm::cuda_stream_view stream,
  std::string_view outcome)
{
  sirius::test::scoped_recording_log_sink logs{"debug"};
  auto output = merge.execute(pipelineable_operator_data(inputs), stream);
  require_passthrough_outcome(logs, outcome, inputs.size());
  return output;
}

std::unique_ptr<operator_data> execute_without_outcome(
  sirius_physical_grouped_aggregate_merge& merge,
  const std::vector<std::shared_ptr<data_batch>>& inputs,
  rmm::cuda_stream_view stream)
{
  sirius::test::scoped_recording_log_sink logs{"debug"};
  auto output = merge.execute(pipelineable_operator_data(inputs), stream);
  require_no_passthrough_outcome(logs);
  return output;
}

std::shared_ptr<data_batch> run_local(sirius_physical_grouped_aggregate& local,
                                      const std::shared_ptr<data_batch>& input,
                                      rmm::cuda_stream_view stream)
{
  auto output = local.execute(
    pipelineable_operator_data(std::vector<std::shared_ptr<data_batch>>{input}), stream);
  auto const& batches = dynamic_cast<const pipelineable_operator_data&>(*output).get_data_batches();
  REQUIRE(batches.size() == 1);
  return batches.front();
}

class merge_fixture {
 public:
  explicit merge_fixture(std::string aggregate           = "sum",
                         uint64_t hash_partition_bytes   = std::numeric_limits<uint64_t>::max(),
                         std::size_t upstream_partitions = 1)
  {
    auto aggregate_definitions =
      sirius::test::create_aggregate_expressions<I64Traits>({0}, {std::move(aggregate)}, {1});
    local_aggregate = std::make_unique<sirius_physical_grouped_aggregate>(
      std::move(aggregate_definitions.output_types),
      std::move(aggregate_definitions.aggregates),
      std::move(aggregate_definitions.groups),
      100);
    merge = std::make_unique<sirius_physical_grouped_aggregate_merge>(local_aggregate.get(),
                                                                      hash_partition_bytes);
    merge->set_disjoint_groupby_passthrough(true);

    if (upstream_partitions > repository.num_partitions()) {
      repository.set_num_partitions(upstream_partitions);
    }
    attach_input_repository(*merge, repository);
  }

  cucascade::shared_data_repository repository;
  std::unique_ptr<sirius_physical_grouped_aggregate> local_aggregate;
  std::unique_ptr<sirius_physical_grouped_aggregate_merge> merge;
};

const std::vector<std::shared_ptr<data_batch>>& output_batches(const operator_data& output)
{
  return dynamic_cast<const pipelineable_operator_data&>(output).get_data_batches();
}

void require_normal_merge(const operator_data& output,
                          const std::vector<std::shared_ptr<data_batch>>& inputs,
                          cudf::table_view expected)
{
  auto const& batches = output_batches(output);
  REQUIRE(batches.size() == 1);
  REQUIRE(std::none_of(inputs.begin(), inputs.end(), [&](auto const& input) {
    return input->get_batch_id() == batches.front()->get_batch_id();
  }));
  REQUIRE(sirius::test::expect_data_batch_equivalent_to_table(batches.front(), expected, true));
}

}  // namespace

TEMPLATE_TEST_CASE("disjoint grouped partials pass through without changing batch identity",
                   "[disjoint_groupby_passthrough]",
                   int64_t,
                   uint64_t,
                   timestamp_us_tag)
{
  using KeyTraits = passthrough_key_traits<TestType>;
  using Key       = typename KeyTraits::type;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();

  Key const base = [] {
    if constexpr (std::is_same_v<TestType, uint64_t>) {
      return static_cast<Key>(std::numeric_limits<int64_t>::max()) + static_cast<Key>(1024);
    } else {
      return Key{0};
    }
  }();
  std::vector<std::vector<Key>> const expected_keys{
    {base + static_cast<Key>(100), base + static_cast<Key>(101)},
    {base, base + static_cast<Key>(1), base + static_cast<Key>(2)},
    {base + static_cast<Key>(50), base + static_cast<Key>(51)}};
  std::vector<std::vector<int64_t>> const expected_values{{1000, 1010}, {10, 20, 30}, {500, 510}};

  std::vector<std::shared_ptr<data_batch>> inputs;
  inputs.reserve(expected_keys.size());
  for (std::size_t index = 0; index < expected_keys.size(); ++index) {
    inputs.push_back(
      make_partial_batch<KeyTraits>(expected_keys[index], expected_values[index], *space, stream));
  }
  std::vector<uint64_t> expected_ids;
  expected_ids.reserve(inputs.size());
  for (auto const& input : inputs) {
    expected_ids.push_back(input->get_batch_id());
  }

  merge_fixture fixture;
  auto output  = execute_with_outcome(*fixture.merge, inputs, stream, "engaged");
  auto batches = dynamic_cast<const pipelineable_operator_data&>(*output).get_read_only_batches();

  stream.synchronize();
  REQUIRE(batches.size() == inputs.size());
  for (std::size_t index = 0; index < batches.size(); ++index) {
    REQUIRE(batches[index].get_batch_id() == expected_ids[index]);
    auto const table = sirius::get_cudf_table_view(batches[index]);
    REQUIRE(table.num_columns() == 2);
    REQUIRE(table.num_rows() == static_cast<cudf::size_type>(expected_keys[index].size()));
    REQUIRE(table.column(0).type().id() == KeyTraits::cudf_type);
    REQUIRE(table.column(1).type().id() == cudf::type_id::INT64);
    REQUIRE(copy_column_to_host<Key>(table.column(0)) == expected_keys[index]);
    REQUIRE(copy_column_to_host<int64_t>(table.column(1)) == expected_values[index]);
  }
}

TEST_CASE("passthrough outcomes cover single input and disabled execution",
          "[disjoint_groupby_passthrough]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();

  SECTION("single input reports its outcome and preserves identity")
  {
    std::vector<std::shared_ptr<data_batch>> inputs{
      make_partial_batch<I64Traits>({0, 1}, {10, 20}, *space, stream)};
    auto const expected_id = inputs.front()->get_batch_id();
    merge_fixture fixture;
    auto output = execute_with_outcome(*fixture.merge, inputs, stream, "single_input");
    stream.synchronize();

    auto const& batches = output_batches(*output);
    REQUIRE(batches.size() == 1);
    REQUIRE(batches.front()->get_batch_id() == expected_id);
  }

  SECTION("disabled passthrough emits no outcome and uses the normal merge")
  {
    std::vector<std::shared_ptr<data_batch>> inputs{
      make_partial_batch<I64Traits>({0, 1}, {10, 20}, *space, stream),
      make_partial_batch<I64Traits>({100, 101}, {30, 40}, *space, stream)};
    auto expected = make_partial_table<I64Traits>(
      {0, 1, 100, 101}, {10, 20, 30, 40}, stream, get_resource_ref(*space));
    merge_fixture fixture;
    fixture.merge->set_disjoint_groupby_passthrough(false);
    auto output = execute_without_outcome(*fixture.merge, inputs, stream);
    stream.synchronize();

    require_normal_merge(*output, inputs, expected->view());
  }
}

TEST_CASE("overlapping grouped partial ranges use the normal merge",
          "[disjoint_groupby_passthrough]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();

  auto run_case = [&](std::vector<int64_t> first_keys,
                      std::vector<int64_t> first_values,
                      std::vector<int64_t> second_keys,
                      std::vector<int64_t> second_values,
                      std::vector<int64_t> expected_keys,
                      std::vector<int64_t> expected_values) {
    std::vector<std::shared_ptr<data_batch>> inputs{
      make_partial_batch<I64Traits>(first_keys, first_values, *space, stream),
      make_partial_batch<I64Traits>(second_keys, second_values, *space, stream)};
    auto expected = make_partial_table<I64Traits>(
      expected_keys, expected_values, stream, get_resource_ref(*space));
    merge_fixture fixture;
    auto output = execute_with_outcome(*fixture.merge, inputs, stream, "overlapping_ranges");
    stream.synchronize();
    require_normal_merge(*output, inputs, expected->view());
  };

  SECTION("equal boundary keys are combined")
  {
    run_case({1, 2}, {10, 20}, {2, 3}, {30, 40}, {1, 2, 3}, {10, 50, 40});
  }
  SECTION("nested ranges fall back")
  {
    run_case({0, 100}, {10, 20}, {25, 75}, {30, 40}, {0, 25, 75, 100}, {10, 30, 40, 20});
  }
  SECTION("interleaved ranges fall back")
  {
    run_case({0, 2, 4},
             {10, 20, 30},
             {1, 3, 5},
             {40, 50, 60},
             {0, 1, 2, 3, 4, 5},
             {10, 40, 20, 50, 30, 60});
  }
}

TEST_CASE("unsupported and nullable leading keys use the normal merge",
          "[disjoint_groupby_passthrough]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();

  SECTION("floating-point keys are unsupported by the proof")
  {
    using F64Traits = gpu_type_traits<double>;
    std::vector<std::shared_ptr<data_batch>> inputs{
      make_partial_batch<F64Traits>({0.0, 1.0}, {10, 20}, *space, stream),
      make_partial_batch<F64Traits>({100.0, 101.0}, {30, 40}, *space, stream)};
    auto expected = make_partial_table<F64Traits>(
      {0.0, 1.0, 100.0, 101.0}, {10, 20, 30, 40}, stream, get_resource_ref(*space));
    merge_fixture fixture;
    auto output = execute_with_outcome(*fixture.merge, inputs, stream, "unsupported_key");
    stream.synchronize();
    require_normal_merge(*output, inputs, expected->view());
  }

  SECTION("null keys are combined only by the normal merge")
  {
    std::vector<std::shared_ptr<data_batch>> inputs{
      make_nullable_partial_batch({0, 2}, {false, true}, {10, 20}, *space, stream),
      make_nullable_partial_batch({0, 3}, {false, true}, {30, 40}, *space, stream)};
    merge_fixture fixture;
    auto output = execute_with_outcome(*fixture.merge, inputs, stream, "unsupported_key");
    stream.synchronize();

    auto const& batches = output_batches(*output);
    REQUIRE(batches.size() == 1);
    REQUIRE(std::none_of(inputs.begin(), inputs.end(), [&](auto const& input) {
      return input->get_batch_id() == batches.front()->get_batch_id();
    }));
    auto table = sirius::get_cudf_table_view(*batches.front());
    REQUIRE(table.num_rows() == 3);
    REQUIRE(table.column(0).null_count() == 1);
  }
}

TEST_CASE("passthrough requires one upstream partition and a fitting task",
          "[disjoint_groupby_passthrough]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();
  std::vector<std::shared_ptr<data_batch>> inputs{
    make_partial_batch<I64Traits>({0, 1}, {10, 20}, *space, stream),
    make_partial_batch<I64Traits>({100, 101}, {30, 40}, *space, stream)};
  auto expected = make_partial_table<I64Traits>(
    {0, 1, 100, 101}, {10, 20, 30, 40}, stream, get_resource_ref(*space));

  SECTION("multiple upstream partitions")
  {
    merge_fixture fixture("sum", std::numeric_limits<uint64_t>::max(), 2);
    auto output =
      execute_with_outcome(*fixture.merge, inputs, stream, "multiple_upstream_partitions");
    stream.synchronize();
    require_normal_merge(*output, inputs, expected->view());
  }
  SECTION("input exceeds hash partition budget")
  {
    merge_fixture fixture("sum", 1, 1);
    auto output = execute_with_outcome(*fixture.merge, inputs, stream, "byte_budget");
    stream.synchronize();
    require_normal_merge(*output, inputs, expected->view());
  }
}

TEST_CASE("AVG grouped partials never use disjoint passthrough", "[disjoint_groupby_passthrough]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();
  auto mr     = get_resource_ref(*space);

  auto make_avg_partial =
    [&](std::vector<int64_t> keys, std::vector<int64_t> sums, std::vector<int64_t> counts) {
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(vector_to_cudf_column<I64Traits>(keys, stream, mr));
      columns.push_back(vector_to_cudf_column<I64Traits>(sums, stream, mr));
      columns.push_back(vector_to_cudf_column<I64Traits>(counts, stream, mr));
      return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                     *space,
                                     stream,
                                     sirius::telemetry::batch_telemetry_info{});
    };
  std::vector<std::shared_ptr<data_batch>> inputs{make_avg_partial({0, 1}, {20, 60}, {2, 3}),
                                                  make_avg_partial({100, 101}, {40, 50}, {2, 5})};

  merge_fixture fixture("avg");
  auto output = execute_with_outcome(*fixture.merge, inputs, stream, "avg");
  stream.synchronize();
  auto const& batches = output_batches(*output);
  REQUIRE(batches.size() == 1);
  REQUIRE(std::none_of(inputs.begin(), inputs.end(), [&](auto const& input) {
    return input->get_batch_id() == batches.front()->get_batch_id();
  }));
}

TEST_CASE("COUNT DISTINCT partials never use disjoint passthrough",
          "[disjoint_groupby_passthrough]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();

  auto aggregate_definitions =
    sirius::test::create_count_distinct_expressions<I64Traits, I64Traits>({0}, 1);
  cucascade::shared_data_repository repository;
  sirius_physical_grouped_aggregate local(std::move(aggregate_definitions.output_types),
                                          std::move(aggregate_definitions.aggregates),
                                          std::move(aggregate_definitions.groups),
                                          4);
  sirius_physical_grouped_aggregate_merge merge(&local, std::numeric_limits<uint64_t>::max());
  merge.set_disjoint_groupby_passthrough(true);
  attach_input_repository(merge, repository);

  auto first_input  = make_partial_batch<I64Traits>({0, 0, 1}, {10, 10, 20}, *space, stream);
  auto second_input = make_partial_batch<I64Traits>({100, 100, 101}, {30, 40, 50}, *space, stream);
  std::vector<std::shared_ptr<data_batch>> partials{run_local(local, first_input, stream),
                                                    run_local(local, second_input, stream)};
  auto expected =
    make_partial_table<I64Traits>({0, 1, 100, 101}, {1, 1, 2, 1}, stream, get_resource_ref(*space));

  auto output = execute_with_outcome(merge, partials, stream, "count_distinct");
  stream.synchronize();
  require_normal_merge(*output, partials, expected->view());
}

TEST_CASE("multiple grouping sets preserve metadata and use the normal merge",
          "[disjoint_groupby_passthrough]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();

  auto aggregate_definitions =
    sirius::test::create_aggregate_expressions<I64Traits>({0}, {"sum"}, {1});
  duckdb::vector<duckdb::GroupingSet> grouping_sets;
  grouping_sets.push_back(duckdb::GroupingSet{0});
  grouping_sets.push_back(duckdb::GroupingSet{});
  cucascade::shared_data_repository repository;
  sirius_physical_grouped_aggregate local(std::move(aggregate_definitions.output_types),
                                          std::move(aggregate_definitions.aggregates),
                                          std::move(aggregate_definitions.groups),
                                          std::move(grouping_sets),
                                          {},
                                          4,
                                          duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
                                          duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES);
  sirius_physical_grouped_aggregate_merge merge(&local, std::numeric_limits<uint64_t>::max());
  REQUIRE(merge.grouping_sets.size() == 2);
  REQUIRE(merge.grouping_sets[0] == duckdb::GroupingSet{0});
  REQUIRE(merge.grouping_sets[1].empty());
  merge.set_disjoint_groupby_passthrough(true);
  attach_input_repository(merge, repository);

  std::vector<std::shared_ptr<data_batch>> partials{
    make_partial_batch<I64Traits>({0, 1}, {10, 20}, *space, stream),
    make_partial_batch<I64Traits>({100, 101}, {30, 40}, *space, stream)};
  auto expected = make_partial_table<I64Traits>(
    {0, 1, 100, 101}, {10, 20, 30, 40}, stream, get_resource_ref(*space));

  auto output = execute_with_outcome(merge, partials, stream, "multiple_grouping_sets");
  stream.synchronize();
  require_normal_merge(*output, partials, expected->view());
}

TEST_CASE("passthrough partials feed straight into TopN", "[disjoint_groupby_passthrough]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();

  std::vector<std::vector<int64_t>> const keys{{100, 101}, {0, 1, 2}, {50, 51}};
  std::vector<std::vector<int64_t>> const values{{1000, 1010}, {10, 20, 30}, {500, 510}};

  std::vector<std::shared_ptr<data_batch>> inputs;
  inputs.reserve(keys.size());
  for (std::size_t index = 0; index < keys.size(); ++index) {
    inputs.push_back(make_partial_batch<I64Traits>(keys[index], values[index], *space, stream));
  }

  merge_fixture fixture;
  auto partials = execute_with_outcome(*fixture.merge, inputs, stream, "engaged");
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*partials).get_data_batches().size() ==
          inputs.size());

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // group key
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // summed value

  auto make_value_orders = [] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(duckdb::BoundOrderByNode(
      duckdb::OrderType::DESCENDING,
      duckdb::OrderByNullType::NULLS_LAST,
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(
        duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), duckdb::idx_t{1})));
    return orders;
  };
  constexpr std::size_t limit  = 2;
  constexpr std::size_t offset = 1;

  sirius_physical_top_n topn(
    sirius::from_duckdb_vec(types), make_value_orders(), limit, offset, nullptr, 0);
  auto local_out = topn.execute(*partials, stream);
  auto const& candidates =
    dynamic_cast<const pipelineable_operator_data&>(*local_out).get_data_batches();
  REQUIRE(candidates.size() == inputs.size());

  sirius_physical_top_n_merge topn_merge(
    sirius::from_duckdb_vec(types), make_value_orders(), limit, offset, nullptr, 0);
  auto merged = topn_merge.execute(pipelineable_operator_data(candidates), stream);
  auto const& merged_batches =
    dynamic_cast<const pipelineable_operator_data&>(*merged).get_data_batches();
  REQUIRE(merged_batches.size() == 1);

  stream.synchronize();
  auto const view = sirius::get_cudf_table_view(*merged_batches.front());
  REQUIRE(copy_column_to_host<int64_t>(view.column(0)) == std::vector<int64_t>{100, 51});
  REQUIRE(copy_column_to_host<int64_t>(view.column(1)) == std::vector<int64_t>{1000, 510});
}
