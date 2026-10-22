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

// [late_mat][port] — the far end of a deferral, against a real pin. GPU required.
//
// The batch that arrives here is what the ride produced: a UINT64 rowid where
// the first deferred column was, INT8 placeholders where the rest were, and
// every other column carrying its own values. What must come back is the table
// the reader would have seen had nothing been deferred — same rows, same order,
// values restored in place and everything else untouched.
//
// The default pin holds row i at value i, so a restored value IS its own global row id:
// a splice that puts a column in the wrong position, or reads the wrong rows,
// produces a column that names what it actually read.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/reference.hpp>
#include <late_mat/port_materialize.hpp>
#include <op/partition_placement.hpp>
#include <op/scan/scan_output_operator_data.hpp>
#include <op/scan/sirius_physical_dynamic_filter.hpp>
#include <op/sirius_physical_dense_count_join.hpp>
#include <op/sirius_physical_partition.hpp>
#include <op/sirius_physical_projection.hpp>
#include <planner/late_mat_plan_pass.hpp>
#include <planner/sirius_physical_plan_generator.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

using sirius::late_mat::column_origin;
using sirius::late_mat::make_defer_pair;
using sirius::late_mat::materialize_at_port;
using sirius::late_mat::pin_entry_handle;
using sirius::late_mat::port_directive_matches;
using sirius::scan_manager::pinned_entry;

namespace {

/// A two-column device pin whose row i holds value i in both columns.
struct fake_entry {
  pinned_entry entry;
  std::shared_ptr<pinned_entry const> entry_owner;
  std::shared_ptr<pin_entry_handle> handle;

  fake_entry(std::vector<std::int64_t> const& batch_rows, ::cuda::stream_ref stream)
  {
    entry.tier = cucascade::memory::Tier::GPU;
    for (auto const& name : {std::string{"c_name"}, std::string{"c_address"}}) {
      entry.cache_info.names.push_back(name);
      std::vector<std::shared_ptr<cudf::column>> chunks;
      std::int64_t next = 0;
      for (auto const rows : batch_rows) {
        std::vector<std::int32_t> host(static_cast<std::size_t>(rows));
        std::iota(host.begin(), host.end(), static_cast<std::int32_t>(next));
        next += rows;
        auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                             static_cast<cudf::size_type>(rows),
                                             cudf::mask_state::UNALLOCATED,
                                             stream);
        cudaMemcpyAsync(col->mutable_view().data<std::int32_t>(),
                        host.data(),
                        host.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice,
                        stream.get());
        chunks.push_back(std::shared_ptr<cudf::column>(std::move(col)));
        if (entry.cache_info.names.size() == 1) {
          entry.num_rows += static_cast<std::size_t>(rows);
        }
      }
      cudaStreamSynchronize(stream.get());
      entry.data_batches_by_column.emplace(name, std::move(chunks));
    }
    entry_owner = std::shared_ptr<pinned_entry const>(&entry, [](pinned_entry const*) {});
    handle      = std::make_shared<pin_entry_handle>("customer", 5);
    handle->set_entry(entry_owner);
  }

  [[nodiscard]] column_origin origin(std::uint32_t pos) const
  {
    column_origin o;
    o.handle     = handle;
    o.column_pos = pos;
    o.generation = handle->generation();
    return o;
  }
};

template <typename T>
std::unique_ptr<cudf::column> upload(std::vector<T> const& host,
                                     cudf::type_id id,
                                     ::cuda::stream_ref stream)
{
  auto col = cudf::make_numeric_column(cudf::data_type{id},
                                       static_cast<cudf::size_type>(host.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream);
  cudaMemcpyAsync(col->mutable_view().template data<T>(),
                  host.data(),
                  host.size() * sizeof(T),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudaStreamSynchronize(stream.get());
  return col;
}

template <typename T>
std::vector<T> read_back(cudf::column_view const& col)
{
  std::vector<T> host(static_cast<std::size_t>(col.size()));
  if (!host.empty()) {
    cudaMemcpy(host.data(), col.data<T>(), host.size() * sizeof(T), cudaMemcpyDeviceToHost);
  }
  return host;
}

/// The batch as it rides: [payload INT32, rowid UINT64, placeholder INT8].
///
/// The deferred pair sits at positions 1 and 2, so position 0 exercises the
/// splice's other half — a column that must come through untouched.
std::unique_ptr<cudf::table> riding_batch(std::vector<std::uint64_t> const& rowids,
                                          std::vector<std::int32_t> const& payload,
                                          ::cuda::stream_ref stream)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload(payload, cudf::type_id::INT32, stream));
  columns.push_back(upload(rowids, cudf::type_id::UINT64, stream));
  columns.push_back(
    upload(std::vector<std::int8_t>(rowids.size(), 0), cudf::type_id::INT8, stream));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::vector<cudf::data_type> riding_schema()
{
  return {cudf::data_type{cudf::type_id::INT32},
          cudf::data_type{cudf::type_id::INT32},
          cudf::data_type{cudf::type_id::INT32}};
}

}  // namespace

TEST_CASE("a rowid becomes its columns again, in the batch's own order", "[late_mat][port]")
{
  auto const stream = ::cuda::stream_ref{cudaStream_t{}};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  fake_entry pin({300, 150, 200}, stream);

  // A join's output: unordered, repeated, spanning all three chunks.
  std::vector<std::uint64_t> const rowids{420, 7, 420, 300, 649, 7, 0, 299, 450};
  std::vector<std::int32_t> payload(rowids.size());
  std::iota(payload.begin(), payload.end(), 1000);

  auto const pair = make_defer_pair(
    riding_schema(), {1, 2}, riding_schema(), {1, 2}, {pin.origin(0), pin.origin(1)});
  REQUIRE(pair.valid());

  auto const batch = riding_batch(rowids, payload, stream);
  REQUIRE(port_directive_matches(pair.port, batch->view()));

  auto const restored = materialize_at_port(pair.port, batch->view(), stream, mr);
  cudaStreamSynchronize(stream.get());
  REQUIRE(restored->num_columns() == 3);
  REQUIRE(restored->num_rows() == static_cast<cudf::size_type>(rowids.size()));

  // The pin holds row i at value i, so each restored value names the row it read.
  std::vector<std::int32_t> expect;
  for (auto const id : rowids) {
    expect.push_back(static_cast<std::int32_t>(id));
  }
  REQUIRE(read_back<std::int32_t>(restored->get_column(1).view()) == expect);
  REQUIRE(read_back<std::int32_t>(restored->get_column(2).view()) == expect);
  // The column that was never deferred is the one it always was.
  REQUIRE(read_back<std::int32_t>(restored->get_column(0).view()) == payload);
}

TEST_CASE("a batch of another shape is declined, not materialized", "[late_mat][port]")
{
  auto const stream = ::cuda::stream_ref{cudaStream_t{}};
  fake_entry pin({64}, stream);

  auto const pair = make_defer_pair(
    riding_schema(), {1, 2}, riding_schema(), {1, 2}, {pin.origin(0), pin.origin(1)});
  REQUIRE(pair.valid());

  // A UINT64 in the right place is not enough: an operator can receive batches
  // from more than one producer, and materializing against a stranger's batch
  // reads arbitrary rows of the pinned table.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload(std::vector<std::int32_t>(4, 0), cudf::type_id::INT32, stream));
  columns.push_back(upload(std::vector<std::uint64_t>(4, 0), cudf::type_id::UINT64, stream));
  columns.push_back(upload(std::vector<std::int32_t>(4, 0), cudf::type_id::INT32, stream));
  cudf::table const stranger(std::move(columns));

  REQUIRE_FALSE(port_directive_matches(pair.port, stranger.view()));
  REQUIRE_THROWS(materialize_at_port(
    pair.port, stranger.view(), stream, rmm::mr::get_current_device_resource_ref()));
}

TEST_CASE(
  "a planned dynamic-filter port restores remapped values before late publication is applied",
  "[late_mat][port][dynamic_filter]")
{
  using sirius::op::sirius_physical_operator;
  using sirius::op::SiriusPhysicalOperatorType;
  auto const stream = ::cuda::stream_ref{cudaStream_t{}};
  auto* space       = sirius::test::operator_utils::get_default_gpu_space();
  auto const mr     = space->get_default_allocator();
  duckdb::vector<sirius::logical_type> const scan_types{
    sirius::logical_type::make(sirius::type_id::INTEGER),
    sirius::logical_type::make(sirius::type_id::BIGINT),
    sirius::logical_type::make(sirius::type_id::BIGINT)};
  auto scan = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::TABLE_SCAN, scan_types, 4);
  auto* scan_ptr                                    = scan.get();
  duckdb::unique_ptr<sirius_physical_operator> tree = std::move(scan);
  sirius_physical_operator key_source(SiriusPhysicalOperatorType::NESTED_LOOP_JOIN, scan_types, 4);
  for (int i = 0; i < 3; ++i) {
    auto partition =
      duckdb::make_uniq<sirius::op::sirius_physical_partition>(scan_types, 4, &key_source);
    REQUIRE(partition->partition_keys().empty());
    partition->children.push_back(std::move(tree));
    tree = std::move(partition);
  }

  std::vector<cudf::size_type> const output_order{2, 0, 1};
  duckdb::vector<sirius::logical_type> port_types;
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions;
  for (auto position : output_order) {
    port_types.push_back(scan_types[position]);
    expressions.push_back(std::make_unique<sirius::ast::node>(
      sirius::ast::reference{static_cast<std::uint32_t>(position), scan_types[position]}));
  }
  auto projection = duckdb::make_uniq<sirius::op::sirius_physical_projection>(
    port_types, std::move(expressions), 4);
  auto* projection_ptr = projection.get();
  projection->children.push_back(std::move(tree));
  auto channel  = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({0, 2});
  channel->freeze_registration();
  auto endpoint = duckdb::make_uniq<sirius::op::scan::sirius_physical_dynamic_filter>(
    port_types,
    4,
    std::make_shared<sirius::op::scan::dynamic_filter_consumer>(
      channel,
      sirius::op::scan::consumer_config{
        .mode = sirius::op::scan::dynamic_filter_apply_mode::MEMBERSHIP_MASKS_ONLY},
      std::vector<sirius::op::scan::binding>{}));
  auto* endpoint_ptr = endpoint.get();
  endpoint->children.push_back(std::move(projection));
  sirius_physical_operator later_reader(SiriusPhysicalOperatorType::HASH_GROUP_BY, port_types, 4);
  later_reader.children.push_back(std::move(endpoint));
  sirius::planner::sirius_physical_plan_generator::set_parent_ops(later_reader, nullptr);

  REQUIRE(channel->snapshot().empty());
  REQUIRE_FALSE(channel->snapshot().terminal());
  sirius::late_mat::defer_policy const policy{
    .min_value_bytes = 1, .min_value_x_boundaries = 0, .min_boundaries = 4};
  auto const planned = sirius::planner::plan_deferral(*scan_ptr, policy);
  REQUIRE(planned.installable());
  REQUIRE(planned.port == endpoint_ptr);
  REQUIRE(planned.port_input == projection_ptr);
  REQUIRE(planned.positions == std::vector<std::size_t>{1, 2});
  REQUIRE(planned.port_positions == std::vector<std::size_t>{2, 0});
  REQUIRE(planned.boundaries == 4);
  REQUIRE(planned.net_value_bytes == 8);
  REQUIRE(planned.join_keys_skipped == 0);

  fake_entry pin({4}, stream);
  pin.entry.data_batches_by_column.at("c_name")    = {std::shared_ptr<cudf::column>(
    upload(std::vector<std::int64_t>{101, 202, 303, 404}, cudf::type_id::INT64, stream))};
  pin.entry.data_batches_by_column.at("c_address") = {std::shared_ptr<cudf::column>(
    upload(std::vector<std::int64_t>{1001, 1002, 1003, 1004}, cudf::type_id::INT64, stream))};
  std::vector<cudf::data_type> const scan_schema{cudf::data_type{cudf::type_id::INT32},
                                                 cudf::data_type{cudf::type_id::INT64},
                                                 cudf::data_type{cudf::type_id::INT64}};
  std::vector<cudf::data_type> const port_schema{scan_schema[2], scan_schema[0], scan_schema[1]};
  std::vector<column_origin> origins;
  for (auto position : planned.positions) {
    origins.push_back(pin.origin(static_cast<std::uint32_t>(position - 1)));
  }
  REQUIRE(sirius::planner::install_deferral(
    *scan_ptr,
    *planned.port,
    make_defer_pair(scan_schema, planned.positions, port_schema, planned.port_positions, origins)));
  auto const& deferred  = scan_ptr->deferred_output();
  auto const& directive = endpoint_ptr->port_directive();
  REQUIRE(deferred.output_positions == std::vector<std::size_t>{1, 2});
  REQUIRE(directive.valid());
  REQUIRE(directive.rowid_position() == 2);
  REQUIRE(later_reader.port_directive().empty());

  std::vector<std::uint64_t> const rowids{2, 1, 0, 3};
  std::vector<std::int32_t> const payload{33, 22, 11, 44};
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (std::size_t position = 0; position < scan_schema.size(); ++position) {
    if (position == deferred.rowid_position()) {
      columns.push_back(upload(rowids, deferred.rowid_type, stream));
    } else if (deferred.defers(position)) {
      columns.push_back(upload(
        std::vector<std::int8_t>(rowids.size(), 0), sirius::late_mat::kPlaceholderType, stream));
    } else {
      columns.push_back(upload(payload, cudf::type_id::INT32, stream));
    }
  }
  cudf::table const substituted(std::move(columns));
  auto const at_endpoint = substituted.view().select(output_order);
  REQUIRE(port_directive_matches(directive, at_endpoint));

  // Publication occurs after the planner has installed the restoration point.
  auto keys = upload(std::vector<std::int64_t>{202, 404}, cudf::type_id::INT64, stream);
  auto filter =
    std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(keys->view(), stream, mr);
  stream.sync();
  REQUIRE(producer.push_filter(2, filter));
  auto restored = materialize_at_port(directive, at_endpoint, stream, mr);
  stream.sync();
  REQUIRE(read_back<std::int64_t>(restored->get_column(2).view()) ==
          std::vector<std::int64_t>{303, 202, 101, 404});
  REQUIRE(read_back<std::int64_t>(restored->get_column(0).view()) ==
          std::vector<std::int64_t>{1003, 1002, 1001, 1004});
  REQUIRE(read_back<std::int32_t>(restored->get_column(1).view()) == payload);

  auto batch = sirius::make_data_batch(std::move(restored), *space, stream, {});
  sirius::op::pipelineable_operator_data input({std::move(batch)});
  input.prepare_for_processing(space, stream);
  auto output          = endpoint_ptr->execute(input, stream);
  auto const* filtered = dynamic_cast<sirius::op::pipelineable_operator_data const*>(output.get());
  REQUIRE(filtered != nullptr);
  auto const output_batches = filtered->get_read_only_batches();
  REQUIRE(output_batches.size() == 1);
  auto const observed = sirius::get_cudf_table_view(output_batches.front());
  stream.sync();
  REQUIRE(observed.num_rows() == 2);
  REQUIRE(read_back<std::int64_t>(observed.column(2)) == std::vector<std::int64_t>{202, 404});
  REQUIRE(read_back<std::int64_t>(observed.column(0)) == std::vector<std::int64_t>{1002, 1004});
  REQUIRE(read_back<std::int32_t>(observed.column(1)) == std::vector<std::int32_t>{22, 44});
}

TEST_CASE("a row-preserving replacement keeps the payload's identity", "[late_mat][port]")
{
  using sirius::op::partitioned_operator_data;
  using sirius::op::pipelineable_operator_data;
  using sirius::op::scan::scan_output_operator_data;
  auto const stream = ::cuda::stream_ref{cudaStream_t{}};
  auto* space       = sirius::test::operator_utils::get_default_gpu_space();
  auto const batch  = [&] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(upload(std::vector<std::int32_t>{1, 2, 3}, cudf::type_id::INT32, stream));
    return sirius::make_data_batch(
      std::make_unique<cudf::table>(std::move(columns)), *space, stream, {});
  };
  auto const original    = batch();
  auto const original_id = original->get_batch_id();

  SECTION("plain")
  {
    pipelineable_operator_data payload({original});
    payload.set_preferred_device_id(0);
    auto const restored = payload.with_batches_preserving_rows({batch()});
    REQUIRE(typeid(*restored) == typeid(pipelineable_operator_data));
    REQUIRE(restored->original_batch_ids() == std::vector<std::uint64_t>{original_id});
    REQUIRE(restored->get_preferred_device_id() == 0);
    // A second replacement, as after a relocation, still reports the first original ID.
    auto const again = restored->with_batches_preserving_rows({batch()});
    REQUIRE(again->original_batch_ids() == std::vector<std::uint64_t>{original_id});
    REQUIRE_THROWS_AS(payload.with_batches_preserving_rows({batch(), batch()}),
                      std::invalid_argument);
    pipelineable_operator_data with_null({nullptr, original});
    auto const kept_null = with_null.with_batches_preserving_rows({nullptr, batch()});
    REQUIRE(kept_null->get_data_batches().front() == nullptr);
    REQUIRE(kept_null->original_batch_ids() == std::vector<std::uint64_t>{original_id});
    REQUIRE_THROWS_AS(with_null.with_batches_preserving_rows({batch(), nullptr}),
                      std::invalid_argument);
  }
  SECTION("partitioned")
  {
    partitioned_operator_data payload(
      {original}, 3, sirius::op::partition_placement::round_robin(4, {0}));
    auto const restored     = payload.with_batches_preserving_rows({batch()});
    auto const* partitioned = dynamic_cast<partitioned_operator_data const*>(restored.get());
    REQUIRE(partitioned != nullptr);
    REQUIRE(partitioned->get_type() == sirius::op::operator_data_type::PARTITIONED);
    REQUIRE(partitioned->get_partition_idx() == 3);
    REQUIRE(partitioned->get_preferred_device_id() == 0);
    REQUIRE(partitioned->original_batch_ids() == std::vector<std::uint64_t>{original_id});
  }
  SECTION("dense count join input")
  {
    auto const counted_first  = batch();
    auto const counted_second = batch();
    sirius::op::dense_count_join_input payload(
      {original},
      {counted_first, counted_second},
      2,
      sirius::op::partition_placement::round_robin(4, {0}));
    auto const replacements = std::vector{batch(), batch(), batch()};
    auto const restored     = payload.with_batches_preserving_rows(replacements);
    REQUIRE(typeid(*restored) == typeid(sirius::op::dense_count_join_input));
    auto const& dense = dynamic_cast<sirius::op::dense_count_join_input const&>(*restored);
    REQUIRE(dense.preserved_count() == 1);
    REQUIRE(dense.counted_count() == 2);
    REQUIRE(dense.get_data_batches() == replacements);
    REQUIRE(dense.get_type() == sirius::op::operator_data_type::PARTITIONED);
    REQUIRE(dense.get_partition_idx() == 2);
    REQUIRE(dense.get_preferred_device_id() == 0);
    REQUIRE(dense.original_batch_ids() ==
            std::vector<std::uint64_t>{
              original_id, counted_first->get_batch_id(), counted_second->get_batch_id()});
    REQUIRE_THROWS_AS(payload.with_batches_preserving_rows({batch(), batch()}),
                      std::invalid_argument);
  }
  SECTION("scan output")
  {
    auto const keys   = upload(std::vector<std::int32_t>{2}, cudf::type_id::INT32, stream);
    auto const filter = std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(
      keys->view(), stream, space->get_default_allocator());
    stream.sync();
    auto const channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    scan_output_operator_data payload(original);
    payload.install_receipt({channel, original_id, {{filter, 0}}, {{filter, 0}}});
    auto const restored = payload.with_batches_preserving_rows({batch()});
    auto const* scanned = dynamic_cast<scan_output_operator_data const*>(restored.get());
    REQUIRE(scanned != nullptr);
    REQUIRE(scanned->receipt().endpoint == channel);
    REQUIRE(scanned->receipt().original_batch_id == original_id);
    REQUIRE(scanned->receipt().applied == payload.receipt().applied);
    REQUIRE(scanned->receipt().decode_attached == payload.receipt().decode_attached);
    REQUIRE(scanned->original_batch_ids() == std::vector<std::uint64_t>{original_id});
    REQUIRE_THROWS_AS(payload.with_batches_preserving_rows({batch(), batch()}),
                      std::invalid_argument);
  }
}

TEST_CASE("decode history survives restoration and keeps conditioned rows out of the gate",
          "[late_mat][dynamic_filter][consumer_integration]")
{
  using sirius::op::sirius_physical_operator;
  using sirius::op::scan::scan_output_operator_data;
  using sirius::op::scan::detail::consumer_test_access;
  using sirius::op::scan::detail::gate_test_access;
  auto const stream = ::cuda::stream_ref{cudaStream_t{}};
  auto* space       = sirius::test::operator_utils::get_default_gpu_space();
  auto const mr     = space->get_default_allocator();
  fake_entry pin({4}, stream);

  // The scan defers positions 1 and 2 and the endpoint restores them; it filters position 1.
  duckdb::vector<sirius::logical_type> const types(
    3, sirius::logical_type::make(sirius::type_id::INTEGER));
  auto channel  = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto producer = channel->register_producer({1});
  channel->freeze_registration();
  auto consumer = std::make_shared<sirius::op::scan::dynamic_filter_consumer>(
    channel,
    sirius::op::scan::consumer_config{},
    sirius::op::scan::identity_bindings(types.size()));
  sirius_physical_operator scan(sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN, types, 4);
  sirius::op::scan::sirius_physical_dynamic_filter endpoint(types, 4, consumer);
  REQUIRE(sirius::planner::install_deferral(
    scan,
    endpoint,
    make_defer_pair(
      riding_schema(), {1, 2}, riding_schema(), {1, 2}, {pin.origin(0), pin.origin(1)})));
  auto const keys = upload(std::vector<std::int32_t>{1, 3}, cudf::type_id::INT32, stream);
  auto const filter =
    std::make_shared<sirius::op::sirius_dynamic_small_in_list_filter>(keys->view(), stream, mr);
  stream.sync();
  REQUIRE(producer.push_filter(1, filter));
  auto& gate = consumer_test_access::gate(*consumer);

  // What the scan emits, and what materialize_deferred_input does with it before the endpoint.
  auto const scan_output = [&](std::vector<std::uint64_t> const& rowids,
                               std::vector<sirius::op::scan::applied_entry> history) {
    auto batch = sirius::make_data_batch(
      riding_batch(rowids, std::vector<std::int32_t>(rowids.size(), 7), stream),
      *space,
      stream,
      {});
    auto const id = batch->get_batch_id();
    auto output   = std::make_unique<scan_output_operator_data>(std::move(batch));
    output->install_receipt({channel, id, {}, std::move(history)});
    return output;
  };
  auto const restore = [&](scan_output_operator_data const& riding) {
    auto const leases = riding.get_read_only_batches();
    auto const view   = sirius::get_cudf_table_view(leases.front());
    REQUIRE(port_directive_matches(endpoint.port_directive(), view));
    auto restored = sirius::make_data_batch(
      materialize_at_port(endpoint.port_directive(), view, stream, mr), *space, stream, {});
    auto input = riding.with_batches_preserving_rows({std::move(restored)});
    input->prepare_for_processing(space, stream);
    return input;
  };
  auto const rows_after_endpoint = [&](sirius::op::pipelineable_operator_data const& input) {
    auto output = endpoint.execute(input, stream);
    auto const ro =
      dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_read_only_batches();
    REQUIRE(ro.size() == 1);
    return sirius::get_cudf_table_view(ro.front()).num_rows();
  };

  // Rows the decode already conditioned on F: every one passes F again.
  auto const conditioned = scan_output({1, 3}, {{filter, 1}});
  auto const restored    = restore(*conditioned);
  auto const* carried    = dynamic_cast<scan_output_operator_data const*>(restored.get());
  REQUIRE(carried != nullptr);
  REQUIRE(carried->receipt().decode_attached == conditioned->receipt().decode_attached);
  REQUIRE(carried->original_batch_ids() == conditioned->original_batch_ids());
  REQUIRE(rows_after_endpoint(*restored) == 2);
  REQUIRE_FALSE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 1}, 1).has_value());
  REQUIRE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));

  // A clean batch is still filtered by F and measures its true keep ratio.
  auto const clean = scan_output({0, 1, 2, 3}, {});
  REQUIRE(rows_after_endpoint(*restore(*clean)) == 2);
  REQUIRE(gate_test_access::filter_keep_ratio(gate, {filter.get(), 1}, 1) == Approx(0.5));
  REQUIRE_FALSE(gate_test_access::needs_combined_sample(gate, channel->snapshot()));
}
