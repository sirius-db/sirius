/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/arrow_materializer.hpp"
#include "op/scan/field_id_mapping.hpp"
#include "op/scan/iceberg_dv_preparation.hpp"
#include "op/scan/iceberg_gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/positional_visibility.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "transparent/read_view_registry.hpp"
#include "transparent/replay_admission.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <cucascade/io/io_errors.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/local_ioctx.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <algorithm>
#include <array>
#include <deque>
#include <stdexcept>
#include <system_error>

using namespace sirius::op::scan;
namespace {
std::unique_ptr<cudf::column> constant(int32_t value, int rows, ::cuda::stream_ref stream)
{
  cudf::numeric_scalar<int32_t> scalar(value, true, stream);
  return cudf::make_column_from_scalar(scalar, rows, stream);
}
int64_t first_int64(cudf::column_view col)
{
  int64_t value = 0;
  REQUIRE(cudaMemcpy(&value, col.head<int64_t>(), sizeof(value), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  return value;
}
struct arrow_fixture {
  ArrowSchema child_schema{}, schema{};
  ArrowArray child{}, array{};
  ArrowSchema* schema_children[1]{&child_schema};
  ArrowArray* children[1]{&child};
  int32_t values[3]{10, 20, 30};
  uint8_t validity = 5;
  void const* buffers[2]{&validity, values};
  void const* root_buffers[1]{nullptr};
  int* releases;
  explicit arrow_fixture(int& count) : releases(&count)
  {
    child_schema.format  = "i";
    child_schema.flags   = ARROW_FLAG_NULLABLE;
    child_schema.release = [](ArrowSchema* p) { p->release = nullptr; };
    schema.format        = "+s";
    schema.n_children    = 1;
    schema.children      = schema_children;
    schema.release       = [](ArrowSchema* p) { p->release = nullptr; };
    child.length         = 3;
    child.null_count     = 1;
    child.n_buffers      = 2;
    child.buffers        = buffers;
    child.release        = [](ArrowArray* p) { p->release = nullptr; };
    array.length         = 3;
    array.n_buffers      = 1;
    array.buffers        = root_buffers;
    array.n_children     = 1;
    array.children       = children;
    array.private_data   = releases;
    array.release        = [](ArrowArray* p) {
      ++*static_cast<int*>(p->private_data);
      p->release = nullptr;
    };
  }
};

struct test_arrow_source final : arrow_batch_source {
  std::deque<arrow_stream_batch> queue;
  bool throw_when_empty = false;
  int next_calls        = 0;
  int cancel_calls      = 0;

  std::optional<arrow_stream_batch> next() override
  {
    ++next_calls;
    if (queue.empty()) {
      if (throw_when_empty) throw std::runtime_error("test source failure");
      return std::nullopt;
    }
    auto item = std::move(queue.front());
    queue.pop_front();
    return item;
  }
  void cancel() noexcept override { ++cancel_calls; }
};

std::shared_ptr<owned_arrow_batch> owned_fixture(std::shared_ptr<arrow_fixture> const& fixture)
{
  return std::make_shared<owned_arrow_batch>(
    fixture->schema, fixture->array, fixture, *std::pmr::new_delete_resource());
}
}  // namespace

TEST_CASE("Row domain preserves file offsets and refuses holes", "[materialization][scan]")
{
  row_domain domain;
  domain.files = {{"generation-A", 180}, {"generation-B", 100}};
  domain.runs  = {{"a", 100, 0, 80, 0}, {"b", 0, 80, 3, 1}};
  REQUIRE_NOTHROW(domain.validate(83));
  domain.runs[1].batch_row_offset = 79;
  REQUIRE_THROWS(domain.validate(83));
  domain.runs.clear();
  domain.positions = std::vector<source_position>{{0, 105}, {1, 2}};
  REQUIRE_NOTHROW(domain.validate(2));
  domain.positions->at(1).row = 100;
  REQUIRE_THROWS(domain.validate(2));
}

TEST_CASE("Arrow layout rejects unsupported metadata and inconsistent buffers",
          "[materialization][scan]")
{
  int released = 0;
  arrow_fixture f(released);
  REQUIRE(validate_arrow_host_flat(f.schema, f.array).size() == 1);
  f.root_buffers[0] = &f.validity;
  REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
  f.child.null_count = 0;
  f.buffers[0]       = nullptr;
  REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
  f.buffers[0] = &f.validity;
  f.validity   = 0;
  REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
  f.validity         = 5;
  f.child.null_count = -1;
  REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
  f.child.n_buffers = 0;
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  f.child.n_buffers     = 2;
  f.child_schema.format = "tsu:UTC";
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  f.child_schema.format = "d:39,2";
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  __int128 decimal_values[3]{10, 20, 30};
  f.buffers[1]          = decimal_values;
  f.child_schema.format = "d:18,2";
  REQUIRE(validate_arrow_host_flat(f.schema, f.array)[0] ==
          cudf::data_type(cudf::type_id::DECIMAL128, -2));
  f.schema.metadata = "extension";
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
}

TEST_CASE("Arrow preflight checks selected null totals and numeric bounds",
          "[materialization][scan][arrow_migration]")
{
  int released = 0;
  arrow_fixture f(released);
  SECTION("known whole-column null total must match")
  {
    f.child.null_count = 2;
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  }
  SECTION("a slice cannot hide more nulls than the remaining rows")
  {
    f.array.offset     = 1;
    f.array.length     = 2;
    f.child.null_count = 3;
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
    f.child.null_count = 2;
    REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
    f.child.null_count = -1;
    REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
  }
  SECTION("non-nullable fields cannot contain selected nulls")
  {
    f.child_schema.flags = 0;
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  }
  SECTION("combined offsets are bounded before reading buffers")
  {
    f.child.offset = std::numeric_limits<cudf::size_type>::max();
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
    f.child.offset = 0;
    f.array.offset = std::numeric_limits<cudf::size_type>::max();
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  }
  SECTION("decimal payloads must fit their declared precision")
  {
    __int128 values[3]{999, 0, -999};
    f.child_schema.format = "d:3,2";
    f.buffers[1]          = values;
    REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
    values[0] = 1000;
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
    values[0] = -1000;
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
    values[0] = -(__int128{1} << 126);
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  }
  SECTION("absent bitmaps normalize both known-zero and unknown null counts")
  {
    f.root_buffers[0] = f.buffers[0] = nullptr;
    for (int root_nulls : {0, -1}) {
      for (int child_nulls : {0, -1}) {
        f.array.null_count = root_nulls;
        f.child.null_count = child_nulls;
        REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
      }
    }
    f.child.null_count = 1;
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
    f.child.null_count = 0;
    f.array.null_count = 1;
    REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  }
}

TEST_CASE("Arrow import owns its source through retirement and shared finish",
          "[materialization][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  int released = 0;
  auto f       = std::make_shared<arrow_fixture>(released);
  auto owner   = owned_fixture(f);
  materialization_program program{11, 7, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
  program.input_types = program.output_types;
  auto batch = import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref());
  REQUIRE(batch.rows == 3);
  REQUIRE(batch.table.view().column(0).null_count() == 1);
  owner.reset();
  f.reset();
  REQUIRE(released == 0);
  int projections = 0;
  materialization_steps steps;
  steps.projection = [&](materialized_batch&) { ++projections; };
  auto output      = finish_materialization(std::move(batch), program, steps);
  REQUIRE(projections == 1);
  REQUIRE(output.projection_complete);
  REQUIRE(lane.quiesce() == cudaSuccess);
  REQUIRE(released == 1);
}

TEST_CASE("Arrow stream keeps empty batches separate from EOS", "[materialization][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  int released            = 0;
  auto empty              = std::make_shared<arrow_fixture>(released);
  empty->array.length     = 0;
  empty->child.null_count = 0;
  auto second             = std::make_shared<arrow_fixture>(released);
  test_arrow_source source;
  source.queue.push_back({owned_fixture(empty), 0, 0, 0});
  source.queue.push_back({owned_fixture(second), 1, 3, sizeof(second->values)});
  materialization_program program{11, 7, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
  program.input_types = program.output_types;
  int received        = 0;
  auto result         = materialize_arrow_stream(source,
                                         program,
                                         lane,
                                         cudf::get_current_device_resource_ref(),
                                         [&](materialized_batch&& batch) {
                                           ++received;
                                           REQUIRE((batch.rows == 0 || batch.rows == 3));
                                         });
  REQUIRE(result.batches == 2);
  REQUIRE(result.rows == 3);
  REQUIRE(received == 2);
  REQUIRE(source.next_calls == 3);
  REQUIRE(source.cancel_calls == 0);
  REQUIRE(lane.quiesce() == cudaSuccess);
  REQUIRE(released == 2);
}

TEST_CASE("Arrow stream validates envelopes and cancels on import or source failure",
          "[materialization][scan]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  materialization_program program{11, 7, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
  program.input_types = program.output_types;

  SECTION("sequence and byte limits")
  {
    sirius::exec::retire_lane lane(stream.value());
    int released = 0;
    auto f       = std::make_shared<arrow_fixture>(released);
    test_arrow_source source;
    source.queue.push_back({owned_fixture(f), 1, 3, sizeof(f->values)});
    REQUIRE_THROWS(materialize_arrow_stream(
      source, program, lane, mr, [](materialized_batch&&) {}, arrow_stream_limits{3, 1}));
    REQUIRE(source.cancel_calls == 1);
    REQUIRE(lane.quiesce() == cudaSuccess);
    REQUIRE(released == 1);
  }

  SECTION("second batch schema failure")
  {
    sirius::exec::retire_lane lane(stream.value());
    int released             = 0;
    auto first               = std::make_shared<arrow_fixture>(released);
    auto bad                 = std::make_shared<arrow_fixture>(released);
    bad->child_schema.format = "l";
    test_arrow_source source;
    source.queue.push_back({owned_fixture(first), 0, 3, sizeof(first->values)});
    source.queue.push_back({owned_fixture(bad), 1, 3, sizeof(bad->values)});
    int received = 0;
    REQUIRE_THROWS(materialize_arrow_stream(
      source, program, lane, mr, [&](materialized_batch&&) { ++received; }));
    REQUIRE(received == 1);
    REQUIRE(source.cancel_calls == 1);
    REQUIRE(lane.quiesce() == cudaSuccess);
    REQUIRE(released == 2);
  }

  SECTION("source failure")
  {
    sirius::exec::retire_lane lane(stream.value());
    int released = 0;
    auto f       = std::make_shared<arrow_fixture>(released);
    test_arrow_source source;
    source.queue.push_back({owned_fixture(f), 0, 3, sizeof(f->values)});
    source.throw_when_empty = true;
    REQUIRE_THROWS(
      materialize_arrow_stream(source, program, lane, mr, [](materialized_batch&&) {}));
    REQUIRE(source.cancel_calls == 1);
    REQUIRE(lane.quiesce() == cudaSuccess);
    REQUIRE(released == 1);
  }

  SECTION("row envelope mismatch")
  {
    sirius::exec::retire_lane lane(stream.value());
    int released = 0;
    auto f       = std::make_shared<arrow_fixture>(released);
    test_arrow_source source;
    source.queue.push_back({owned_fixture(f), 0, 2, sizeof(f->values)});
    REQUIRE_THROWS(
      materialize_arrow_stream(source, program, lane, mr, [](materialized_batch&&) {}));
    REQUIRE(source.cancel_calls == 1);
    REQUIRE(lane.quiesce() == cudaSuccess);
    REQUIRE(released == 1);
  }

  SECTION("sink failure")
  {
    sirius::exec::retire_lane lane(stream.value());
    int released = 0;
    auto f       = std::make_shared<arrow_fixture>(released);
    test_arrow_source source;
    source.queue.push_back({owned_fixture(f), 0, 3, sizeof(f->values)});
    REQUIRE_THROWS(materialize_arrow_stream(
      source, program, lane, mr, [](materialized_batch&&) { throw std::runtime_error("sink"); }));
    REQUIRE(source.cancel_calls == 1);
    REQUIRE(lane.quiesce() == cudaSuccess);
    REQUIRE(released == 1);
  }
}

TEST_CASE("Direct GPU reader skips completed steps and retains zero column cardinality",
          "[materialization][scan]")
{
  rmm::cuda_stream stream;
  materialized_batch batch;
  batch.contract                = 31;
  batch.generation              = 2;
  batch.rows                    = 4;
  batch.logical_schema_complete = true;
  batch.visibility_complete     = true;
  batch.predicates              = {{0, sirius::op::predicate_stage::enforced, {}}};
  ensure_row_count_carrier(batch, stream, cudf::get_current_device_resource_ref());
  materialization_program program{31, 2, {}, 1, true};
  program.input_types = std::vector<cudf::data_type>{};
  materialization_steps steps;
  steps.mapping    = steps.visibility =
    steps.residual = [](materialized_batch&) { FAIL("completed work repeated"); };
  steps.projection = [](materialized_batch&) {};
  auto out         = finish_materialization(std::move(batch), program, steps);
  REQUIRE(out.table.num_rows() == 4);
  out.generation = 3;
  REQUIRE_THROWS(validate_materialized_batch(out, program));
  stream.synchronize();
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Real Parquet ID mapping widens reorders and fills explicit null",
                 "[materialization][scan][integration]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(constant(11, 3, stream));
  columns.push_back(constant(22, 3, stream));
  cudf::table table(std::move(columns));
  cudf::io::table_input_metadata metadata(table.view());
  metadata.column_metadata[0].set_name("old.dot").set_parquet_field_id(4);
  metadata.column_metadata[1].set_name("right").set_parquet_field_id(9);
  sirius::test::scratch_dir dir("materialization_field_ids");
  auto path   = (dir.path() / "data.parquet").string();
  auto writer = cudf::io::parquet_writer_options::builder(cudf::io::sink_info(path), table.view())
                  .metadata(std::move(metadata))
                  .build();
  cudf::io::write_parquet(writer, stream);
  stream.synchronize();
  auto state    = sirius::test::get_registered_sirius_context(*con);
  auto source   = state->get_scan_manager().create_datasource(path);
  auto options  = cudf::io::parquet_reader_options::builder().build();
  auto resolved = resolve_parquet_metadata_with_identity(*source, 77, path, options);
  materialization_program program{77,
                                  1,
                                  {cudf::data_type{cudf::type_id::INT64},
                                   cudf::data_type{cudf::type_id::INT32},
                                   cudf::data_type{cudf::type_id::INT32}},
                                  0,
                                  false};
  program.input_types = program.output_types;
  std::vector<field_requirement> fields{
    {9, duckdb::LogicalType::BIGINT, {}, true},
    {4, duckdb::LogicalType::INTEGER, {}, false},
    {12, duckdb::LogicalType::INTEGER, duckdb::Value(duckdb::LogicalType::INTEGER), false}};
  auto mapping = field_id_mapping::compile(resolved.artifact, resolved.identity, program, fields);
  REQUIRE(mapping.additional_bytes(3) >= 3 * (sizeof(int64_t) + sizeof(int32_t)));
  parquet_source input{source, resolved.artifact->file_metadata(), {0}};
  bound_table_scan contract;
  contract.contract_id = 77;
  contract.profiles    = std::make_shared<physical_profile_table>();
  std::vector<std::size_t> groups{0};
  auto batch = materialize_mapped_parquet(input, mapping, contract, groups, stream, mr);
  REQUIRE(first_int64(batch.table.view().column(0)) == 22);
  REQUIRE(batch.table.view().column(2).null_count() == 3);
  materialization_steps steps;
  steps.projection = [](materialized_batch&) {};
  REQUIRE(finish_materialization(std::move(batch), program, steps).rows == 3);
  auto second_path = dir.file("second.parquet");
  std::filesystem::copy_file(path, second_path);
  auto second_source = state->get_scan_manager().create_datasource(second_path);
  auto second_resolved =
    resolve_parquet_metadata_with_identity(*second_source, 77, second_path, options);
  auto second_mapping =
    field_id_mapping::compile(second_resolved.artifact, second_resolved.identity, program, fields);
  REQUIRE(second_mapping.identity() != mapping.identity());
  REQUIRE(mapping.compatible_with(second_mapping));
  std::vector<mapped_parquet_unit> units{
    {input, &mapping},
    {{second_source, second_resolved.artifact->file_metadata(), {0}}, &second_mapping}};
  auto coalesced = materialize_mapped_parquet_runs(units, contract, stream, mr);
  REQUIRE(coalesced.size() == 1);
  REQUIRE(coalesced[0].rows == 6);
  REQUIRE(coalesced[0].lineage.files.size() == 2);
  REQUIRE(coalesced[0].lineage.runs[1].batch_row_offset == 3);
  REQUIRE(first_int64(coalesced[0].table.view().column(0)) == 22);
  auto defaults             = fields;
  defaults[2].missing_value = duckdb::Value::INTEGER(99);
  auto different            = field_id_mapping::compile(
    second_resolved.artifact, second_resolved.identity, program, defaults);
  REQUIRE_FALSE(mapping.compatible_with(different));
  units[1].mapping = &different;
  auto flushed     = materialize_mapped_parquet_runs(units, contract, stream, mr);
  REQUIRE(flushed.size() == 2);
  REQUIRE(flushed[0].table.view().column(2).null_count() == 3);
  REQUIRE(flushed[1].table.view().column(2).null_count() == 0);
  int32_t default_value = 0;
  REQUIRE(cudaMemcpy(&default_value,
                     flushed[1].table.view().column(2).head<int32_t>(),
                     sizeof(default_value),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(default_value == 99);
  units[1]      = units[0];  // repeated files represent repeated rows, not deduplication
  auto repeated = materialize_mapped_parquet_runs(units, contract, stream, mr);
  REQUIRE(repeated.size() == 1);
  REQUIRE(repeated[0].rows == 6);
  auto floating_fields             = fields;
  floating_fields[2].type          = duckdb::LogicalType::DOUBLE;
  floating_fields[2].missing_value = duckdb::Value::DOUBLE(0.0);
  auto floating_program            = program;
  floating_program.output_types[2] = cudf::data_type{cudf::type_id::FLOAT64};
  floating_program.input_types     = floating_program.output_types;
  auto positive_zero               = field_id_mapping::compile(
    resolved.artifact, resolved.identity, floating_program, floating_fields);
  floating_fields[2].missing_value = duckdb::Value::DOUBLE(-0.0);
  auto negative_zero               = field_id_mapping::compile(
    resolved.artifact, resolved.identity, floating_program, floating_fields);
  REQUIRE_FALSE(positive_zero.compatible_with(negative_zero));
  fields[0].allow_int32_to_int64 = false;
  REQUIRE_THROWS(field_id_mapping::compile(resolved.artifact, resolved.identity, program, fields));
  REQUIRE_THROWS(field_id_mapping::compile(resolved.artifact, "other-version", program, fields));
  contract.contract_id = 78;
  REQUIRE_THROWS(materialize_mapped_parquet(input, mapping, contract, groups, stream, mr));
}

TEST_CASE("Arrow nullable slice retains values and rejects schema drift", "[materialization][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  int released    = 0;
  auto f          = std::make_shared<arrow_fixture>(released);
  f->array.offset = 1;
  f->array.length = 2;
  auto owner      = owned_fixture(f);
  materialization_program program{42, 1, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
  program.input_types = std::vector{cudf::data_type{cudf::type_id::INT64}};
  REQUIRE_THROWS(import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref()));
  program.input_types = program.output_types;
  auto batch = import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref());
  REQUIRE(batch.rows == 2);
  REQUIRE(batch.table.view().column(0).null_count() == 1);
  REQUIRE(lane.quiesce() == cudaSuccess);
  int32_t values[2];
  REQUIRE(cudaMemcpy(values,
                     batch.table.view().column(0).head<int32_t>(),
                     sizeof(values),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(values[1] == 30);
}

TEST_CASE("Arrow all-valid slices ignore absent or unused bitmaps", "[materialization][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  for (bool bitmap : {false, true}) {
    for (bool sliced : {false, true}) {
      int released        = 0;
      auto f              = std::make_shared<arrow_fixture>(released);
      f->child.null_count = 0;
      f->validity         = 0;
      f->buffers[0]       = bitmap ? &f->validity : nullptr;
      f->root_buffers[0]  = &f->validity;
      f->array.offset     = sliced ? 1 : 0;
      f->array.length     = sliced ? 2 : 3;
      auto owner          = owned_fixture(f);
      materialization_program program{42, 1, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
      program.input_types = program.output_types;
      auto batch = import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref());
      REQUIRE(lane.quiesce() == cudaSuccess);
      REQUIRE(batch.table.view().column(0).null_count() == 0);
      int32_t values[3]{};
      REQUIRE(cudaMemcpy(values,
                         batch.table.view().column(0).head<int32_t>(),
                         batch.rows * sizeof(int32_t),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(values[0] == (sliced ? 20 : 10));
      REQUIRE(values[batch.rows - 1] == 30);
    }
  }
}

TEST_CASE("Arrow metadata uses separate bounds and rejects every truncated prefix",
          "[materialization][scan]")
{
  int released = 0;
  arrow_fixture f(released);
  std::string metadata;
  auto integer = [&](int32_t n) { metadata.append(reinterpret_cast<char const*>(&n), sizeof(n)); };
  integer(1);
  integer(1);
  metadata += 'k';
  integer(3);
  metadata += "abc";
  std::string root(4, '\0');
  f.schema.metadata       = root.data();
  f.child_schema.metadata = metadata.data();
  arrow_metadata_lengths bounds{{root.data(), root.size()}, {metadata.data(), metadata.size()}};
  REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array, bounds));
  for (std::size_t bytes = 0; bytes < metadata.size(); ++bytes) {
    INFO("metadata bytes: " << bytes);
    bounds[metadata.data()] = bytes;
    REQUIRE_THROWS_AS(validate_arrow_host_flat(f.schema, f.array, bounds), std::invalid_argument);
  }
  bounds.erase(metadata.data());
  REQUIRE_THROWS_AS(validate_arrow_host_flat(f.schema, f.array, bounds), std::invalid_argument);
  f.array.null_count = -1;
  f.root_buffers[0]  = &f.validity;
  f.schema.metadata = f.child_schema.metadata = nullptr;
  REQUIRE_THROWS_AS(validate_arrow_host_flat(f.schema, f.array), std::invalid_argument);
}

TEST_CASE("Arrow decimal imports normalize the approved carrier", "[materialization][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  for (auto format : {"d:9,2", "d:18,2", "d:38,2"}) {
    int released = 0;
    auto f       = std::make_shared<arrow_fixture>(released);
    __int128 values[3]{123, 456, 789};
    f->child_schema.format = format;
    f->buffers[1]          = values;
    auto owner             = owned_fixture(f);
    materialization_program program{
      43, 1, {cudf::data_type{cudf::type_id::DECIMAL128, -2}}, 0, false};
    program.input_types        = program.output_types;
    program.decimal_precisions = {1};
    REQUIRE_THROWS(
      import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref()));
    program.decimal_precisions = {std::stoi(std::string(format).substr(2))};
    auto batch = import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref());
    REQUIRE(batch.table.view().column(0).type() == program.output_types[0]);
    REQUIRE(lane.quiesce() == cudaSuccess);
    __int128 first = 0;
    REQUIRE(cudaMemcpy(
              &first, batch.table.view().column(0).head(), sizeof(first), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(static_cast<int64_t>(first) == 123);
  }
}

TEST_CASE("Arrow empty and zero-column batches preserve cardinality", "[materialization][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  for (int rows : {0, 3}) {
    int released         = 0;
    auto f               = std::make_shared<arrow_fixture>(released);
    f->schema.n_children = f->array.n_children = 0;
    f->array.length                            = rows;
    auto owner                                 = owned_fixture(f);
    materialization_program program{44, 1, {}, 0, false};
    program.input_types = std::vector<cudf::data_type>{};
    auto batch = import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref());
    REQUIRE(batch.rows == rows);
    REQUIRE(batch.table.num_rows() == rows);
    REQUIRE(lane.quiesce() == cudaSuccess);
  }
}

TEST_CASE("Explicit GPU positions feed shared visibility before residual",
          "[materialization][scan]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(constant(7, 3, stream));
  materialized_batch batch;
  batch.table                   = owning_table_view(std::make_unique<cudf::table>(std::move(cols)));
  batch.contract                = 45;
  batch.generation              = 1;
  batch.rows                    = 3;
  batch.logical_schema_complete = true;
  batch.lineage.files           = {{"file-v1", 100}};
  batch.lineage.positions       = std::vector<source_position>{{0, 10}, {0, 70}, {0, 30}};
  materialization_program program{45, 1, {cudf::data_type{cudf::type_id::INT32}}, 1, true};
  program.input_types = program.output_types;
  std::vector<int64_t> deleted{30, 70};
  materialization_steps steps;
  steps.visibility = [&](materialized_batch& b) {
    apply_positional_visibility(
      b,
      [&](std::string const& identity) -> std::span<int64_t const> {
        REQUIRE(identity == "file-v1");
        return deleted;
      },
      stream,
      mr);
  };
  steps.residual   = [&](materialized_batch& b) { REQUIRE(b.rows == 1); };
  steps.projection = [](materialized_batch&) {};
  auto result      = finish_materialization(std::move(batch), program, steps);
  REQUIRE(result.rows == 1);
  REQUIRE(result.visibility_complete);
  REQUIRE(result.predicates[0].stage == sirius::op::predicate_stage::enforced);
  stream.synchronize();
}

TEST_CASE("Arrow imports every flat primitive with nulls", "[materialization][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  for (auto format : {"b", "c", "s", "i", "l", "C", "S", "I", "L", "f", "g", "tdD", "tsu:", "u"}) {
    INFO(format);
    int released = 0;
    auto f       = std::make_shared<arrow_fixture>(released);
    alignas(16) uint8_t data[48]{};
    int32_t offsets[]{0, 1, 1, 3};
    void const* buffers[]{&f->validity, data, "abc"};
    f->child_schema.format = format;
    f->child.buffers       = buffers;
    f->child.null_count    = -1;
    if (std::string_view(format) == "u") {
      buffers[1]         = offsets;
      f->child.n_buffers = 3;
    }
    auto types = validate_arrow_host_flat(f->schema, f->array);
    auto owner = owned_fixture(f);
    materialization_program program{46, 1, types, 0, false};
    program.input_types = types;
    auto batch = import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref());
    REQUIRE(batch.rows == 3);
    REQUIRE(batch.table.view().column(0).type() == types[0]);
    REQUIRE(batch.table.view().column(0).null_count() == 1);
    REQUIRE(lane.quiesce() == cudaSuccess);
  }
}

TEST_CASE("Arrow rejects nested dictionary extensions and invalid string offsets",
          "[materialization][scan]")
{
  int released = 0;
  arrow_fixture f(released);
  SECTION("nested") { f.child_schema.n_children = 1; }
  SECTION("dictionary") { f.child_schema.dictionary = &f.child_schema; }
  SECTION("timezone") { f.child_schema.format = "tsu:UTC"; }
  SECTION("large string") { f.child_schema.format = "U"; }
  SECTION("binary") { f.child_schema.format = "z"; }
  SECTION("missing mask") { f.buffers[0] = nullptr; }
  SECTION("negative offset") { f.child.offset = -1; }
  SECTION("short child") { f.child.length = 2; }
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
}

TEST_CASE("Arrow metadata and UTF8 offsets are qualified before import", "[materialization][scan]")
{
  int released = 0;
  arrow_fixture f(released);
  std::string metadata;
  auto integer = [&](int32_t n) { metadata.append(reinterpret_cast<char*>(&n), sizeof(n)); };
  auto entry   = [&](std::string const& key, std::string const& value) {
    integer(static_cast<int32_t>(key.size()));
    metadata += key;
    integer(static_cast<int32_t>(value.size()));
    metadata += value;
  };
  integer(1);
  entry("description", "plain column");
  f.child_schema.metadata = metadata.data();
  REQUIRE_NOTHROW(
    validate_arrow_host_flat(f.schema, f.array, {{metadata.data(), metadata.size()}}));
  metadata.clear();
  integer(1);
  entry("ARROW:extension:name", "opaque");
  f.child_schema.metadata = metadata.data();
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array, {{metadata.data(), metadata.size()}}));
  f.child_schema.metadata = metadata.data();
  REQUIRE_THROWS(
    validate_arrow_host_flat(f.schema, f.array, {{metadata.data(), metadata.size() - 1}}));
  metadata.assign(1, '\x01');
  f.child_schema.metadata = metadata.data();
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array, {{metadata.data(), metadata.size()}}));
  int32_t entries    = 1;
  int32_t key_length = 100;
  metadata.clear();
  metadata.append(reinterpret_cast<char*>(&entries), sizeof(entries));
  metadata.append(reinterpret_cast<char*>(&key_length), sizeof(key_length));
  f.child_schema.metadata = metadata.data();
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array, {{metadata.data(), metadata.size()}}));
  metadata.clear();
  integer(1);
  integer(1);
  metadata.push_back('k');
  int32_t value_length = 100;
  integer(value_length);
  f.child_schema.metadata = metadata.data();
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array, {{metadata.data(), metadata.size()}}));
  f.child_schema.metadata = nullptr;
  int32_t offsets[]{0, 2, 1, 3};
  void const* buffers[]{&f.validity, offsets, "abc"};
  f.child_schema.format = "u";
  f.child.n_buffers     = 3;
  f.child.buffers       = buffers;
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  offsets[2] = 2;
  REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
  buffers[2] = nullptr;
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
}

TEST_CASE("Retiring an abandoned Arrow import releases its owner exactly once",
          "[materialization][scan]")
{
  rmm::cuda_stream stream;
  int released = 0;
  {
    sirius::exec::retire_lane lane(stream.value());
    auto f     = std::make_shared<arrow_fixture>(released);
    auto owner = owned_fixture(f);
    materialization_program program{47, 1, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
    program.input_types = program.output_types;
    auto batch = import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref());
    owner.reset();
    f.reset();
    REQUIRE(released == 0);
    // Cancellation stops consumers. Destroy the output, then let lane teardown
    // wait for the pending import before releasing its independent source owner.
  }
  REQUIRE(released == 1);
}

TEST_CASE("Arrow query drain retires cancelled batches and returns their credit",
          "[materialization][scan]")
{
  struct credit_token {
    int* returns;
    explicit credit_token(int* target) : returns(target) {}
    ~credit_token() { ++*returns; }
  };

  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  int released  = 0;
  int credits   = 0;
  auto fixture  = std::make_shared<arrow_fixture>(released);
  auto capacity = std::make_shared<credit_token>(&credits);
  // The admission owner must retain producer storage, not just the accounting token.
  struct admitted_input {
    std::shared_ptr<arrow_fixture> fixture;
    std::shared_ptr<credit_token> credit;
  };
  auto admitted = std::make_shared<admitted_input>(admitted_input{fixture, capacity});
  test_arrow_source source;
  source.queue.push_back(
    {std::make_shared<owned_arrow_batch>(
       fixture->schema, fixture->array, admitted, *std::pmr::new_delete_resource()),
     0,
     3,
     12});

  materialization_program program{49, 1, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
  program.input_types = program.output_types;
  REQUIRE_THROWS(materialize_arrow_stream(
    source, program, lane, cudf::get_current_device_resource_ref(), [](materialized_batch&&) {
      throw std::runtime_error("query sink cancelled");
    }));
  REQUIRE(source.cancel_calls == 1);
  fixture.reset();
  capacity.reset();
  admitted.reset();
  // Cancellation stops production, but cannot release an owner submitted to the GPU lane.
  CHECK(released == 0);
  CHECK(credits == 0);

  // The query-level drain is the last consumer barrier. It must retire the import owner and
  // return the producer reservation exactly once after the submitted work is complete.
  REQUIRE(lane.quiesce() == cudaSuccess);
  CHECK(released == 1);
  CHECK(credits == 1);
}

TEST_CASE("Finish refuses unproved visibility and predicate reports", "[materialization][scan]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(constant(9, 2, stream));
  materialized_batch batch;
  batch.table                   = owning_table_view(std::make_unique<cudf::table>(std::move(cols)));
  batch.rows                    = 2;
  batch.contract                = 48;
  batch.generation              = 1;
  batch.logical_schema_complete = true;
  materialization_program program{48, 1, {cudf::data_type{cudf::type_id::INT32}}, 1, true};
  program.input_types = program.output_types;
  SECTION("wrong answer dtype")
  {
    batch.predicates = {{0, sirius::op::predicate_stage::answer, 0}};
    REQUIRE_THROWS(validate_materialized_batch(batch, program));
  }
  SECTION("duplicate progress")
  {
    batch.predicates = {{0, sirius::op::predicate_stage::enforced, {}},
                        {0, sirius::op::predicate_stage::enforced, {}}};
    REQUIRE_THROWS(validate_materialized_batch(batch, program));
  }
  SECTION("unknown progress")
  {
    batch.predicates = {{1, sirius::op::predicate_stage::enforced, {}}};
    REQUIRE_THROWS(validate_materialized_batch(batch, program));
  }
  SECTION("no positions")
  {
    materialization_steps steps;
    steps.visibility = [](materialized_batch&) { FAIL("unproved lineage reached visibility"); };
    REQUIRE_THROWS(finish_materialization(std::move(batch), program, steps));
  }
  SECTION("missing residual")
  {
    batch.visibility_complete = true;
    REQUIRE_THROWS(finish_materialization(std::move(batch), program, {}));
  }
  stream.synchronize();
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Empty evolved file preserves its schema and completion split",
                 "[materialization][scan][integration]")
{
  sirius::test::scratch_dir directory("materialization_empty_evolved");
  auto path    = directory.file("empty.parquet");
  auto written = con->Query("COPY (SELECT 1::INTEGER old_id WHERE false) TO " +
                            directory.file_literal("empty.parquet") +
                            " (FORMAT PARQUET, FIELD_IDS {old_id: 1})");
  REQUIRE(written);
  REQUIRE_FALSE(written->HasError());
  auto info                    = std::make_unique<parquet_ingestible_table_info>();
  info->contract_id            = 91;
  info->resolved_file_paths    = {path};
  info->names                  = {"renamed"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::BIGINT)};
  info->column_ids             = {duckdb::ColumnIndex(0)};
  info->projected_output_count = 1;
  info->physical_schema        = iceberg_table_schema{{{"renamed", 1, "BIGINT"}}};
  info->field_mapping_requirements =
    std::vector<field_requirement>{{1, duckdb::LogicalType::BIGINT, {}, true}};
  auto ingestible = make_ingestible(std::move(info));
  auto ioctx      = sirius::test::make_test_local_ioctx();
  auto prepare    = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> { return ioctx; });
  auto prepared    = prepare({});
  auto const& file = dynamic_cast<parquet_file_scan_info const&>(*prepared);
  REQUIRE(file.field_mapping);
  REQUIRE(file.row_groups.empty());
  auto coalescer = ingestible->create_batch_coalescer();
  REQUIRE(coalescer->push(std::move(prepared)).empty());
  auto splits = coalescer->flush();
  REQUIRE(splits.size() == 1);
  auto const& split = dynamic_cast<parquet_split_info const&>(*splits[0]);
  REQUIRE(split.field_mapping);
  REQUIRE(split.rg_slices.size() == 1);
  auto const& slice = split.rg_slices[0];
  bound_table_scan contract;
  contract.contract_id = 91;
  contract.profiles    = std::make_shared<physical_profile_table>();
  rmm::cuda_stream stream;
  auto batch = materialize_mapped_parquet({slice.datasource, slice.file_metadata, {}},
                                          *split.field_mapping,
                                          contract,
                                          {},
                                          stream,
                                          cudf::get_current_device_resource_ref());
  REQUIRE(batch.rows == 0);
  REQUIRE(batch.table.view().num_columns() == 1);
  CHECK(batch.table.view().column(0).type().id() == cudf::type_id::INT64);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Adapter mapping hook runs without an Iceberg physical schema",
                 "[materialization][scan][mapping][adapter]")
{
  sirius::test::scratch_dir directory("materialization_adapter_mapping");
  auto path = directory.file("input.parquet");
  auto written =
    con->Query("COPY (SELECT 7::INTEGER old_id) TO " + directory.file_literal("input.parquet") +
               " (FORMAT PARQUET, FIELD_IDS {old_id: 1})");
  REQUIRE(written);
  REQUIRE_FALSE(written->HasError());
  auto info                    = std::make_unique<parquet_ingestible_table_info>();
  info->contract_id            = 92;
  info->resolved_file_paths    = {path};
  info->names                  = {"renamed"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::BIGINT)};
  info->column_ids             = {duckdb::ColumnIndex(0)};
  info->projected_output_count = 1;
  info->field_mapping_requirements =
    std::vector<field_requirement>{{1, duckdb::LogicalType::BIGINT, {}, true}};
  info->prepare_field_mapping = [](std::shared_ptr<parquet_metadata const> metadata,
                                   std::string const& identity,
                                   materialization_program program,
                                   std::vector<field_requirement> fields) {
    return std::make_shared<field_id_mapping const>(field_id_mapping::compile(
      std::move(metadata), identity, std::move(program), std::move(fields)));
  };
  auto ingestible = make_ingestible(std::move(info));
  auto ioctx      = sirius::test::make_test_local_ioctx();
  auto prepare    = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> { return ioctx; });
  auto prepared = prepare({});
  REQUIRE(prepared);
  auto const& file = dynamic_cast<parquet_file_scan_info const&>(*prepared);
  REQUIRE(file.field_mapping);
  CHECK(file.field_mapping->read_names() == std::vector<std::string>{"old_id"});
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Directly bound virtual multi-column predicates stay out of footer AST",
                 "[materialization][scan][virtual_columns][integration]")
{
  auto const sibling = GENERATE(0, 1, 2);
  CAPTURE(sibling);
  sirius::test::scratch_dir directory("materialization_virtual_filter");
  auto path = directory.file("input.parquet");
  {
    sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
    auto written = con->Query("COPY (SELECT i::INTEGER id FROM range(1,6145) t(i) ORDER BY i) TO " +
                              directory.file_literal("input.parquet") +
                              " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    REQUIRE(written);
    REQUIRE_FALSE(written->HasError());
  }
  auto info                    = std::make_unique<parquet_ingestible_table_info>();
  auto const virtual_id        = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  info->resolved_file_paths    = {path};
  info->names                  = {"id"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(virtual_id)};
  info->projected_output_count = 2;
  info->approximate_batch_size = std::size_t{1} << 30;
  info->virtual_columns        = {{virtual_id,
                                   "file_row_number",
                                   sirius::logical_type::make(sirius::type_id::BIGINT),
                                   scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER}};
  auto filters                 = duckdb::make_uniq<duckdb::TableFilterSet>();
  // (file_row_number >= 5000 OR id <= 2): the physical branch must not be
  // extracted from this OR for pruning. Binding is constructed without an optimizer.
  auto predicate =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_OR);
  predicate->GetChildrenMutable().push_back(duckdb::BoundComparisonExpression::Create(
    duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0),
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::BIGINT(5000))));
  predicate->GetChildrenMutable().push_back(duckdb::BoundComparisonExpression::Create(
    duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 1),
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(2))));
  filters->PushMultiColumnFilter(duckdb::make_uniq<duckdb::ExpressionFilter>(
    std::move(predicate),
    duckdb::vector<duckdb::ProjectionIndex>{duckdb::ProjectionIndex(1),
                                            duckdb::ProjectionIndex(0)}));
  if (sibling == 1)
    filters->PushFilter(
      duckdb::ProjectionIndex(0),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                                    duckdb::Value::INTEGER(4097)));
  if (sibling == 2)
    filters->PushFilter(duckdb::ProjectionIndex(1),
                        sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                                      duckdb::Value::BIGINT(6000)));
  info->table_filters = std::move(filters);
  auto ingestible     = make_ingestible(std::move(info));
  auto ioctx          = sirius::test::make_test_local_ioctx();
  auto prepare        = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> { return ioctx; });
  REQUIRE(prepare);
  auto prepared    = prepare({});  // includes the footer AST name-resolution path
  auto const& file = dynamic_cast<parquet_file_scan_info const&>(*prepared);
  REQUIRE(file.row_groups.size() == (sibling == 1 ? 1 : 3));
  if (sibling == 1) CHECK(file.row_groups.front().index == 2);
  auto coalescer = ingestible->create_batch_coalescer();
  auto splits    = coalescer->push(std::move(prepared));
  auto tail      = coalescer->flush();
  for (auto& split : tail)
    splits.push_back(std::move(split));
  REQUIRE(splits.size() == 1);
  auto state  = sirius::test::get_registered_sirius_context(*con);
  auto* space = state->get_memory_manager().get_memory_space(
    cucascade::memory::Tier::GPU, rmm::get_current_cuda_device().value());
  REQUIRE(space);
  rmm::cuda_stream stream;
  auto batch =
    ingestible->materialize_metadata_to_table(*splits.front(), *space, stream, false, nullptr);
  REQUIRE(batch.state != filter_state::ROW_FILTERED_AND_PROJECTED);
  auto output        = ingestible->post_filter_and_project(std::move(batch), *space, stream);
  auto expected_rows = sibling == 1 ? 1144 : sibling == 2 ? 1002 : 1146;
  REQUIRE(output->num_rows() == expected_rows);
  REQUIRE(output->num_columns() == 2);
  std::vector<int32_t> ids(expected_rows);
  std::vector<int64_t> rownums(expected_rows);
  REQUIRE(cudaMemcpyAsync(ids.data(),
                          output->view().column(0).head<int32_t>(),
                          ids.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(rownums.data(),
                          output->view().column(1).head<int64_t>(),
                          rownums.size() * sizeof(int64_t),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  for (int i = 0; i < expected_rows; ++i) {
    auto expected_id = sibling == 1 ? 5001 + i : i < 2 ? i + 1 : 4999 + i;
    CHECK(ids[i] == expected_id);
    CHECK(rownums[i] == expected_id - 1);
  }
}

namespace {
// Exercise the production Iceberg wrapper with failures from an actual datasource,
// after the independent Parquet footer has been prepared successfully.
class failing_dv_context final : public cucascade::io::ioctx {
 public:
  enum class failure { corrupt, reader_io, object_changed, descriptor, cancelled, resource };
  explicit failing_dv_context(failure mode) : mode_(mode) {}
  ~failing_dv_context() override { pre_destroy(); }
  cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::uring;
  }
  void shutdown() noexcept override {}
  bool supports(std::string_view) const noexcept override { return true; }
  bool supports_device_read() const noexcept override { return false; }
  bool supports_host_to_device_read() const noexcept override { return false; }
  bool supports_vector_host_read() const noexcept override { return false; }
  bool supports_device_range_read() const noexcept override { return false; }
  std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges,
    std::optional<size_t>) const noexcept override
  {
    return {ranges.begin(), ranges.end()};
  }
  size_t host_read_io(cucascade::io::io_object const& object,
                      size_t,
                      size_t bytes,
                      uint8_t* dst) override
  {
    switch (mode_) {
      case failure::corrupt: std::fill_n(dst, bytes, 0); return bytes;
      case failure::reader_io:
        throw sirius::transparent::classified_execution_error(
          sirius::transparent::late_failure_cause::reader_io, "DV read failed");
      case failure::object_changed:
        throw cucascade::io::object_changed_error(object.object_path(), "v1", "v2");
      case failure::cancelled:
        throw std::system_error(std::make_error_code(std::errc::operation_canceled));
      case failure::resource: throw std::bad_alloc();
      case failure::descriptor: throw std::logic_error("descriptor failure should occur at open");
    }
    return 0;
  }
  cucascade::exec::semi_future<size_t> host_read_async_io(
    cucascade::io::io_object const& object,
    size_t offset,
    size_t bytes,
    uint8_t* dst,
    cucascade::io::io_priority) noexcept override
  {
    try {
      return cucascade::exec::make_semi_future<size_t>(host_read_io(object, offset, bytes, dst));
    } catch (...) {
      return cucascade::exec::make_semi_future<size_t>(std::current_exception());
    }
  }
  cucascade::exec::semi_future<size_t> mixed_readv_async_io(
    cucascade::io::io_object const&,
    std::vector<cucascade::io::prepared_io_slice>&&) noexcept override
  {
    return cucascade::exec::make_semi_future<size_t>(
      std::make_exception_ptr(std::logic_error("unexpected vectored DV read")));
  }

 protected:
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string path) override
  {
    if (mode_ == failure::descriptor)
      throw cucascade::io::descriptor_mismatch_error("DV descriptor changed");
    struct object final : cucascade::io::io_object {
      std::string path;
      explicit object(std::string name) : path(std::move(name)) {}
      std::string const& raw_file_cache_id() const noexcept override { return path; }
      std::string const& object_path() const noexcept override { return path; }
      size_t size() const noexcept override { return 400; }
    };
    return std::make_shared<object>(std::move(path));
  }

 private:
  failure mode_;
};
}  // namespace

TEST_CASE("Iceberg Puffin wrapper retains failure diagnostics", "[materialization][scan][iceberg]")
{
  using namespace sirius::scan_manager;
  using failure = failing_dv_context::failure;
  using cause   = sirius::transparent::late_failure_cause;
  auto mode     = GENERATE(failure::corrupt,
                       failure::reader_io,
                       failure::object_changed,
                       failure::descriptor,
                       failure::cancelled,
                       failure::resource);
  CAPTURE(static_cast<int>(mode));
  auto path =
    std::filesystem::absolute("test/cpp/integration/data/parquet/nation.parquet").string();
  auto dvpath = path + ".puffin";
  test::test_reservation_provider provider;
  provider.grant_bytes = 64 * 1024;
  auto ledger          = std::make_shared<preparation_ledger>(provider);
  auto discovery       = std::make_shared<iceberg_delete_discovery>();
  IcebergDeleteFileEntry dv;
  dv.file_path             = dvpath;
  dv.file_format           = "puffin";
  dv.referenced_data_file  = path;
  dv.content_offset        = 4;
  dv.content_size_in_bytes = 44;
  dv.file_size_in_bytes    = 400;
  dv.record_count          = 2;
  discovery->deletion_vector_entries.push_back(std::move(dv));
  std::vector<std::string> paths{path};
  auto envelope = iceberg_dv_preparation::envelope(*discovery, paths);
  REQUIRE(ledger->admit(std::array{envelope}, memory_space_id(cucascade::memory::Tier::HOST, 0))
            .deferred);
  auto views    = std::make_shared<sirius::transparent::read_view_registry>();
  auto contract = allocate_scan_contract(
    *views, {}, 0, 1, nullptr, {}, {}, {materializer_kind::iceberg, "iceberg.v1"});
  auto plan =
    std::make_shared<iceberg_dv_preparation>(contract, "table", paths, discovery, *ledger);
  auto info                    = std::make_unique<iceberg_ingestible_table_info>();
  info->contract_id            = contract;
  info->read_views             = views;
  info->table_path             = "table";
  info->resolved_file_paths    = paths;
  info->names                  = {"n_nationkey", "n_name", "n_regionkey", "n_comment"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::INTEGER),
                                  sirius::logical_type::make(sirius::type_id::VARCHAR),
                                  sirius::logical_type::make(sirius::type_id::INTEGER),
                                  sirius::logical_type::make(sirius::type_id::VARCHAR)};
  info->column_ids             = {duckdb::ColumnIndex(0)};
  info->projected_output_count = 1;
  info->deferred               = plan;
  info->profiles               = std::make_shared<physical_profile_table>();
  auto counters                = std::make_shared<physical_check_counters>();
  info->profiles->counters     = counters;
  auto ingestible              = make_ingestible(std::move(info));
  ingestible->attach_preparation_ledger(ledger);
  auto dataio  = sirius::test::make_test_local_ioctx();
  auto dvio    = std::make_shared<failing_dv_context>(mode);
  auto claimed = ingestible->try_claim_split(
    [&](std::string_view name) -> std::shared_ptr<cucascade::io::ioctx> {
      return name == dvpath ? std::static_pointer_cast<cucascade::io::ioctx>(dvio) : dataio;
    });
  REQUIRE(std::holds_alternative<preparation_job>(claimed));
  auto work = std::move(std::get<preparation_job>(claimed));
  work.unit = std::make_shared<preparation_unit>(work.requirement->key, work.requirement->required);
  try {
    (void)work.run({});
    FAIL("DV failure was not reported");
  } catch (unsupported_physical_input const& error) {
    REQUIRE(mode == failure::corrupt);
    CHECK(error.reason == verdict_reason::iceberg_delete_corrupt);
    CHECK(std::string(error.what()).find("bad leading magic") != std::string::npos);
  } catch (preparation_resource_error const& error) {
    REQUIRE(mode == failure::resource);
    CHECK(error.cause == cause::resource);
    CHECK(std::string(error.what()) == "DV preparation control allocation failed");
  } catch (sirius::transparent::classified_execution_error const& error) {
    REQUIRE((mode == failure::reader_io || mode == failure::object_changed ||
             mode == failure::descriptor));
    CHECK(error.cause == cause::reader_io);
    auto expected = mode == failure::object_changed
                      ? std::string(cucascade::io::object_changed_error(dvpath, "v1", "v2").what())
                    : mode == failure::descriptor ? "DV descriptor changed"
                                                  : "DV read failed";
    CHECK(std::string(error.what()) == expected);
  } catch (std::system_error const& error) {
    REQUIRE(mode == failure::cancelled);
    CHECK(error.code() == std::errc::operation_canceled);
  }
  CHECK(counters->iceberg_delete_payload_loads == 1);
  CHECK(counters->rejections[static_cast<size_t>(verdict_reason::iceberg_delete_corrupt)] ==
        (mode == failure::corrupt ? 1 : 0));
  CHECK(counters->iceberg_delete_corrupt_inputs == (mode == failure::corrupt ? 1 : 0));
  // The open-time descriptor failure passes through the outer classified-error catch;
  // object-changed conversion is thrown from that catch chain and counted only as identity.
  CHECK(counters->iceberg_delete_reader_io_failures ==
        (mode == failure::reader_io || mode == failure::descriptor ? 1 : 0));
  CHECK(counters->iceberg_delete_identity_failures == (mode == failure::object_changed ? 1 : 0));
  CHECK(counters->descriptor_mismatches == (mode == failure::descriptor ? 1 : 0));
  work = {};
  ingestible->finish_preparation();
  CHECK(ledger->tracked_units() == 0);
  ledger->close();
  ledger.reset();
  CHECK(provider.outstanding() == 0);
}
