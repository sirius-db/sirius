/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/arrow_materializer.hpp"
#include "op/scan/field_id_mapping.hpp"
#include "op/scan/positional_visibility.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

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
}  // namespace

TEST_CASE("R4 row domain preserves file offsets and refuses holes", "[r4][scan]")
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

TEST_CASE("R4 Arrow layout rejects unsupported metadata and inconsistent buffers", "[r4][scan]")
{
  int released = 0;
  arrow_fixture f(released);
  REQUIRE(validate_arrow_host_flat(f.schema, f.array).size() == 1);
  f.child.null_count = -1;
  REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
  f.child.n_buffers = 0;
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  f.child.n_buffers     = 2;
  f.child_schema.format = "tsu:UTC";
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  f.child_schema.format = "d:39,2";
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
  f.child_schema.format = "d:18,2";
  REQUIRE(validate_arrow_host_flat(f.schema, f.array)[0] ==
          cudf::data_type(cudf::type_id::DECIMAL128, -2));
  f.schema.metadata = "extension";
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
}

TEST_CASE("R4 Arrow import owns its source through retirement and shared finish", "[r4][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  int released = 0;
  auto f       = std::make_shared<arrow_fixture>(released);
  auto owner   = std::make_shared<owned_arrow_batch>(f->schema, f->array, f);
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

TEST_CASE("R4 direct GPU reader skips completed steps and retains zero column cardinality",
          "[r4][scan]")
{
  rmm::cuda_stream stream;
  materialized_batch batch;
  batch.contract                = 31;
  batch.generation              = 2;
  batch.rows                    = 4;
  batch.logical_schema_complete = true;
  batch.visibility_complete     = true;
  batch.predicates              = {{0, sirius::op::predicate_stage::enforced, {}}};
  ensure_row_count_carrier(batch, stream.view(), cudf::get_current_device_resource_ref());
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
                 "R4 real Parquet ID mapping widens reorders and fills explicit null",
                 "[r4][scan][integration]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(constant(11, 3, stream.view()));
  columns.push_back(constant(22, 3, stream.view()));
  cudf::table table(std::move(columns));
  cudf::io::table_input_metadata metadata(table.view());
  metadata.column_metadata[0].set_name("old.dot").set_parquet_field_id(4);
  metadata.column_metadata[1].set_name("right").set_parquet_field_id(9);
  sirius::test::scratch_dir dir("r4_field_ids");
  auto path   = (dir.path() / "data.parquet").string();
  auto writer = cudf::io::parquet_writer_options::builder(cudf::io::sink_info(path), table.view())
                  .metadata(std::move(metadata))
                  .build();
  cudf::io::write_parquet(writer, stream.view());
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
  auto batch = materialize_mapped_parquet(input, mapping, contract, groups, stream.view(), mr);
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
  auto coalesced = materialize_mapped_parquet_runs(units, contract, stream.view(), mr);
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
  auto flushed     = materialize_mapped_parquet_runs(units, contract, stream.view(), mr);
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
  auto repeated = materialize_mapped_parquet_runs(units, contract, stream.view(), mr);
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
  REQUIRE_THROWS(materialize_mapped_parquet(input, mapping, contract, groups, stream.view(), mr));
}

TEST_CASE("R4 Arrow nullable slice retains values and rejects schema drift", "[r4][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  int released    = 0;
  auto f          = std::make_shared<arrow_fixture>(released);
  f->array.offset = 1;
  f->array.length = 2;
  auto owner      = std::make_shared<owned_arrow_batch>(f->schema, f->array, f);
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

TEST_CASE("R4 Arrow decimal imports normalize the approved carrier", "[r4][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  for (auto format : {"d:9,2", "d:18,2", "d:38,2"}) {
    int released = 0;
    auto f       = std::make_shared<arrow_fixture>(released);
    __int128 values[3]{123, 456, 789};
    f->child_schema.format = format;
    f->buffers[1]          = values;
    auto owner             = std::make_shared<owned_arrow_batch>(f->schema, f->array, f);
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

TEST_CASE("R4 Arrow empty and zero-column batches preserve cardinality", "[r4][scan]")
{
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  for (int rows : {0, 3}) {
    int released         = 0;
    auto f               = std::make_shared<arrow_fixture>(released);
    f->schema.n_children = f->array.n_children = 0;
    f->array.length                            = rows;
    auto owner = std::make_shared<owned_arrow_batch>(f->schema, f->array, f);
    materialization_program program{44, 1, {}, 0, false};
    program.input_types = std::vector<cudf::data_type>{};
    auto batch = import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref());
    REQUIRE(batch.rows == rows);
    REQUIRE(batch.table.num_rows() == rows);
    REQUIRE(lane.quiesce() == cudaSuccess);
  }
}

TEST_CASE("R4 explicit GPU positions feed shared visibility before residual", "[r4][scan]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(constant(7, 3, stream.view()));
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
      stream.view(),
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

TEST_CASE("R4 Arrow imports every flat primitive with nulls", "[r4][scan]")
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
    auto owner = std::make_shared<owned_arrow_batch>(f->schema, f->array, f);
    materialization_program program{46, 1, types, 0, false};
    program.input_types = types;
    auto batch = import_arrow_host(owner, program, lane, cudf::get_current_device_resource_ref());
    REQUIRE(batch.rows == 3);
    REQUIRE(batch.table.view().column(0).type() == types[0]);
    REQUIRE(batch.table.view().column(0).null_count() == 1);
    REQUIRE(lane.quiesce() == cudaSuccess);
  }
}

TEST_CASE("R4 Arrow rejects nested dictionary extensions and invalid string offsets", "[r4][scan]")
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

TEST_CASE("R4 Arrow metadata and UTF8 offsets are qualified before import", "[r4][scan]")
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
  REQUIRE_NOTHROW(validate_arrow_host_flat(f.schema, f.array));
  metadata.clear();
  integer(1);
  entry("ARROW:extension:name", "opaque");
  f.child_schema.metadata = metadata.data();
  REQUIRE_THROWS(validate_arrow_host_flat(f.schema, f.array));
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

TEST_CASE("R4 retiring an abandoned Arrow import releases its owner exactly once", "[r4][scan]")
{
  rmm::cuda_stream stream;
  int released = 0;
  {
    sirius::exec::retire_lane lane(stream.value());
    auto f     = std::make_shared<arrow_fixture>(released);
    auto owner = std::make_shared<owned_arrow_batch>(f->schema, f->array, f);
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

TEST_CASE("R4 finish refuses unproved visibility and predicate reports", "[r4][scan]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(constant(9, 2, stream.view()));
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
