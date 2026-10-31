/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/table_scan/field_mapper.hpp"
#include "sirius_context.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <bit>

using namespace sirius::op::scan;
namespace {
std::shared_ptr<parquet_metadata const> mapper_footer(duckdb::Connection& con,
                                                      std::string const& path,
                                                      ::cuda::stream_ref stream)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  cudf::numeric_scalar<int32_t> value(7, true, stream);
  for (int i = 0; i < 2; ++i)
    columns.push_back(cudf::make_column_from_scalar(value, 3, stream));
  cudf::table data(std::move(columns));
  cudf::io::table_input_metadata metadata(data.view());
  metadata.column_metadata[0].set_name("a").set_parquet_field_id(1);
  metadata.column_metadata[1].set_name("b").set_parquet_field_id(2);
  auto options = cudf::io::parquet_writer_options::builder(cudf::io::sink_info(path), data.view())
                   .metadata(std::move(metadata))
                   .build();
  cudf::io::write_parquet(options, stream);
  stream.sync();
  auto context = sirius::test::get_registered_sirius_context(con);
  auto source  = context->get_scan_manager().create_datasource(path);
  return resolve_parquet_metadata_with_identity(
           *source, 77, path, cudf::io::parquet_reader_options::builder().build())
    .artifact;
}
std::shared_ptr<parquet_metadata const> replace_footer(
  parquet_metadata const& original, std::shared_ptr<cudf::io::parquet::FileMetaData const> footer)
{
  return std::make_shared<parquet_metadata const>(std::move(footer),
                                                  original.footer_byte_len(),
                                                  original.encryption_evidence,
                                                  original.original_schema,
                                                  original.arrow_schema,
                                                  original.original_logical_annotations,
                                                  original.built_for_identity);
}
}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Public mapper counts each actual flat refusal exactly once",
                 "[scan][field_mapper][integration]")
{
  auto rule = GENERATE(refusal_rule::not_flat,
                       refusal_rule::untagged_field,
                       refusal_rule::invalid_or_duplicate_id,
                       refusal_rule::ambiguous_physical_name,
                       refusal_rule::unqualified_conversion,
                       refusal_rule::missing_typed_default,
                       refusal_rule::wrong_default_type,
                       refusal_rule::no_qualified_carrier,
                       refusal_rule::program_target_mismatch,
                       refusal_rule::unbound_generation,
                       refusal_rule::partition_constant_unproved);
  CAPTURE(static_cast<int>(rule));
  rmm::cuda_stream stream;
  sirius::test::scratch_dir dir("mapper_refusal");
  auto path     = dir.file("input.parquet");
  auto original = mapper_footer(*con, path, stream);
  auto footer   = std::make_shared<cudf::io::parquet::FileMetaData>(*original->file_metadata());
  scan_plan plan;
  plan.data_columns = {{0, "a"}, {1, "b"}};
  std::vector<field_requirement> fields{{1, duckdb::LogicalType::INTEGER},
                                        {2, duckdb::LogicalType::INTEGER}};
  materialization_program program;
  program.contract    = 77;
  program.generation  = 1;
  program.input_types = std::vector<cudf::data_type>(2, cudf::data_type{cudf::type_id::INT32});
  std::unordered_set<std::string> proof{path};
  switch (rule) {
    case refusal_rule::not_flat:
      footer->schema[0].num_children = 1;
      footer->schema[0].children_idx = {1};
      footer->schema[1].num_children = 1;
      footer->schema[1].children_idx = {2};
      footer->schema[1].type         = cudf::io::parquet::Type::UNDEFINED;
      footer->schema[1].logical_type.reset();
      footer->schema[1].converted_type.reset();
      break;
    case refusal_rule::untagged_field: footer->schema[1].field_id.reset(); break;
    case refusal_rule::invalid_or_duplicate_id: footer->schema[2].field_id = 1; break;
    case refusal_rule::ambiguous_physical_name: footer->schema[2].name = "a"; break;
    case refusal_rule::unqualified_conversion:
      fields[0].type             = duckdb::LogicalType::BIGINT;
      program.input_types->at(0) = cudf::data_type{cudf::type_id::INT64};
      break;
    case refusal_rule::missing_typed_default: fields[0].id = 99; break;
    case refusal_rule::wrong_default_type:
      fields[0].id            = 99;
      fields[0].missing_value = duckdb::Value::BIGINT(0);
      break;
    case refusal_rule::no_qualified_carrier:
      footer->schema.resize(1);
      footer->schema[0].num_children = 0;
      footer->schema[0].children_idx.clear();
      fields.resize(1);
      fields[0].id            = 99;
      fields[0].missing_value = duckdb::Value(duckdb::LogicalType::INTEGER);
      plan.data_columns.resize(1);
      program.input_types->resize(1);
      break;
    case refusal_rule::program_target_mismatch:
      program.input_types->at(0) = cudf::data_type{cudf::type_id::INT64};
      break;
    case refusal_rule::unbound_generation: program.generation = 0; break;
    case refusal_rule::partition_constant_unproved:
      fields[0].id            = 99;
      fields[0].missing_value = duckdb::Value(duckdb::LogicalType::INTEGER);
      proof.clear();
      break;
  }
  mapping_request request{
    rule_set_id::iceberg_flat_v1, replace_footer(*original, footer), path, plan, program};
  request.requirements                   = fields;
  request.files_without_partition_values = &proof;
  physical_check_counters counters;
  auto result = map_file(request, &counters);
  REQUIRE(std::holds_alternative<mapping_refusal>(result));
  CHECK(std::get<mapping_refusal>(result).rule == rule);
  CHECK_FALSE(std::get<mapping_refusal>(result).text.empty());
  for (std::size_t ruleset = 0; ruleset < 3; ++ruleset)
    for (std::size_t code = 0; code < 11; ++code)
      CHECK(counters.mapper_refusals[ruleset][code].load() ==
            (ruleset == 2 && code == static_cast<std::size_t>(rule) ? 1 : 0));
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Mapper profiles preserve legacy evidence and per-name leaf grouping",
                 "[scan][field_mapper][integration]")
{
  rmm::cuda_stream stream;
  sirius::test::scratch_dir dir("mapper_profiles");
  auto path   = dir.file("input.parquet");
  auto footer = mapper_footer(*con, path, stream);
  scan_plan plan;
  plan.data_columns = {{0, "a"}, {1, "b"}};
  materialization_program program;
  program.contract   = 77;
  program.generation = 1;
  physical_check_counters counters;
  mapping_request request{rule_set_id::parquet_name_v1, footer, path, plan, program};
  auto names = map_file(request, &counters);
  REQUIRE(std::holds_alternative<std::shared_ptr<mapping_result const>>(names));
  CHECK(std::get<std::shared_ptr<mapping_result const>>(names)->leaves ==
        std::vector<std::vector<std::size_t>>{{0}, {1}});
  iceberg_table_schema schema{{{"a", 1, "INTEGER"}, {"b", 2, "INTEGER"}}};
  request.rules          = rule_set_id::iceberg_field_id_v1;
  request.iceberg_schema = &schema;
  auto compare           = [&] {
    auto legacy = check_iceberg_file_schema(
      *request.footer->file_metadata(), schema, path, request.footer->original_logical_annotations);
    auto outcome = map_file(request, &counters);
    physical_profile_result actual;
    if (auto accepted = std::get_if<std::shared_ptr<mapping_result const>>(&outcome))
      actual = (*accepted)->evidence;
    else
      actual = std::get<mapping_refusal>(outcome).evidence;
    CHECK(actual.approved == legacy.approved);
    CHECK(actual.reason == legacy.reason);
    CHECK(actual.text == legacy.text);
    CHECK(actual.validation == legacy.validation);
    CHECK(actual.profile == legacy.profile);
    CHECK(actual.type_mismatches == legacy.type_mismatches);
    return actual.reason;
  };
  CHECK(compare() == verdict_reason::none);
  schema.fields[0].type = "BIGINT";
  CHECK(compare() == verdict_reason::iceberg_schema_promoted_type);
  schema.fields[0].type = "INTEGER";
  auto changed = std::make_shared<cudf::io::parquet::FileMetaData>(*footer->file_metadata());
  for (auto& element : changed->schema)
    element.field_id.reset();
  request.footer = replace_footer(*footer, changed);
  CHECK(compare() == verdict_reason::iceberg_schema_no_field_ids);
  changed = std::make_shared<cudf::io::parquet::FileMetaData>(*footer->file_metadata());
  changed->schema.clear();
  request.footer = replace_footer(*footer, changed);
  CHECK(compare() == verdict_reason::iceberg_schema_no_rows);
  changed = std::make_shared<cudf::io::parquet::FileMetaData>(*footer->file_metadata());
  changed->schema[2].field_id = 1;
  request.footer              = replace_footer(*footer, changed);
  CHECK(compare() == verdict_reason::iceberg_schema_invalid_field_id);
  for (auto const& row : counters.mapper_refusals)
    for (auto const& counter : row)
      CHECK(counter.load() == 0);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Name mapper preserves nested leaf groups projection and zero-output carrier",
                 "[scan][field_mapper][integration]")
{
  sirius::test::scratch_dir dir("mapper_nested");
  auto path = dir.file("nested.parquet");
  run_ok("COPY (SELECT {'x': 7::INTEGER, 'y': 9::BIGINT} nested, 11::INTEGER scalar) TO " +
         dir.file_literal("nested.parquet") + " (FORMAT PARQUET)");
  auto context = sirius::test::get_registered_sirius_context(*con);
  auto source  = context->get_scan_manager().create_datasource(path);
  auto footer  = resolve_parquet_metadata_with_identity(
                  *source, 77, path, cudf::io::parquet_reader_options::builder().build())
                  .artifact;
  materialization_program program;
  program.contract   = 77;
  program.generation = 1;
  scan_plan plan;
  plan.data_columns = {{0, "nested"}, {1, "scalar"}};
  physical_check_counters counts;
  auto map = [&] {
    mapping_request request{rule_set_id::parquet_name_v1, footer, path, plan, program};
    auto mapped = map_file(request, &counts);
    REQUIRE(std::holds_alternative<std::shared_ptr<mapping_result const>>(mapped));
    return std::get<std::shared_ptr<mapping_result const>>(mapped);
  };
  auto full = map();
  CHECK(full->selection == mapping_selection::full_correspondence);
  CHECK(full->leaves == std::vector<std::vector<std::size_t>>{{0, 1}, {2}});
  plan.needs_reader_projection = true;
  plan.data_columns            = {{1, "scalar"}};
  auto projected               = map();
  CHECK(projected->selection == mapping_selection::reader_projection);
  CHECK(projected->leaves == std::vector<std::vector<std::size_t>>{{2}});
  CHECK(projected->physical_leaves.at("nested") == std::vector<std::size_t>{0, 1});
  plan.output_layout.clear();
  plan.carrier_batch_index = 0;
  auto carrier             = map();
  CHECK(carrier->leaves == projected->leaves);
  CHECK(carrier->footer == footer);
  plan.data_columns = {{0, "absent"}};
  auto fallback     = map();
  CHECK(fallback->selection == mapping_selection::full_correspondence);
  CHECK(fallback->reader_projection.natural_read);
  mapping_request virtual_request{rule_set_id::parquet_name_v1, footer, path, plan, program};
  plan.virtual_columns               = {{99,
                                         "file",
                                         sirius::logical_type::make(sirius::type_id::VARCHAR),
                                         scan_plan::parquet_virtual_column_kind::FILENAME,
                                         1}};
  virtual_request.allow_file_carrier = true;
  auto virtual_mapping               = map_file(virtual_request);
  REQUIRE(std::holds_alternative<std::shared_ptr<mapping_result const>>(virtual_mapping));
  auto actual = std::get<std::shared_ptr<mapping_result const>>(virtual_mapping);
  CHECK(actual->file_carrier == std::optional<std::string>{"scalar"});
  CHECK_FALSE(actual->reader_projection.natural_read);
  CHECK(actual->reader_projection.names == std::vector<std::string>{"scalar"});
  for (auto const& row : counts.mapper_refusals)
    for (auto const& count : row)
      CHECK(count.load() == 0);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Public flat mapping preserves typed defaults and floating bit patterns",
                 "[scan][field_mapper][integration]")
{
  auto bits             = GENERATE(uint64_t{0},
                       uint64_t{0x8000000000000000},
                       uint64_t{0x7ff8000000000001},
                       uint64_t{0x7ff8000000000002});
  bool const typed_null = GENERATE(false, true);
  CAPTURE(bits, typed_null);
  rmm::cuda_stream stream;
  sirius::test::scratch_dir dir("mapper_defaults");
  auto path   = dir.file("input.parquet");
  auto footer = mapper_footer(*con, path, stream);
  scan_plan plan;
  plan.data_columns = {{0, "renamed_a"}, {1, "renamed_b"}, {2, "added"}};
  materialization_program program;
  program.contract    = 77;
  program.generation  = 1;
  program.input_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::FLOAT64}};
  auto value          = typed_null ? duckdb::Value(duckdb::LogicalType::DOUBLE)
                                   : duckdb::Value::DOUBLE(std::bit_cast<double>(bits));
  std::vector<field_requirement> fields{{1, duckdb::LogicalType::INTEGER},
                                        {2, duckdb::LogicalType::INTEGER},
                                        {99, duckdb::LogicalType::DOUBLE, value}};
  std::unordered_set<std::string> proof{path};
  mapping_request request{rule_set_id::iceberg_flat_v1, footer, path, plan, program};
  request.requirements                   = fields;
  request.files_without_partition_values = &proof;
  auto outcome                           = map_file(request);
  REQUIRE(std::holds_alternative<std::shared_ptr<mapping_result const>>(outcome));
  auto mapped = std::get<std::shared_ptr<mapping_result const>>(outcome);
  std::vector<std::unique_ptr<cudf::column>> columns;
  cudf::numeric_scalar<int32_t> scalar(7, true, static_cast<::cuda::stream_ref>(stream));
  columns.push_back(cudf::make_column_from_scalar(scalar, 3, stream));
  columns.push_back(cudf::make_column_from_scalar(scalar, 3, stream));
  auto result =
    mapped->compiled->apply(owning_table_view{std::make_unique<cudf::table>(std::move(columns))},
                            3,
                            stream,
                            cudf::get_current_device_resource_ref());
  auto added = result.table.view().column(2);
  CHECK(added.null_count() == (typed_null ? 3 : 0));
  if (!typed_null) {
    std::vector<uint64_t> actual(3);
    REQUIRE(cudaMemcpyAsync(actual.data(),
                            added.head(),
                            actual.size() * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
    CHECK(actual == std::vector<uint64_t>(3, bits));
  }
  auto same = map_file(request);
  REQUIRE(std::holds_alternative<std::shared_ptr<mapping_result const>>(same));
  CHECK(mapped->compiled->compatible_with(
    *std::get<std::shared_ptr<mapping_result const>>(same)->compiled));
  fields[2].missing_value = duckdb::Value::INTEGER(7).DefaultCastAs(duckdb::LogicalType::DOUBLE);
  auto cast               = map_file(request);
  REQUIRE(std::holds_alternative<std::shared_ptr<mapping_result const>>(cast));
  CHECK_FALSE(mapped->compiled->compatible_with(
    *std::get<std::shared_ptr<mapping_result const>>(cast)->compiled));
  // A write default from an unrelated schema is never consulted by this request.
  fields[2].missing_value.reset();
  auto absent = map_file(request);
  REQUIRE(std::holds_alternative<mapping_refusal>(absent));
  CHECK(std::get<mapping_refusal>(absent).rule == refusal_rule::missing_typed_default);
  // Actual fields need no partition/default proof.
  plan.data_columns.pop_back();
  program.input_types->pop_back();
  request.program = program;
  proof.clear();
  auto present = map_file(request);
  CHECK(std::holds_alternative<std::shared_ptr<mapping_result const>>(present));
}
