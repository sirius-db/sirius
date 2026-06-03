/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "helper/logical_type.hpp"
#include "op/scan/scan_info.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "sirius_config.hpp"
#include "utils/utils.hpp"

#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

template <typename Tag, typename Tag::type Member>
struct private_member_access {
  friend typename Tag::type get(Tag) { return Member; }
};

struct split_table_scan_source_access {
  using type = void (sirius::pipeline::sirius_pipeline_converter::*)(
    duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>&);
  friend type get(split_table_scan_source_access);
};

template struct private_member_access<
  split_table_scan_source_access,
  &sirius::pipeline::sirius_pipeline_converter::split_table_scan_source>;

struct inserted_operators_access {
  using type = duckdb::vector<duckdb::unique_ptr<sirius::op::sirius_physical_operator>>
    sirius::pipeline::sirius_pipeline_converter::*;
  friend type get(inserted_operators_access);
};

template struct private_member_access<
  inserted_operators_access,
  &sirius::pipeline::sirius_pipeline_converter::inserted_operators_>;

struct scan_info_access {
  using type = std::unique_ptr<sirius::op::scan::scan_info>
    sirius::op::scan::sirius_gpu_parquet_scan_operator::*;
  friend type get(scan_info_access);
};

template struct private_member_access<
  scan_info_access,
  &sirius::op::scan::sirius_gpu_parquet_scan_operator::_scan_info>;

void split_table_scan_source(sirius::pipeline::sirius_pipeline_converter& converter,
                             duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>& pipeline)
{
  (converter.*get(split_table_scan_source_access{}))(pipeline);
}

auto& inserted_operators(sirius::pipeline::sirius_pipeline_converter& converter)
{
  return converter.*get(inserted_operators_access{});
}

auto const& scan_info(sirius::op::scan::sirius_gpu_parquet_scan_operator const& op)
{
  return op.*get(scan_info_access{});
}

void require_ok(std::unique_ptr<duckdb::QueryResult> result)
{
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

fs::path fresh_tmp_dir(std::string const& tag)
{
  auto dir = fs::temp_directory_path() / ("sirius_pr6_converter_" + tag);
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir);
  return dir;
}

fs::path write_parquet_file(duckdb::Connection& con, fs::path const& dir)
{
  auto const path = dir / "local.parquet";
  require_ok(
    con.Query("CREATE OR REPLACE TABLE converter_local AS "
              "SELECT 1::INTEGER AS id, 'one'::VARCHAR AS name"));
  require_ok(con.Query("COPY converter_local TO '" + path.string() + "' (FORMAT PARQUET)"));
  return path;
}

duckdb::vector<sirius::logical_type> one_int_type()
{
  return {sirius::logical_type::make(sirius::type_id::INTEGER)};
}

duckdb::vector<duckdb::ColumnIndex> one_column_id()
{
  duckdb::vector<duckdb::ColumnIndex> ids;
  ids.emplace_back(duckdb::ColumnIndex(0));
  return ids;
}

duckdb::vector<std::string> one_column_name() { return {"id"}; }

std::unique_ptr<sirius::op::sirius_physical_table_scan> make_sirius_read_parquet_scan(
  std::string const& uri)
{
  duckdb::TableFunction fn;
  fn.name = "sirius_read_parquet";

  duckdb::vector<duckdb::Value> parameters;
  parameters.emplace_back(uri);

  return std::make_unique<sirius::op::sirius_physical_table_scan>(
    one_int_type(),
    std::move(fn),
    nullptr,
    one_int_type(),
    one_column_id(),
    /*projection_ids*/ duckdb::vector<std::size_t>{},
    one_column_name(),
    nullptr,
    /*estimated_cardinality*/ 1,
    duckdb::ExtraOperatorInfo{},
    std::move(parameters),
    duckdb::virtual_column_map_t{});
}

duckdb::unique_ptr<duckdb::FunctionData> bind_read_parquet(
  duckdb::ClientContext& ctx,
  std::string const& parquet_path,
  duckdb::TableFunction& table_function,
  duckdb::vector<duckdb::LogicalType>& types,
  duckdb::vector<std::string>& names)
{
  duckdb::unique_ptr<duckdb::FunctionData> bind_data;
  ctx.RunFunctionInTransaction([&] {
    auto& entry = duckdb::Catalog::GetEntry<duckdb::TableFunctionCatalogEntry>(
      ctx, INVALID_CATALOG, DEFAULT_SCHEMA, "read_parquet");

    duckdb::vector<duckdb::LogicalType> arg_types;
    arg_types.emplace_back(duckdb::LogicalType::VARCHAR);
    table_function = entry.functions.GetFunctionByArguments(ctx, arg_types);

    duckdb::vector<duckdb::Value> inputs;
    inputs.emplace_back(parquet_path);

    duckdb::named_parameter_map_t named_parameters;
    duckdb::vector<duckdb::LogicalType> input_table_types;
    duckdb::vector<std::string> input_table_names;

    duckdb::TableFunctionRef ref;
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
    children.push_back(duckdb::make_uniq<duckdb::ConstantExpression>(duckdb::Value(parquet_path)));
    ref.function = duckdb::make_uniq<duckdb::FunctionExpression>(
      "read_parquet", std::move(children), nullptr, nullptr, false, false, false);

    duckdb::TableFunctionBindInput bind_input(inputs,
                                              named_parameters,
                                              input_table_types,
                                              input_table_names,
                                              nullptr,
                                              nullptr,
                                              table_function,
                                              ref);
    bind_data = table_function.bind(ctx, bind_input, types, names);
  });
  return bind_data;
}

std::unique_ptr<sirius::op::sirius_physical_table_scan> make_read_parquet_scan(
  duckdb::ClientContext& ctx, std::string const& parquet_path)
{
  duckdb::TableFunction fn;
  duckdb::vector<duckdb::LogicalType> duck_types;
  duckdb::vector<std::string> names;
  auto bind_data = bind_read_parquet(ctx, parquet_path, fn, duck_types, names);

  duckdb::vector<sirius::logical_type> returned_types;
  returned_types.reserve(duck_types.size());
  for (auto const& type : duck_types) {
    returned_types.push_back(sirius::logical_type::make(type.id() == duckdb::LogicalTypeId::INTEGER
                                                          ? sirius::type_id::INTEGER
                                                          : sirius::type_id::VARCHAR));
  }

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  for (std::size_t i = 0; i < returned_types.size(); ++i) {
    column_ids.emplace_back(duckdb::ColumnIndex(i));
  }

  duckdb::vector<duckdb::Value> parameters;
  parameters.emplace_back(parquet_path);

  return std::make_unique<sirius::op::sirius_physical_table_scan>(
    returned_types,
    std::move(fn),
    std::move(bind_data),
    returned_types,
    std::move(column_ids),
    /*projection_ids*/ duckdb::vector<std::size_t>{},
    std::move(names),
    nullptr,
    /*estimated_cardinality*/ 1,
    duckdb::ExtraOperatorInfo{},
    std::move(parameters),
    duckdb::virtual_column_map_t{});
}

struct converted_scan_summary {
  std::vector<std::string> file_paths;
  std::size_t partition_index_count{0};
};

converted_scan_summary convert_single_scan(sirius::op::sirius_physical_table_scan& scan)
{
  sirius::pipeline::pipeline_build_context ctx{};
  sirius::operator_params params{};
  sirius::pipeline::sirius_pipeline_converter converter(ctx, params);
  sirius::pipeline::sirius_pipeline_build_state build_state;
  auto pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(ctx);
  build_state.set_pipeline_source(*pipeline, scan);

  split_table_scan_source(converter, pipeline);
  auto& inserted = inserted_operators(converter);
  REQUIRE(inserted.size() == 1);

  auto* gpu_scan =
    dynamic_cast<sirius::op::scan::sirius_gpu_parquet_scan_operator*>(inserted.front().get());
  REQUIRE(gpu_scan != nullptr);
  auto const& info = scan_info(*gpu_scan);
  REQUIRE(info != nullptr);
  return converted_scan_summary{info->file_paths, info->partition_indices.size()};
}

}  // namespace

TEST_CASE("pipeline converter routes sirius_read_parquet using scan parameters",
          "[pipeline][s3][sql]")
{
  auto const uri = std::string{"s3://bucket/parquet/nation.parquet"};
  auto scan      = make_sirius_read_parquet_scan(uri);

  auto scan_info = convert_single_scan(*scan);
  REQUIRE(scan_info.file_paths.size() == 1);
  CHECK(scan_info.file_paths.front() == uri);
  CHECK(scan_info.partition_index_count == 0);
}

TEST_CASE("pipeline converter leaves local read_parquet MultiFileBindData path intact",
          "[pipeline][sql]")
{
  auto const dir       = fresh_tmp_dir("local");
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto const path      = write_parquet_file(con, dir);
  auto scan            = make_read_parquet_scan(*con.context, path.string());

  auto scan_info = convert_single_scan(*scan);
  REQUIRE(scan_info.file_paths.size() == 1);
  CHECK(scan_info.file_paths.front() == path.string());
}
