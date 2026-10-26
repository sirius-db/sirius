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

#include "catch.hpp"
#include "helper/type_conversions.hpp"
#include "op/result/host_table_chunk_reader.hpp"
#include "op/sirius_physical_gpu_values.hpp"
#include "operator_test_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream.hpp>

#include <duckdb.hpp>

namespace {

void require_empty_string(cudf::column_view column, ::cuda::stream_ref stream)
{
  REQUIRE(column.type().id() == cudf::type_id::STRING);
  REQUIRE(column.size() == 0);
  // cuDF stores chars in the parent's data buffer, with offsets as its only child.
  REQUIRE(column.num_children() == 1);
  cudf::strings_column_view strings(column);
  auto offsets = strings.offsets();
  REQUIRE(offsets.type().id() == cudf::type_id::INT32);
  REQUIRE(offsets.size() == 1);
  int32_t zero = -1;
  REQUIRE(cudaMemcpyAsync(
            &zero, offsets.data<int32_t>(), sizeof(zero), cudaMemcpyDeviceToHost, stream.get()) ==
          cudaSuccess);
  stream.sync();
  CHECK(zero == 0);
  CHECK(strings.chars_size(stream) == 0);
}

}  // namespace

TEST_CASE("GPU VALUES empty collections preserve nested schemas through host conversion",
          "[operator][gpu_values][empty_collection]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu    = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto* host   = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
  REQUIRE(gpu);
  REQUIRE(host);
  rmm::cuda_stream stream;
  using type         = duckdb::LogicalType;
  auto const strings = type::LIST(type::VARCHAR);
  auto const records = type::LIST(type::STRUCT({{"label", type::VARCHAR}}));
  // Keep ARRAY in the bound schema; a Parquet round trip can normalize it to LIST.
  duckdb::vector<type> schema{type::BIGINT,
                              type::DECIMAL(18, 2),
                              type::VARCHAR,
                              type::ARRAY(type::VARCHAR, 2),
                              type::MAP(type::VARCHAR, strings),
                              type::STRUCT({{"label", type::VARCHAR}, {"records", records}})};
  auto collection =
    duckdb::make_uniq<duckdb::ColumnDataCollection>(duckdb::Allocator::DefaultAllocator(), schema);
  REQUIRE(collection->Count() == 0);
  REQUIRE(collection->Types()[3].id() == duckdb::LogicalTypeId::ARRAY);
  sirius::op::sirius_physical_gpu_values source(
    sirius::from_duckdb_vec(schema),
    0,
    duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection>(std::move(collection)));
  auto const narrow = GENERATE(false, true);
  if (narrow) {
    source.set_physical_types({cudf::data_type{cudf::type_id::INT8},
                               cudf::data_type{cudf::type_id::DECIMAL32, -2},
                               cudf::data_type{cudf::type_id::STRING},
                               cudf::data_type{cudf::type_id::LIST},
                               cudf::data_type{cudf::type_id::LIST},
                               cudf::data_type{cudf::type_id::STRUCT}});
  }
  auto input = source.get_next_task_input_data();
  REQUIRE(input);
  input->prepare_for_processing(gpu, stream);
  auto output   = source.execute(*input, stream);
  auto& batches = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output).get_data_batches();
  REQUIRE(batches.size() == 1);
  {
    auto ro    = batches[0]->to_read_only();
    auto table = sirius::get_cudf_table_view(ro);
    REQUIRE(table.num_rows() == 0);
    REQUIRE(table.num_columns() == schema.size());
    CHECK(table.column(0).type().id() == (narrow ? cudf::type_id::INT8 : cudf::type_id::INT64));
    CHECK(table.column(1).type() ==
          cudf::data_type{narrow ? cudf::type_id::DECIMAL32 : cudf::type_id::DECIMAL64, -2});
    require_empty_string(table.column(2), stream);
    REQUIRE(table.column(3).type().id() == cudf::type_id::LIST);
    require_empty_string(table.column(3).child(1), stream);
    auto entries = table.column(4).child(1);
    REQUIRE(entries.type().id() == cudf::type_id::STRUCT);
    require_empty_string(entries.child(0), stream);
    require_empty_string(entries.child(1).child(1), stream);
    auto record = table.column(5);
    REQUIRE(record.type().id() == cudf::type_id::STRUCT);
    require_empty_string(record.child(0), stream);
    require_empty_string(record.child(1).child(1).child(0), stream);
  }
  {
    auto mut = batches[0]->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), host, stream);
  }
  auto ro          = batches[0]->to_read_only();
  auto const& data = ro.get_data()->cast<cucascade::host_data_representation>();
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  sirius::op::result::host_table_chunk_reader reader(*con.context, data, schema);
  duckdb::DataChunk chunk;
  CHECK_FALSE(reader.get_next_chunk(chunk));
  CHECK_FALSE(source.get_next_task_input_data());
}

TEST_CASE("GPU VALUES EMPTY_RESULT preserves native and overridden scalar carriers",
          "[operator][gpu_values][empty_collection]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu    = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(gpu);
  rmm::cuda_stream stream;
  duckdb::vector<duckdb::LogicalType> schema{
    duckdb::LogicalType::BIGINT, duckdb::LogicalType::DECIMAL(18, 2), duckdb::LogicalType::VARCHAR};
  sirius::op::sirius_physical_gpu_values source(sirius::from_duckdb_vec(schema), 0, false);
  auto const narrow = GENERATE(false, true);
  if (narrow) {
    source.set_physical_types({cudf::data_type{cudf::type_id::INT8},
                               cudf::data_type{cudf::type_id::DECIMAL32, -2},
                               cudf::data_type{cudf::type_id::STRING}});
  }
  auto input = source.get_next_task_input_data();
  input->prepare_for_processing(gpu, stream);
  auto output   = source.execute(*input, stream);
  auto& batches = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto ro    = batches[0]->to_read_only();
  auto table = sirius::get_cudf_table_view(ro);
  REQUIRE(table.num_rows() == 0);
  REQUIRE(table.num_columns() == schema.size());
  CHECK(table.column(0).type().id() == (narrow ? cudf::type_id::INT8 : cudf::type_id::INT64));
  CHECK(table.column(1).type() ==
        cudf::data_type{narrow ? cudf::type_id::DECIMAL32 : cudf::type_id::DECIMAL64, -2});
  CHECK(table.column(2).type().id() == cudf::type_id::STRING);
}
