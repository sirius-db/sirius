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

/**
 * @file test_dictionary_carrier.cpp
 * @brief A dictionary carrier is a hint: every place that meets one may meet its strings instead.
 *
 * Decoding, empty tables, concatenation, and partitioning all hand back STRING where a carrier
 * cannot stay, and the values never change.
 */

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "helper/numeric_narrowing.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "operator/operator_test_utils.hpp"
#include "operator/operator_type_traits.hpp"
#include "utils/data_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

using sirius::test::vector_to_cudf_column;
using sirius::test::operator_utils::gpu_type_traits;
using sirius::test::operator_utils::string_tag;

std::unique_ptr<cudf::column> strings(std::vector<std::string> const& values)
{
  return vector_to_cudf_column<gpu_type_traits<string_tag>>(values);
}

std::unique_ptr<cudf::column> encode(cudf::column_view column)
{
  return cudf::dictionary::encode(column, cudf::data_type{cudf::type_id::INT32});
}

std::vector<std::string> to_host(cudf::column_view column)
{
  if (column.size() == 0) { return {}; }
  auto const stream = cudf::get_default_stream();
  cudf::strings_column_view const view(column);
  // A column rebuilt from host memory may have 64-bit offsets.
  std::vector<int64_t> offsets(column.size() + 1);
  if (view.offsets().type().id() == cudf::type_id::INT64) {
    cudaMemcpy(offsets.data(),
               view.offsets().data<int64_t>() + column.offset(),
               offsets.size() * sizeof(int64_t),
               cudaMemcpyDeviceToHost);
  } else {
    std::vector<int32_t> narrow(offsets.size());
    cudaMemcpy(narrow.data(),
               view.offsets().data<int32_t>() + column.offset(),
               narrow.size() * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    std::ranges::copy(narrow, offsets.begin());
  }
  std::string chars(static_cast<std::size_t>(view.chars_size(stream)), '\0');
  cudaMemcpy(chars.data(), view.chars_begin(stream), chars.size(), cudaMemcpyDeviceToHost);
  std::vector<std::string> values;
  for (cudf::size_type row = 0; row < column.size(); ++row) {
    values.push_back(chars.substr(offsets[row], offsets[row + 1] - offsets[row]));
  }
  return values;
}

std::vector<std::string> repeated(std::size_t distinct, std::size_t rows, std::size_t width)
{
  std::vector<std::string> values;
  values.reserve(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    auto value = "v" + std::to_string(row % distinct);
    value.resize(std::max(width, value.size()), '_');
    values.push_back(std::move(value));
  }
  return values;
}

}  // namespace

TEST_CASE("a dictionary carrier decodes to its strings", "[dictionary_carrier]")
{
  auto const stream = cudf::get_default_stream();
  SECTION("zero rows decode to an empty STRING column, with or without children")
  {
    auto const childless = cudf::make_empty_column(cudf::type_id::DICTIONARY32);
    auto const encoded   = encode(cudf::make_empty_column(cudf::type_id::STRING)->view());
    for (auto const* column : {childless.get(), encoded.get()}) {
      auto const decoded = sirius::decode_dictionary_carrier(column->view(), stream);
      CHECK(decoded->type().id() == cudf::type_id::STRING);
      CHECK(decoded->size() == 0);
    }
    CHECK(sirius::dictionary_keys_bytes(*childless) == 0);
  }
  SECTION("rows decode to their values, and the keys count their offsets and chars")
  {
    std::vector<std::string> const values{"a", "bb", "ccc", "a", "bb"};
    auto const source  = strings(values);
    auto const encoded = encode(source->view());
    CHECK(sirius::dictionary_keys_bytes(*encoded) == 4 * sizeof(int32_t) + 6);
    auto const decoded = sirius::decode_dictionary_carrier(encoded->view(), stream);
    CHECK(to_host(decoded->view()) == values);
  }
}

TEST_CASE("an empty table of a dictionary carrier holds an empty STRING column",
          "[dictionary_carrier]")
{
  auto const table = sirius::make_empty_table(std::vector<cudf::data_type>{
    cudf::data_type{cudf::type_id::DICTIONARY32}, cudf::data_type{cudf::type_id::INT32}});
  CHECK(table->get_column(0).type().id() == cudf::type_id::STRING);
  CHECK(table->get_column(1).type().id() == cudf::type_id::INT32);
}

TEST_CASE("concatenating carriers decodes only a column that mixes the two forms",
          "[dictionary_carrier]")
{
  auto const stream            = cudf::get_default_stream();
  auto const first             = strings({"a", "b"});
  auto const second            = strings({"b", "c"});
  auto const first_dictionary  = encode(first->view());
  auto const second_dictionary = encode(second->view());
  SECTION("a dictionary beside strings leaves as strings")
  {
    auto const out = sirius::concatenate_carriers(
      {cudf::table_view{{first_dictionary->view()}}, cudf::table_view{{second->view()}}}, stream);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::STRING);
    CHECK(to_host(out->get_column(0).view()) == std::vector<std::string>{"a", "b", "b", "c"});
  }
  SECTION("dictionaries throughout stay one")
  {
    auto const out = sirius::concatenate_carriers(
      {cudf::table_view{{first_dictionary->view()}}, cudf::table_view{{second_dictionary->view()}}},
      stream);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::DICTIONARY32);
    auto const decoded = sirius::decode_dictionary_carrier(out->get_column(0).view(), stream);
    CHECK(to_host(decoded->view()) == std::vector<std::string>{"a", "b", "b", "c"});
  }
  SECTION("an empty dictionary without keys beside a dictionary leaves the dictionary")
  {
    auto const childless = cudf::make_empty_column(cudf::type_id::DICTIONARY32);
    auto const out       = sirius::concatenate_carriers(
      {cudf::table_view{{childless->view()}}, cudf::table_view{{second_dictionary->view()}}},
      stream);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::DICTIONARY32);
    auto const decoded = sirius::decode_dictionary_carrier(out->get_column(0).view(), stream);
    CHECK(to_host(decoded->view()) == std::vector<std::string>{"b", "c"});
  }
  SECTION("empty tables alone leave an empty table")
  {
    auto const childless = cudf::make_empty_column(cudf::type_id::DICTIONARY32);
    auto const out       = sirius::concatenate_carriers(
      {cudf::table_view{{childless->view()}}, cudf::table_view{{childless->view()}}}, stream);
    CHECK(out->num_rows() == 0);
  }
  SECTION("dictionaries whose merged keys exceed the cap leave as strings")
  {
    // 6,000 distinct 64-byte keys each, about 400 KiB: under the cap alone, over it merged.
    auto values    = repeated(6'000, 6'000, 64);
    auto const low = encode(strings(values)->view());
    for (auto& value : values) {
      value.front() = 'w';
    }
    auto const high = encode(strings(values)->view());
    REQUIRE(sirius::dictionary_keys_bytes(*low) <= sirius::dictionary_carrier_max_keys_bytes);
    auto const out = sirius::concatenate_carriers(
      {cudf::table_view{{low->view()}}, cudf::table_view{{high->view()}}}, stream);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::STRING);
    CHECK(out->num_rows() == 12'000);
    CHECK(to_host(out->get_column(0).view()).back() == values.back());
  }
}

TEST_CASE("hash partitioning decodes a carrier key and a carrier whose keys every split copies",
          "[dictionary_carrier]")
{
  auto* space                = sirius::test::operator_utils::get_default_gpu_space();
  auto const stream          = cudf::get_default_stream();
  constexpr std::size_t rows = 20'000;
  // Few short keys: far under an eighth of the batch however many partitions copy them.
  auto const small = encode(strings(repeated(4, rows, 8))->view());
  // Distinct 64-byte keys: as large as the batch's strings, so every partition copying them is too
  // much.
  auto const large = encode(strings(repeated(rows, rows, 64))->view());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::make_unique<cudf::column>(small->view()));
  columns.push_back(std::make_unique<cudf::column>(large->view()));
  columns.push_back(std::make_unique<cudf::column>(small->view()));
  auto const batch = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                             *space,
                                             stream,
                                             sirius::telemetry::batch_telemetry_info{});
  auto const parts =
    sirius::op::gpu_partition_impl::hash_partition(batch->to_read_only(), {2}, 4, stream, *space);
  REQUIRE(parts.size() == 4);
  std::size_t total = 0;
  for (auto const& part : parts) {
    auto const view = sirius::get_cudf_table_view(*part);
    total += static_cast<std::size_t>(view.num_rows());
    CHECK(view.column(0).type().id() == cudf::type_id::DICTIONARY32);
    CHECK(view.column(1).type().id() == cudf::type_id::STRING);
    CHECK(view.column(2).type().id() == cudf::type_id::STRING);
  }
  CHECK(total == rows);
}

TEST_CASE("even partitioning decodes a carrier whose keys every split copies",
          "[dictionary_carrier]")
{
  auto* space                = sirius::test::operator_utils::get_default_gpu_space();
  auto const stream          = cudf::get_default_stream();
  constexpr std::size_t rows = 20'000;
  auto const small_values    = repeated(4, rows, 8);
  auto const large_values    = repeated(rows, rows, 64);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(encode(strings(small_values)->view()));
  columns.push_back(encode(strings(large_values)->view()));
  auto const batch = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                             *space,
                                             stream,
                                             sirius::telemetry::batch_telemetry_info{});
  auto const parts =
    sirius::op::gpu_partition_impl::evenly_partition(batch->to_read_only(), 4, stream, *space);
  REQUIRE(parts.size() == 4);
  std::vector<std::string> small_out;
  std::vector<std::string> large_out;
  for (auto const& part : parts) {
    auto const view = sirius::get_cudf_table_view(*part);
    REQUIRE(view.column(0).type().id() == cudf::type_id::DICTIONARY32);
    REQUIRE(view.column(1).type().id() == cudf::type_id::STRING);
    auto const decoded = sirius::decode_dictionary_carrier(view.column(0), stream);
    auto const small   = to_host(decoded->view());
    auto const large   = to_host(view.column(1));
    small_out.insert(small_out.end(), small.begin(), small.end());
    large_out.insert(large_out.end(), large.begin(), large.end());
  }
  CHECK(small_out == small_values);
  CHECK(large_out == large_values);
}

TEST_CASE("a dictionary carrier survives a spill to the host and back", "[dictionary_carrier]")
{
  auto const manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu          = const_cast<cucascade::memory::memory_space*>(
    manager->get_memory_space(cucascade::memory::Tier::GPU, 0));
  auto* host = const_cast<cucascade::memory::memory_space*>(
    manager->get_memory_space(cucascade::memory::Tier::HOST, 0));
  REQUIRE(gpu != nullptr);
  REQUIRE(host != nullptr);
  // The spill's batched copy needs a stream other than the default one.
  rmm::cuda_stream const owned_stream;
  ::cuda::stream_ref const stream{owned_stream.value()};
  auto const values = repeated(5, 1'000, 8);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(encode(strings(values)->view()));
  columns.push_back(strings(values));
  auto const batch = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                             *gpu,
                                             stream,
                                             sirius::telemetry::batch_telemetry_info{});
  CUDF_CUDA_TRY(cudaStreamSynchronize(cudf::get_default_stream().get()));
  auto& registry = sirius::converter_registry::get();
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, host, stream);
  }
  REQUIRE(batch->to_read_only().get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(registry, gpu, stream);
  }
  owned_stream.synchronize();

  auto const view = sirius::get_cudf_table_view(*batch);
  REQUIRE(view.column(0).type().id() == cudf::type_id::DICTIONARY32);
  auto const decoded =
    sirius::decode_dictionary_carrier(view.column(0), cudf::get_default_stream());
  CHECK(to_host(decoded->view()) == values);
  CHECK(to_host(view.column(1)) == values);
}
