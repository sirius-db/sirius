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

// Spill compression: GPU batches downgraded to the host or disk tier are stored
// Simpatico-compressed and decoded on the way back, or spill uncompressed when
// compression is off, does not pay, or fails.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_disk_representation.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/spill_compression.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

namespace utils = sirius::test::operator_utils;
using sirius::compression::spill_compression_settings;

bool has_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// GPU, host and disk spaces, with the converters registered.
struct spill_env {
  fs::path dir = fs::temp_directory_path() / "sirius_spill_compression_test";
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;
  cucascade::memory::memory_space* disk = nullptr;
  rmm::cuda_stream stream;

  spill_env()
  {
    fs::create_directories(dir);
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    // Per-thread tracking, as Sirius configures it, so the encode runs column-parallel.
    builder.set_number_of_gpus(1)
      .track_reservation_per_stream(false)
      .set_gpu_usage_limit(1ull << 30)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(1ull << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, 4ull << 30, dir.string());
    mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    gpu  = mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
    host = mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
    disk = const_cast<cucascade::memory::memory_space*>(
      mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).front());
  }
  ~spill_env() { fs::remove_all(dir); }
};

spill_env& env()
{
  static spill_env e;
  return e;
}

/// Installs @p settings for one test and switches spill compression off afterwards.
struct scoped_settings {
  explicit scoped_settings(spill_compression_settings settings)
  {
    sirius::compression::set_spill_compression_settings(settings);
  }
  ~scoped_settings() { sirius::compression::set_spill_compression_settings({}); }
};

spill_compression_settings enabled_settings()
{
  spill_compression_settings s;
  s.enabled         = true;
  s.min_batch_bytes = 0;
  return s;
}

/// A column with every seventh row null.
void null_every_seventh(cudf::column& column, ::cuda::stream_ref stream)
{
  auto const rows = column.size();
  // cudf sizes a mask to its padded allocation, not to the bits it needs.
  std::vector<std::uint32_t> words(
    cudf::bitmask_allocation_size_bytes(rows) / sizeof(std::uint32_t), 0);
  cudf::size_type nulls = 0;
  for (cudf::size_type i = 0; i < rows; ++i) {
    if (i % 7 == 0) {
      ++nulls;
    } else {
      words[static_cast<std::size_t>(i) / 32] |= 1u << (i % 32);
    }
  }
  rmm::device_buffer mask(words.data(), words.size() * sizeof(std::uint32_t), stream);
  column.set_null_mask(std::move(mask), nulls);
}

template <typename T>
std::unique_ptr<cudf::column> numeric_column(cudf::data_type type,
                                             std::vector<T> const& values,
                                             ::cuda::stream_ref stream)
{
  auto column = cudf::make_fixed_width_column(type,
                                              static_cast<cudf::size_type>(values.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              utils::get_resource_ref(*env().gpu));
  cudaMemcpy(column->mutable_view().head(),
             values.data(),
             values.size() * sizeof(T),
             cudaMemcpyHostToDevice);
  return column;
}

/// A GPU batch whose producer has recorded its writer event, as an operator's output has.
std::shared_ptr<cucascade::data_batch> gpu_batch(std::unique_ptr<cudf::table> table)
{
  auto stream = utils::default_stream();
  auto batch  = sirius::make_data_batch(
    std::move(table), *env().gpu, stream, sirius::telemetry::batch_telemetry_info{});
  batch->to_mutable().get_data()->record_writer_event(stream);
  return batch;
}

/// One million rows, one column per default plan: a run of INT64s (delta -> rle), sorted
/// INT64 keys (delta -> bitpack), nullable small INT32s, DECIMAL(38,2) values (bitpack), a
/// low-cardinality string column (dictionary) and a nullable high-cardinality string
/// column with empty values (str_split).
std::unique_ptr<cudf::table> compressible_table()
{
  constexpr std::size_t rows = 1'000'000;
  auto stream                = utils::default_stream();

  std::vector<std::int64_t> keys(rows);
  std::iota(keys.begin(), keys.end(), std::int64_t{1'000'000});
  std::vector<std::int32_t> small(rows);
  std::vector<__int128> amounts(rows);
  std::vector<std::int64_t> sorted(rows);
  std::vector<std::string> flags(rows);
  std::vector<std::string> names(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    sorted[i]  = static_cast<std::int64_t>(i * 100 + i % 7);
    small[i]   = static_cast<std::int32_t>(i % 1000);
    amounts[i] = static_cast<__int128>(i % 50'000) * 100 + 25;
    flags[i]   = (i % 3 == 0) ? "RETURNED" : (i % 3 == 1 ? "NONE" : "DELIVER IN PERSON");
    names[i]   = (i % 11 == 0) ? "" : "Customer#" + std::to_string(i * 7919 % rows);
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT64}, keys, stream));
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT64}, sorted, stream));
  columns.push_back(numeric_column(cudf::data_type{cudf::type_id::INT32}, small, stream));
  null_every_seventh(*columns.back(), stream);
  columns.push_back(
    numeric_column(cudf::data_type{cudf::type_id::DECIMAL128, -2}, amounts, stream));
  columns.push_back(utils::make_string_column(flags, stream, utils::get_resource_ref(*env().gpu)));
  columns.push_back(utils::make_string_column(names, stream, utils::get_resource_ref(*env().gpu)));
  null_every_seventh(*columns.back(), stream);
  return std::make_unique<cudf::table>(std::move(columns));
}

/// Row hashes and per-column null counts: equal for equal tables of any column types.
struct table_digest {
  std::vector<std::uint32_t> row_hashes;
  std::vector<cudf::size_type> null_counts;
  bool operator==(table_digest const&) const = default;
};

table_digest digest(cudf::table_view const& view)
{
  auto stream = utils::default_stream();
  auto hashes = cudf::hashing::murmurhash3_x86_32(view, cudf::DEFAULT_HASH_SEED, stream);
  table_digest out;
  out.row_hashes = utils::copy_column_to_host<std::uint32_t>(hashes->view(), stream);
  for (auto const& column : view) {
    out.null_counts.push_back(column.null_count());
  }
  return out;
}

table_digest digest(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return digest(ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view());
}

void move_to(std::shared_ptr<cucascade::data_batch> const& batch,
             cucascade::memory::memory_space* space)
{
  sirius::convertible_data_batch convertible(batch);
  REQUIRE(convertible.convert({space}, env().stream.view(), *env().mgr, true).has_value());
}

template <typename Representation>
bool holds(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return dynamic_cast<Representation const*>(ro.get_data()) != nullptr;
}

}  // namespace

TEST_CASE("spill compression: default plans follow the column type and a sample",
          "[compression][spill]")
{
  if (!has_gpu()) { return; }
  auto stream = utils::default_stream();
  auto mr     = utils::get_resource_ref(*env().gpu);
  auto plan   = [&](cudf::column_view const& c) {
    return sirius::compression::default_spill_plan(c, stream);
  };

  auto const contains = [](std::string const& plan, char const* op) {
    return plan.find(op) != std::string::npos;
  };

  // Integer-valued columns: the sample decides between bitpack and the delta plans.
  std::mt19937_64 rng{7};
  std::vector<std::int64_t> random(100'000), sorted(100'000), repeated_keys(100'000);
  for (std::size_t i = 0; i < random.size(); ++i) {
    random[i]        = static_cast<std::int64_t>(rng() % 1'000'000'000);
    sorted[i]        = static_cast<std::int64_t>(i * 1000 + rng() % 7);
    repeated_keys[i] = static_cast<std::int64_t>(i / 4 * 1000);
  }
  auto const int64       = cudf::data_type{cudf::type_id::INT64};
  auto const random_plan = plan(numeric_column(int64, random, stream)->view());
  CHECK(contains(random_plan, "bitpack"));
  CHECK_FALSE(contains(random_plan, "delta"));
  auto const sorted_plan = plan(numeric_column(int64, sorted, stream)->view());
  CHECK(contains(sorted_plan, "delta"));
  CHECK_FALSE(contains(sorted_plan, "rle"));
  auto const repeated_column = numeric_column(int64, repeated_keys, stream);
  CHECK(contains(plan(repeated_column->view()), "rle"));
  // The sample starts at the column's offset, not at its buffer.
  std::vector<std::int64_t> random_then_repeated(random);
  random_then_repeated.insert(
    random_then_repeated.end(), repeated_keys.begin(), repeated_keys.end());
  auto const spliced = numeric_column(int64, random_then_repeated, stream);
  auto const tail    = cudf::slice(spliced->view(),
                                   {static_cast<cudf::size_type>(random.size()),
                                    static_cast<cudf::size_type>(random_then_repeated.size())},
                                stream)
                      .front();
  CHECK(contains(plan(tail), "rle"));

  // Other fixed-width types are bitpacked as they are.
  auto const doubles = numeric_column(
    cudf::data_type{cudf::type_id::FLOAT64}, std::vector<double>(10'000, 1.5), stream);
  auto const decimals = numeric_column(
    cudf::data_type{cudf::type_id::DECIMAL128, -2}, std::vector<__int128>(10'000, 42), stream);
  CHECK(plan(doubles->view()) == plan(decimals->view()));
  CHECK(contains(plan(doubles->view()), "bitpack"));
  CHECK_FALSE(contains(plan(doubles->view()), "delta"));

  std::vector<std::string> repeated(10'000), unique(10'000);
  for (std::size_t i = 0; i < repeated.size(); ++i) {
    repeated[i] = "value_" + std::to_string(i % 10);
    unique[i]   = "value_" + std::to_string(i);
  }
  CHECK(plan(utils::make_string_column(repeated, stream, mr)->view()).find("dictionary") !=
        std::string::npos);
  CHECK(plan(utils::make_string_column(unique, stream, mr)->view()).find("str_split") !=
        std::string::npos);
  // Too short for a dictionary to pay for its keys.
  std::vector<std::string> few(100, "x");
  CHECK(plan(utils::make_string_column(few, stream, mr)->view()).find("identity") !=
        std::string::npos);
}

TEST_CASE("spill compression: GPU -> compressed host -> GPU", "[compression][spill]")
{
  if (!has_gpu()) { return; }
  scoped_settings settings(enabled_settings());
  auto batch                     = gpu_batch(compressible_table());
  auto const expected            = digest(*batch);
  std::size_t const device_bytes = batch->to_read_only().get_data()->get_size_in_bytes();
  auto* gpu_mr               = env().gpu->get_memory_resource_of<cucascade::memory::Tier::GPU>();
  auto const reserved_before = gpu_mr->get_total_reserved_bytes();

  move_to(batch, env().host);
  REQUIRE(holds<sirius::compressed_host_representation>(*batch));
  {
    auto ro = batch->to_read_only();
    CHECK(ro.get_data()->get_size_in_bytes() < device_bytes);
    CHECK(ro.get_data()->get_uncompressed_data_size_in_bytes() == device_bytes);
  }
  // The encode's reservation is detached and released once the batch is encoded.
  CHECK_FALSE(gpu_mr->is_stream_tracked(env().stream.view()));
  CHECK(gpu_mr->get_total_reserved_bytes() == reserved_before);

  move_to(batch, env().gpu);
  REQUIRE(holds<cucascade::gpu_table_representation>(*batch));
  CHECK(digest(*batch) == expected);
}

TEST_CASE("spill compression: GPU -> compressed disk -> GPU, and the file goes with the batch",
          "[compression][spill]")
{
  if (!has_gpu()) { return; }
  scoped_settings settings(enabled_settings());
  auto batch          = gpu_batch(compressible_table());
  auto const expected = digest(*batch);

  auto const rows = static_cast<std::size_t>(expected.row_hashes.size());

  move_to(batch, env().disk);
  REQUIRE(holds<sirius::compressed_disk_representation>(*batch));
  std::string const path = [&] {
    auto ro = batch->to_read_only();
    return ro.get_data()->cast<sirius::compressed_disk_representation>().path();
  }();
  CHECK(fs::exists(path));
  {
    auto ro = batch->to_read_only();
    CHECK(sirius::representation_num_rows(*ro.get_data()) == rows);
    CHECK(sirius::representation_num_rows(ro.get_data()) == rows);
  }

  move_to(batch, env().gpu);
  CHECK(digest(*batch) == expected);
  // The decoded batch replaced the disk representation, which deleted its file.
  CHECK_FALSE(fs::exists(path));
}

TEST_CASE("spill compression: a compressed host batch moves to disk without a decode",
          "[compression][spill]")
{
  if (!has_gpu()) { return; }
  scoped_settings settings(enabled_settings());
  auto batch          = gpu_batch(compressible_table());
  auto const expected = digest(*batch);

  move_to(batch, env().host);
  REQUIRE(holds<sirius::compressed_host_representation>(*batch));
  move_to(batch, env().disk);
  REQUIRE(holds<sirius::compressed_disk_representation>(*batch));
  move_to(batch, env().gpu);
  CHECK(digest(*batch) == expected);
}

TEST_CASE("spill compression: batches spill uncompressed when it does not apply",
          "[compression][spill]")
{
  if (!has_gpu()) { return; }

  SECTION("disabled")
  {
    scoped_settings settings(spill_compression_settings{});
    auto batch = gpu_batch(compressible_table());
    move_to(batch, env().host);
    CHECK(holds<cucascade::host_data_representation>(*batch));
  }
  SECTION("smaller than min_batch_bytes")
  {
    auto s            = enabled_settings();
    s.min_batch_bytes = 1ull << 40;
    scoped_settings settings(s);
    auto batch = gpu_batch(compressible_table());
    move_to(batch, env().disk);
    CHECK(holds<cucascade::disk_data_representation>(*batch));
  }
  SECTION("the compressed form saves too little")
  {
    scoped_settings settings(enabled_settings());
    // Random 64-bit values: bitpacking cannot shrink them.
    std::mt19937_64 rng{42};
    std::vector<std::int64_t> noise(1 << 20);
    for (auto& v : noise) {
      v = static_cast<std::int64_t>(rng());
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(
      numeric_column(cudf::data_type{cudf::type_id::INT64}, noise, utils::default_stream()));
    auto batch          = gpu_batch(std::make_unique<cudf::table>(std::move(columns)));
    auto const expected = digest(*batch);
    move_to(batch, env().host);
    CHECK(holds<cucascade::host_data_representation>(*batch));
    move_to(batch, env().gpu);
    CHECK(digest(*batch) == expected);
  }
  SECTION("no column has a compressing plan")
  {
    scoped_settings settings(enabled_settings());
    auto stream = utils::default_stream();
    // Too few rows for any string plan.
    std::vector<std::string> unique(1000);
    for (std::size_t i = 0; i < unique.size(); ++i) {
      unique[i] = "value_" + std::to_string(i);
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(
      utils::make_string_column(unique, stream, utils::get_resource_ref(*env().gpu)));
    auto batch          = gpu_batch(std::make_unique<cudf::table>(std::move(columns)));
    auto const expected = digest(*batch);
    move_to(batch, env().host);
    CHECK(holds<cucascade::host_data_representation>(*batch));
    move_to(batch, env().gpu);
    CHECK(digest(*batch) == expected);
  }
}
