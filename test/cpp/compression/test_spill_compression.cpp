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
#include "spill_io_test_context.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_disk_representation.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/compression_converters.hpp>
#include <compression/spill_compression.hpp>
#include <compression/spill_io.hpp>
#include <cucascade/cudf/disk_io_context.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <liburing.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <system_error>
#include <utility>
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
  fs::path dir =
    fs::temp_directory_path() / ("sirius_spill_compression_test_" + std::to_string(::getpid()));
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

/// The environment is process-static. Retain a non-null previous/default context so neither
/// an explicitly selected backend nor a test proxy can leak into later tests.
struct scoped_disk_context {
  std::shared_ptr<cucascade::io::ioctx> previous;

  explicit scoped_disk_context(std::shared_ptr<cucascade::io::ioctx> context)
    : previous(env().disk->get_or_init_io_context(cucascade::default_disk_io_context))
  {
    env().disk->set_io_context(std::move(context));
  }
  ~scoped_disk_context() { env().disk->set_io_context(std::move(previous)); }
  scoped_disk_context(scoped_disk_context const&)            = delete;
  scoped_disk_context& operator=(scoped_disk_context const&) = delete;
};

std::shared_ptr<cucascade::io::ioctx> make_spill_disk_context(cucascade::disk_io_backend backend)
{
  if (backend == cucascade::disk_io_backend::uring) {
    // Probe independently of the factory: only known process/kernel restrictions may skip.
    // Resource exhaustion, unexpected setup errors and every backend construction error fail.
    io_uring ring{};
    auto const result = io_uring_queue_init(1, &ring, 0);
    if (result == 0) { io_uring_queue_exit(&ring); }
    if (result == -EPERM || result == -ENOSYS || result == -EACCES) {
      SKIP("forced uring is unavailable: io_uring_queue_init returned " << result);
    }
    REQUIRE(result == 0);
  }
  cucascade::disk_io_context_options options;
  options.backend = backend;
  return cucascade::make_disk_io_context(options);
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
  auto mask = cudf::create_null_mask(
    rows, cudf::mask_state::UNINITIALIZED, stream, utils::get_resource_ref(*env().gpu));
  REQUIRE(cudaMemcpyAsync(mask.data(),
                          words.data(),
                          words.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice,
                          stream.get()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
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
  REQUIRE(convertible.convert({space}, env().stream, *env().mgr, true).has_value());
}

template <typename Representation>
bool holds(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return dynamic_cast<Representation const*>(ro.get_data()) != nullptr;
}

/// Drive registered converters directly: convertible_data_batch intentionally falls
/// back to uncompressed spill and would hide the errors these tests must observe.
struct direct_spill_input {
  cucascade::representation_converter_registry registry;
  std::shared_ptr<cucascade::data_batch> batch = gpu_batch(compressible_table());
  std::unique_ptr<sirius::compressed_host_representation> host;

  explicit direct_spill_input(bool via_host)
  {
    sirius::register_compression_converters(registry);
    if (via_host) {
      auto ro = batch->to_read_only();
      host    = registry.convert<sirius::compressed_host_representation>(
        *ro.get_data(), env().host, env().stream);
    }
  }

  std::unique_ptr<sirius::compressed_disk_representation> to_disk()
  {
    auto ro = batch->to_read_only();
    auto const& source =
      host ? static_cast<cucascade::idata_representation const&>(*host) : *ro.get_data();
    return registry.convert<sirius::compressed_disk_representation>(
      source, env().disk, env().stream);
  }
};

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
  CHECK_FALSE(gpu_mr->is_stream_tracked(env().stream));
  CHECK(gpu_mr->get_total_reserved_bytes() == reserved_before);

  move_to(batch, env().gpu);
  REQUIRE(holds<cucascade::gpu_table_representation>(*batch));
  CHECK(digest(*batch) == expected);
}

TEST_CASE("spill compression: GPU -> compressed disk -> GPU, and the file goes with the batch",
          "[compression][spill]")
{
  if (!has_gpu()) { return; }
#ifdef CUCASCADE_HAS_CUFILE
  auto const backend = GENERATE(cucascade::disk_io_backend::automatic,
                                cucascade::disk_io_backend::uring,
                                cucascade::disk_io_backend::cufile);
#else
  auto const backend =
    GENERATE(cucascade::disk_io_backend::automatic, cucascade::disk_io_backend::uring);
#endif
  CAPTURE(backend);
  auto context = make_spill_disk_context(backend);
  REQUIRE(context != nullptr);
  REQUIRE(context->supports_write());
  REQUIRE(context->supports_device_write());
  scoped_disk_context disk_context(context);
  scoped_settings settings(enabled_settings());
  auto batch          = gpu_batch(compressible_table());
  auto const expected = digest(*batch);

  auto const rows         = static_cast<std::size_t>(expected.row_hashes.size());
  auto const device_bytes = batch->to_read_only().get_data()->get_size_in_bytes();

  move_to(batch, env().disk);
  REQUIRE(holds<sirius::compressed_disk_representation>(*batch));
  CHECK(env().disk->get_io_context() == context);
  std::string const path = [&] {
    auto ro         = batch->to_read_only();
    auto const& rep = ro.get_data()->cast<sirius::compressed_disk_representation>();
    CHECK(rep.header_bytes() > 0);
    // The format is not padded for direct IO: the host header and device payload meet here.
    CHECK(rep.header_bytes() % 4096 != 0);
    CHECK(rep.get_size_in_bytes() < device_bytes);
    CHECK(rep.get_uncompressed_data_size_in_bytes() == device_bytes);
    CHECK(fs::file_size(rep.path()) == rep.get_size_in_bytes());
    return rep.path();
  }();
  CHECK(fs::exists(path));
  auto clone = [&] {
    auto ro = batch->to_read_only();
    CHECK(sirius::representation_num_rows(*ro.get_data()) == rows);
    CHECK(sirius::representation_num_rows(ro.get_data()) == rows);
    return ro.get_data()->clone(env().stream);
  }();
  CHECK(clone->cast<sirius::compressed_disk_representation>().path() == path);
  auto last_clone = clone->clone(env().stream);
  CHECK(last_clone->cast<sirius::compressed_disk_representation>().path() == path);

  move_to(batch, env().gpu);
  REQUIRE(holds<cucascade::gpu_table_representation>(*batch));
  CHECK(digest(*batch) == expected);
  CHECK(env().disk->get_io_context() == context);
  // Replacing the original representation must not unlink a file that clones still own.
  CHECK(fs::exists(path));
  clone.reset();
  CHECK(fs::exists(path));
  last_clone.reset();
  CHECK_FALSE(fs::exists(path));
}

TEST_CASE("spill compression: a compressed host batch moves to disk without a decode",
          "[compression][spill]")
{
  if (!has_gpu()) { return; }
#ifdef CUCASCADE_HAS_CUFILE
  auto const backend = GENERATE(cucascade::disk_io_backend::automatic,
                                cucascade::disk_io_backend::uring,
                                cucascade::disk_io_backend::cufile);
#else
  auto const backend =
    GENERATE(cucascade::disk_io_backend::automatic, cucascade::disk_io_backend::uring);
#endif
  CAPTURE(backend);
  auto context = make_spill_disk_context(backend);
  REQUIRE(context != nullptr);
  REQUIRE(context->supports_write());
  REQUIRE(context->supports_device_read());
  scoped_disk_context disk_context(context);
  scoped_settings settings(enabled_settings());
  auto batch          = gpu_batch(compressible_table());
  auto const expected = digest(*batch);

  move_to(batch, env().host);
  REQUIRE(holds<sirius::compressed_host_representation>(*batch));
  std::size_t header_bytes  = 0;
  std::size_t payload_bytes = 0;
  {
    auto ro         = batch->to_read_only();
    auto const& rep = ro.get_data()->cast<sirius::compressed_host_representation>();
    header_bytes    = rep.header().size();
    payload_bytes   = rep.get_size_in_bytes();
    CHECK(header_bytes % 4096 != 0);
    // The non-repeating string payload must span pinned-block write slices.
    CHECK(rep.payload().size() > 1);
  }
  move_to(batch, env().disk);
  REQUIRE(holds<sirius::compressed_disk_representation>(*batch));
  CHECK(env().disk->get_io_context() == context);
  std::string const path = [&] {
    auto ro         = batch->to_read_only();
    auto const& rep = ro.get_data()->cast<sirius::compressed_disk_representation>();
    CHECK(rep.header_bytes() == header_bytes);
    CHECK(rep.get_size_in_bytes() == header_bytes + payload_bytes);
    CHECK(fs::file_size(rep.path()) == rep.get_size_in_bytes());
    return rep.path();
  }();
  move_to(batch, env().gpu);
  REQUIRE(holds<cucascade::gpu_table_representation>(*batch));
  CHECK(digest(*batch) == expected);
  CHECK(env().disk->get_io_context() == context);
  CHECK_FALSE(fs::exists(path));
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

TEST_CASE("spill IO: compressed disk conversion lazily adopts the shared default context",
          "[compression][spill][io]")
{
  if (!has_gpu()) { return; }
  scoped_settings settings(enabled_settings());
  direct_spill_input input(false);
  auto const expected = digest(*input.batch);
  cucascade::memory::memory_space disk_space(cucascade::memory::disk_memory_space_config{
    .disk_id = 99, .memory_capacity = 4ull << 30, .mount_paths = env().dir.string()});
  // Do not use scoped_disk_context: only the compressed converter may install it.
  REQUIRE(disk_space.get_io_context() == nullptr);

  auto ro   = input.batch->to_read_only();
  auto disk = input.registry.convert<sirius::compressed_disk_representation>(
    *ro.get_data(), &disk_space, env().stream);
  REQUIRE(disk != nullptr);
  CHECK(&disk->get_memory_space() == &disk_space);
  CHECK(disk->get_size_in_bytes() < ro.get_data()->get_size_in_bytes());
  auto adopted = disk_space.get_io_context();
  REQUIRE(adopted != nullptr);
  CHECK(adopted == cucascade::default_disk_io_context());
  auto const path = disk->path();
  CHECK(fs::exists(path));

  auto decoded =
    input.registry.convert<cucascade::gpu_table_representation>(*disk, env().gpu, env().stream);
  CHECK(digest(decoded->get_table_view()) == expected);
  CHECK(disk_space.get_io_context() == adopted);
  disk.reset();
  CHECK_FALSE(fs::exists(path));
}

TEST_CASE("spill IO: each disk conversion submits exactly one vectored request",
          "[compression][spill][io]")
{
  if (!has_gpu()) { return; }
  auto const via_host = GENERATE(false, true);
  CAPTURE(via_host);
  scoped_settings settings(enabled_settings());
  auto context = std::make_shared<sirius::test::spill_io_test_context>();
  scoped_disk_context install(context);
  direct_spill_input input(via_host);
  auto const expected = digest(*input.batch);

  auto disk           = input.to_disk();
  auto const observed = context->observed();
  CHECK(observed.calls == 1);
  if (via_host) {
    CHECK(observed.host_slices > 1);  // Header plus pinned payload blocks.
    CHECK(observed.device_slices == 0);
  } else {
    CHECK(observed.host_slices == 1);  // Header only; payload stays on the device.
    CHECK(observed.device_slices > 0);
  }
  CHECK(observed.requested_bytes == disk->get_size_in_bytes());
  CHECK(observed.inner_bytes == observed.requested_bytes);
  CHECK(observed.header_bytes == disk->header_bytes());
  CHECK(observed.last_path == disk->path());
  CHECK(fs::file_size(disk->path()) == observed.requested_bytes);
  auto decoded =
    input.registry.convert<cucascade::gpu_table_representation>(*disk, env().gpu, env().stream);
  CHECK(digest(decoded->get_table_view()) == expected);
}

TEST_CASE("spill IO: incorrect counts and failures reject both disk conversion routes",
          "[compression][spill][io]")
{
  if (!has_gpu()) { return; }
  using context_type  = sirius::test::spill_io_test_context;
  auto const via_host = GENERATE(false, true);
  auto const failure =
    GENERATE(context_type::completion::short_count, context_type::completion::failure);
  CAPTURE(via_host, static_cast<int>(failure));
  scoped_settings settings(enabled_settings());
  auto context              = std::make_shared<context_type>();
  context->write_completion = failure;
  scoped_disk_context install(context);
  direct_spill_input input(via_host);

  if (failure == context_type::completion::short_count) {
    REQUIRE_THROWS_WITH(input.to_disk(),
                        Catch::Matchers::ContainsSubstring("compressed spill IO transferred"));
  } else {
    REQUIRE_THROWS_WITH(input.to_disk(), "injected spill write failure");
  }
  auto const observed = context->observed();
  CHECK(observed.calls == 1);
  CHECK(observed.requested_bytes > 0);
  // The real write drained before either injected result was delivered.
  CHECK(observed.inner_bytes == observed.requested_bytes);
  REQUIRE_FALSE(observed.last_path.empty());
  CHECK_FALSE(fs::exists(observed.last_path));
  CHECK(holds<cucascade::gpu_table_representation>(*input.batch));
}

TEST_CASE("spill IO: unsupported write capabilities reject before submission",
          "[compression][spill][io]")
{
  if (!has_gpu()) { return; }
  auto const via_host = GENERATE(false, true);
  CAPTURE(via_host);
  scoped_settings settings(enabled_settings());
  auto context = std::make_shared<sirius::test::spill_io_test_context>();
  scoped_disk_context install(context);
  direct_spill_input input(via_host);

  SECTION("no host write support")
  {
    context->write_enabled = false;
    REQUIRE_THROWS_WITH(input.to_disk(), Catch::Matchers::ContainsSubstring("cannot write"));
  }
  SECTION("no device write support")
  {
    context->device_write_enabled = false;
    if (via_host) {
      // Flushing an already compressed host batch must not require device writes.
      auto disk = input.to_disk();
      CHECK(context->observed().calls == 1);
      CHECK(context->observed().device_slices == 0);
      return;
    }
    REQUIRE_THROWS_WITH(input.to_disk(), Catch::Matchers::ContainsSubstring("cannot write"));
  }
  CHECK(context->observed().calls == 0);
  CHECK(context->observed().last_path.empty());
}

TEST_CASE("spill IO: delayed failed submission retains the file and encoded buffers until drain",
          "[compression][spill][io]")
{
  if (!has_gpu()) { return; }
  using namespace std::chrono_literals;
  scoped_settings settings(enabled_settings());
  auto context              = std::make_shared<sirius::test::spill_io_test_context>();
  context->write_completion = sirius::test::spill_io_test_context::completion::failure;
  context->gate             = std::make_shared<sirius::test::spill_write_gate>();
  scoped_disk_context install(context);
  direct_spill_input input(false);
  auto const expected           = digest(*input.batch);
  auto const uncompressed_bytes = input.batch->to_read_only().get_data()->get_size_in_bytes();

  sirius::test::joined_spill_task<std::unique_ptr<sirius::compressed_disk_representation>>
    conversion(context->gate, [&] { return input.to_disk(); });
  REQUIRE(context->gate->submitted.wait_for(30s) == std::future_status::ready);
  auto const submitted = context->observed();
  REQUIRE(submitted.calls == 1);
  REQUIRE(submitted.device_slices > 0);
  REQUIRE_FALSE(submitted.last_path.empty());
  CHECK(conversion.result.wait_for(50ms) == std::future_status::timeout);
  CHECK(fs::exists(submitted.last_path));
  CHECK(context->observed().inner_bytes == 0);

  // Only now may the worker read the converter's encoded header/device buffers.
  context->gate->allow_submit();
  REQUIRE(context->gate->inner_completed.wait_for(30s) == std::future_status::ready);
  auto const drained = context->observed();
  REQUIRE(drained.inner_bytes == drained.requested_bytes);
  CHECK(conversion.result.wait_for(50ms) == std::future_status::timeout);
  REQUIRE(fs::exists(drained.last_path));

  // Decode a copy of the physically written image while the original future is
  // still pending. Matching the input proves the encoded buffers survived the
  // delayed consumption, rather than merely checking that their addresses existed.
  auto const copy_path = drained.last_path + ".drained-copy";
  REQUIRE(fs::copy_file(drained.last_path, copy_path));
  {
    sirius::compressed_disk_representation copy(
      *env().disk,
      copy_path,
      drained.header_bytes,
      drained.requested_bytes,
      uncompressed_bytes,
      static_cast<std::int64_t>(expected.row_hashes.size()));
    auto decoded =
      input.registry.convert<cucascade::gpu_table_representation>(copy, env().gpu, env().stream);
    CHECK(digest(decoded->get_table_view()) == expected);
  }
  CHECK_FALSE(fs::exists(copy_path));
  CHECK(fs::exists(drained.last_path));

  context->gate->allow_completion();
  REQUIRE(conversion.result.wait_for(30s) == std::future_status::ready);
  REQUIRE_THROWS_WITH(conversion.result.get(), "injected spill write failure");
  CHECK_FALSE(fs::exists(drained.last_path));
}

TEST_CASE("spill IO: unexpected unwinding drains the outstanding request before its owners",
          "[compression][spill][io]")
{
  using namespace std::chrono_literals;
  auto gate = std::make_shared<sirius::test::spill_write_gate>();
  cucascade::exec::promise<std::size_t> completion;
  auto future = completion.get_semi_future();
  std::atomic<bool> owner_destroyed{false};
  std::atomic<bool> saw_live_owner{false};

  // Declare the producer first: the conversion is joined before captured owners
  // can disappear, and either task's destructor releases both gate phases.
  sirius::test::joined_spill_task<void> producer(gate, [&] {
    gate->wait_to_submit();
    saw_live_owner.store(!owner_destroyed.load());
    completion.set_value(std::size_t{1});
  });
  sirius::test::joined_spill_task<void> unwinding(gate, [&, future = std::move(future)]() mutable {
    struct owner {
      std::atomic<bool>& destroyed;
      ~owner() { destroyed.store(true); }
    } buffers{owner_destroyed};
    sirius::compression::detail::spill_request request{std::move(future)};
    gate->submitted_promise.set_value();
    throw std::runtime_error("unwind after submitting spill");
  });

  REQUIRE(gate->submitted.wait_for(30s) == std::future_status::ready);
  CHECK(unwinding.result.wait_for(50ms) == std::future_status::timeout);
  CHECK_FALSE(owner_destroyed.load());
  gate->allow_submit();
  REQUIRE(unwinding.result.wait_for(30s) == std::future_status::ready);
  CHECK_THROWS_WITH(unwinding.result.get(), "unwind after submitting spill");
  CHECK(saw_live_owner.load());
  CHECK(owner_destroyed.load());
}

TEST_CASE("spill IO: direct disk decode rejects a truncated header or payload",
          "[compression][spill][io]")
{
  if (!has_gpu()) { return; }
  auto const truncate_header     = GENERATE(true, false);
  auto const truncate_after_open = GENERATE(false, true);
  CAPTURE(truncate_header, truncate_after_open);
  scoped_settings settings(enabled_settings());
  auto context = std::make_shared<sirius::test::spill_io_test_context>();
  scoped_disk_context install(context);
  direct_spill_input input(false);
  auto disk       = input.to_disk();
  auto const path = disk->path();
  REQUIRE(disk->header_bytes() > 1);
  REQUIRE(disk->get_size_in_bytes() > disk->header_bytes());
  // Leave a complete header for the payload case, but no payload byte at all.
  auto const truncated_bytes = truncate_header ? disk->header_bytes() - 1 : disk->header_bytes();
  if (!truncate_after_open) { fs::resize_file(path, truncated_bytes); }
  // Explicitly reopen for reading: this is never the writable object whose size grew with
  // note_written(). Test both an accurate truncated size and a size made stale after open.
  auto reopened = context->open_io_object(path);
  REQUIRE_FALSE(reopened->writable());
  REQUIRE(reopened->size() == (truncate_after_open ? disk->get_size_in_bytes() : truncated_bytes));
  if (truncate_after_open) { fs::resize_file(path, truncated_bytes); }
  context->next_read_object = reopened;
  auto decode               = [&] {
    return input.registry.convert<cucascade::gpu_table_representation>(
      *disk, env().gpu, env().stream);
  };
  if (truncate_after_open && !truncate_header) {
    // A stale open promised payload bytes that no longer exist. The reactor must report
    // its physical short read as EIO, rather than expose a partly filled device buffer.
    try {
      (void)decode();
      FAIL("a payload truncated after open must fail IO");
    } catch (std::system_error const& error) {
      CHECK(error.code() == std::errc::io_error);
    }
  } else {
    REQUIRE_THROWS_WITH(decode(),
                        Catch::Matchers::ContainsSubstring("compressed spill IO transferred"));
  }
  CHECK(context->next_read_object == nullptr);  // The decode consumed exactly this fresh open.
  auto const observed = context->observed();
  CHECK(observed.read_opens == 2);
  CHECK(observed.opened_read_size == reopened->size());
  CHECK(observed.host_read_calls == 1);
  if (truncate_header) {
    CHECK(observed.host_read_bytes == truncated_bytes);
    CHECK(observed.device_read_calls == 0);  // Rejected by byte count before header parsing.
  } else {
    CHECK(observed.host_read_bytes == disk->header_bytes());
    CHECK(observed.device_read_calls == 1);
    CHECK(observed.device_read_bytes == 0);  // Short count or EIO: no payload reaches decode.
  }
  // A failed decode does not consume the source representation or orphan its file.
  CHECK(fs::exists(path));
  disk.reset();
  CHECK_FALSE(fs::exists(path));
}
