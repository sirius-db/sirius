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

#ifdef CUCASCADE_HAS_CUFILE

#include "cuda/device_health.hpp"
#include "io/bulk_read.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/parquet_materialize.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/io/parquet_io_utils.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cufile/cufile_api.hpp>
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/detail.hpp>
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using cucascade::io::io_context_type;
using sirius::op::scan::parquet_source;
using sirius::scan_manager::io_backend;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::sirius_scan_manager;
using namespace std::chrono_literals;

std::shared_ptr<const sirius::memory::topology_index> single_gpu_index()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

scan_manager_config reader_config(io_backend backend)
{
  scan_manager_config config;
  config.backend                 = backend;
  config.thread_pool.num_threads = 3;
  config.uring_n_reactors        = 1;
  // A small arena: (2 read + 1 write) slots x 4 lanes of 1 MiB.
  config.cufile.lanes      = 4;
  config.cufile.lane_bytes = 1UL << 20;
  // Request a cache deliberately: native must get one, cuFile must still bypass it.
  config.cache.mode          = cucascade::io::cache::cache_mode::cucs;
  config.max_readahead_scans = 2;
  config.apply_cache_mode();
  return config;
}

/// Whether a cuFile context can start here at all (compatibility mode is enough).
bool cufile_usable()
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) { return false; }
  namespace cufile = cucascade::io::cufile;
  return cufile::cufile_api::get() != nullptr && cufile::cufile_api::open_driver();
}

/// Whether cuFile registered the file behind @p source: its device reads go through host memory
/// otherwise.
bool registered(cucascade::io::datasource const& source)
{
  auto const* file =
    dynamic_cast<cucascade::io::cufile::cufile_io_object const*>(&source.get_io_object());
  return file != nullptr && file->cufile_registered();
}

/// No cuFile arena is kept for any GPU during the scope, nor after it (see
/// cucascade::io::cufile::reserve_arena): the case's reservations allocate their own.
struct fresh_arenas {
  fresh_arenas() { cucascade::io::cufile::detail::release_arenas_for_testing(); }
  ~fresh_arenas() { cucascade::io::cufile::detail::release_arenas_for_testing(); }
  fresh_arenas(fresh_arenas const&)            = delete;
  fresh_arenas& operator=(fresh_arenas const&) = delete;
};

std::atomic<int> buffer_registrations{0};
/// cuFileRead of this file handle fails with EIO (null: none).
std::atomic<void const*> failing_handle{nullptr};
std::mutex files_in_flight_mutex;
std::map<void const*, int> files_in_flight;  // cuFile handle -> its cuFileRead calls in flight
std::size_t most_files_in_flight{0};

/// libcufile's table, counting cuFileBufRegister calls and the files whose cuFileRead calls are
/// in flight at once.  A read lingers (up to 300 ms) until another file's read has joined it, so
/// reads issued together are seen together.  Installed for a scope: declare it before the
/// contexts and files it serves.
class counting_cufile_api {
 public:
  counting_cufile_api()
  {
    namespace cufile   = cucascade::io::cufile;
    table              = *cufile::cufile_api::loaded();
    table.buf_register = [](void const* base, std::size_t size, int flags) -> CUfileError_t {
      ++buffer_registrations;
      return cufile::cufile_api::loaded()->buf_register(base, size, flags);
    };
    table.read = [](CUfileHandle_t handle,
                    void* base,
                    std::size_t size,
                    off_t file_offset,
                    off_t base_offset) -> ssize_t {
      if (handle == failing_handle.load()) {
        errno = EIO;
        return -1;
      }
      auto const files = [] {
        std::lock_guard const lock(files_in_flight_mutex);
        return files_in_flight.size();
      };
      {
        std::lock_guard const lock(files_in_flight_mutex);
        ++files_in_flight[handle];
        most_files_in_flight = std::max(most_files_in_flight, files_in_flight.size());
      }
      auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{300};
      while (files() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
      }
      auto const n =
        cufile::cufile_api::loaded()->read(handle, base, size, file_offset, base_offset);
      {
        std::lock_guard const lock(files_in_flight_mutex);
        if (--files_in_flight[handle] == 0) { files_in_flight.erase(handle); }
      }
      return n;
    };
    buffer_registrations = 0;
    failing_handle       = nullptr;
    {
      std::lock_guard const lock(files_in_flight_mutex);
      files_in_flight.clear();
      most_files_in_flight = 0;
    }
    cufile::detail::set_cufile_api_for_testing(&table);
  }
  ~counting_cufile_api() { cucascade::io::cufile::detail::set_cufile_api_for_testing(nullptr); }
  counting_cufile_api(counting_cufile_api const&)            = delete;
  counting_cufile_api& operator=(counting_cufile_api const&) = delete;

  cucascade::io::cufile::cufile_api table;
};

/// Holds every byte of the current GPU's memory it can get, in 16 MiB pieces, for its scope.
struct gpu_memory_hog {
  gpu_memory_hog()
  {
    for (void* chunk = nullptr; cudaMalloc(&chunk, 16UL << 20) == cudaSuccess; chunk = nullptr) {
      chunks.push_back(chunk);
    }
    std::ignore = cudaGetLastError();
  }
  ~gpu_memory_hog()
  {
    for (auto* chunk : chunks) {
      std::ignore = cudaFree(chunk);
    }
  }
  gpu_memory_hog(gpu_memory_hog const&)            = delete;
  gpu_memory_hog& operator=(gpu_memory_hog const&) = delete;

  std::vector<void*> chunks;
};

std::size_t free_device_bytes(int device)
{
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
  std::size_t free  = 0;
  std::size_t total = 0;
  REQUIRE(cudaMemGetInfo(&free, &total) == cudaSuccess);
  return free;
}

void write_parquet(sirius::test::scratch_dir const& directory,
                   std::string const& name = "input.parquet")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection connection(db);
  // Four groups with distinct, non-repeating values and an unselected column.
  // Uncompressed BIGINT chunks (512 KiB each) are large enough to span several small lanes.
  auto result = connection.Query(
    "COPY (SELECT i::BIGINT id, (17*i*i + 31*i + 7)::BIGINT value, -i AS ignored "
    "FROM range(262144) t(i) ORDER BY i) TO " +
    directory.file_literal(name) +
    " (FORMAT PARQUET, COMPRESSION UNCOMPRESSED, ROW_GROUP_SIZE 65536)");
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

// A datasource read's future is a deferred join (cuCascade #255): wait_for() on it only ever
// reports `deferred`, so it cannot be timed out. Get it on its own thread to poll it.
template <typename T>
std::future<T> launched(std::future<T> read)
{
  return std::async(std::launch::async, [read = std::move(read)]() mutable { return read.get(); });
}

// A deterministic caller-stream gate. Its destructor releases even on assertion failure,
// and drains the callback before its atomic flag is destroyed. It never calls CUDA in a callback.
class stream_gate {
 public:
  explicit stream_gate(rmm::cuda_stream const& stream) : _stream(stream)
  {
    REQUIRE(cudaLaunchHostFunc(
              stream.value(),
              [](void* flag) {
                auto& released = *static_cast<std::atomic<bool>*>(flag);
                while (!released.load(std::memory_order_acquire)) {
                  std::this_thread::yield();
                }
              },
              &_released) == cudaSuccess);
  }

  ~stream_gate()
  {
    release();
    (void)cudaStreamSynchronize(_stream.value());
  }

  void release() noexcept { _released.store(true, std::memory_order_release); }

 private:
  rmm::cuda_stream const& _stream;
  std::atomic<bool> _released{false};
};

struct device_chunks {
  std::vector<rmm::device_buffer> buffers;
  std::vector<std::vector<std::uint8_t>> observed_bytes;
};

device_chunks read_chunks(cucascade::io::datasource& source,
                          std::span<cudf::io::text::byte_range_info const> ranges,
                          rmm::cuda_stream const& caller,
                          rmm::cuda_stream const& observer,
                          bool check_stream_gate)
{
  device_chunks chunks;
  chunks.buffers.reserve(ranges.size());
  std::vector<cucascade::io::slice> reads;
  std::size_t total = 0;
  for (auto const& range : ranges) {
    auto const size = static_cast<std::size_t>(range.size());
    chunks.buffers.emplace_back(size, caller);
    reads.emplace_back(static_cast<std::size_t>(range.offset()),
                       size,
                       static_cast<std::uint8_t*>(chunks.buffers.back().data()));
    total += size;
  }

  std::unique_ptr<stream_gate> gate;
  if (check_stream_gate) { gate = std::make_unique<stream_gate>(caller); }
  // Reads must wait for these overwrites already queued on the caller stream.
  for (auto& buffer : chunks.buffers) {
    REQUIRE(cudaMemsetAsync(buffer.data(), 0xab, buffer.size(), caller.value()) == cudaSuccess);
  }
  auto future = source.device_read_ranges_async(reads, caller);
  if (gate) {
    // Only a launched read can be seen still pending behind the gated caller stream.
    future = launched(std::move(future));
    CHECK(future.wait_for(20ms) == std::future_status::timeout);
    gate->release();
  }
  // Settle before checking counts or releasing any destination memory.
  auto const bytes = future.get();
  REQUIRE(bytes == total);

  // No caller-stream sync or event dependency here. A successful future must mean all
  // destinations are host-complete, including the copies on the lanes' private streams.
  chunks.observed_bytes.resize(chunks.buffers.size());
  for (std::size_t i = 0; i < chunks.buffers.size(); ++i) {
    auto const& buffer = chunks.buffers[i];
    chunks.observed_bytes[i].resize(buffer.size());
    REQUIRE(cudaMemcpyAsync(chunks.observed_bytes[i].data(),
                            buffer.data(),
                            buffer.size(),
                            cudaMemcpyDeviceToHost,
                            observer.value()) == cudaSuccess);
  }
  observer.synchronize();
  return chunks;
}

std::vector<std::vector<std::int64_t>> host_columns(cudf::table_view table,
                                                    rmm::cuda_stream const& stream)
{
  REQUIRE(table.num_columns() == 2);
  std::vector<std::vector<std::int64_t>> columns(2);
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    auto const column = table.column(i);
    REQUIRE(column.type().id() == cudf::type_id::INT64);
    REQUIRE(column.null_count() == 0);
    columns[i].resize(table.num_rows());
    REQUIRE(cudaMemcpyAsync(columns[i].data(),
                            column.data<std::int64_t>(),
                            columns[i].size() * sizeof(std::int64_t),
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
  }
  stream.synchronize();
  return columns;
}

std::vector<std::vector<std::int64_t>> decode_chunks(
  device_chunks const& chunks,
  cudf::io::parquet::FileMetaData const& metadata,
  cudf::io::parquet_reader_options const& options,
  std::vector<cudf::size_type> const& groups,
  rmm::cuda_stream const& stream)
{
  std::vector<cudf::device_span<std::uint8_t const>> spans;
  for (auto const& buffer : chunks.buffers) {
    spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
  }
  cudf::io::parquet::experimental::hybrid_scan_reader reader(metadata, options);
  auto decoded = reader.materialize_all_columns(
    groups, spans, options, stream, cudf::get_current_device_resource_ref());
  return host_columns(decoded.tbl->view(), stream);
}

/// The id and value columns write_parquet() stores in @p groups.
std::vector<std::vector<std::int64_t>> expected_rows(
  cudf::io::parquet::FileMetaData const& metadata, std::vector<cudf::size_type> const& groups)
{
  std::vector<std::vector<std::int64_t>> expected(2);
  std::int64_t first_row = 0;
  for (std::size_t group = 0; group < metadata.row_groups.size(); ++group) {
    auto const rows = metadata.row_groups[group].num_rows;
    if (std::find(groups.begin(), groups.end(), static_cast<cudf::size_type>(group)) !=
        groups.end()) {
      for (auto id = first_row; id < first_row + rows; ++id) {
        expected[0].push_back(id);
        expected[1].push_back(17 * id * id + 31 * id + 7);
      }
    }
    first_row += rows;
  }
  return expected;
}

}  // namespace

TEST_CASE("cuFile scan reads and decodes selected Parquet chunks like native", "[cufile][scan]")
{
  if (!cufile_usable()) {
    SKIP("cuFile scan integration requires a CUDA device and a cuFile driver");
  }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  sirius::test::scratch_dir directory("cufile_scan");
  write_parquet(directory);

  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index();
  auto config   = reader_config(io_backend::cufile);
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager cufile{lifecycle, config, *memory, topology};
  sirius_scan_manager native{lifecycle, reader_config(io_backend::native), *memory, topology};

  auto const path = directory.file("input.parquet");
  cucascade::io::io_context_registry routes(config.to_io_config(), *memory);
  CHECK(routes.lookup_path(path) == io_context_type::cufile);
  CHECK(routes.lookup_path("s3://unconfigured-bucket/input.parquet") == io_context_type::restful);

  auto source   = cufile.create_datasource(path);
  auto baseline = native.create_datasource(path);
  REQUIRE(source);
  REQUIRE(baseline);
  REQUIRE(source->io_ctx().get() == cufile.io_ctx());
  CHECK(std::dynamic_pointer_cast<cucascade::io::cufile::cufile_ioctx>(source->io_ctx()));
  CHECK(source->io_ctx()->type() == io_context_type::cufile);
  CHECK(baseline->io_ctx()->type() == io_context_type::uring);
  CHECK(cufile.create_datasource("file://" + path)->io_ctx() == source->io_ctx());
  CHECK(source->supports_device_read());
  CHECK(source->io_ctx()->supports_device_range_read());
  CHECK_FALSE(source->io_ctx()->can_use_fs_cache());
  CHECK(source->io_ctx()->cache() == nullptr);
  CHECK_FALSE(source->uses_fs_cache());
  REQUIRE(baseline->io_ctx()->cache() != nullptr);  // the cache request is not vacuous
  CHECK(baseline->uses_fs_cache());

  // Host reads go through the context's host-only io_uring reactor: the footer matches.
  auto options = cudf::io::parquet_reader_options::builder().column_names({"id", "value"}).build();
  auto footer  = cudf::io::parquet::fetch_footer_to_host(*baseline);
  auto const cufile_footer = cudf::io::parquet::fetch_footer_to_host(*source);
  REQUIRE(cufile_footer->size() == footer->size());
  CHECK(std::equal(footer->data(), footer->data() + footer->size(), cufile_footer->data()));
  cudf::io::parquet::experimental::hybrid_scan_reader reader(
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), options);
  auto metadata =
    std::make_shared<cudf::io::parquet::FileMetaData const>(reader.parquet_metadata());
  REQUIRE(metadata->row_groups.size() >= 4);
  std::vector<cudf::size_type> const groups{
    1, static_cast<cudf::size_type>(metadata->row_groups.size() - 1)};
  auto const ranges = sirius::op::scan::column_chunk_ranges(*metadata, options, groups);
  REQUIRE(ranges.size() == groups.size() * 2);  // projection excludes "ignored"
  REQUIRE(std::any_of(
    ranges.begin(), ranges.end(), [](auto const& range) { return range.size() > (256 << 10); }));
  source->fadvise(ranges, 0);
  CHECK(source->cache_chunk_count() == 0);
  CHECK(source->prepare_prefetch(false) == cucascade::io::prepare_result::nothing_to_prepare);
  if (!registered(*source)) { SKIP("cuFile did not register the file here"); }

  rmm::cuda_stream caller(rmm::cuda_stream::flags::non_blocking);
  rmm::cuda_stream observer(rmm::cuda_stream::flags::non_blocking);
  rmm::cuda_stream decoder(rmm::cuda_stream::flags::non_blocking);
  auto const native_chunks = read_chunks(*baseline, ranges, caller, observer, false);
  auto const cufile_chunks = read_chunks(*source, ranges, caller, observer, true);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    CAPTURE(i, ranges[i].offset(), ranges[i].size());
    auto const& got      = cufile_chunks.observed_bytes[i];
    auto const& expected = native_chunks.observed_bytes[i];
    CHECK(std::equal(got.begin(), got.end(), expected.begin(), expected.end()));
  }

  auto const expected = expected_rows(*metadata, groups);
  REQUIRE_FALSE(expected[0].empty());
  auto const decoded_native = decode_chunks(native_chunks, *metadata, options, groups, decoder);
  CHECK(std::equal(decoded_native.begin(), decoded_native.end(), expected.begin(), expected.end()));
  auto const decoded_cufile = decode_chunks(cufile_chunks, *metadata, options, groups, decoder);
  CHECK(std::equal(
    decoded_cufile.begin(), decoded_cufile.end(), decoded_native.begin(), decoded_native.end()));

  // Sirius's materializer: native keeps cuDF's general reader, cuFile takes the bulk route
  // (cufile_bulk_read: on), one vectored request per file.
  for (auto const& datasource : {baseline, source}) {
    std::array<parquet_source, 1> const sources{{{datasource, metadata, groups}}};
    CHECK(sirius::op::scan::prefers_bulk_materialize(sources, options) == (datasource == source));
    auto table = sirius::op::scan::materialize_parquet(
      sources, options, {}, decoder, cudf::get_current_device_resource_ref());
    REQUIRE(table);
    auto const actual = host_columns(table->view(), decoder);
    CHECK(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
  }
}

TEST_CASE("mandatory cuFile construction preserves the actual exception cause",
          "[cufile][scan][scan_manager]")
{
  auto config = reader_config(io_backend::cufile);
  // Validation accepts any device list; building the context for a GPU Sirius does not
  // read into is what fails.
  config.cufile.devices = {7};
  std::string cause;
  try {
    std::ignore = config.resolved_cufile(std::vector<int>{0});
  } catch (std::exception const& e) {
    cause = e.what();
  }
  REQUIRE_FALSE(cause.empty());
  auto memory = initialize_memory_manager(1);
  sirius::exec::query_lifecycle_registry lifecycle;
  REQUIRE_THROWS_WITH(
    (sirius_scan_manager{lifecycle, config, *memory, single_gpu_index()}),
    Catch::Matchers::ContainsSubstring("failed to create cufile io_context: " + cause));
}

TEST_CASE("cuFile lanes read every column chunk of one vectored request", "[cufile][scan]")
{
  if (!cufile_usable()) {
    SKIP("cuFile scan integration requires a CUDA device and a cuFile driver");
  }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  sirius::test::scratch_dir directory("cufile_lanes_scan");
  write_parquet(directory);

  auto config = reader_config(io_backend::cufile);
  // A chunk is cut into several segments, served by more segments than there are lanes.
  SECTION("lanes smaller than a column chunk") { config.cufile.lane_bytes = 64UL << 10; }
  // Each row group's adjacent id and value chunks share one segment.
  SECTION("lanes larger than a row group") { config.cufile.lane_bytes = 8UL << 20; }

  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager reader{lifecycle, config, *memory, topology};
  sirius_scan_manager native{lifecycle, reader_config(io_backend::native), *memory, topology};

  auto const path = directory.file("input.parquet");
  auto source     = reader.create_datasource(path);
  auto baseline   = native.create_datasource(path);
  REQUIRE(source);
  REQUIRE(baseline);
  REQUIRE(source->io_ctx().get() == reader.io_ctx());
  if (!registered(*source)) { SKIP("cuFile did not register the file here"); }

  auto options = cudf::io::parquet_reader_options::builder().column_names({"id", "value"}).build();
  auto footer  = cudf::io::parquet::fetch_footer_to_host(*baseline);
  cudf::io::parquet::experimental::hybrid_scan_reader footer_reader(
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), options);
  auto const metadata = footer_reader.parquet_metadata();
  std::vector<cudf::size_type> groups(metadata.row_groups.size());
  std::iota(groups.begin(), groups.end(), 0);
  auto const ranges = sirius::op::scan::column_chunk_ranges(metadata, options, groups);

  // One vectored read of every selected chunk, behind work queued on the caller's stream.
  rmm::cuda_stream caller(rmm::cuda_stream::flags::non_blocking);
  rmm::cuda_stream observer(rmm::cuda_stream::flags::non_blocking);
  auto const expected = read_chunks(*baseline, ranges, caller, observer, false);
  auto const actual   = read_chunks(*source, ranges, caller, observer, true);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    CAPTURE(i, ranges[i].offset(), ranges[i].size());
    auto const& got  = actual.observed_bytes[i];
    auto const& want = expected.observed_bytes[i];
    CHECK(std::equal(got.begin(), got.end(), want.begin(), want.end()));
  }
}

// [whole_gpu]: it measures the GPUs' free memory, which tests running beside it would move.
TEST_CASE("the cuFile arena is reserved ahead of the pool, on Sirius's GPU only",
          "[cufile][scan][scan_manager][whole_gpu]")
{
  if (!cufile_usable()) { SKIP("the cuFile arena requires a CUDA device and a cuFile driver"); }
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  auto memory = initialize_memory_manager(1);
  std::vector<int> const gpu_0{0};

  // An idle arena is kept per GPU for the process, and serves any later context whose
  // segments fit its slots: start without one, so an arena an earlier case left behind cannot
  // satisfy the checks below.
  fresh_arenas const fresh;

  auto config              = reader_config(io_backend::cufile);
  config.cufile.lanes      = 3;
  config.cufile.lane_bytes = 16UL << 20;
  auto const arena         = config.cufile.arena_bytes();
  REQUIRE(arena == (144UL << 20));  // (2 read slots + 1 write slot) x 3 lanes x 16 MiB
  auto const used_since = [](int gpu, std::size_t free_before) {
    auto const free_now = free_device_bytes(gpu);
    return free_before > free_now ? free_before - free_now : std::size_t{0};
  };
  auto const before = free_device_bytes(0);
  auto const other  = device_count > 1 ? free_device_bytes(1) : std::size_t{0};

  // Nothing for another backend.
  auto native              = reader_config(io_backend::native);
  native.cufile.lanes      = config.cufile.lanes;
  native.cufile.lane_bytes = config.cufile.lane_bytes;
  sirius::scan_manager::reserve_cufile_arenas(native, gpu_0);
  CHECK(used_since(0, before) < arena / 2);

  sirius::scan_manager::reserve_cufile_arenas(config, gpu_0);
  CHECK(used_since(0, before) >= arena / 2);
  if (device_count > 1) { CHECK(used_since(1, other) < arena / 2); }

  // The context built afterwards takes the reserved arena instead of allocating another.
  auto const reserved = free_device_bytes(0);
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager reader{lifecycle, config, *memory, single_gpu_index()};
  CHECK(used_since(0, reserved) < arena / 2);
}

TEST_CASE("spills through cuFile get their arena reserved ahead of the pool",
          "[cufile][scan][scan_manager]")
{
  if (!cufile_usable()) { SKIP("the cuFile arena requires a CUDA device and a cuFile driver"); }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  std::vector<int> const gpu_0{0};
  fresh_arenas const fresh;
  counting_cufile_api const api;
  // The shape of the DISK tier's shared cuFile context: cuCascade's defaults.
  auto disk    = cucascade::io::io_config{}.cufile;
  disk.devices = gpu_0;
  auto const registrations =
    static_cast<int>((disk.read_slots_per_lane + 1) * disk.lanes);  // one per slot

  // Native scans: only the spills want an arena.
  auto const native = reader_config(io_backend::native);
  sirius::scan_manager::reserve_cufile_arenas(native, gpu_0, /*spills_through_cufile=*/false);
  CHECK(buffer_registrations == 0);
  sirius::scan_manager::reserve_cufile_arenas(native, gpu_0, /*spills_through_cufile=*/true);
  CHECK(buffer_registrations == registrations);

  // The shared context, as cuCascade builds it at the first spill, takes that arena.
  cucascade::io::cufile::cufile_ioctx spill_context{disk};
  spill_context.start();
  CHECK(buffer_registrations == registrations);
  spill_context.shutdown();
}

// [whole_gpu]: it takes all of GPU 0's free memory, which tests running beside it would miss.
TEST_CASE("a spill arena that cannot be reserved leaves startup and the scan arena alone",
          "[cufile][scan][scan_manager][whole_gpu]")
{
  if (!cufile_usable()) { SKIP("the cuFile arena requires a CUDA device and a cuFile driver"); }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  std::vector<int> const gpu_0{0};
  fresh_arenas const fresh;
  counting_cufile_api const api;
  // A scan arena too small for the DISK context: covering both takes another allocation.
  auto const config = reader_config(io_backend::cufile);
  sirius::scan_manager::reserve_cufile_arenas(config, gpu_0);
  auto const reserved = buffer_registrations.load();
  REQUIRE(reserved > 0);
  {
    gpu_memory_hog const hog;
    CHECK_NOTHROW(
      sirius::scan_manager::reserve_cufile_arenas(config, gpu_0, /*spills_through_cufile=*/true));
  }

  // The scan arena is still the GPU's: a scan context of its shape registers nothing new.
  cucascade::io::cufile::cufile_ioctx scans{config.resolved_cufile(gpu_0)};
  scans.start();
  CHECK(buffer_registrations == reserved);
  scans.shutdown();
}

TEST_CASE("a spill arena registered differently from the scan arena is not reserved",
          "[cufile][scan][scan_manager]")
{
  if (!cufile_usable()) { SKIP("the cuFile arena requires a CUDA device and a cuFile driver"); }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  std::vector<int> const gpu_0{0};
  fresh_arenas const fresh;
  counting_cufile_api const api;
  auto config                     = reader_config(io_backend::cufile);
  config.cufile.register_per_slot = false;  // the DISK context registers per slot
  sirius::scan_manager::reserve_cufile_arenas(config, gpu_0, /*spills_through_cufile=*/true);
  CHECK(buffer_registrations == 1);  // the scan arena, registered as a whole, and nothing else

  // The scan context takes the reserved arena instead of allocating after the pools.
  cucascade::io::cufile::cufile_ioctx scans{config.resolved_cufile(gpu_0)};
  scans.start();
  CHECK(buffer_registrations == 1);
  scans.shutdown();
}

TEST_CASE("cufile_bulk_read picks the Parquet read route of cuFile scans",
          "[cufile][scan][scan_manager][bulk_read]")
{
  using mode = sirius::scan_manager::cufile_bulk_read_mode;
  if (!cufile_usable()) {
    SKIP("cuFile scan integration requires a CUDA device and a cuFile driver");
  }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  sirius::test::scratch_dir directory("cufile_bulk_route");
  write_parquet(directory);
  auto const path = directory.file("input.parquet");
  auto memory     = initialize_memory_manager(1);
  auto topology   = single_gpu_index();
  sirius::exec::query_lifecycle_registry lifecycle;

  // off keeps cuDF's general reader.
  auto const [bulk_mode, expected] = GENERATE(table<mode, bool>({
    {mode::on, true},
    {mode::off, false},
  }));
  CAPTURE(static_cast<int>(bulk_mode));
  auto config             = reader_config(io_backend::cufile);
  config.cufile_bulk_read = bulk_mode;
  CHECK(config.reads_cufile_in_bulk() == expected);

  {
    sirius_scan_manager reader{lifecycle, config, *memory, topology};
    auto source = reader.create_datasource(path);
    REQUIRE(source);
    CHECK_FALSE(source->prefers_bulk_io());  // cuCascade's own answer for cuFile
    CHECK(sirius::io::prefers_bulk_read(*source) == expected);
    CHECK(sirius::io::prefers_bulk_read(*reader.io_ctx()) == expected);

    auto options =
      cudf::io::parquet_reader_options::builder().column_names({"id", "value"}).build();
    auto footer = cudf::io::parquet::fetch_footer_to_host(*source);
    cudf::io::parquet::experimental::hybrid_scan_reader footer_reader(
      cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), options);
    auto const metadata =
      std::make_shared<cudf::io::parquet::FileMetaData const>(footer_reader.parquet_metadata());
    std::vector<cudf::size_type> groups(metadata->row_groups.size());
    std::iota(groups.begin(), groups.end(), 0);
    std::array<parquet_source, 1> const sources{{{source, metadata, groups}}};
    CHECK(sirius::op::scan::prefers_bulk_materialize(sources, options) == expected);

    // Both routes return the same rows.
    rmm::cuda_stream decoder(rmm::cuda_stream::flags::non_blocking);
    auto table = sirius::op::scan::materialize_parquet(
      sources, options, {}, decoder, cudf::get_current_device_resource_ref());
    REQUIRE(table);
    auto const actual = host_columns(table->view(), decoder);
    auto const rows   = expected_rows(*metadata, groups);
    CHECK(std::equal(actual.begin(), actual.end(), rows.begin(), rows.end()));
  }
  // The registration ends with the scan manager that made it.
  CHECK(sirius::io::detail::bulk_read_registry().count.load() == 0);

  // The native backend is never affected (REST keeps its own compile-time preference).
  sirius_scan_manager native{lifecycle, reader_config(io_backend::native), *memory, topology};
  auto local = native.create_datasource(path);
  REQUIRE(local);
  CHECK_FALSE(sirius::io::prefers_bulk_read(*local));
}

TEST_CASE("a multi-file split on the cuFile bulk route reads every file at once",
          "[cufile][scan][scan_manager][bulk_read]")
{
  if (!cufile_usable()) {
    SKIP("cuFile scan integration requires a CUDA device and a cuFile driver");
  }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  sirius::test::scratch_dir directory("cufile_bulk_files");
  std::array<std::string, 3> const names{"first.parquet", "second.parquet", "third.parquet"};
  for (auto const& name : names) {
    write_parquet(directory, name);
  }
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index();
  sirius::exec::query_lifecycle_registry lifecycle;
  counting_cufile_api const api;
  sirius_scan_manager reader{lifecycle, reader_config(io_backend::cufile), *memory, topology};

  auto options = cudf::io::parquet_reader_options::builder().column_names({"id", "value"}).build();
  std::vector<parquet_source> sources;
  std::vector<std::vector<std::int64_t>> rows(2);
  for (std::size_t i = 0; i < names.size(); ++i) {
    auto source = reader.create_datasource(directory.file(names[i]));
    REQUIRE(source);
    auto footer = cudf::io::parquet::fetch_footer_to_host(*source);
    cudf::io::parquet::experimental::hybrid_scan_reader footer_reader(
      cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), options);
    auto const metadata =
      std::make_shared<cudf::io::parquet::FileMetaData const>(footer_reader.parquet_metadata());
    // A different row group of each file.
    std::vector<cudf::size_type> const groups{static_cast<cudf::size_type>(i)};
    auto const expected = expected_rows(*metadata, groups);
    for (std::size_t column = 0; column < rows.size(); ++column) {
      rows[column].insert(rows[column].end(), expected[column].begin(), expected[column].end());
    }
    sources.push_back(parquet_source{std::move(source), metadata, groups});
  }
  REQUIRE(sirius::op::scan::prefers_bulk_materialize(sources, options));

  rmm::cuda_stream decoder(rmm::cuda_stream::flags::non_blocking);
  auto table = sirius::op::scan::materialize_parquet(
    sources, options, {}, decoder, cudf::get_current_device_resource_ref());
  REQUIRE(table);
  CHECK(host_columns(table->view(), decoder) == rows);
  // The files' reads were in flight together, not one file after another.
  std::lock_guard const lock(files_in_flight_mutex);
  CHECK(most_files_in_flight >= 2);
}

TEST_CASE("a multi-file split reports a fatal device failure over an ordinary one",
          "[cufile][scan][scan_manager][bulk_read]")
{
  if (!cufile_usable()) {
    SKIP("cuFile scan integration requires a CUDA device and a cuFile driver");
  }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  sirius::test::scratch_dir directory("cufile_bulk_fatal");
  std::array<std::string, 2> const names{"ordinary.parquet", "fatal.parquet"};
  for (auto const& name : names) {
    write_parquet(directory, name);
  }
  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index();
  sirius::exec::query_lifecycle_registry lifecycle;
  fresh_arenas const fresh;  // the fatal read retires a slot
  counting_cufile_api const api;
  sirius_scan_manager reader{lifecycle, reader_config(io_backend::cufile), *memory, topology};

  auto options = cudf::io::parquet_reader_options::builder().column_names({"id", "value"}).build();
  std::vector<parquet_source> sources;
  for (auto const& name : names) {
    auto source = reader.create_datasource(directory.file(name));
    REQUIRE(source);
    if (!registered(*source)) { SKIP("cuFile did not register the file here"); }
    auto footer = cudf::io::parquet::fetch_footer_to_host(*source);
    cudf::io::parquet::experimental::hybrid_scan_reader footer_reader(
      cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), options);
    auto const metadata =
      std::make_shared<cudf::io::parquet::FileMetaData const>(footer_reader.parquet_metadata());
    sources.push_back(parquet_source{std::move(source), metadata, {0}});
  }
  // The first file's cuFileRead fails (an ordinary I/O error); the second file's copies end in a
  // status CUDA documents as terminal.
  failing_handle = dynamic_cast<cucascade::io::cufile::cufile_io_object const&>(
                     sources[0].datasource->get_io_object())
                     .cufile_handle();
  struct lift_injection {
    ~lift_injection() { cucascade::io::cufile::detail::fail_copy_queries_for_testing(0, 0); }
  } const lifted;
  cucascade::io::cufile::detail::fail_copy_queries_for_testing(cudaErrorLaunchFailure,
                                                               cudaErrorIllegalAddress);

  rmm::cuda_stream decoder(rmm::cuda_stream::flags::non_blocking);
  try {
    std::ignore = sirius::op::scan::materialize_parquet(
      sources, options, {}, decoder, cudf::get_current_device_resource_ref());
    FAIL("the split was read despite its failures");
  } catch (...) {
    CHECK(sirius::fatal_device_status(std::current_exception()) == cudaErrorIllegalAddress);
  }
}

TEST_CASE("a SQL scan through the cuFile backend reads on its lanes",
          "[cufile][scan][isolated_context]")
{
  if (!cufile_usable()) {
    SKIP("cuFile scan integration requires a CUDA device and a cuFile driver");
  }
  // auto = the bulk route, off = cuDF's general reader.
  auto const bulk_read = GENERATE(as<std::string>{}, "auto", "off");
  CAPTURE(bulk_read);
  sirius::test::scratch_dir directory("cufile_lanes_sql");
  write_parquet(directory);
  auto const config_path = directory.file("sirius_cufile_lanes.yaml");
  {
    std::ofstream yaml(config_path);
    yaml << "sirius:\n"
            "  topology:\n"
            "    num_gpus: 1\n"
            "  memory:\n"
            "    gpu:\n"
            "      usage_limit_bytes: 2GiB\n"
            "    host:\n"
            "      capacity_bytes: 1GiB\n"
            "  executor:\n"
            "    pipeline:\n"
            "      num_threads: 2\n"
            "    scan_manager:\n"
            "      backend: cufile\n"
            "      uring_n_reactors: 1\n"
            "      cufile_bulk_read: "
         << bulk_read
         << "\n"
            "      cufile:\n"
            "        lanes: 4\n"
            "        lane_bytes: 1MiB\n";
    REQUIRE(yaml);
  }

  auto db = sirius::test::open_sirius_db(nullptr, config_path);
  {
    duckdb::Connection con(*db);
    auto context = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(context);
    auto& scans = context->get_scan_manager();
    REQUIRE(dynamic_cast<cucascade::io::cufile::cufile_ioctx const*>(scans.io_ctx()) != nullptr);
    auto const source = scans.create_datasource(directory.file("input.parquet"));
    REQUIRE(source);
    if (!registered(*source)) { SKIP("cuFile did not register the file here"); }

    auto const run = [&](std::string const& sql) {
      auto result = con.Query(sql);
      REQUIRE(result);
      INFO((result->HasError() ? result->GetError() : std::string{}));
      REQUIRE_FALSE(result->HasError());
      return result;
    };
    // A GPU failure must surface rather than fall back to DuckDB's own reader.
    std::ignore = run("SET enable_duckdb_fallback = false");
    auto const query =
      "SELECT count(*), min(id), max(id), min(value), max(value) FROM "
      "read_parquet(" +
      directory.file_literal("input.parquet") + ") WHERE id >= 100000";
    auto gpu    = run(query);
    std::ignore = run("SET gpu_execution = false");
    auto cpu    = run(query);
    REQUIRE(gpu->RowCount() == 1);
    REQUIRE(cpu->RowCount() == 1);
    REQUIRE(gpu->ColumnCount() == cpu->ColumnCount());
    for (duckdb::idx_t c = 0; c < gpu->ColumnCount(); ++c) {
      CAPTURE(c);
      CHECK(gpu->Collection().GetValue(c, 0).ToString() ==
            cpu->Collection().GetValue(c, 0).ToString());
    }
    constexpr std::int64_t last = 262143;
    CHECK(gpu->Collection().GetValue(0, 0).GetValue<std::int64_t>() == last - 100000 + 1);
    CHECK(gpu->Collection().GetValue(4, 0).GetValue<std::int64_t>() ==
          17 * last * last + 31 * last + 7);
  }
  db.reset();
}

#endif  // CUCASCADE_HAS_CUFILE
