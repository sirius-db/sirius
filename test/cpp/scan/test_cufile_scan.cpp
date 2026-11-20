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
#include <cucascade/io/cufile/cufile_ioctx.hpp>
#include <cucascade/io/cufile/placement.hpp>
#include <cucascade/io/cufile/platform.hpp>
#include <cucascade/io/cufile/staging.hpp>
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
#include <cstdlib>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
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
  config.cufile_n_reactors       = 4;
  // Exercise the forced POSIX/compatibility path, not real-GDS acceptance.
  config.cufile.compat_mode = cucascade::io::cufile::compat::on;
  config.cufile.use_odirect = false;
  // Request a cache deliberately: native must get one, cuFile must still bypass it.
  config.cache.mode          = cucascade::io::cache::cache_mode::cucs;
  config.max_readahead_scans = 2;
  config.apply_cache_mode();
  return config;
}

void write_parquet(sirius::test::scratch_dir const& directory)
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection connection(db);
  // Four groups with distinct, non-repeating values and an unselected column.
  // Uncompressed BIGINT chunks are large enough to exercise staged asynchronous H2D.
  auto result = connection.Query(
    "COPY (SELECT i::BIGINT id, (17*i*i + 31*i + 7)::BIGINT value, -i AS ignored "
    "FROM range(262144) t(i) ORDER BY i) TO " +
    directory.file_literal("input.parquet") +
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
  // destinations are host-complete, including H2D on the reactor's private copy streams.
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

}  // namespace

TEST_CASE("cuFile scan reads and decodes selected Parquet chunks like native", "[cufile][scan]")
{
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count == 0) { SKIP("cuFile scan integration requires a CUDA device"); }
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
  auto cufile_context =
    std::dynamic_pointer_cast<cucascade::io::cufile::cufile_ioctx>(source->io_ctx());
  REQUIRE(cufile_context);
  auto const capability = cufile_context->path_capability(source->get_io_object());
  CHECK(capability.capability == cucascade::io::cufile::path_capability::compatibility_only);
  CHECK(capability.compat_mode == cucascade::io::cufile::compat::on);
  CHECK_FALSE(capability.registered);
  CHECK_FALSE(capability.reason.empty());
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

  auto options = cudf::io::parquet_reader_options::builder().column_names({"id", "value"}).build();
  auto footer  = cudf::io::parquet::fetch_footer_to_host(*baseline);
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

  std::vector<std::vector<std::int64_t>> expected(2);
  std::int64_t first_row = 0;
  for (std::size_t group = 0; group < metadata->row_groups.size(); ++group) {
    auto const rows = metadata->row_groups[group].num_rows;
    auto const selected =
      std::find(groups.begin(), groups.end(), static_cast<cudf::size_type>(group));
    if (selected != groups.end()) {
      for (auto id = first_row; id < first_row + rows; ++id) {
        expected[0].push_back(id);
        expected[1].push_back(17 * id * id + 31 * id + 7);
      }
    }
    first_row += rows;
  }
  REQUIRE_FALSE(expected[0].empty());
  auto const decoded_native = decode_chunks(native_chunks, *metadata, options, groups, decoder);
  CHECK(std::equal(decoded_native.begin(), decoded_native.end(), expected.begin(), expected.end()));
  auto const decoded_cufile = decode_chunks(cufile_chunks, *metadata, options, groups, decoder);
  CHECK(std::equal(
    decoded_cufile.begin(), decoded_cufile.end(), decoded_native.begin(), decoded_native.end()));

  // Also exercise Sirius's normal local materializer: cuFile does not advertise bulk IO,
  // so it must retain the general datasource decode route rather than pretending to be REST.
  for (auto const& datasource : {baseline, source}) {
    std::array<parquet_source, 1> const sources{{{datasource, metadata, groups}}};
    CHECK_FALSE(sirius::op::scan::prefers_bulk_materialize(sources, options));
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
  // Guaranteed to exceed vector::max_size(), without attempting an enormous
  // allocation. Validation accepts positive counts; the backend owns this limit.
  config.cufile_n_reactors = std::numeric_limits<std::size_t>::max();
  std::string cause;
  try {
    std::ignore = cucascade::io::cufile::make_cufile_ioctx(config.cufile_n_reactors, config.cufile);
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

TEST_CASE("the default cuFile pool is planned for Sirius's GPU and reads correct bytes",
          "[cufile][scan]")
{
  namespace cufile = cucascade::io::cufile;
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count == 0) { SKIP("cuFile scan integration requires a CUDA device"); }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  sirius::test::scratch_dir directory("cufile_planned_scan");
  write_parquet(directory);

  auto config              = reader_config(io_backend::cufile);
  config.cufile_n_reactors = 0;  // the default: a pinned group of reactors per GPU
  // Let cuFile register the file, so device reads go through cuFileRead.
  config.cufile.compat_mode.reset();
  config.cufile.use_odirect = true;
  bool forced               = false;
  SECTION("automatic staging") {}
  SECTION("forced staging")
  {
    // No GDS on most CI hosts: automatic staging stays off there, so force it.
    config.cufile.staging_mode = cufile::staging::always;
    forced                     = true;
  }

  auto memory   = initialize_memory_manager(1);
  auto topology = single_gpu_index();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager reader{lifecycle, config, *memory, topology};
  sirius_scan_manager native{lifecycle, reader_config(io_backend::native), *memory, topology};

  auto const* ioctx = dynamic_cast<cufile::cufile_ioctx const*>(reader.io_ctx());
  REQUIRE(ioctx != nullptr);
  // cufile.devices is empty: the plan covers the GPU Sirius reads into, not every
  // visible GPU, and sizes it for unknown storage (no storage_hint_path).
  auto const& plan = ioctx->plan();
  REQUIRE(plan.groups.size() == 1);
  CHECK(plan.groups.front().device == 0);
  CHECK(plan.groups.front().n_reactors == cufile::default_reactors_per_device);
  CHECK(plan.total_reactors() == plan.groups.front().n_reactors);

  auto const path = directory.file("input.parquet");
  auto source     = reader.create_datasource(path);
  auto baseline   = native.create_datasource(path);
  REQUIRE(source);
  REQUIRE(baseline);
  REQUIRE(source->io_ctx().get() == reader.io_ctx());

  auto options = cudf::io::parquet_reader_options::builder().column_names({"id", "value"}).build();
  auto footer  = cudf::io::parquet::fetch_footer_to_host(*baseline);
  cudf::io::parquet::experimental::hybrid_scan_reader footer_reader(
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), options);
  auto const metadata = footer_reader.parquet_metadata();
  std::vector<cudf::size_type> groups(metadata.row_groups.size());
  std::iota(groups.begin(), groups.end(), 0);
  auto const ranges = sirius::op::scan::column_chunk_ranges(metadata, options, groups);
  // Above gds_threshold, so they are cuFile reads rather than POSIX ones.
  REQUIRE(std::any_of(
    ranges.begin(), ranges.end(), [](auto const& range) { return range.size() > (256 << 10); }));

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

  auto const stats = ioctx->stats();
  CHECK(stats.cufile_failures == 0);
  if (!ioctx->path_capability(source->get_io_object()).registered) {
    SKIP("cuFile did not register the file here; its device reads used POSIX");
  }
  if (forced) {
    CHECK(stats.staged_bytes > 0);
    // Every read went to GPU 0's group.
    CHECK(ioctx->stats(0).staged_bytes == stats.staged_bytes);
  } else if (!cufile::gds_available()) {
    CHECK(stats.staged_bytes == 0);
  }
}

TEST_CASE("cuFile staging is reserved ahead of the pool, on Sirius's GPU only",
          "[cufile][scan][scan_manager]")
{
  namespace cufile = cucascade::io::cufile;
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count == 0) { SKIP("cuFile staging requires a CUDA device"); }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover());
  auto const& topology = discovery.get_topology();

  auto config                = reader_config(io_backend::cufile);
  config.cufile_n_reactors   = 0;
  config.cufile.staging_mode = cufile::staging::always;
  std::vector<int> const gpu_0{0};
  // Staging is process-wide: start from none, so slots earlier tests left behind
  // cannot satisfy the checks below.
  std::ignore = cufile::release_staging(0);
  REQUIRE(cufile::staging_usage_of(0).leased_slots == 0);
  REQUIRE(cufile::staging_usage_of(0).allocated_bytes == 0);
  auto const other_before =
    device_count > 1 ? cufile::staging_usage_of(1).allocated_bytes : std::size_t{0};

  // Nothing for a flat pool, which leases its slots lazily, nor for another backend.
  auto flat              = config;
  flat.cufile_n_reactors = 4;
  sirius::scan_manager::reserve_cufile_staging(flat, gpu_0, topology);
  sirius::scan_manager::reserve_cufile_staging(reader_config(io_backend::native), gpu_0, topology);
  CHECK(cufile::staging_usage_of(0).allocated_bytes == 0);

  sirius::scan_manager::reserve_cufile_staging(config, gpu_0, topology);
  if (device_count > 1) { CHECK(cufile::staging_usage_of(1).allocated_bytes == other_before); }
  auto const reserved = cufile::staging_usage_of(0);
  if (reserved.registered_bytes == 0) { SKIP("cuFile cannot register device memory here"); }
  auto const plan =
    cufile::plan_cufile_pool(config.resolved_cufile(gpu_0), &topology, cufile::probe_platform());
  REQUIRE(plan.groups.size() == 1);
  CHECK(reserved.free_slots == plan.groups.front().staging_slots);

  // The pool built afterwards starts on the reserved slots instead of allocating more.
  auto ioctx = cufile::make_cufile_ioctx(0, config.resolved_cufile(gpu_0));
  ioctx->start();
  CHECK(cufile::staging_usage_of(0).allocated_bytes == reserved.allocated_bytes);
  ioctx->shutdown();
}

// ---------------------------------------------------------------------------
// cufile.fused_reads and cufile.host_batch (opt-in).  cuFile runs in compatibility mode
// on hosts without GDS, where lanes still run (cuFile bounces each read through host
// memory) and the batch API still serves host reads.
// ---------------------------------------------------------------------------

namespace {

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

/// A planned pool on GPU 0 that lets cuFile register the file, so device reads can take
/// lanes, with a small lane set (2 lane reactors x 4 lanes of the default 8 MiB).
scan_manager_config fused_reads_config()
{
  auto config              = reader_config(io_backend::cufile);
  config.cufile_n_reactors = 0;
  config.cufile.compat_mode.reset();
  config.cufile.use_odirect                     = true;
  config.cufile.fused_reads.enabled             = true;
  config.cufile.fused_reads.reactors_per_device = 2;
  config.cufile.fused_reads.lanes_per_reactor   = 4;
  return config;
}

}  // namespace

TEST_CASE("fused lane reads serve the cuFile scan and decode the same rows",
          "[cufile][scan][fused_reads]")
{
  namespace cufile = cucascade::io::cufile;
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count == 0) { SKIP("cuFile scan integration requires a CUDA device"); }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  sirius::test::scratch_dir directory("cufile_fused_scan");
  write_parquet(directory);

  auto const config = fused_reads_config();
  auto memory       = initialize_memory_manager(1);
  auto topology     = single_gpu_index();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager reader{lifecycle, config, *memory, topology};
  sirius_scan_manager native{lifecycle, reader_config(io_backend::native), *memory, topology};

  auto const* ioctx = dynamic_cast<cufile::cufile_ioctx const*>(reader.io_ctx());
  REQUIRE(ioctx != nullptr);
  // The plan carries the lane reactors next to the synchronous ones, whose automatic
  // count drops to the local default with fused reads on.
  auto const& plan = ioctx->plan();
  REQUIRE(plan.groups.size() == 1);
  auto const& group = plan.groups.front();
  CHECK(group.device == 0);
  CHECK(group.lane_reactors == 2);
  CHECK(group.lanes_per_reactor == 4);
  CHECK(group.lane_slots == 8);
  CHECK(group.n_reactors == cufile::local_reactors_per_device);
  CHECK(plan.lane_slot_bytes == config.cufile.fused_reads.slot_bytes);
  CHECK(plan.total_reactors() == group.lane_reactors + group.n_reactors);

  auto const path = directory.file("input.parquet");
  auto source     = reader.create_datasource(path);
  auto baseline   = native.create_datasource(path);
  REQUIRE(source);
  REQUIRE(baseline);
  REQUIRE(source->io_ctx().get() == reader.io_ctx());
  if (!ioctx->path_capability(source->get_io_object()).registered) {
    SKIP("cuFile did not register the file here; its device reads use POSIX, not lanes");
  }

  auto options = cudf::io::parquet_reader_options::builder().column_names({"id", "value"}).build();
  auto footer  = cudf::io::parquet::fetch_footer_to_host(*baseline);
  cudf::io::parquet::experimental::hybrid_scan_reader footer_reader(
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), options);
  auto const metadata =
    std::make_shared<cudf::io::parquet::FileMetaData const>(footer_reader.parquet_metadata());
  std::vector<cudf::size_type> groups(metadata->row_groups.size());
  std::iota(groups.begin(), groups.end(), 0);
  auto const ranges = sirius::op::scan::column_chunk_ranges(*metadata, options, groups);

  // One vectored read of every selected chunk, behind work queued on the caller's stream
  // (two-phase lane reads): each row group's adjacent id and value chunks share a window.
  rmm::cuda_stream caller(rmm::cuda_stream::flags::non_blocking);
  rmm::cuda_stream observer(rmm::cuda_stream::flags::non_blocking);
  rmm::cuda_stream decoder(rmm::cuda_stream::flags::non_blocking);
  auto const want = read_chunks(*baseline, ranges, caller, observer, false);
  auto const got  = read_chunks(*source, ranges, caller, observer, true);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    CAPTURE(i, ranges[i].offset(), ranges[i].size());
    auto const& actual   = got.observed_bytes[i];
    auto const& expected = want.observed_bytes[i];
    CHECK(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
  }
  auto const vectored = ioctx->stats();
  CHECK(vectored.lane_reads > 0);
  CHECK(vectored.lane_parts > vectored.lane_reads);  // neighbours were fused
  CHECK(vectored.lane_issued_bytes >= vectored.lane_bytes);
  CHECK(vectored.lane_failures == 0);
  CHECK(vectored.cufile_failures == 0);

  // Sirius's scan route for this backend: cuDF's reader, one device read per contiguous run.
  std::array<parquet_source, 1> const sources{{{source, metadata, groups}}};
  CHECK_FALSE(sirius::op::scan::prefers_bulk_materialize(sources, options));
  auto table = sirius::op::scan::materialize_parquet(
    sources, options, {}, decoder, cudf::get_current_device_resource_ref());
  REQUIRE(table);
  auto const actual   = host_columns(table->view(), decoder);
  auto const expected = expected_rows(*metadata, groups);
  CHECK(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
  auto const scanned = ioctx->stats() - vectored;
  CHECK(scanned.lane_reads > 0);
  CHECK(scanned.lane_failures == 0);
  CHECK(scanned.cufile_failures == 0);
}

TEST_CASE("cuFile fused-read lane slots are reserved ahead of the pool",
          "[cufile][scan][scan_manager][fused_reads]")
{
  namespace cufile = cucascade::io::cufile;
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count == 0) { SKIP("cuFile staging requires a CUDA device"); }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover());
  auto const& topology = discovery.get_topology();

  auto config = fused_reads_config();
  // A slot size of its own, so the lane slots are told apart from the synchronous ones.
  config.cufile.fused_reads.slot_bytes = 1UL << 20;
  std::vector<int> const gpu_0{0};
  // Staging is process-wide: start from none.
  std::ignore = cufile::release_staging(0);
  REQUIRE(cufile::staging_usage_of(0).leased_slots == 0);
  REQUIRE(cufile::staging_usage_of(0).allocated_bytes == 0);

  auto const cfg  = config.resolved_cufile(gpu_0);
  auto const plan = cufile::plan_cufile_pool(cfg, &topology, cufile::probe_platform());
  REQUIRE(plan.groups.size() == 1);
  auto const& group = plan.groups.front();
  REQUIRE(group.lane_slots == 8);
  CHECK(plan.lane_slot_bytes == (1UL << 20));
  // The lanes are budgeted on top of the synchronous cap; the plan's cap includes them.
  auto const lane_bytes = cufile::fused_lane_staging_bytes(cfg);
  CHECK(lane_bytes == 8 * ((1UL << 20) + (64UL << 10)));
  CHECK(group.staging_cap_bytes >= lane_bytes);

  // A flat pool leases its lane slots at its first reads: nothing is reserved for it.
  auto flat              = config;
  flat.cufile_n_reactors = 4;
  sirius::scan_manager::reserve_cufile_staging(flat, gpu_0, topology);
  CHECK(cufile::staging_usage_of(0).allocated_bytes == 0);

  sirius::scan_manager::reserve_cufile_staging(config, gpu_0, topology);
  auto const reserved = cufile::staging_usage_of(0);
  CHECK(reserved.leased_slots == 0);
  CHECK(reserved.free_slots == group.lane_slots + group.staging_slots);
  if (group.staging_slots == 0) {
    // No GDS here: automatic staging reserves no synchronous slots, only the lanes'.
    CHECK(reserved.allocated_bytes == lane_bytes);
  } else {
    CHECK(reserved.allocated_bytes > lane_bytes);
  }

  // The pool built afterwards starts its lanes on the reserved slots.
  auto ioctx = cufile::make_cufile_ioctx(0, cfg);
  ioctx->start();
  CHECK(cufile::staging_usage_of(0).allocated_bytes == reserved.allocated_bytes);
  ioctx->shutdown();
}

TEST_CASE("cuFile host batch serves an aligned host read through Sirius's datasource",
          "[cufile][scan][host_batch]")
{
  namespace cufile = cucascade::io::cufile;
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count == 0) { SKIP("cuFile scan integration requires a CUDA device"); }
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{0}};
  sirius::test::scratch_dir directory("cufile_host_batch");
  write_parquet(directory);

  auto config              = reader_config(io_backend::cufile);
  config.cufile_n_reactors = 0;
  config.cufile.compat_mode.reset();
  config.cufile.use_odirect                = true;
  config.cufile.host_batch.enabled         = true;
  config.cufile.host_batch.max_entry_bytes = 256UL << 10;  // several entries per read
  auto memory                              = initialize_memory_manager(1);
  auto topology                            = single_gpu_index();
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager reader{lifecycle, config, *memory, topology};
  sirius_scan_manager native{lifecycle, reader_config(io_backend::native), *memory, topology};

  auto const* ioctx = dynamic_cast<cufile::cufile_ioctx const*>(reader.io_ctx());
  REQUIRE(ioctx != nullptr);
  CHECK(ioctx->plan().host_batch_reactors == config.cufile.host_batch.reactors);

  auto const path = directory.file("input.parquet");
  auto source     = reader.create_datasource(path);
  auto baseline   = native.create_datasource(path);
  REQUIRE(source);
  REQUIRE(baseline);
  if (!ioctx->path_capability(source->get_io_object()).registered) {
    SKIP("cuFile did not register the file here; its host reads use POSIX");
  }
  constexpr std::size_t offset = 4096;
  constexpr std::size_t bytes  = 1UL << 20;
  REQUIRE(source->size() >= offset + bytes);
  std::vector<std::uint8_t> expected(bytes);
  REQUIRE(baseline->host_read(offset, bytes, expected.data()) == bytes);

  // A 4 KiB-aligned destination for a 4 KiB-aligned range: the batch path's shape.
  auto const release = [](std::uint8_t* p) { std::free(p); };
  std::unique_ptr<std::uint8_t, decltype(release)> aligned{
    static_cast<std::uint8_t*>(std::aligned_alloc(4096, bytes)), release};
  REQUIRE(aligned);
  REQUIRE(source->host_read(offset, bytes, aligned.get()) == bytes);
  CHECK(std::equal(aligned.get(), aligned.get() + bytes, expected.begin(), expected.end()));
  auto const batched = ioctx->stats();
  CHECK(batched.batch_reads == 1);
  CHECK(batched.batch_bytes == bytes);
  CHECK(batched.batch_entries >= bytes / config.cufile.host_batch.max_entry_bytes);
  CHECK(batched.batch_failures == 0);
  CHECK(batched.batch_partial_status == 0);

  // An incongruent destination keeps the synchronous path.
  std::vector<std::uint8_t> unaligned(bytes + 1);
  REQUIRE(source->host_read(offset, bytes, unaligned.data() + 1) == bytes);
  CHECK(std::equal(unaligned.begin() + 1, unaligned.end(), expected.begin(), expected.end()));
  auto const after = ioctx->stats();
  CHECK(after.batch_reads == batched.batch_reads);
  CHECK(after.host_reads > batched.host_reads);
}

TEST_CASE("a SQL scan through the cuFile backend reads on fused lanes",
          "[cufile][scan][fused_reads][isolated_context]")
{
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  if (device_count == 0) { SKIP("cuFile scan integration requires a CUDA device"); }
  sirius::test::scratch_dir directory("cufile_fused_sql");
  write_parquet(directory);
  auto const config_path = directory.file("sirius_cufile_fused.yaml");
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
            "      cufile_n_reactors: 0\n"
            "      uring_n_reactors: 1\n"
            "      cufile:\n"
            "        fused_reads:\n"
            "          enabled: true\n"
            "          reactors_per_device: 2\n"
            "          lanes_per_reactor: 4\n";
    REQUIRE(yaml);
  }

  auto db = sirius::test::open_sirius_db(nullptr, config_path);
  {
    duckdb::Connection con(*db);
    auto context = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(context);
    auto& scans       = context->get_scan_manager();
    auto const* ioctx = dynamic_cast<cucascade::io::cufile::cufile_ioctx const*>(scans.io_ctx());
    REQUIRE(ioctx != nullptr);
    REQUIRE(ioctx->plan().groups.size() == 1);
    CHECK(ioctx->plan().groups.front().lane_reactors == 2);
    CHECK(ioctx->plan().groups.front().lanes_per_reactor == 4);
    auto const path       = directory.file("input.parquet");
    auto const registered = [&] {
      auto source = scans.create_datasource(path);
      REQUIRE(source);
      return ioctx->path_capability(source->get_io_object()).registered;
    }();

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
    auto const before = ioctx->stats();
    auto gpu          = run(query);
    auto const read   = ioctx->stats() - before;
    std::ignore       = run("SET gpu_execution = false");
    auto cpu          = run(query);
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

    CHECK(read.cufile_failures == 0);
    CHECK(read.lane_failures == 0);
    if (registered) {
      // The GPU scan read its column chunks through the lanes.
      CHECK(read.lane_reads > 0);
      CHECK(read.lane_bytes > 0);
    } else {
      WARN("cuFile did not register the file here; the scan's device reads used POSIX");
    }
  }
  db.reset();
}

#endif  // CUCASCADE_HAS_CUFILE
