#include "io/prefetching_cache.hpp"
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"
#include "io/uring/uring_ioctx.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <fcntl.h>
#include <glob.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// 4 columns used by the classic TPC-H lineitem aggregations (Q1, Q6, …).
static const std::vector<std::string> COLUMNS = {
  "l_orderkey",
  "l_extendedprice",
  "l_discount",
  "l_shipdate",
};

enum class DataSource { cudf, uring };

static DataSource parse_source(std::string_view s)
{
  if (s == "cudf") return DataSource::cudf;
  if (s == "uring") return DataSource::uring;
  throw std::invalid_argument(std::string("unknown datasource: ") + std::string(s) +
                              "  (expected: cudf | uring)");
}

static bool drop_caches()
{
  ::sync();
  int fd = ::open("/proc/sys/vm/drop_caches", O_WRONLY);
  if (fd < 0) return false;
  bool ok = ::write(fd, "3", 1) == 1;
  ::close(fd);
  return ok;
}

// Expand @p spec to a sorted list of file paths.  If it contains '*', POSIX
// glob expansion is used; otherwise the spec is returned as a single path.
static std::vector<std::string> expand_paths(std::string const& spec)
{
  if (spec.find('*') == std::string::npos) return {spec};

  std::vector<std::string> paths;
  glob_t g{};
  int rc = ::glob(spec.c_str(), GLOB_TILDE, nullptr, &g);
  if (rc == 0) {
    paths.reserve(g.gl_pathc);
    for (size_t i = 0; i < g.gl_pathc; ++i)
      paths.emplace_back(g.gl_pathv[i]);
  }
  ::globfree(&g);
  std::sort(paths.begin(), paths.end());
  return paths;
}

static void usage(char const* prog)
{
  std::cerr << "usage: " << prog << " <path|glob> <cudf|uring> <num_rows> [n_reactors]\n"
            << "  path|glob  – parquet file path, or shell glob (e.g. "
               "'dir/*.parquet')\n"
            << "  cudf       – cudf default (mmap/pread)\n"
            << "  uring      – O_DIRECT io_uring + DMA to GPU\n"
            << "  num_rows   – rows to read (0 = all)\n"
            << "  n_reactors – uring reactor threads (default 2; uring only)\n";
}

int main(int argc, char** argv)
{
  if (argc != 4 && argc != 5) {
    usage(argv[0]);
    return 1;
  }

  std::string path_spec = argv[1];

  DataSource source;
  try {
    source = parse_source(argv[2]);
  } catch (std::invalid_argument const& e) {
    std::cerr << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  long long num_rows_arg = std::stoll(argv[3]);
  if (num_rows_arg < 0) {
    std::cerr << "num_rows must be >= 0\n";
    return 1;
  }
  size_t num_rows = static_cast<size_t>(num_rows_arg);  // 0 means all

  size_t n_reactors = 2;
  if (argc == 5) {
    long long n_reactors_arg = std::stoll(argv[4]);
    if (n_reactors_arg <= 0) {
      std::cerr << "n_reactors must be > 0\n";
      return 1;
    }
    n_reactors = static_cast<size_t>(n_reactors_arg);
  }

  auto paths = expand_paths(path_spec);
  if (paths.empty()) {
    std::cerr << "no files matched: " << path_spec << "\n";
    return 1;
  }

  spdlog::set_level(spdlog::level::info);  // show per-read trace

  std::cout << "Source : " << argv[2] << "\n"
            << "Files  : " << paths.size() << "\n";
  for (auto const& p : paths)
    std::cout << "  " << p << "\n";
  std::cout << "Rows   : " << (num_rows == 0 ? "all" : std::to_string(num_rows)) << "\n"
            << "Columns: ";
  for (auto const& c : COLUMNS)
    std::cout << c << "  ";
  std::cout << "\n\n";

  bool can_drop = drop_caches();
  if (!can_drop) std::cout << "WARNING: cannot drop caches (run as root for cold results)\n\n";

  cudaFree(nullptr);

  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::mr::set_current_device_resource(&async_mr);

  auto time_ms = [](auto fn) -> double {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0)
      .count();
  };

  // Approximate device-memory footprint of a materialized table: data buffer
  // for fixed-width columns + null-mask buffer where present.  Variable-width
  // columns (strings/lists) are not in our schema so we skip their children.
  auto table_bytes = [](cudf::table_view const& tv) -> size_t {
    size_t bytes = 0;
    for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
      auto const& col = tv.column(i);
      if (cudf::is_fixed_width(col.type()))
        bytes += static_cast<size_t>(col.size()) * cudf::size_of(col.type());
      if (col.nullable()) bytes += cudf::bitmask_allocation_size_bytes(col.size());
    }
    return bytes;
  };

  auto log_table = [&](cudf::io::table_with_metadata const& t) {
    auto bytes = table_bytes(t.tbl->view());
    std::cout << "Schema : " << t.tbl->num_columns() << " columns, " << t.tbl->num_rows()
              << " rows, " << std::fixed << std::setprecision(2)
              << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB materialized\n\n";
  };
  (void)log_table;

  auto scan_opts = cudf::io::parquet_reader_options::builder().column_names(COLUMNS).build();

  // Per-file: probe the footer with cudf's default datasource (so timed paths
  // don't pay metadata cost), then run hybrid_scan to collect the byte ranges
  // we'd touch for the selected columns.  The num_rows budget is consumed
  // in file order — we stop adding row groups once we've covered it.
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  metadatas.reserve(paths.size());
  size_t total_range_bytes = 0;
  size_t total_row_groups  = 0;
  size_t total_ranges      = 0;
  int64_t accumulated_rows = 0;

  for (auto const& path : paths) {
    std::vector<uint8_t> footer_buf;
    {
      auto probe_sources = cudf::io::make_datasources(cudf::io::source_info{{path}});
      auto& probe_ds     = *probe_sources.front();
      auto file_size     = probe_ds.size();
      cudf::io::parquet::file_ender_s ender{};
      probe_ds.host_read(
        file_size - sizeof(ender), sizeof(ender), reinterpret_cast<uint8_t*>(&ender));
      footer_buf.resize(ender.footer_len);
      probe_ds.host_read(
        file_size - sizeof(ender) - ender.footer_len, ender.footer_len, footer_buf.data());
    }

    cudf::io::parquet::experimental::hybrid_scan_reader scanner{
      cudf::host_span<uint8_t const>{footer_buf.data(), footer_buf.size()}, scan_opts};
    auto file_metadata = scanner.parquet_metadata();

    std::vector<cudf::size_type> selected_row_groups;
    if (num_rows == 0) {
      selected_row_groups = scanner.all_row_groups(scan_opts);
    } else if (accumulated_rows < static_cast<int64_t>(num_rows)) {
      for (cudf::size_type i = 0;
           i < static_cast<cudf::size_type>(file_metadata.row_groups.size()) &&
           accumulated_rows < static_cast<int64_t>(num_rows);
           ++i) {
        selected_row_groups.push_back(i);
        accumulated_rows += file_metadata.row_groups[i].num_rows;
      }
    }

    auto byte_ranges = scanner.all_column_chunks_byte_ranges(selected_row_groups, scan_opts);
    std::sort(byte_ranges.begin(), byte_ranges.end(), [](auto const& a, auto const& b) {
      return a.offset() < b.offset();
    });
    std::vector<cudf::io::text::byte_range_info> coalesced;
    coalesced.reserve(byte_ranges.size());
    for (auto const& br : byte_ranges) {
      if (br.size() <= 0) continue;
      if (!coalesced.empty()) {
        auto& back       = coalesced.back();
        int64_t back_end = back.offset() + back.size();
        if (br.offset() <= back_end) {
          int64_t br_end = br.offset() + br.size();
          if (br_end > back_end)
            back = cudf::io::text::byte_range_info{back.offset(), br_end - back.offset()};
          continue;
        }
      }
      coalesced.push_back(br);
    }

    for (auto const& br : coalesced)
      total_range_bytes += static_cast<size_t>(br.size());
    total_row_groups += selected_row_groups.size();
    total_ranges += coalesced.size();

    metadatas.push_back(std::move(file_metadata));
  }

  std::cout << "Hybrid scan: " << total_row_groups << " row group(s), " << total_ranges
            << " byte range(s), " << std::fixed << std::setprecision(2)
            << static_cast<double>(total_range_bytes) / (1024.0 * 1024.0) << " MiB total\n\n";

  // Build read options (no source — provided separately as a datasource
  // vector).
  auto read_opts_builder = cudf::io::parquet_reader_options::builder().column_names(COLUMNS);
  if (num_rows > 0) read_opts_builder.num_rows(num_rows);
  auto read_opts = read_opts_builder.build();

  // cudaMemcpyBatchAsync rejects the legacy / per-thread default streams with
  // cudaMemLocation hints — pass an explicit stream so all H2D copies inside
  // the datasource and cudf share the same one.
  rmm::cuda_stream stream;

  // Timed run.
  double ms = 0;
  if (source == DataSource::cudf) {
    std::vector<std::string> path_list = paths;
    auto sources = cudf::io::make_datasources(cudf::io::source_info{path_list});
    ms           = time_ms([&] {
      auto tbl =
        cudf::io::read_parquet(std::move(sources), std::move(metadatas), read_opts, stream.view());
    });
  } else {
    // Size the buffer pool to fit the working set, plus headroom.
    constexpr uint32_t POOL_MAX_SLABS       = 20;
    constexpr size_t INFLIGHT_BUDGET_CHUNKS = 2048;
    constexpr size_t POOL_CAPACITY          = static_cast<size_t>(POOL_MAX_SLABS) *
                                     static_cast<size_t>(sirius::io::buffer_pool::CHUNKS_PER_SLAB) *
                                     sirius::io::CHUNK_SIZE;

    cucascade::memory::numa_region_pinned_host_memory_resource upstream(0, /*make_portable=*/true);
    cucascade::memory::fixed_size_host_memory_resource host_mr(
      0,                                                              // device_id
      upstream,                                                       // upstream allocator
      POOL_CAPACITY,                                                  // mem_limit
      POOL_CAPACITY,                                                  // capacity
      sirius::io::CHUNK_SIZE,                                         // block_size = 1 MiB
      static_cast<size_t>(sirius::io::buffer_pool::CHUNKS_PER_SLAB),  // pool_size
      1);                                                             // initial_pools

    sirius::io::buffer_pool pool(host_mr, POOL_MAX_SLABS);

    auto io_ctx =
      std::make_shared<sirius::io::uring_ioctx>(n_reactors, 2 * sirius::io::NUM_CHUNKS, host_mr);
    // io_ctx->initialize_cache(pool, INFLIGHT_BUDGET_CHUNKS);

    std::vector<std::unique_ptr<cudf::io::datasource>> sources;
    sources.reserve(paths.size());
    for (auto const& path : paths) {
      auto io_obj = io_ctx->create_io_object(path);
      sources.push_back(std::make_unique<sirius::io::sirius_datasource>(io_ctx, io_obj));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));

    ms = time_ms([&] {
      auto tbl =
        cudf::io::read_parquet(std::move(sources), std::move(metadatas), read_opts, stream.view());
    });

    // std::cout << "cache summary : " << io_ctx->cache()->summary() <<
    // std::endl;
  }

  std::cout << std::fixed << std::setprecision(1) << ms << " ms\n";
  return 0;
}
