#include "io/uring/uring_ioctx.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <fcntl.h>
#include <unistd.h>

// 4 columns used by the classic TPC-H lineitem aggregations (Q1, Q6, …).
static const std::vector<std::string> COLUMNS = {
    "l_orderkey",
    "l_extendedprice",
    "l_discount",
    "l_shipdate",
};

enum class DataSource { cudf, uring };

static DataSource parse_source(std::string_view s) {
  if (s == "cudf")
    return DataSource::cudf;
  if (s == "uring")
    return DataSource::uring;
  throw std::invalid_argument(std::string("unknown datasource: ") +
                              std::string(s) +
                              "  (expected: cudf | uring)");
}

static bool drop_caches() {
  ::sync();
  int fd = ::open("/proc/sys/vm/drop_caches", O_WRONLY);
  if (fd < 0)
    return false;
  bool ok = ::write(fd, "3", 1) == 1;
  ::close(fd);
  return ok;
}

static void usage(char const *prog) {
  std::cerr << "usage: " << prog << " <cudf|uring> <num_rows>\n"
            << "  cudf     – cudf default (mmap/pread)\n"
            << "  uring    – O_DIRECT io_uring + DMA to GPU\n"
            << "  num_rows – rows to read (0 = all)\n";
}

int main(int argc, char **argv) {
  if (argc != 3) {
    usage(argv[0]);
    return 1;
  }

  DataSource source;
  try {
    source = parse_source(argv[1]);
  } catch (std::invalid_argument const &e) {
    std::cerr << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  long long num_rows_arg = std::stoll(argv[2]);
  if (num_rows_arg < 0) {
    std::cerr << "num_rows must be >= 0\n";
    return 1;
  }
  size_t num_rows = static_cast<size_t>(num_rows_arg); // 0 means all

  std::string path =
      "/home/aaramoon/Documents/tpch/sf100/parquet/lineitem_indexed.parquet";

  spdlog::set_level(spdlog::level::err); // show device read pattern

  std::cout << "Source : " << argv[1] << "\n"
            << "Rows   : " << (num_rows == 0 ? "all" : std::to_string(num_rows))
            << "\n"
            << "Columns: ";
  for (auto const &c : COLUMNS)
    std::cout << c << "  ";
  std::cout << "\n\n";

  bool can_drop = drop_caches();
  if (!can_drop)
    std::cout
        << "WARNING: cannot drop caches (run as root for cold results)\n\n";

  cudaFree(nullptr);

  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::mr::set_current_device_resource(&async_mr);

  auto time_ms = [](auto fn) -> double {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t0)
        .count();
  };

  // Read Parquet footer metadata once using cudf's default datasource so that
  // neither the cudf nor the uring timed path pays the cost of a metadata scan.
  cudf::io::parquet::FileMetaData file_metadata = [&] {
    auto probe_sources =
        cudf::io::make_datasources(cudf::io::source_info{{path}});
    auto &probe_ds = *probe_sources.front();
    auto file_size = probe_ds.size();
    cudf::io::parquet::file_ender_s ender{};
    probe_ds.host_read(file_size - sizeof(ender), sizeof(ender),
                       reinterpret_cast<uint8_t *>(&ender));
    std::vector<uint8_t> footer_buf(ender.footer_len);
    probe_ds.host_read(file_size - sizeof(ender) - ender.footer_len,
                       ender.footer_len, footer_buf.data());
    auto base_opts =
        cudf::io::parquet_reader_options::builder().columns(COLUMNS).build();
    cudf::io::parquet::experimental::hybrid_scan_reader scanner{
        cudf::host_span<uint8_t const>{footer_buf.data(), footer_buf.size()},
        base_opts};
    return scanner.parquet_metadata();
  }();

  // Build read options (no source — provided separately as a datasource
  // vector).
  auto read_opts_builder =
      cudf::io::parquet_reader_options::builder().columns(COLUMNS);
  if (num_rows > 0)
    read_opts_builder.num_rows(num_rows);
  auto read_opts = read_opts_builder.build();

  // Timed run.
  double ms = 0;
  if (source == DataSource::cudf) {
    auto sources = cudf::io::make_datasources(cudf::io::source_info{{path}});
    ms = time_ms([&] {
      auto tbl = cudf::io::read_parquet(std::move(sources), {file_metadata},
                                        read_opts);
      std::cout << "Schema : " << tbl.tbl->num_columns() << " columns, "
                << tbl.tbl->num_rows() << " rows read\n\n";
    });
  } else {
    auto io_ctx = std::make_shared<sirius::io::uring_ioctx>();
    auto ds = io_ctx->make_datasource(
        std::make_unique<sirius::io::uring_io_object>(path));

    std::vector<std::unique_ptr<cudf::io::datasource>> sources;
    sources.push_back(std::move(ds));

    ms = time_ms([&] {
      auto tbl = cudf::io::read_parquet(std::move(sources), {file_metadata},
                                        read_opts);
      std::cout << "Schema : " << tbl.tbl->num_columns() << " columns, "
                << tbl.tbl->num_rows() << " rows read\n\n";
    });
  }

  std::cout << std::fixed << std::setprecision(1) << ms << " ms\n";
  return 0;
}
