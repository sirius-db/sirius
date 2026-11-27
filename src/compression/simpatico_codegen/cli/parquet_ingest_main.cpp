// SPDX-License-Identifier: Apache-2.0
//
// parquet_ingest_bench — timed cudf::io::read_parquet decode over an
// in-memory Parquet buffer.
//
// This exists to get a plain-Parquet-ingestion throughput number that is
// directly comparable to `simpatico benchmark`'s compress/decompress GB/s:
// both load their input into device memory once, untimed, then time only the
// GPU-bound compute (decode vs. compress/decompress), bounded by
// cudaDeviceSynchronize() so wall time reflects GPU-visible completion, not
// host-side bookkeeping.
//
// The timed read_parquet() calls read from a device_span-backed source_info,
// with the encoded bytes pre-staged to device memory untimed — this matches
// simpatico's decompress input (compressed_representation), which is
// device-resident from the moment compress_with_plan produced it. A separate
// host-backed source is used only for footer/metadata parsing
// (read_parquet_metadata, read_parquet_footers), since device-buffer sources
// don't support host_read().
//
// Usage:
//   parquet_ingest_bench --input FILE.parquet [--mode per-column|full-table]
//                         [--warmup N] [--iters N]
//                         [--table-out PATH] [--csv-out PATH]
//
// cudf::io::read_parquet re-parses the file footer (schema/row-group stats)
// on every call, with no public API to inject pre-parsed metadata and skip
// it; simpatico's compress/decompress never touch file metadata at all. This
// tool times cudf::io::read_parquet_metadata() (footer parse only) in
// isolation and reports "decode_only_GBps" = decode time with that cost
// subtracted, alongside the raw number and the measured metadata_ms.

// cudf/ast/expressions.hpp (pulled in by cudf/io/parquet.hpp) uses std::variant without
// including it.
// clang-format off
#include <variant>
// clang-format on

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

[[noreturn]] void die(std::string const& msg, int code = 1)
{
  std::fprintf(stderr, "parquet_ingest_bench: %s\n", msg.c_str());
  std::exit(code);
}

struct pool_mr_guard {
  using any_device_resource = ::cuda::mr::any_resource<::cuda::mr::device_accessible>;
  rmm::mr::cuda_async_memory_resource mr{};
  std::optional<any_device_resource> previous;

  void install()
  {
    previous =
      rmm::mr::set_current_device_resource(any_device_resource{rmm::device_async_resource_ref{mr}});
  }
  ~pool_mr_guard()
  {
    if (previous) rmm::mr::set_current_device_resource(std::move(*previous));
  }
};

pool_mr_guard g_mr;

void init_gpu()
{
  if (cudaSetDevice(0) != cudaSuccess) die("cudaSetDevice(0) failed");
  g_mr.install();
}

void cuda_sync()
{
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(err));
}

// ── File I/O (untimed load) ───────────────────────────────────────────────

std::vector<uint8_t> read_binary_file(std::string const& path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  auto const size = in.tellg();
  in.seekg(0);
  std::vector<uint8_t> data(static_cast<std::size_t>(size));
  if (size > 0) {
    in.read(reinterpret_cast<char*>(data.data()), size);
    if (!in) throw std::runtime_error("read failed: " + path);
  }
  return data;
}

// ── Byte accounting (decoded columnar bytes — matches simpatico's metric) ──

std::string dtype_name(cudf::data_type const& t)
{
  switch (t.id()) {
    case cudf::type_id::INT8: return "i8";
    case cudf::type_id::INT16: return "i16";
    case cudf::type_id::INT32: return "i32";
    case cudf::type_id::INT64: return "i64";
    case cudf::type_id::UINT8: return "u8";
    case cudf::type_id::UINT16: return "u16";
    case cudf::type_id::UINT32: return "u32";
    case cudf::type_id::UINT64: return "u64";
    case cudf::type_id::FLOAT32: return "f32";
    case cudf::type_id::FLOAT64: return "f64";
    case cudf::type_id::STRING: return "str";
    case cudf::type_id::DECIMAL32: return "dec32";
    case cudf::type_id::DECIMAL64: return "dec64";
    case cudf::type_id::DECIMAL128: return "dec128";
    case cudf::type_id::TIMESTAMP_DAYS: return "date";
    default: return "t" + std::to_string(static_cast<int>(t.id()));
  }
}

std::size_t column_bytes(cudf::column_view col,
                         ::cuda::stream_ref stream = cudf::get_default_stream())
{
  if (col.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view scv(col);
    return static_cast<std::size_t>(col.size() + 1) * sizeof(int32_t) +
           static_cast<std::size_t>(scv.chars_size(stream));
  }
  return static_cast<std::size_t>(col.size()) * static_cast<std::size_t>(cudf::size_of(col.type()));
}

std::size_t table_bytes(cudf::table_view tv, ::cuda::stream_ref stream = cudf::get_default_stream())
{
  std::size_t total = 0;
  for (int i = 0; i < tv.num_columns(); ++i)
    total += column_bytes(tv.column(i), stream);
  return total;
}

double gbps(std::size_t bytes, double ms)
{
  if (ms <= 0.0) return 0.0;
  return (static_cast<double>(bytes) / 1.0e9) / (ms / 1000.0);
}

// Parquet is itself compressed on disk (snappy/dictionary/RLE by default), so
// it has its own compressed-vs-decoded ratio, directly comparable to
// simpatico's `ratio` field. Pulled from the real Thrift footer via
// read_parquet_footers() (total_compressed_size per column chunk, summed
// across row groups and files) rather than assumed from the file size.
std::unordered_map<std::string, int64_t> parquet_compressed_bytes_by_column(
  std::vector<std::vector<uint8_t>> const& file_bytes)
{
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.reserve(file_bytes.size());
  for (auto const& bytes : file_bytes) {
    cudf::host_span<std::byte const> span(reinterpret_cast<std::byte const*>(bytes.data()),
                                          bytes.size());
    sources.push_back(cudf::io::datasource::create(span));
  }
  cudf::host_span<std::unique_ptr<cudf::io::datasource> const> sources_view(sources.data(),
                                                                            sources.size());
  auto footers = cudf::io::read_parquet_footers(sources_view);

  std::unordered_map<std::string, int64_t> out;
  bool printed_codecs = std::getenv("DUMP_CODECS") == nullptr;
  for (auto const& footer : footers) {
    for (auto const& rg : footer.row_groups) {
      for (auto const& col : rg.columns) {
        if (col.meta_data.path_in_schema.empty()) continue;
        std::string const name = col.meta_data.path_in_schema.back();
        out[name] += col.meta_data.total_compressed_size;
        if (!printed_codecs) {
          static char const* codec_names[] = {
            "UNCOMPRESSED", "SNAPPY", "GZIP", "LZO", "BROTLI", "LZ4", "ZSTD", "LZ4_RAW"};
          auto idx = static_cast<int>(col.meta_data.codec);
          std::fprintf(stderr,
                       "codec %-16s %-15s uncompressed=%12lld compressed=%12lld encodings=",
                       name.c_str(),
                       (idx >= 0 && idx < 8) ? codec_names[idx] : "?",
                       static_cast<long long>(col.meta_data.total_uncompressed_size),
                       static_cast<long long>(col.meta_data.total_compressed_size));
          for (auto e : col.meta_data.encodings)
            std::fprintf(stderr, "%d,", static_cast<int>(e));
          std::fprintf(stderr, "\n");
        }
      }
    }
    printed_codecs = true;
  }
  return out;
}

// ── Timing stats (same shape as simpatico_main's) ───────────────────────────

struct timing_stats {
  double min    = 0;
  double median = 0;
  double mean   = 0;
};

timing_stats compute_stats(std::vector<double> const& ms)
{
  timing_stats s;
  if (ms.empty()) return s;
  s.min      = *std::min_element(ms.begin(), ms.end());
  double sum = 0;
  for (double v : ms)
    sum += v;
  s.mean      = sum / static_cast<double>(ms.size());
  auto sorted = ms;
  std::sort(sorted.begin(), sorted.end());
  std::size_t const n = sorted.size();
  s.median            = (n % 2 == 1) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  return s;
}

template <typename Body>
timing_stats time_iters(int warmup, int iters, Body&& body)
{
  for (int w = 0; w < warmup; ++w)
    body();
  cuda_sync();

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    body();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return compute_stats(samples);
}

// ── Bench row + reporting ───────────────────────────────────────────────────

struct bench_row {
  std::string column;
  std::string dtype;
  std::int64_t rows             = 0;
  std::size_t bytes             = 0;  // decoded (columnar) bytes
  std::int64_t compressed_bytes = 0;  // on-disk Parquet bytes for this column, from the footer
  timing_stats decode_ms;
  double metadata_ms = 0;  // measured once per file set, subtracted here for reporting

  double decode_gbps_median() const { return gbps(bytes, decode_ms.median); }
  double decode_only_ms_median() const { return std::max(0.0, decode_ms.median - metadata_ms); }
  double decode_only_gbps_median() const { return gbps(bytes, decode_only_ms_median()); }
  double ratio() const
  {
    return compressed_bytes > 0 ? static_cast<double>(bytes) / static_cast<double>(compressed_bytes)
                                : 0.0;
  }
};

bench_row make_total_row(std::vector<bench_row> const& rows, double metadata_ms)
{
  bench_row total;
  total.column = "TOTAL";
  total.dtype  = "(all)";
  total.rows   = rows.empty() ? 0 : rows.front().rows;
  total.metadata_ms =
    metadata_ms * static_cast<double>(rows.size());  // paid once per per-column call
  for (auto const& r : rows) {
    total.bytes += r.bytes;
    total.compressed_bytes += r.compressed_bytes;
    total.decode_ms.min += r.decode_ms.min;
    total.decode_ms.median += r.decode_ms.median;
    total.decode_ms.mean += r.decode_ms.mean;
  }
  return total;
}

void write_row_csv(std::ostream& os, bench_row const& r)
{
  os << r.column << ',' << r.dtype << ',' << r.rows << ',' << r.bytes << ',' << r.compressed_bytes
     << ',' << r.ratio() << ',' << r.decode_ms.min << ',' << r.decode_ms.median << ','
     << r.decode_ms.mean << ',' << r.decode_gbps_median() << ',' << r.metadata_ms << ','
     << r.decode_only_ms_median() << ',' << r.decode_only_gbps_median() << '\n';
}

void write_csv(std::string const& path, std::vector<bench_row> const& rows)
{
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write csv: " + path);
  out << "column,dtype,rows,bytes,compressed_bytes,ratio,decode_ms_min,decode_ms_median,"
         "decode_ms_mean,decode_gbps_median,metadata_ms,decode_only_ms_median,"
         "decode_only_gbps_median\n";
  for (auto const& r : rows)
    write_row_csv(out, r);
}

struct bench_config {
  std::vector<std::string> input_paths;
  enum class mode_t { per_column, full_table } mode = mode_t::per_column;
  int warmup                                        = 3;
  int iters                                         = 10;
  std::string table_out;
  std::string csv_out;
  std::string concat_out;
  std::vector<std::string> select_columns;
};

void write_bench_table(std::ostream& os,
                       std::vector<bench_row> const& rows,
                       bench_config const& cfg)
{
  using mode_t = bench_config::mode_t;
  os << "# parquet_ingest_bench mode="
     << (cfg.mode == mode_t::per_column ? "per-column" : "full-table") << " warmup=" << cfg.warmup
     << " iters=" << cfg.iters << " files=" << cfg.input_paths.size() << '\n';
  os << "# column | dtype | rows | bytes | compressed_bytes | ratio | decode_ms(min/med/mean) | "
        "decode_GBps | metadata_ms | decode_only_GBps\n";
  auto fmt3 = [](double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", v);
    return std::string(buf);
  };
  auto fmt2 = [](double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return std::string(buf);
  };
  for (auto const& r : rows) {
    os << r.column << " | " << r.dtype << " | " << r.rows << " | " << r.bytes << " | "
       << r.compressed_bytes << " | " << fmt3(r.ratio()) << "x | " << fmt3(r.decode_ms.min) << "/"
       << fmt3(r.decode_ms.median) << "/" << fmt3(r.decode_ms.mean) << " | "
       << fmt2(r.decode_gbps_median()) << " | " << fmt3(r.metadata_ms) << " | "
       << fmt2(r.decode_only_gbps_median()) << '\n';
  }
}

// ── CLI ──────────────────────────────────────────────────────────────────────

void usage()
{
  std::fprintf(
    stderr,
    "Usage: parquet_ingest_bench --input FILE.parquet [--input FILE2.parquet ...] [options]\n"
    "\n"
    "  --input PATH              Parquet file; repeatable to decode multiple files\n"
    "                            as one logical table (required, >= 1)\n"
    "  --mode {per-column|full-table}\n"
    "                            Benchmark granularity (default: per-column)\n"
    "  --warmup N                Warmup iterations (default: 3)\n"
    "  --iters N                 Timed iterations (default: 10)\n"
    "  --table-out PATH          Human-readable output file (default: stdout)\n"
    "  --csv-out PATH            CSV output file\n"
    "  --concat-out PATH         Utility mode: decode all --input files once and write\n"
    "                            them out as a single Parquet file at PATH (untimed; no\n"
    "                            benchmark runs). Use this to build a single-file input\n"
    "                            for `simpatico benchmark`, which only accepts one file.\n");
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    using mode_t = bench_config::mode_t;
    bench_config cfg;

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto need       = [&](char const* flag) -> std::string {
        if (i + 1 >= argc) die(std::string(flag) + " requires a value");
        return argv[++i];
      };
      if (arg == "--help" || arg == "-h") {
        usage();
        return 0;
      } else if (arg == "--input") {
        cfg.input_paths.push_back(need("--input"));
      } else if (arg == "--mode") {
        auto v = need("--mode");
        if (v == "per-column")
          cfg.mode = mode_t::per_column;
        else if (v == "full-table")
          cfg.mode = mode_t::full_table;
        else
          die("--mode: use per-column|full-table");
      } else if (arg == "--warmup") {
        cfg.warmup = std::stoi(need("--warmup"));
      } else if (arg == "--iters") {
        cfg.iters = std::stoi(need("--iters"));
      } else if (arg == "--table-out") {
        cfg.table_out = need("--table-out");
      } else if (arg == "--csv-out") {
        cfg.csv_out = need("--csv-out");
      } else if (arg == "--concat-out") {
        cfg.concat_out = need("--concat-out");
      } else if (arg == "--select-column") {
        cfg.select_columns.push_back(need("--select-column"));
      } else {
        die("unknown flag '" + arg + "'");
      }
    }

    if (cfg.input_paths.empty()) die("--input required (repeat for multiple files)");

    init_gpu();

    // Untimed: read every file's raw bytes into its own host buffer once.
    // Every timed decode below reads from these in-memory buffers — no file
    // I/O, no page cache effects, in the timed region. cudf treats multiple
    // sources passed together as one logical table (concatenated row groups),
    // so N partition files decode as a single N-times-larger table.
    std::vector<std::vector<uint8_t>> file_bytes;
    file_bytes.reserve(cfg.input_paths.size());
    for (auto const& path : cfg.input_paths)
      file_bytes.push_back(read_binary_file(path));

    std::vector<cudf::host_span<uint8_t const>> spans;
    spans.reserve(file_bytes.size());
    for (auto const& bytes : file_bytes)
      spans.emplace_back(bytes.data(), bytes.size());
    cudf::host_span<cudf::host_span<uint8_t const>> spans_view(spans.data(), spans.size());

    // Host-backed source: used only for footer/metadata parsing (discover, concat-out,
    // metadata_ms, read_parquet_footers), which needs host_read() — device-buffer sources
    // don't support that.
    auto make_host_source = [&]() { return cudf::io::source_info{spans_view}; };

    // Untimed: copy every file's bytes to device memory once. The timed decode calls read
    // from here instead of the host buffers above, so no host->device copy of compressed
    // page bytes happens inside the timed region — matching simpatico's decompress, whose
    // input (compressed_representation) is already device-resident before timing starts.
    std::vector<rmm::device_buffer> device_bufs;
    device_bufs.reserve(file_bytes.size());
    for (auto const& bytes : file_bytes)
      device_bufs.emplace_back(bytes.data(),
                               bytes.size(),
                               cudf::get_default_stream(),
                               rmm::mr::get_current_device_resource_ref());
    cuda_sync();

    std::vector<cudf::device_span<std::byte const>> device_spans;
    device_spans.reserve(device_bufs.size());
    for (auto const& buf : device_bufs)
      device_spans.emplace_back(static_cast<std::byte const*>(buf.data()), buf.size());
    cudf::host_span<cudf::device_span<std::byte const>> device_spans_view(device_spans.data(),
                                                                          device_spans.size());

    auto make_source = [&]() { return cudf::io::source_info{device_spans_view}; };

    // Untimed: one full decode to discover schema/column names and per-column
    // decoded byte sizes (used as the GB/s denominator for every column).
    auto discover_builder = cudf::io::parquet_reader_options::builder(make_host_source());
    if (!cfg.select_columns.empty()) discover_builder.column_names(cfg.select_columns);
    auto discover_opts = discover_builder.build();
    auto discovered    = cudf::io::read_parquet(discover_opts);
    cuda_sync();

    if (!cfg.concat_out.empty()) {
      cudf::io::table_input_metadata meta(discovered.tbl->view());
      for (std::size_t k = 0;
           k < discovered.metadata.schema_info.size() && k < meta.column_metadata.size();
           ++k)
        meta.column_metadata[k].set_name(discovered.metadata.schema_info[k].name);
      auto write_opts = cudf::io::parquet_writer_options::builder(
                          cudf::io::sink_info{cfg.concat_out}, discovered.tbl->view())
                          .metadata(meta)
                          .build();
      cudf::io::write_parquet(write_opts);
      std::printf("wrote %d cols x %lld rows -> %s\n",
                  discovered.tbl->num_columns(),
                  static_cast<long long>(discovered.tbl->num_rows()),
                  cfg.concat_out.c_str());
      return 0;
    }

    int const ncols = discovered.tbl->num_columns();
    std::vector<std::string> col_names;
    col_names.reserve(discovered.metadata.schema_info.size());
    for (auto const& c : discovered.metadata.schema_info)
      col_names.push_back(c.name);

    // Isolate the footer/schema-parse-only cost: read_parquet_metadata() does
    // no column decode and no device work, so its time is exactly the piece
    // that read_parquet() re-pays on every call and that simpatico never
    // pays at all. Timed the same way (host-side wall clock; no GPU sync
    // needed since there is no GPU work).
    double const metadata_ms = [&]() {
      std::vector<double> samples;
      for (int w = 0; w < cfg.warmup; ++w)
        cudf::io::read_parquet_metadata(make_host_source());
      for (int i = 0; i < cfg.iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        cudf::io::read_parquet_metadata(make_host_source());
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      return compute_stats(samples).median;
    }();

    auto const compressed_bytes_by_col = parquet_compressed_bytes_by_column(file_bytes);
    auto compressed_bytes_for          = [&](std::string const& name) -> int64_t {
      auto it = compressed_bytes_by_col.find(name);
      return it != compressed_bytes_by_col.end() ? it->second : 0;
    };

    std::vector<bench_row> rows;

    if (cfg.mode == mode_t::per_column) {
      for (int i = 0; i < ncols; ++i) {
        std::string const name = (static_cast<std::size_t>(i) < col_names.size() &&
                                  !col_names[static_cast<std::size_t>(i)].empty())
                                   ? col_names[static_cast<std::size_t>(i)]
                                   : ("col" + std::to_string(i));

        bench_row row;
        row.column           = name;
        row.dtype            = dtype_name(discovered.tbl->view().column(i).type());
        row.rows             = discovered.tbl->num_rows();
        row.bytes            = column_bytes(discovered.tbl->view().column(i));
        row.compressed_bytes = compressed_bytes_for(name);
        row.metadata_ms      = metadata_ms;

        row.decode_ms = time_iters(cfg.warmup, cfg.iters, [&]() {
          auto opts =
            cudf::io::parquet_reader_options::builder(make_source()).column_names({name}).build();
          auto result = cudf::io::read_parquet(opts);
          cuda_sync();  // wait for GPU-visible decode completion
        });
        rows.push_back(std::move(row));
      }
      rows.push_back(make_total_row(rows, metadata_ms));
    } else {
      bench_row row;
      row.column      = "TOTAL";
      row.dtype       = "(all)";
      row.rows        = discovered.tbl->num_rows();
      row.bytes       = table_bytes(discovered.tbl->view());
      row.metadata_ms = metadata_ms;
      for (auto const& [name, bytes] : compressed_bytes_by_col)
        row.compressed_bytes += bytes;

      row.decode_ms = time_iters(cfg.warmup, cfg.iters, [&]() {
        auto opts   = cudf::io::parquet_reader_options::builder(make_source()).build();
        auto result = cudf::io::read_parquet(opts);
        cuda_sync();  // wait for GPU-visible decode completion
      });
      rows.push_back(std::move(row));
    }

    if (cfg.table_out.empty()) {
      write_bench_table(std::cout, rows, cfg);
    } else {
      std::ofstream out(cfg.table_out);
      if (!out) die("cannot write table-out: " + cfg.table_out);
      write_bench_table(out, rows, cfg);
    }
    if (!cfg.csv_out.empty()) write_csv(cfg.csv_out, rows);

    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "parquet_ingest_bench: fatal: %s\n", e.what());
    return 1;
  }
}
