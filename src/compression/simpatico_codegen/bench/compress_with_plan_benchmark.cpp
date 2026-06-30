// SPDX-License-Identifier: Apache-2.0
// compress_with_plan_benchmark — compress/decompress benchmark harness for simpatico_codegen.
// Timed compress + decompress loops with warmup, roundtrip verification,
// and human-readable / CSV output.

#include "api/simpatico_codegen.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"
#include "codegen/util/stream_pool.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using simpatico::compress_with_plan;
using simpatico::compressed_table;
using simpatico::decompress;
using simpatico::plan_compound;
using simpatico::split_plan_dsl;

// ── CLI ──────────────────────────────────────────────────────────────────────

enum class input_format { parquet, binary };
enum class bench_mode { per_column, full_table };

struct config {
  std::string input_path;
  std::optional<input_format> format;
  std::optional<std::string> dtype;  // required for binary
  std::string plan_path;
  bench_mode mode = bench_mode::per_column;
  int threads     = 0;  // 0 => column count (full-table)
  int warmup      = 3;
  int iters       = 10;
  std::string table_out;  // empty => stdout
  std::string csv_out;
};

[[noreturn]] void die(char const* msg, int code = 1)
{
  std::fprintf(stderr, "compress_with_plan_benchmark: %s\n", msg);
  std::exit(code);
}

[[noreturn]] void die_fmt(char const* fmt, std::string const& arg, int code = 1)
{
  std::fprintf(stderr, "compress_with_plan_benchmark: ");
  std::fprintf(stderr, fmt, arg.c_str());
  std::fprintf(stderr, "\n");
  std::exit(code);
}

void usage()
{
  std::fprintf(
    stderr,
    "Usage: compress_with_plan_benchmark --input PATH --plan PATH [options]\n"
    "\n"
    "  --input PATH          Parquet or raw binary column file (required)\n"
    "  --plan PATH           Plan DSL file, one '---'-separated block per column (required)\n"
    "  --format {parquet|binary}\n"
    "                        Input format (default: infer from extension)\n"
    "  --dtype {i32|i64|f32|f64}\n"
    "                        Element type for binary input (required for binary)\n"
    "  --mode {per-column|full-table}\n"
    "                        Benchmark granularity (default: per-column)\n"
    "  --threads N           Parallel column threads for full-table mode\n"
    "                        (default: number of columns)\n"
    "  --warmup N            Warmup iterations (default: 3)\n"
    "  --iters N             Timed iterations (default: 10)\n"
    "  --table-out PATH      Human-readable table output (default: stdout)\n"
    "  --csv-out PATH        CSV output file (optional)\n");
}

std::string read_file(std::string const& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<uint8_t> read_binary_file(std::string const& path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  auto const size = in.tellg();
  in.seekg(0);
  std::vector<uint8_t> data(static_cast<std::size_t>(size));
  if (size > 0) {
    in.read(reinterpret_cast<char*>(data.data()), size);
    if (!in) throw std::runtime_error("read failed: " + path);
  }
  return data;
}

input_format infer_format(std::string const& path)
{
  auto dot = path.rfind('.');
  if (dot != std::string::npos) {
    std::string ext = path.substr(dot);
    for (auto& c : ext)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == ".parquet" || ext == ".pq") return input_format::parquet;
  }
  return input_format::binary;
}

cudf::data_type parse_dtype(std::string const& s)
{
  if (s == "i32") return cudf::data_type{cudf::type_id::INT32};
  if (s == "i64") return cudf::data_type{cudf::type_id::INT64};
  if (s == "f32") return cudf::data_type{cudf::type_id::FLOAT32};
  if (s == "f64") return cudf::data_type{cudf::type_id::FLOAT64};
  throw std::runtime_error("unsupported --dtype '" + s + "' (use i32|i64|f32|f64)");
}

std::string dtype_name(cudf::data_type const& t)
{
  switch (t.id()) {
    case cudf::type_id::INT32: return "i32";
    case cudf::type_id::INT64: return "i64";
    case cudf::type_id::FLOAT32: return "f32";
    case cudf::type_id::FLOAT64: return "f64";
    default: return type_id_to_name(t);
  }
}

bool is_fixed_width_numeric(cudf::data_type const& t)
{
  return cudf::is_numeric(t) && !cudf::is_boolean(t);
}

config parse_args(int argc, char** argv)
{
  config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need       = [&](char const* flag) -> std::string {
      if (i + 1 >= argc) die((std::string(flag) + " requires a value").c_str());
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage();
      std::exit(0);
    } else if (arg == "--input") {
      cfg.input_path = need("--input");
    } else if (arg == "--format") {
      auto v = need("--format");
      if (v == "parquet")
        cfg.format = input_format::parquet;
      else if (v == "binary")
        cfg.format = input_format::binary;
      else
        die("unknown --format (use parquet|binary)");
    } else if (arg == "--dtype") {
      cfg.dtype = need("--dtype");
    } else if (arg == "--plan") {
      cfg.plan_path = need("--plan");
    } else if (arg == "--mode") {
      auto v = need("--mode");
      if (v == "per-column")
        cfg.mode = bench_mode::per_column;
      else if (v == "full-table")
        cfg.mode = bench_mode::full_table;
      else
        die("unknown --mode (use per-column|full-table)");
    } else if (arg == "--threads") {
      cfg.threads = std::stoi(need("--threads"));
    } else if (arg == "--warmup") {
      cfg.warmup = std::stoi(need("--warmup"));
    } else if (arg == "--iters") {
      cfg.iters = std::stoi(need("--iters"));
    } else if (arg == "--table-out") {
      cfg.table_out = need("--table-out");
    } else if (arg == "--csv-out") {
      cfg.csv_out = need("--csv-out");
    } else {
      die_fmt("unknown flag '%s'", arg);
    }
  }
  if (cfg.input_path.empty()) die("--input required");
  if (cfg.plan_path.empty()) die("--plan required");
  if (!cfg.format) cfg.format = infer_format(cfg.input_path);
  if (*cfg.format == input_format::binary && !cfg.dtype) {
    die("--dtype required for binary input");
  }
  if (cfg.warmup < 0 || cfg.iters < 1) die("--warmup must be >= 0 and --iters must be >= 1");
  return cfg;
}

// ── Input loading ─────────────────────────────────────────────────────────────

struct loaded_table {
  std::unique_ptr<cudf::table> table;
  std::vector<std::string> column_names;
};

loaded_table load_parquet(std::string const& path)
{
  auto source  = cudf::io::source_info{path};
  auto options = cudf::io::parquet_reader_options::builder(source).build();
  auto result  = cudf::io::read_parquet(options);
  loaded_table out;
  out.table = std::move(result.tbl);
  out.column_names.reserve(result.metadata.schema_info.size());
  for (auto const& col : result.metadata.schema_info) {
    out.column_names.push_back(col.name);
  }
  for (int i = 0; i < out.table->num_columns(); ++i) {
    auto const& t = out.table->view().column(i).type();
    if (!is_fixed_width_numeric(t)) {
      throw std::runtime_error("parquet column " + std::to_string(i) + " (" +
                               out.column_names[static_cast<std::size_t>(i)] +
                               ") is not fixed-width numeric: " + type_id_to_name(t));
    }
  }
  return out;
}

loaded_table load_binary(std::string const& path, cudf::data_type dtype)
{
  auto bytes             = read_binary_file(path);
  std::size_t const elem = static_cast<std::size_t>(cudf::size_of(dtype));
  if (elem == 0 || bytes.size() % elem != 0) {
    throw std::runtime_error("binary file size " + std::to_string(bytes.size()) +
                             " is not a multiple of element size " + std::to_string(elem));
  }
  cudf::size_type const nrows = static_cast<cudf::size_type>(bytes.size() / elem);
  auto col = cudf::make_numeric_column(dtype, nrows, cudf::mask_state::UNALLOCATED);
  if (!bytes.empty()) {
    cudaMemcpy(
      col->mutable_view().head<void>(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  loaded_table out;
  out.table        = std::make_unique<cudf::table>(std::move(cols));
  out.column_names = {"col0"};
  return out;
}

// ── Memory / sync helpers ───────────────────────────────────────────────────────

// Owns a cuda_async_memory_resource and installs it as the current device
// resource. RMM 26.x resource refs are non-owning, so the resource object
// (`mr`) must outlive the period it is the current resource — the destructor
// restores `previous` before `mr` is torn down. `previous` is captured at
// construction (install() runs immediately after), so install() does not need
// the set_current_device_resource_ref return value — which is `device_async_resource_ref`
// on some RMM versions and a deprecated `any_resource` on newer ones.
struct pool_mr_guard {
  rmm::mr::cuda_async_memory_resource mr{};
  rmm::device_async_resource_ref previous{rmm::mr::get_current_device_resource_ref()};
  bool installed = false;

  void install()
  {
    rmm::mr::set_current_device_resource_ref(mr);
    installed = true;
  }

  ~pool_mr_guard()
  {
    if (installed) rmm::mr::set_current_device_resource_ref(previous);
  }
};

void cuda_sync()
{
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(err));
}

std::size_t column_input_bytes(cudf::column_view col)
{
  return static_cast<std::size_t>(col.size()) * static_cast<std::size_t>(cudf::size_of(col.type()));
}

std::size_t table_input_bytes(cudf::table_view tv)
{
  std::size_t total = 0;
  for (int i = 0; i < tv.num_columns(); ++i)
    total += column_input_bytes(tv.column(i));
  return total;
}

// ── Compressed size ─────────────────────────────────────────────────────────────

std::size_t rep_bytes(simpatico::compressed_representation const* rep)
{
  return rep ? rep->compressed_size_bytes() : 0;
}

std::size_t compound_compressed_bytes(plan_compound const& compound)
{
  std::size_t total = 0;
  for (auto const& node : compound.tree.nodes) {
    total += rep_bytes(node.rep.get());
    for (auto const& [path, rep] : node.channels) {
      (void)path;
      total += rep_bytes(rep.get());
    }
  }
  return total;
}

std::size_t compressed_table_bytes(compressed_table const& ct)
{
  std::size_t total = 0;
  for (auto const& col : ct.columns) {
    if (col.compound) total += compound_compressed_bytes(*col.compound);
  }
  return total;
}

// ── Stats ─────────────────────────────────────────────────────────────────────

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
  if (n % 2 == 1)
    s.median = sorted[n / 2];
  else
    s.median = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  return s;
}

double gbps(std::size_t bytes, double ms)
{
  if (ms <= 0.0) return 0.0;
  return (static_cast<double>(bytes) / 1.0e9) / (ms / 1000.0);
}

double ratio(std::size_t input_bytes, std::size_t compressed_bytes)
{
  if (compressed_bytes == 0) return 0.0;
  return static_cast<double>(input_bytes) / static_cast<double>(compressed_bytes);
}

// ── Verify ────────────────────────────────────────────────────────────────────

bool columns_equal(cudf::column_view a, cudf::column_view b)
{
  if (a.type() != b.type() || a.size() != b.size()) return false;
  std::size_t const nbytes =
    static_cast<std::size_t>(a.size()) * static_cast<std::size_t>(cudf::size_of(a.type()));
  std::vector<uint8_t> ha(nbytes), hb(nbytes);
  if (nbytes > 0) {
    cudaMemcpy(ha.data(), a.head<uint8_t>(), nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b.head<uint8_t>(), nbytes, cudaMemcpyDeviceToHost);
  }
  return ha == hb;
}

bool verify_roundtrip(cudf::table_view input, compressed_table const& ct)
{
  auto out = decompress(ct, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
  if (!out || out->num_columns() != input.num_columns()) return false;
  for (int i = 0; i < input.num_columns(); ++i) {
    if (!columns_equal(input.column(i), out->view().column(i))) return false;
  }
  return true;
}

// ── Benchmark result row ────────────────────────────────────────────────────────

struct bench_row {
  std::string column;
  std::string dtype;
  std::int64_t rows            = 0;
  std::size_t input_bytes      = 0;
  std::size_t compressed_bytes = 0;
  timing_stats compress_ms;
  timing_stats decompress_ms;
  bool verify_ok = false;

  double ratio_val() const { return ratio(input_bytes, compressed_bytes); }
  double compress_gbps_median() const { return gbps(input_bytes, compress_ms.median); }
  double decompress_gbps_median() const { return gbps(input_bytes, decompress_ms.median); }
};

// ── Timing loops ──────────────────────────────────────────────────────────────

template <typename Fn>
timing_stats time_loop(int warmup, int iters, Fn&& fn)
{
  for (int w = 0; w < warmup; ++w)
    fn();
  cuda_sync();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    cuda_sync();
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return compute_stats(samples);
}

bench_row bench_single_column(cudf::column_view col,
                              std::string_view plan_block,
                              std::string const& col_name,
                              int warmup,
                              int iters)
{
  bench_row row;
  row.column      = col_name;
  row.dtype       = dtype_name(col.type());
  row.rows        = col.size();
  row.input_bytes = column_input_bytes(col);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::make_unique<cudf::column>(col));
  cudf::table single(std::move(cols));

  // Compress warmup + timed
  compressed_table last_ct;
  for (int w = 0; w < warmup; ++w) {
    last_ct = compress_with_plan(single.view(),
                                 plan_block,
                                 cudf::get_default_stream(),
                                 rmm::mr::get_current_device_resource_ref());
  }
  cuda_sync();
  std::vector<double> compress_samples;
  compress_samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    last_ct = compressed_table{};  // drop previous outside timer
    cuda_sync();
    auto t0 = std::chrono::steady_clock::now();
    last_ct = compress_with_plan(single.view(),
                                 plan_block,
                                 cudf::get_default_stream(),
                                 rmm::mr::get_current_device_resource_ref());
    auto t1 = std::chrono::steady_clock::now();
    compress_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  row.compress_ms      = compute_stats(compress_samples);
  row.compressed_bytes = compressed_table_bytes(last_ct);

  // Decompress warmup + timed (reuse last_ct)
  auto decompress_once = [&]() {
    auto out =
      decompress(last_ct, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
    (void)out;
  };
  for (int w = 0; w < warmup; ++w)
    decompress_once();
  cuda_sync();
  std::vector<double> decompress_samples;
  decompress_samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    cuda_sync();
    auto t0 = std::chrono::steady_clock::now();
    decompress_once();
    auto t1 = std::chrono::steady_clock::now();
    decompress_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  row.decompress_ms = compute_stats(decompress_samples);

  row.verify_ok = verify_roundtrip(single.view(), last_ct);
  return row;
}

bench_row bench_full_table(
  cudf::table_view tv, std::string_view plan_dsl, int threads, int warmup, int iters)
{
  bench_row row;
  row.column      = "TOTAL";
  row.dtype       = "(all)";
  row.rows        = tv.num_rows();
  row.input_bytes = table_input_bytes(tv);

  // Caller-owned pool must outlive parallel compress/decompress calls — the
  // int-thread overload destroys its internal pool on return, which races
  // cuda_async_memory_resource frees tied to worker streams.
  simpatico::stream_pool pool;
  if (!pool.init(static_cast<std::size_t>(std::max(1, threads)))) {
    throw std::runtime_error("failed to initialize stream_pool");
  }

  compressed_table last_ct;
  for (int w = 0; w < warmup; ++w) {
    last_ct = compress_with_plan(tv, plan_dsl, pool, rmm::mr::get_current_device_resource_ref());
  }
  cuda_sync();
  std::vector<double> compress_samples;
  compress_samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    last_ct = compressed_table{};
    cuda_sync();  // let prior compound teardown finish before timing
    auto t0 = std::chrono::steady_clock::now();
    last_ct = compress_with_plan(tv, plan_dsl, pool, rmm::mr::get_current_device_resource_ref());
    auto t1 = std::chrono::steady_clock::now();
    compress_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  row.compress_ms      = compute_stats(compress_samples);
  row.compressed_bytes = compressed_table_bytes(last_ct);

  auto decompress_once = [&]() {
    auto out = decompress(last_ct, pool, rmm::mr::get_current_device_resource_ref());
    cuda_sync();
  };
  for (int w = 0; w < warmup; ++w)
    decompress_once();
  cuda_sync();
  std::vector<double> decompress_samples;
  decompress_samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    cuda_sync();
    auto t0 = std::chrono::steady_clock::now();
    decompress_once();
    auto t1 = std::chrono::steady_clock::now();
    decompress_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  row.decompress_ms = compute_stats(decompress_samples);

  row.verify_ok = verify_roundtrip(tv, last_ct);
  return row;
}

bench_row make_total_row(std::vector<bench_row> const& rows)
{
  bench_row total;
  total.column    = "TOTAL";
  total.dtype     = "(all)";
  total.rows      = rows.empty() ? 0 : rows.front().rows;
  total.verify_ok = true;
  for (auto const& r : rows) {
    total.input_bytes += r.input_bytes;
    total.compressed_bytes += r.compressed_bytes;
    total.compress_ms.min += r.compress_ms.min;
    total.compress_ms.median += r.compress_ms.median;
    total.compress_ms.mean += r.compress_ms.mean;
    total.decompress_ms.min += r.decompress_ms.min;
    total.decompress_ms.median += r.decompress_ms.median;
    total.decompress_ms.mean += r.decompress_ms.mean;
    total.verify_ok = total.verify_ok && r.verify_ok;
  }
  return total;
}

// ── Output ──────────────────────────────────────────────────────────────────────

void write_row_csv(std::ostream& os, bench_row const& r)
{
  os << r.column << ',' << r.dtype << ',' << r.rows << ',' << r.input_bytes << ','
     << r.compressed_bytes << ',' << r.ratio_val() << ',' << r.compress_ms.min << ','
     << r.compress_ms.median << ',' << r.compress_ms.mean << ',' << r.compress_gbps_median() << ','
     << r.decompress_ms.min << ',' << r.decompress_ms.median << ',' << r.decompress_ms.mean << ','
     << r.decompress_gbps_median() << ',' << (r.verify_ok ? 1 : 0) << '\n';
}

void write_csv(std::string const& path, std::vector<bench_row> const& rows)
{
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write csv: " + path);
  out << "column,dtype,rows,input_bytes,compressed_bytes,ratio,"
         "compress_ms_min,compress_ms_median,compress_ms_mean,compress_gbps_median,"
         "decompress_ms_min,decompress_ms_median,decompress_ms_mean,decompress_gbps_median,"
         "verify_ok\n";
  for (auto const& r : rows)
    write_row_csv(out, r);
}

void write_table(std::ostream& os, std::vector<bench_row> const& rows, config const& cfg)
{
  os << "# compress_with_plan_benchmark mode="
     << (cfg.mode == bench_mode::per_column ? "per-column" : "full-table")
     << " warmup=" << cfg.warmup << " iters=" << cfg.iters << '\n';
  os << "# column | dtype | rows | input_bytes | compressed_bytes | ratio | "
        "comp_ms(min/med/mean) | comp_GBps | decomp_ms(min/med/mean) | decomp_GBps | verify\n";

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
    os << r.column << " | " << r.dtype << " | " << r.rows << " | " << r.input_bytes << " | "
       << r.compressed_bytes << " | " << fmt3(r.ratio_val()) << "x | " << fmt3(r.compress_ms.min)
       << "/" << fmt3(r.compress_ms.median) << "/" << fmt3(r.compress_ms.mean) << " | "
       << fmt2(r.compress_gbps_median()) << " | " << fmt3(r.decompress_ms.min) << "/"
       << fmt3(r.decompress_ms.median) << "/" << fmt3(r.decompress_ms.mean) << " | "
       << fmt2(r.decompress_gbps_median()) << " | " << (r.verify_ok ? "ok" : "FAIL") << '\n';
  }
}

void emit_outputs(config const& cfg, std::vector<bench_row> const& rows)
{
  if (cfg.table_out.empty()) {
    write_table(std::cout, rows, cfg);
  } else {
    std::ofstream out(cfg.table_out);
    if (!out) throw std::runtime_error("cannot write table: " + cfg.table_out);
    write_table(out, rows, cfg);
  }
  if (!cfg.csv_out.empty()) write_csv(cfg.csv_out, rows);
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    config cfg = parse_args(argc, argv);

    if (cudaSetDevice(0) != cudaSuccess) die("cudaSetDevice(0) failed");
    pool_mr_guard mr;
    mr.install();
    codegen::jit::ensure_cuda_context();

    loaded_table loaded;
    if (*cfg.format == input_format::parquet) {
      loaded = load_parquet(cfg.input_path);
    } else {
      loaded = load_binary(cfg.input_path, parse_dtype(*cfg.dtype));
    }

    std::string plan_dsl = read_file(cfg.plan_path);
    auto plan_blocks     = split_plan_dsl(plan_dsl);
    auto const ncols     = loaded.table->num_columns();
    if (static_cast<int>(plan_blocks.size()) != ncols) {
      std::fprintf(stderr,
                   "compress_with_plan_benchmark: plan has %zu blocks but input has %d columns\n",
                   plan_blocks.size(),
                   ncols);
      return 1;
    }

    int threads = cfg.threads > 0 ? cfg.threads : ncols;
    std::vector<bench_row> rows;

    if (cfg.mode == bench_mode::per_column) {
      for (int i = 0; i < ncols; ++i) {
        std::string name = (static_cast<std::size_t>(i) < loaded.column_names.size() &&
                            !loaded.column_names[static_cast<std::size_t>(i)].empty())
                             ? loaded.column_names[static_cast<std::size_t>(i)]
                             : ("col" + std::to_string(i));
        rows.push_back(bench_single_column(loaded.table->view().column(i),
                                           plan_blocks[static_cast<std::size_t>(i)],
                                           name,
                                           cfg.warmup,
                                           cfg.iters));
      }
      rows.push_back(make_total_row(std::vector<bench_row>(rows.begin(), rows.end())));
    } else {
      rows.push_back(
        bench_full_table(loaded.table->view(), plan_dsl, threads, cfg.warmup, cfg.iters));
    }

    emit_outputs(cfg, rows);

    bool all_ok = true;
    for (auto const& r : rows) {
      if (!r.verify_ok) all_ok = false;
    }
    return all_ok ? 0 : 5;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "plan_runner: %s\n", e.what());
    return 1;
  }
}
