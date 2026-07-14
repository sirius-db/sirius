// SPDX-License-Identifier: Apache-2.0
//
// simpatico — multi-mode CLI for simpatico_codegen.
//
// Modes:
//   benchmark  Timed compress+decompress over a Parquet/binary/CSV input
//   explore    BFS cascade search for a single column
//   compress   Compress input to a .hpln file (--verify round-trips it)
//   decompress Decompress a .hpln file to Parquet (--verify checks vs source)
//   plan       Print each column's compression plan (DSL) from a .hpln file
//
// The driver helpers below (input loading / dtype parsing, roundtrip
// verification, and the `benchmark` mode machinery — the latter originally in
// bench/compress_with_plan_benchmark.cpp) live directly in this translation
// unit: this is the only consumer, so an anonymous namespace gives them
// internal linkage without a separate header.

#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "codegen/plan/plan_interpreter.hpp"  // plan_compound, render_plan_tree
#include "codegen/util/stream_pool.hpp"
#include "explore/compression_explorer.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
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
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ── Error helpers ─────────────────────────────────────────────────────────────

[[noreturn]] void die(std::string const& msg, int code = 1)
{
  std::fprintf(stderr, "simpatico: %s\n", msg.c_str());
  std::exit(code);
}

// ── RMM pool guard ────────────────────────────────────────────────────────────

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

// ── Input format / dtype helpers ──────────────────────────────────────────────

enum class input_format { parquet, binary, csv };

input_format infer_format(std::string const& path)
{
  auto dot = path.rfind('.');
  if (dot != std::string::npos) {
    std::string ext = path.substr(dot);
    for (auto& c : ext)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == ".parquet" || ext == ".pq") return input_format::parquet;
    if (ext == ".csv" || ext == ".tbl") return input_format::csv;
  }
  return input_format::binary;
}

cudf::data_type parse_dtype(std::string const& s)
{
  if (s == "i8") return cudf::data_type{cudf::type_id::INT8};
  if (s == "i16") return cudf::data_type{cudf::type_id::INT16};
  if (s == "i32") return cudf::data_type{cudf::type_id::INT32};
  if (s == "i64") return cudf::data_type{cudf::type_id::INT64};
  if (s == "u8") return cudf::data_type{cudf::type_id::UINT8};
  if (s == "u16") return cudf::data_type{cudf::type_id::UINT16};
  if (s == "u32") return cudf::data_type{cudf::type_id::UINT32};
  if (s == "u64") return cudf::data_type{cudf::type_id::UINT64};
  if (s == "f32") return cudf::data_type{cudf::type_id::FLOAT32};
  if (s == "f64") return cudf::data_type{cudf::type_id::FLOAT64};
  throw std::runtime_error("unsupported --dtype '" + s +
                           "' (use i8/i16/i32/i64/u8/u16/u32/u64/f32/f64)");
}

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
    default: return "t" + std::to_string(static_cast<int>(t.id()));
  }
}

// ── Shared argument parsing ───────────────────────────────────────────────────

input_format parse_input_format(std::string const& v)
{
  if (v == "parquet") return input_format::parquet;
  if (v == "csv") return input_format::csv;
  if (v == "binary") return input_format::binary;
  die("--format: use parquet|csv|binary");
}

// Handle the input-loading flags shared by the benchmark/explore/compress modes
// (--input / --format / --dtype). `need(flag)` returns the flag's value and
// advances the caller's scan index. Returns true if `arg` was one of these
// flags (and has been consumed), false otherwise so the caller can keep matching
// its own flags.
template <typename NeedFn>
bool parse_input_flag(std::string const& arg,
                      NeedFn&& need,
                      std::string& path,
                      std::optional<input_format>& fmt,
                      std::optional<std::string>& dtype)
{
  if (arg == "--input") {
    path = need("--input");
  } else if (arg == "--format") {
    fmt = parse_input_format(need("--format"));
  } else if (arg == "--dtype") {
    dtype = need("--dtype");
  } else {
    return false;
  }
  return true;
}

// ── File I/O helpers ──────────────────────────────────────────────────────────

std::string read_file(std::string const& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

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

// ── Table loading ─────────────────────────────────────────────────────────────

struct loaded_table {
  std::unique_ptr<cudf::table> table;
  std::vector<std::string> column_names;
};

/// Parquet loader — accepts all column types (numerics, strings, …).
loaded_table load_parquet(std::string const& path)
{
  auto source  = cudf::io::source_info{path};
  auto options = cudf::io::parquet_reader_options::builder(source).build();
  auto result  = cudf::io::read_parquet(options);
  loaded_table out;
  out.table = std::move(result.tbl);
  out.column_names.reserve(result.metadata.schema_info.size());
  for (auto const& col : result.metadata.schema_info)
    out.column_names.push_back(col.name);
  return out;
}

/// Binary loader — flat raw array of a single dtype, single column.
loaded_table load_binary(std::string const& path, cudf::data_type dtype)
{
  auto bytes             = read_binary_file(path);
  std::size_t const elem = static_cast<std::size_t>(cudf::size_of(dtype));
  if (elem == 0 || bytes.size() % elem != 0)
    throw std::runtime_error("binary file size " + std::to_string(bytes.size()) +
                             " is not a multiple of element size " + std::to_string(elem));
  cudf::size_type const nrows = static_cast<cudf::size_type>(bytes.size() / elem);
  auto col = cudf::make_numeric_column(dtype, nrows, cudf::mask_state::UNALLOCATED);
  if (!bytes.empty())
    cudaMemcpy(
      col->mutable_view().head<void>(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  loaded_table out;
  out.table        = std::make_unique<cudf::table>(std::move(cols));
  out.column_names = {"col0"};
  return out;
}

/// CSV loader. Delimiter defaults to ',' for .csv, '|' for .tbl (TPC-H).
/// header=true treats the first row as column names.
loaded_table load_csv(std::string const& path, char delimiter = ',', bool has_header = true)
{
  auto source = cudf::io::source_info{path};
  auto opts   = cudf::io::csv_reader_options::builder(source)
                .delimiter(delimiter)
                .header(has_header ? 0 : -1)
                .build();
  auto result = cudf::io::read_csv(opts);
  loaded_table out;
  out.table = std::move(result.tbl);
  for (auto const& col : result.metadata.schema_info)
    out.column_names.push_back(col.name);
  if (out.column_names.empty())
    for (int i = 0; i < out.table->num_columns(); ++i)
      out.column_names.push_back("col" + std::to_string(i));
  return out;
}

/// Dispatch to the right loader based on detected/specified format.
loaded_table load_input(std::string const& path,
                        input_format fmt,
                        std::optional<std::string> const& dtype_str = std::nullopt)
{
  switch (fmt) {
    case input_format::parquet: return load_parquet(path);
    case input_format::csv: {
      // .tbl = TPC-H pipe-separated, no header
      bool const is_tbl = path.size() >= 4 && path.substr(path.size() - 4) == ".tbl";
      return load_csv(path, is_tbl ? '|' : ',', !is_tbl);
    }
    case input_format::binary:
      if (!dtype_str) throw std::runtime_error("--dtype required for binary input");
      return load_binary(path, parse_dtype(*dtype_str));
  }
  throw std::runtime_error("unknown input format");
}

// ── CUDA / sync helpers ────────────────────────────────────────────────────────

void cuda_sync()
{
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(err));
}

std::size_t column_input_bytes(cudf::column_view col,
                               rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  if (col.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view scv(col);
    return static_cast<std::size_t>(col.size() + 1) * sizeof(int32_t) +
           static_cast<std::size_t>(scv.chars_size(stream));
  }
  return static_cast<std::size_t>(col.size()) * static_cast<std::size_t>(cudf::size_of(col.type()));
}

std::size_t table_input_bytes(cudf::table_view tv,
                              rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  std::size_t total = 0;
  for (int i = 0; i < tv.num_columns(); ++i)
    total += column_input_bytes(tv.column(i), stream);
  return total;
}

// ── Statistics ─────────────────────────────────────────────────────────────────

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

double gbps(std::size_t bytes, double ms)
{
  if (ms <= 0.0) return 0.0;
  return (static_cast<double>(bytes) / 1.0e9) / (ms / 1000.0);
}

double compression_ratio(std::size_t input_bytes, std::size_t compressed_bytes)
{
  if (compressed_bytes == 0) return 0.0;
  return static_cast<double>(input_bytes) / static_cast<double>(compressed_bytes);
}

// ── Column equality (fixed-width + strings) ───────────────────────────────────

// Per-row validity as host bools (true = valid); maskless / 0-null columns are
// all-valid, so a (mask, 0-null) column and a maskless one compare equal.
std::vector<bool> host_validity_bits(cudf::column_view v)
{
  std::vector<bool> valid(static_cast<std::size_t>(v.size()), true);
  if (v.null_mask() == nullptr || v.null_count() == 0) return valid;
  std::size_t const first_word = static_cast<std::size_t>(v.offset()) / 32;
  std::size_t const last_word  = static_cast<std::size_t>(v.offset() + v.size() + 31) / 32;
  std::vector<uint32_t> words(last_word - first_word);
  cudaMemcpy(words.data(),
             reinterpret_cast<uint32_t const*>(v.null_mask()) + first_word,
             words.size() * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  for (cudf::size_type r = 0; r < v.size(); ++r) {
    std::size_t const bit              = static_cast<std::size_t>(v.offset() + r);
    valid[static_cast<std::size_t>(r)] = (words[bit / 32 - first_word] >> (bit % 32)) & 1u;
  }
  return valid;
}

bool columns_equal(cudf::column_view a, cudf::column_view b)
{
  if (a.type() != b.type() || a.size() != b.size()) return false;
  // Validity must match too: a roundtrip that drops or moves nulls is not a
  // PASS even when the payload bytes agree.
  if (a.null_count() != b.null_count() || host_validity_bits(a) != host_validity_bits(b))
    return false;
  if (a.size() == 0) return true;

  if (a.type().id() == cudf::type_id::STRING) {
    auto stream = cudf::get_default_stream();
    cudf::strings_column_view sa(a), sb(b);
    auto const ca = sa.chars_size(stream);
    auto const cb = sb.chars_size(stream);
    if (ca != cb) return false;
    std::size_t const n_off = static_cast<std::size_t>(a.size()) + 1;
    std::vector<int32_t> oa(n_off), ob(n_off);
    cudaMemcpy(
      oa.data(), sa.offsets().head<int32_t>(), n_off * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(
      ob.data(), sb.offsets().head<int32_t>(), n_off * sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (oa != ob) return false;
    if (ca > 0) {
      std::vector<uint8_t> pa(static_cast<std::size_t>(ca)), pb(static_cast<std::size_t>(ca));
      cudaMemcpy(
        pa.data(), sa.chars_begin(stream), static_cast<std::size_t>(ca), cudaMemcpyDeviceToHost);
      cudaMemcpy(
        pb.data(), sb.chars_begin(stream), static_cast<std::size_t>(ca), cudaMemcpyDeviceToHost);
      if (pa != pb) return false;
    }
    return true;
  }

  std::size_t const nbytes =
    static_cast<std::size_t>(a.size()) * static_cast<std::size_t>(cudf::size_of(a.type()));
  std::vector<uint8_t> ha(nbytes), hb(nbytes);
  if (nbytes > 0) {
    cudaMemcpy(ha.data(), a.head<uint8_t>(), nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b.head<uint8_t>(), nbytes, cudaMemcpyDeviceToHost);
  }
  return ha == hb;
}

bool tables_equal(cudf::table_view a, cudf::table_view b)
{
  if (a.num_columns() != b.num_columns()) return false;
  for (int i = 0; i < a.num_columns(); ++i)
    if (!columns_equal(a.column(i), b.column(i))) return false;
  return true;
}

/// Decompress @p ct and compare every column byte-exactly to @p source.
bool verify_roundtrip(cudf::table_view source, simpatico::compressed_table const& ct)
{
  auto out = simpatico::decompress(
    ct, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
  if (!out || out->num_columns() != source.num_columns()) return false;
  return tables_equal(source, out->view());
}

// ══ `benchmark` mode ══════════════════════════════════════════════════════════
// Timed compress/decompress over a loaded table, with per-column or full-table
// modes and CSV / human-readable reporting.

// ── Compressed-size accounting ────────────────────────────────────────────────

std::size_t rep_bytes(simpatico::compressed_representation const* rep,
                      rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  return rep ? rep->compressed_size_bytes(stream) : 0;
}

std::size_t compound_compressed_bytes(simpatico::PlanTree const& tree,
                                      rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  std::size_t total = 0;
  for (auto const& node : tree.nodes) {
    total += rep_bytes(node.rep.get(), stream);
    for (auto const& [path, rep] : node.channels)
      total += rep_bytes(rep.get(), stream);
  }
  return total;
}

std::size_t compressed_table_bytes(simpatico::compressed_table const& ct,
                                   rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  std::size_t total = 0;
  for (auto const& col : ct.columns)
    if (col.compound) total += compound_compressed_bytes(*col.compound, stream);
  return total;
}

// ── Benchmark result row ──────────────────────────────────────────────────────

struct bench_row {
  std::string column;
  std::string dtype;
  std::int64_t rows            = 0;
  std::size_t input_bytes      = 0;
  std::size_t compressed_bytes = 0;
  timing_stats compress_ms;
  timing_stats decompress_ms;
  bool verify_ok = false;

  double ratio_val() const { return compression_ratio(input_bytes, compressed_bytes); }
  double compress_gbps_median() const { return gbps(input_bytes, compress_ms.median); }
  double decompress_gbps_median() const { return gbps(input_bytes, decompress_ms.median); }
};

// Time `body` over `iters` runs after `warmup` untimed runs, returning the
// per-run stats. `reset` runs before each timed iteration (untimed, e.g. to drop
// the previous result); a drain sync then separates iterations. `body` owns any
// GPU-completion sync it wants counted in its own timing — so a compress body
// that measures launch-only omits it, while one measuring completion ends with
// cuda_sync().
template <typename Reset, typename Body>
timing_stats time_iters(int warmup, int iters, Reset&& reset, Body&& body)
{
  for (int w = 0; w < warmup; ++w)
    body();
  cuda_sync();

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    reset();
    cuda_sync();  // drain previous iter
    auto t0 = std::chrono::steady_clock::now();
    body();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return compute_stats(samples);
}

// ── Per-column benchmark ─────────────────────────────────────────────────────

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

  simpatico::compressed_table last_ct;
  row.compress_ms = time_iters(
    warmup,
    iters,
    [&]() { last_ct = simpatico::compressed_table{}; },
    [&]() {
      last_ct = simpatico::compress_with_plan(single.view(),
                                              plan_block,
                                              cudf::get_default_stream(),
                                              rmm::mr::get_current_device_resource_ref());
      cuda_sync();  // wait for GPU completion
    });
  row.compressed_bytes = compressed_table_bytes(last_ct);

  row.decompress_ms = time_iters(
    warmup,
    iters,
    []() {},
    [&]() {
      auto out = simpatico::decompress(
        last_ct, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
      (void)out;
      cuda_sync();  // wait for GPU completion
    });
  row.verify_ok = verify_roundtrip(single.view(), last_ct);
  return row;
}

// ── Full-table benchmark ──────────────────────────────────────────────────────

bench_row bench_full_table(
  cudf::table_view tv, std::string_view plan_dsl, int threads, int warmup, int iters)
{
  bench_row row;
  row.column      = "TOTAL";
  row.dtype       = "(all)";
  row.rows        = tv.num_rows();
  row.input_bytes = table_input_bytes(tv);

  simpatico::stream_pool pool;
  if (!pool.init(static_cast<std::size_t>(std::max(1, threads))))
    throw std::runtime_error("failed to initialize stream_pool");

  // The pooled compress_with_plan already joins its worker streams, so the
  // compress body measures completion without an extra sync.
  simpatico::compressed_table last_ct;
  row.compress_ms = time_iters(
    warmup,
    iters,
    [&]() { last_ct = simpatico::compressed_table{}; },
    [&]() {
      last_ct = simpatico::compress_with_plan(
        tv, plan_dsl, pool, rmm::mr::get_current_device_resource_ref());
    });
  row.compressed_bytes = compressed_table_bytes(last_ct);

  row.decompress_ms = time_iters(
    warmup,
    iters,
    []() {},
    [&]() {
      auto out = simpatico::decompress(last_ct, pool, rmm::mr::get_current_device_resource_ref());
      (void)out;
      cuda_sync();  // wait for GPU completion
    });
  row.verify_ok = verify_roundtrip(tv, last_ct);
  return row;
}

// ── Aggregation ───────────────────────────────────────────────────────────────

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

// ── Output formatting ─────────────────────────────────────────────────────────

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

struct bench_config {
  std::string input_path;
  std::optional<input_format> format;
  std::optional<std::string> dtype;
  std::string plan_path;
  enum class mode_t { per_column, full_table } mode = mode_t::per_column;
  int threads                                       = 0;
  int warmup                                        = 3;
  int iters                                         = 10;
  std::string table_out;
  std::string csv_out;
};

void write_bench_table(std::ostream& os,
                       std::vector<bench_row> const& rows,
                       bench_config const& cfg)
{
  using mode_t = bench_config::mode_t;
  os << "# benchmark mode=" << (cfg.mode == mode_t::per_column ? "per-column" : "full-table")
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

// ── Top-level usage ───────────────────────────────────────────────────────────

void usage_top()
{
  std::fprintf(stderr,
               "Usage: simpatico <mode> [options]\n"
               "\n"
               "Modes:\n"
               "  benchmark   Timed compress+decompress (Parquet/binary/CSV input, plan file)\n"
               "  explore     BFS cascade search for the best plan for a single column\n"
               "  compress    Compress input to a .hpln file (--verify round-trips it)\n"
               "  decompress  Decompress a .hpln file to Parquet (--verify checks vs source)\n"
               "  plan        Print each column's compression plan (DSL) from a .hpln file\n"
               "\n"
               "Run 'simpatico <mode> --help' for per-mode options.\n");
}

// ── Shared GPU/RMM init ───────────────────────────────────────────────────────

pool_mr_guard g_mr;

void init_gpu()
{
  if (cudaSetDevice(0) != cudaSuccess) die("cudaSetDevice(0) failed");
  g_mr.install();
}

/// Explicit, non-default stream for all driver work, created once as a
/// function-local static and never torn down before process exit (destroyed via
/// the regular atexit path). The nvcomp-backed operators drive the low-level
/// batched API with all memory owned by RMM and hold no stream-bound state, so
/// this stream just needs to outlive the work enqueued on it.
rmm::cuda_stream_view driver_stream()
{
  static rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  return stream.view();
}

// ── BENCHMARK mode ────────────────────────────────────────────────────────────

void usage_benchmark()
{
  std::fprintf(stderr,
               "Usage: simpatico benchmark --input PATH --plan PATH [options]\n"
               "\n"
               "  --input PATH              Parquet, CSV/.tbl, or raw binary file (required)\n"
               "  --plan PATH               Plan DSL file, '---'-separated per column (required)\n"
               "  --format {parquet|csv|binary}\n"
               "                            Input format (default: infer from extension)\n"
               "  --dtype {i32|i64|f32|f64|u8|...}\n"
               "                            Element type for binary input\n"
               "  --mode {per-column|full-table}\n"
               "                            Benchmark granularity (default: per-column)\n"
               "  --threads N               Worker threads for full-table mode\n"
               "  --warmup N                Warmup iterations (default: 3)\n"
               "  --iters N                 Timed iterations (default: 10)\n"
               "  --table-out PATH          Human-readable output file (default: stdout)\n"
               "  --csv-out PATH            CSV output file\n");
}

int run_benchmark(int argc, char** argv)
{
  using mode_t = bench_config::mode_t;
  bench_config cfg;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need       = [&](char const* flag) -> std::string {
      if (i + 1 >= argc) die(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage_benchmark();
      return 0;
    } else if (parse_input_flag(arg, need, cfg.input_path, cfg.format, cfg.dtype)) {
      // handled: --input / --format / --dtype
    } else if (arg == "--plan") {
      cfg.plan_path = need("--plan");
    } else if (arg == "--mode") {
      auto v = need("--mode");
      if (v == "per-column")
        cfg.mode = mode_t::per_column;
      else if (v == "full-table")
        cfg.mode = mode_t::full_table;
      else
        die("--mode: use per-column|full-table");
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
      die("benchmark: unknown flag '" + arg + "'");
    }
  }

  if (cfg.input_path.empty()) die("benchmark: --input required");
  if (cfg.plan_path.empty()) die("benchmark: --plan required");
  if (!cfg.format) cfg.format = infer_format(cfg.input_path);
  if (*cfg.format == input_format::binary && !cfg.dtype)
    die("benchmark: --dtype required for binary input");

  init_gpu();
  auto loaded   = load_input(cfg.input_path, *cfg.format, cfg.dtype);
  auto plan_dsl = read_file(cfg.plan_path);
  auto blocks   = simpatico::split_plan_dsl(plan_dsl);
  int ncols     = loaded.table->num_columns();

  if (static_cast<int>(blocks.size()) != ncols) {
    std::fprintf(stderr,
                 "simpatico benchmark: plan has %zu blocks but input has %d columns\n",
                 blocks.size(),
                 ncols);
    return 1;
  }

  int threads = cfg.threads > 0 ? cfg.threads : ncols;
  std::vector<bench_row> rows;

  if (cfg.mode == mode_t::per_column) {
    for (int i = 0; i < ncols; ++i) {
      std::string name = (static_cast<std::size_t>(i) < loaded.column_names.size() &&
                          !loaded.column_names[static_cast<std::size_t>(i)].empty())
                           ? loaded.column_names[static_cast<std::size_t>(i)]
                           : ("col" + std::to_string(i));
      rows.push_back(bench_single_column(loaded.table->view().column(i),
                                         blocks[static_cast<std::size_t>(i)],
                                         name,
                                         cfg.warmup,
                                         cfg.iters));
    }
    rows.push_back(make_total_row(rows));
  } else {
    rows.push_back(
      bench_full_table(loaded.table->view(), plan_dsl, threads, cfg.warmup, cfg.iters));
  }

  if (cfg.table_out.empty()) {
    write_bench_table(std::cout, rows, cfg);
  } else {
    std::ofstream out(cfg.table_out);
    if (!out) die("cannot write table-out: " + cfg.table_out);
    write_bench_table(out, rows, cfg);
  }
  if (!cfg.csv_out.empty()) write_csv(cfg.csv_out, rows);

  bool all_ok = true;
  for (auto const& r : rows)
    if (!r.verify_ok) all_ok = false;
  return all_ok ? 0 : 5;
}

// ── EXPLORE mode ──────────────────────────────────────────────────────────────

void usage_explore()
{
  std::fprintf(
    stderr,
    "Usage: simpatico explore --input PATH [options]\n"
    "\n"
    "  --input PATH              Parquet, CSV/.tbl, or binary file (required)\n"
    "  --col N                   Column index to explore (default: all)\n"
    "  --format {parquet|csv|binary}\n"
    "  --dtype {i32|i64|...}     Element type for binary input\n"
    "  --beam-width N            BFS beam width (default: 100)\n"
    "  --max-depth N             Maximum cascade depth (default: 10)\n"
    "  --score {weighted|pareto} Ranking mode (default: weighted)\n"
    "  --weight-ratio W          Compression-ratio exponent (default: 1.0)\n"
    "  --weight-comp W           Compress-throughput exponent (default: 1.0)\n"
    "  --weight-decomp W         Decompress-throughput exponent (default: 1.0)\n"
    "  --rerank-top N            Number of finalists to time, selected by ratio (default: 8)\n"
    "  --simplicity-slots N      Top-N completed plans per cascade-depth level\n"
    "                            injected into the rerank pool, giving lighter-weight\n"
    "                            plans a fair shot against deep cascades (default: 4)\n"
    "  --sample-rows N           Approximate speedup: run the ratio search on an\n"
    "                            N-row prefix (finalists still measured on the full\n"
    "                            column). Default 0 = full column. May pick worse\n"
    "                            plans for sorted/monotonic columns.\n"
    "  --max-col-bytes N         Per-column byte budget for BFS + rerank; larger\n"
    "                            columns are trimmed to a representative prefix so no\n"
    "                            codec OOMs. Default 2147483648 (2 GiB). 0 = off.\n"
    "  --verbose                 Print BFS progress\n");
}

struct explore_cfg {
  std::string input_path;
  std::optional<input_format> format;
  std::optional<std::string> dtype;
  int col = -1;  // -1 = all
  simpatico::exploration_config ecfg;
};

int run_explore(int argc, char** argv)
{
  explore_cfg cfg;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need       = [&](char const* flag) -> std::string {
      if (i + 1 >= argc) die(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage_explore();
      return 0;
    } else if (parse_input_flag(arg, need, cfg.input_path, cfg.format, cfg.dtype)) {
      // handled: --input / --format / --dtype
    } else if (arg == "--col") {
      cfg.col = std::stoi(need("--col"));
    } else if (arg == "--beam-width") {
      cfg.ecfg.beam_width = static_cast<std::size_t>(std::stoul(need("--beam-width")));
    } else if (arg == "--max-depth") {
      cfg.ecfg.max_depth = static_cast<std::size_t>(std::stoul(need("--max-depth")));
    } else if (arg == "--score") {
      auto v = need("--score");
      if (v == "weighted")
        cfg.ecfg.rerank_mode = simpatico::score_mode::Weighted;
      else if (v == "pareto")
        cfg.ecfg.rerank_mode = simpatico::score_mode::Pareto;
      else
        die("--score: use weighted|pareto");
    } else if (arg == "--weight-ratio") {
      cfg.ecfg.rerank_weights[0] = std::stod(need("--weight-ratio"));
    } else if (arg == "--weight-comp") {
      cfg.ecfg.rerank_weights[1] = std::stod(need("--weight-comp"));
    } else if (arg == "--weight-decomp") {
      cfg.ecfg.rerank_weights[2] = std::stod(need("--weight-decomp"));
    } else if (arg == "--rerank-top") {
      cfg.ecfg.rerank_top = static_cast<std::size_t>(std::stoul(need("--rerank-top")));
    } else if (arg == "--simplicity-slots") {
      cfg.ecfg.simplicity_slots = static_cast<std::size_t>(std::stoul(need("--simplicity-slots")));
    } else if (arg == "--sample-rows") {
      cfg.ecfg.sample_rows = static_cast<std::size_t>(std::stoul(need("--sample-rows")));
    } else if (arg == "--max-col-bytes") {
      cfg.ecfg.max_explore_bytes = static_cast<std::size_t>(std::stoull(need("--max-col-bytes")));
    } else if (arg == "--verbose") {
      cfg.ecfg.verbose = true;
    } else {
      die("explore: unknown flag '" + arg + "'");
    }
  }

  if (cfg.input_path.empty()) die("explore: --input required");
  if (!cfg.format) cfg.format = infer_format(cfg.input_path);

  init_gpu();
  auto loaded = load_input(cfg.input_path, *cfg.format, cfg.dtype);
  auto stream = driver_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();
  int ncols   = loaded.table->num_columns();

  // Determine which columns to explore
  std::vector<int> col_indices;
  if (cfg.col < 0) {
    col_indices.resize(static_cast<std::size_t>(ncols));
    for (int i = 0; i < ncols; ++i)
      col_indices[static_cast<std::size_t>(i)] = i;
  } else {
    if (cfg.col >= ncols)
      die("explore: --col " + std::to_string(cfg.col) + " out of range (table has " +
          std::to_string(ncols) + " columns)");
    col_indices = {cfg.col};
  }

  // Keep the table_view alive for the entire loop — column() may return a
  // const-ref into it and the temporary would be destroyed otherwise.
  auto const tv = loaded.table->view();

  // Multi-column output: emit one block per column separated by ---
  bool first = true;
  for (int ci : col_indices) {
    auto const col_view  = tv.column(ci);
    std::string col_name = (static_cast<std::size_t>(ci) < loaded.column_names.size())
                             ? loaded.column_names[static_cast<std::size_t>(ci)]
                             : ("col" + std::to_string(ci));

    if (!first) std::printf("\n---\n");
    first = false;

    std::fprintf(stderr,
                 "# exploring column %d (%s, dtype=%s, rows=%d)\n",
                 ci,
                 col_name.c_str(),
                 dtype_name(col_view.type()).c_str(),
                 col_view.size());

    // A column may be too large to explore within device memory even though the
    // codecs fail cleanly (RMM throws rmm::out_of_memory). Such a throw here is
    // a healthy-context OOM (not a fatal CUDA error), so catch it, emit an
    // identity plan for the column, and keep going rather than aborting the run.
    simpatico::exploration_result result;
    try {
      result = simpatico::explore_column_compression(col_view, cfg.ecfg, stream, mr);
    } catch (std::exception const& e) {
      std::fprintf(stderr,
                   "# column %d (%s): exploration failed (%s); emitting identity\n",
                   ci,
                   col_name.c_str(),
                   e.what());
      std::printf("# column: %s  dtype: %s  ratio: 1.000x  depth: 0\n",
                  col_name.c_str(),
                  dtype_name(col_view.type()).c_str());
      std::printf("# note: exploration skipped (out of memory); using identity\n");
      std::printf("input -> identity\n");
      continue;
    }

    std::printf("# column: %s  dtype: %s  ratio: %.3fx  depth: %zu\n",
                col_name.c_str(),
                dtype_name(col_view.type()).c_str(),
                result.compression_ratio,
                result.cascade_depth);
    if (result.compress_throughput_gbps > 0.0)
      std::printf("# comp: %.2f GB/s  decomp: %.2f GB/s\n",
                  result.compress_throughput_gbps,
                  result.decompress_throughput_gbps);
    std::printf("%s\n", result.plan_dsl.c_str());

    if (!result.pareto_alternates_summary.empty())
      std::fprintf(stderr, "%s\n", result.pareto_alternates_summary.c_str());
  }
  return 0;
}

// ── COMPRESS mode ─────────────────────────────────────────────────────────────

void usage_compress()
{
  std::fprintf(stderr,
               "Usage: simpatico compress --input PATH --plan PATH --out FILE.hpln [options]\n"
               "\n"
               "  --input PATH              Parquet, CSV/.tbl, or binary file (required)\n"
               "  --plan PATH               Plan DSL file (required)\n"
               "  --out PATH                Output .hpln file (required)\n"
               "  --format {parquet|csv|binary}\n"
               "  --dtype {i32|i64|...}     Element type for binary input\n"
               "  --verify                  After writing, decompress and check it round-trips\n");
}

int run_compress(int argc, char** argv)
{
  std::string input_path, plan_path, out_path;
  std::optional<input_format> fmt;
  std::optional<std::string> dtype;
  bool verify = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need       = [&](char const* flag) -> std::string {
      if (i + 1 >= argc) die(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage_compress();
      return 0;
    } else if (parse_input_flag(arg, need, input_path, fmt, dtype)) {
      // handled: --input / --format / --dtype
    } else if (arg == "--plan") {
      plan_path = need("--plan");
    } else if (arg == "--out") {
      out_path = need("--out");
    } else if (arg == "--verify") {
      verify = true;
    } else {
      die("compress: unknown flag '" + arg + "'");
    }
  }

  if (input_path.empty()) die("compress: --input required");
  if (plan_path.empty()) die("compress: --plan required");
  if (out_path.empty()) die("compress: --out required");
  if (!fmt) fmt = infer_format(input_path);

  init_gpu();
  auto loaded   = load_input(input_path, *fmt, dtype);
  auto plan_dsl = read_file(plan_path);

  auto stream = driver_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();
  auto ct =
    simpatico::compress_with_plan(loaded.table->view(), plan_dsl, stream, mr, loaded.column_names);

  auto err = simpatico::write_compressed_table(ct, out_path, stream);
  if (!err.empty()) die("compress: write failed: " + err);

  std::size_t input_b = table_input_bytes(loaded.table->view(), stream);
  std::size_t comp_b  = 0;
  for (auto const& col : ct.columns)
    if (col.compound) comp_b += compound_compressed_bytes(*col.compound, stream);

  std::printf("compressed %zu -> %zu bytes (%.3fx)  -> %s\n",
              input_b,
              comp_b,
              compression_ratio(input_b, comp_b),
              out_path.c_str());

  if (verify) {
    bool ok = verify_roundtrip(loaded.table->view(), ct);
    std::printf("verify: %s\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;
  }
  return 0;
}

// ── DECOMPRESS mode ───────────────────────────────────────────────────────────

void usage_decompress()
{
  std::fprintf(stderr,
               "Usage: simpatico decompress --input FILE.hpln [--out PATH.parquet] [options]\n"
               "\n"
               "  --input PATH   .hpln compressed file (required)\n"
               "  --out PATH     Output Parquet file (optional; prints stats if omitted)\n"
               "  --verify PATH  Source file to check the decompressed output against\n"
               "  --format {parquet|csv|binary}   Format of the --verify source\n"
               "  --dtype {i32|i64|...}           Element type for a binary --verify source\n");
}

int run_decompress(int argc, char** argv)
{
  std::string in_path, out_path, verify_path;
  std::optional<input_format> fmt;
  std::optional<std::string> dtype;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need       = [&](char const* flag) -> std::string {
      if (i + 1 >= argc) die(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage_decompress();
      return 0;
    } else if (arg == "--input") {
      in_path = need("--input");
    } else if (arg == "--out") {
      out_path = need("--out");
    } else if (arg == "--verify") {
      verify_path = need("--verify");
    } else if (arg == "--format") {
      fmt = parse_input_format(need("--format"));
    } else if (arg == "--dtype") {
      dtype = need("--dtype");
    } else {
      die("decompress: unknown flag '" + arg + "'");
    }
  }

  if (in_path.empty()) die("decompress: --input required");

  init_gpu();
  auto stream = driver_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();

  std::string err;
  auto ct = simpatico::read_compressed_table(in_path, stream, mr, &err);
  if (!err.empty()) die("decompress: read failed: " + err);

  auto t0  = std::chrono::steady_clock::now();
  auto out = simpatico::decompress(ct, stream, mr);
  cuda_sync();
  auto t1 = std::chrono::steady_clock::now();

  if (!out) die("decompress: decompress returned null");

  double ms             = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::size_t out_bytes = table_input_bytes(out->view(), stream);
  std::printf("decompressed %d cols x %lld rows  %zu bytes  %.1f ms  %.2f GB/s\n",
              out->num_columns(),
              static_cast<long long>(out->num_rows()),
              out_bytes,
              ms,
              gbps(out_bytes, ms));

  if (!out_path.empty()) {
    // Build column metadata from the compressed_table names
    std::vector<std::string> names;
    for (auto const& col : ct.columns)
      names.push_back(col.name.value_or("col" + std::to_string(names.size())));

    cudf::io::table_input_metadata meta(out->view());
    for (std::size_t k = 0; k < names.size() && k < meta.column_metadata.size(); ++k)
      meta.column_metadata[k].set_name(names[k]);

    auto opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{out_path}, out->view())
        .metadata(meta)
        .build();
    cudf::io::write_parquet(opts);
    std::printf("wrote %s\n", out_path.c_str());
  }

  if (!verify_path.empty()) {
    if (!fmt) fmt = infer_format(verify_path);
    auto source = load_input(verify_path, *fmt, dtype);
    if (source.table->num_columns() != out->num_columns())
      die("verify: source has " + std::to_string(source.table->num_columns()) +
          " columns but .hpln decompressed to " + std::to_string(out->num_columns()));
    bool ok = tables_equal(source.table->view(), out->view());
    std::printf("verify: %s\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;
  }
  return 0;
}

// ── PLAN mode ───────────────────────────────────────────────────────────────

void usage_plan()
{
  std::fprintf(stderr,
               "Usage: simpatico plan --input FILE.hpln\n"
               "\n"
               "  --input PATH   .hpln compressed file (required)\n"
               "\n"
               "Prints each column's compression plan as DSL, rendered from the\n"
               "stored plan tree (the file holds no DSL text of its own).\n");
}

int run_plan(int argc, char** argv)
{
  std::string in_path;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need       = [&](char const* flag) -> std::string {
      if (i + 1 >= argc) die(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage_plan();
      return 0;
    } else if (arg == "--input") {
      in_path = need("--input");
    } else {
      die("plan: unknown flag '" + arg + "'");
    }
  }
  if (in_path.empty()) die("plan: --input required");

  init_gpu();
  auto stream = driver_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();

  std::string err;
  auto ct = simpatico::read_compressed_table(in_path, stream, mr, &err);
  if (!err.empty()) die("plan: read failed: " + err);

  for (std::size_t ci = 0; ci < ct.columns.size(); ++ci) {
    if (ci > 0) std::printf("---\n");
    auto const& col = ct.columns[ci];
    std::printf("# column %zu: %s\n", ci, col.name.value_or("").c_str());
    if (col.compound) std::printf("%s", simpatico::render_plan_tree(*col.compound).c_str());
  }
  return 0;
}

}  // namespace

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
  try {
    if (argc < 2) {
      usage_top();
      return 1;
    }

    std::string mode = argv[1];
    if (mode == "--help" || mode == "-h") {
      usage_top();
      return 0;
    }

    // Pass (argc-1, argv+1): argv[0] stays "simpatico", mode word is gone.
    if (mode == "benchmark") return run_benchmark(argc - 1, argv + 1);
    if (mode == "explore") return run_explore(argc - 1, argv + 1);
    if (mode == "compress") return run_compress(argc - 1, argv + 1);
    if (mode == "decompress") return run_decompress(argc - 1, argv + 1);
    if (mode == "plan") return run_plan(argc - 1, argv + 1);

    std::fprintf(stderr, "simpatico: unknown mode '%s'\n", mode.c_str());
    usage_top();
    return 1;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "simpatico: fatal: %s\n", e.what());
    return 1;
  }
}
