// SPDX-License-Identifier: Apache-2.0
//
// Shared front-end utilities for the simpatico CLI driver.
// Included only by cli/simpatico_main.cpp (via cli/benchmark.hpp); all
// functions are inline so no ODR issues arise.
#pragma once

#include "api/simpatico_codegen.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Error helpers ─────────────────────────────────────────────────────────────

[[noreturn]] inline void die(std::string const& msg, int code = 1)
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

inline input_format infer_format(std::string const& path)
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

inline cudf::data_type parse_dtype(std::string const& s)
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

inline std::string dtype_name(cudf::data_type const& t)
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

// ── File I/O helpers ──────────────────────────────────────────────────────────

inline std::string read_file(std::string const& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

inline std::vector<uint8_t> read_binary_file(std::string const& path)
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
inline loaded_table load_parquet(std::string const& path)
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
inline loaded_table load_binary(std::string const& path, cudf::data_type dtype)
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
inline loaded_table load_csv(std::string const& path, char delimiter = ',', bool has_header = true)
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
inline loaded_table load_input(std::string const& path,
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

inline void cuda_sync()
{
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(err));
}

inline std::size_t column_input_bytes(cudf::column_view col,
                                      rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  if (col.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view scv(col);
    return static_cast<std::size_t>(col.size() + 1) * sizeof(int32_t) +
           static_cast<std::size_t>(scv.chars_size(stream));
  }
  return static_cast<std::size_t>(col.size()) * static_cast<std::size_t>(cudf::size_of(col.type()));
}

inline std::size_t table_input_bytes(cudf::table_view tv,
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

inline timing_stats compute_stats(std::vector<double> const& ms)
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

inline double gbps(std::size_t bytes, double ms)
{
  if (ms <= 0.0) return 0.0;
  return (static_cast<double>(bytes) / 1.0e9) / (ms / 1000.0);
}

inline double compression_ratio(std::size_t input_bytes, std::size_t compressed_bytes)
{
  if (compressed_bytes == 0) return 0.0;
  return static_cast<double>(input_bytes) / static_cast<double>(compressed_bytes);
}

// ── Column equality (fixed-width + strings) ───────────────────────────────────

inline bool columns_equal(cudf::column_view a, cudf::column_view b)
{
  if (a.type() != b.type() || a.size() != b.size()) return false;

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

inline bool tables_equal(cudf::table_view a, cudf::table_view b)
{
  if (a.num_columns() != b.num_columns()) return false;
  for (int i = 0; i < a.num_columns(); ++i)
    if (!columns_equal(a.column(i), b.column(i))) return false;
  return true;
}

/// Decompress @p ct and compare every column byte-exactly to @p source.
inline bool verify_roundtrip(cudf::table_view source, simpatico::compressed_table const& ct)
{
  auto out = simpatico::decompress(
    ct, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
  if (!out || out->num_columns() != source.num_columns()) return false;
  return tables_equal(source, out->view());
}
