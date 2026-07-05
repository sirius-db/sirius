// SPDX-License-Identifier: Apache-2.0
//
// Benchmark helpers for the simpatico CLI `benchmark` mode.
// Extracted from bench/compress_with_plan_benchmark.cpp.
#pragma once

#include "api/simpatico_codegen.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/util/stream_pool.hpp"
#include "driver_common.hpp"

#include <cudf/table/table.hpp>

namespace {  // TU-local — included only from simpatico_main.cpp

// ── Compressed-size accounting ────────────────────────────────────────────────

inline std::size_t rep_bytes(simpatico::compressed_representation const* rep,
                             rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  return rep ? rep->compressed_size_bytes(stream) : 0;
}

inline std::size_t compound_compressed_bytes(
  simpatico::plan_compound const& compound,
  rmm::cuda_stream_view stream = cudf::get_default_stream())
{
  std::size_t total = 0;
  for (auto const& node : compound.tree.nodes) {
    total += rep_bytes(node.rep.get(), stream);
    for (auto const& [path, rep] : node.channels)
      total += rep_bytes(rep.get(), stream);
  }
  return total;
}

inline std::size_t compressed_table_bytes(simpatico::compressed_table const& ct,
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

// ── Per-column benchmark ─────────────────────────────────────────────────────

inline bench_row bench_single_column(cudf::column_view col,
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
  auto compress_once = [&]() {
    last_ct = simpatico::compress_with_plan(single.view(),
                                            plan_block,
                                            cudf::get_default_stream(),
                                            rmm::mr::get_current_device_resource_ref());
  };
  for (int w = 0; w < warmup; ++w)
    compress_once();
  cuda_sync();

  std::vector<double> compress_samples;
  compress_samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    last_ct = simpatico::compressed_table{};
    cuda_sync();  // drain previous iter
    auto t0 = std::chrono::steady_clock::now();
    compress_once();
    cuda_sync();  // wait for GPU completion
    auto t1 = std::chrono::steady_clock::now();
    compress_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  row.compress_ms      = compute_stats(compress_samples);
  row.compressed_bytes = compressed_table_bytes(last_ct);

  auto decompress_once = [&]() {
    auto out = simpatico::decompress(
      last_ct, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
    (void)out;
  };
  for (int w = 0; w < warmup; ++w)
    decompress_once();
  cuda_sync();

  std::vector<double> decompress_samples;
  decompress_samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    cuda_sync();  // drain previous iter
    auto t0 = std::chrono::steady_clock::now();
    decompress_once();
    cuda_sync();  // wait for GPU completion
    auto t1 = std::chrono::steady_clock::now();
    decompress_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  row.decompress_ms = compute_stats(decompress_samples);
  row.verify_ok     = verify_roundtrip(single.view(), last_ct);
  return row;
}

// ── Full-table benchmark ──────────────────────────────────────────────────────

inline bench_row bench_full_table(
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

  simpatico::compressed_table last_ct;
  for (int w = 0; w < warmup; ++w)
    last_ct =
      simpatico::compress_with_plan(tv, plan_dsl, pool, rmm::mr::get_current_device_resource_ref());
  cuda_sync();

  std::vector<double> compress_samples;
  compress_samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    last_ct = simpatico::compressed_table{};
    cuda_sync();
    auto t0 = std::chrono::steady_clock::now();
    last_ct =
      simpatico::compress_with_plan(tv, plan_dsl, pool, rmm::mr::get_current_device_resource_ref());
    auto t1 = std::chrono::steady_clock::now();
    compress_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  row.compress_ms      = compute_stats(compress_samples);
  row.compressed_bytes = compressed_table_bytes(last_ct);

  auto decompress_once = [&]() {
    auto out = simpatico::decompress(last_ct, pool, rmm::mr::get_current_device_resource_ref());
    cuda_sync();
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
  row.verify_ok     = verify_roundtrip(tv, last_ct);
  return row;
}

// ── Aggregation ───────────────────────────────────────────────────────────────

inline bench_row make_total_row(std::vector<bench_row> const& rows)
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

inline void write_row_csv(std::ostream& os, bench_row const& r)
{
  os << r.column << ',' << r.dtype << ',' << r.rows << ',' << r.input_bytes << ','
     << r.compressed_bytes << ',' << r.ratio_val() << ',' << r.compress_ms.min << ','
     << r.compress_ms.median << ',' << r.compress_ms.mean << ',' << r.compress_gbps_median() << ','
     << r.decompress_ms.min << ',' << r.decompress_ms.median << ',' << r.decompress_ms.mean << ','
     << r.decompress_gbps_median() << ',' << (r.verify_ok ? 1 : 0) << '\n';
}

inline void write_csv(std::string const& path, std::vector<bench_row> const& rows)
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

inline void write_bench_table(std::ostream& os,
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

}  // anonymous namespace
