// SPDX-License-Identifier: Apache-2.0
//
// simpatico — multi-mode CLI for simpatico_codegen.
//
// Modes:
//   benchmark  Timed compress+decompress over a Parquet/binary/CSV input
//   explore    BFS cascade search for a single column
//   compress   Compress input to a .hpln file
//   decompress Decompress a .hpln file to Parquet
//   verify     Compress (or read .hpln) and check byte-exact roundtrip

#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "benchmark.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/plan/plan_interpreter.hpp"  // plan_compound, render_plan_tree
#include "driver_common.hpp"
#include "explore/compression_explorer.hpp"

#include <cudf/io/parquet.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ── Top-level usage ───────────────────────────────────────────────────────────

static void usage_top()
{
  std::fprintf(stderr,
               "Usage: simpatico <mode> [options]\n"
               "\n"
               "Modes:\n"
               "  benchmark   Timed compress+decompress (Parquet/binary/CSV input, plan file)\n"
               "  explore     BFS cascade search for the best plan for a single column\n"
               "  compress    Compress input to a .hpln file\n"
               "  decompress  Decompress a .hpln file to Parquet\n"
               "  plan        Print each column's compression plan (DSL) from a .hpln file\n"
               "  verify      Roundtrip equality check\n"
               "\n"
               "Run 'simpatico <mode> --help' for per-mode options.\n");
}

// ── Shared GPU/RMM init ───────────────────────────────────────────────────────

static pool_mr_guard g_mr;

static void init_gpu()
{
  if (cudaSetDevice(0) != cudaSuccess) die("cudaSetDevice(0) failed");
  g_mr.install();
  codegen::jit::ensure_cuda_context();
}

/// Explicit, non-default stream for all driver work, created once and never
/// torn down before process exit.
///
/// nvcomp-backed operators (ans/bitcomp/cascaded/snappy/deflate/lz4) cache
/// their nvcomp::*Manager thread-locally, keyed by the stream they were built
/// with (see e.g. ans_compressor.cu) — the Manager must not outlive that
/// stream. Those caches are only torn down at thread exit, so a
/// function-scoped stream (destroyed when e.g. run_compress() returns) can
/// go dangling before the cache that still references it. A function-local
/// static is destroyed via the regular __exit_funcs/atexit path, which glibc
/// runs strictly after __call_tls_dtors (thread-local destructors) — so it
/// outlives every thread_local manager cache and is safe to hand out here.
static rmm::cuda_stream_view driver_stream()
{
  static rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  return stream.view();
}

// ── BENCHMARK mode ────────────────────────────────────────────────────────────

static void usage_benchmark()
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

static int run_benchmark(int argc, char** argv)
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
    } else if (arg == "--input") {
      cfg.input_path = need("--input");
    } else if (arg == "--plan") {
      cfg.plan_path = need("--plan");
    } else if (arg == "--format") {
      auto v = need("--format");
      if (v == "parquet")
        cfg.format = input_format::parquet;
      else if (v == "csv")
        cfg.format = input_format::csv;
      else if (v == "binary")
        cfg.format = input_format::binary;
      else
        die("--format: use parquet|csv|binary");
    } else if (arg == "--dtype") {
      cfg.dtype = need("--dtype");
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

static void usage_explore()
{
  std::fprintf(stderr,
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
               "  --rerank-top N            Number of finalists to time (default: 8)\n"
               "  --sample-rows N           Approximate speedup: run the ratio search on an\n"
               "                            N-row prefix (finalists still measured on the full\n"
               "                            column). Default 0 = full column. May pick worse\n"
               "                            plans for sorted/monotonic columns.\n"
               "  --verbose                 Print BFS progress\n");
}

struct explore_cfg {
  std::string input_path;
  std::optional<input_format> format;
  std::optional<std::string> dtype;
  int col = -1;  // -1 = all
  simpatico::exploration_config ecfg;
};

static int run_explore(int argc, char** argv)
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
    } else if (arg == "--input") {
      cfg.input_path = need("--input");
    } else if (arg == "--col") {
      cfg.col = std::stoi(need("--col"));
    } else if (arg == "--format") {
      auto v = need("--format");
      if (v == "parquet")
        cfg.format = input_format::parquet;
      else if (v == "csv")
        cfg.format = input_format::csv;
      else if (v == "binary")
        cfg.format = input_format::binary;
      else
        die("--format: use parquet|csv|binary");
    } else if (arg == "--dtype") {
      cfg.dtype = need("--dtype");
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
    } else if (arg == "--sample-rows") {
      cfg.ecfg.sample_rows = static_cast<std::size_t>(std::stoul(need("--sample-rows")));
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

    auto result = simpatico::explore_column_compression(col_view, cfg.ecfg, stream, mr);

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

static void usage_compress()
{
  std::fprintf(stderr,
               "Usage: simpatico compress --input PATH --plan PATH --out FILE.hpln [options]\n"
               "\n"
               "  --input PATH              Parquet, CSV/.tbl, or binary file (required)\n"
               "  --plan PATH               Plan DSL file (required)\n"
               "  --out PATH                Output .hpln file (required)\n"
               "  --format {parquet|csv|binary}\n"
               "  --dtype {i32|i64|...}     Element type for binary input\n");
}

static int run_compress(int argc, char** argv)
{
  std::string input_path, plan_path, out_path;
  std::optional<input_format> fmt;
  std::optional<std::string> dtype;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need       = [&](char const* flag) -> std::string {
      if (i + 1 >= argc) die(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage_compress();
      return 0;
    } else if (arg == "--input") {
      input_path = need("--input");
    } else if (arg == "--plan") {
      plan_path = need("--plan");
    } else if (arg == "--out") {
      out_path = need("--out");
    } else if (arg == "--format") {
      auto v = need("--format");
      if (v == "parquet")
        fmt = input_format::parquet;
      else if (v == "csv")
        fmt = input_format::csv;
      else if (v == "binary")
        fmt = input_format::binary;
      else
        die("--format: use parquet|csv|binary");
    } else if (arg == "--dtype") {
      dtype = need("--dtype");
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
  return 0;
}

// ── DECOMPRESS mode ───────────────────────────────────────────────────────────

static void usage_decompress()
{
  std::fprintf(stderr,
               "Usage: simpatico decompress --input FILE.hpln [--out PATH.parquet]\n"
               "\n"
               "  --input PATH   .hpln compressed file (required)\n"
               "  --out PATH     Output Parquet file (optional; prints stats if omitted)\n");
}

static int run_decompress(int argc, char** argv)
{
  std::string in_path, out_path;

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
  return 0;
}

// ── PLAN mode ───────────────────────────────────────────────────────────────

static void usage_plan()
{
  std::fprintf(stderr,
               "Usage: simpatico plan --input FILE.hpln\n"
               "\n"
               "  --input PATH   .hpln compressed file (required)\n"
               "\n"
               "Prints each column's compression plan as DSL, rendered from the\n"
               "stored plan tree (the file holds no DSL text of its own).\n");
}

static int run_plan(int argc, char** argv)
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
    if (col.compound) std::printf("%s", simpatico::render_plan_tree(col.compound->tree).c_str());
  }
  return 0;
}

// ── VERIFY mode ───────────────────────────────────────────────────────────────

static void usage_verify()
{
  std::fprintf(stderr,
               "Usage: simpatico verify --input SRC (--plan PATH | --hpln FILE) [options]\n"
               "\n"
               "  --input PATH              Source data (Parquet/CSV/.tbl/binary) (required)\n"
               "  --plan PATH               Compress in-memory with this plan, then check\n"
               "  --hpln PATH               Read this .hpln file, then check\n"
               "  --format {parquet|csv|binary}\n"
               "  --dtype {i32|i64|...}     Element type for binary input\n");
}

static int run_verify(int argc, char** argv)
{
  std::string input_path, plan_path, hpln_path;
  std::optional<input_format> fmt;
  std::optional<std::string> dtype;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need       = [&](char const* flag) -> std::string {
      if (i + 1 >= argc) die(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage_verify();
      return 0;
    } else if (arg == "--input") {
      input_path = need("--input");
    } else if (arg == "--plan") {
      plan_path = need("--plan");
    } else if (arg == "--hpln") {
      hpln_path = need("--hpln");
    } else if (arg == "--format") {
      auto v = need("--format");
      if (v == "parquet")
        fmt = input_format::parquet;
      else if (v == "csv")
        fmt = input_format::csv;
      else if (v == "binary")
        fmt = input_format::binary;
      else
        die("--format: use parquet|csv|binary");
    } else if (arg == "--dtype") {
      dtype = need("--dtype");
    } else {
      die("verify: unknown flag '" + arg + "'");
    }
  }

  if (input_path.empty()) die("verify: --input required");
  if (plan_path.empty() && hpln_path.empty()) die("verify: one of --plan or --hpln required");
  if (!plan_path.empty() && !hpln_path.empty())
    die("verify: --plan and --hpln are mutually exclusive");
  if (!fmt) fmt = infer_format(input_path);

  init_gpu();
  auto loaded = load_input(input_path, *fmt, dtype);
  auto stream = driver_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();

  simpatico::compressed_table ct;
  if (!plan_path.empty()) {
    auto plan_dsl = read_file(plan_path);
    ct            = simpatico::compress_with_plan(
      loaded.table->view(), plan_dsl, stream, mr, loaded.column_names);
  } else {
    std::string err;
    ct = simpatico::read_compressed_table(hpln_path, stream, mr, &err);
    if (!err.empty()) die("verify: read_compressed_table: " + err);
    if (ct.num_columns() != static_cast<std::size_t>(loaded.table->num_columns()))
      die("verify: .hpln has " + std::to_string(ct.num_columns()) + " columns but input has " +
          std::to_string(loaded.table->num_columns()));
  }

  bool ok = verify_roundtrip(loaded.table->view(), ct);
  std::printf("verify: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

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
    if (mode == "verify") return run_verify(argc - 1, argv + 1);

    std::fprintf(stderr, "simpatico: unknown mode '%s'\n", mode.c_str());
    usage_top();
    return 1;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "simpatico: fatal: %s\n", e.what());
    return 1;
  }
}
