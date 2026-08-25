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

// columnar_parquet_poc — per-column IO/decode pipelining.
//
// prefetch_hybrid_scan_benchmark issues ONE vectored device read covering every
// column chunk of a file, then decodes all four columns in a single
// materialize_all_columns call.  Decode therefore cannot start until the last
// byte of the last column has landed.
//
// This POC splits that per column: N vectored reads (one per column), and each
// column's decode is deployed to the thread pool the moment that column's
// buffers are on the device — on its own stream.  A latch joins the N
// single-column tables back into one table.  The slowest column bounds the
// file, not the sum of all of them.
//
// Usage:
//   ./columnar_parquet_poc --dir DIR [--n_files N] --mode <b|c> [--nthreads N]
//
// Modes:
//   b  baseline  — enqueue all read_parquet calls directly to the thread pool
//                  (identical to prefetch_hybrid_scan_benchmark's baseline)
//   c  columnar  — columnar_parquet_parser per file, as described above

#include "exec/scoped_dispatcher.hpp"
#include "exec/semi_future.hpp"
#include "exec/thread_pool.hpp"
#include "exec/try.hpp"
#include "io/io_context.hpp"
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"

#include <cudf/column/column.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/stream_pool.hpp>
#include <glob.h>
#include <log/logging.hpp>
#include <log/spdlog_owning_sink.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <latch>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using clock_type         = std::chrono::high_resolution_clock;
using stream_pool_t      = cucascade::memory::exclusive_stream_pool;
using acquire_pol        = stream_pool_t::stream_acquire_policy;
using byte_range         = cudf::io::text::byte_range_info;
using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;
namespace pq             = cudf::io::parquet;

double ms_since(clock_type::time_point t0) noexcept
{
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

double us_since(clock_type::time_point t0) noexcept
{
  return std::chrono::duration<double, std::micro>(clock_type::now() - t0).count();
}

std::vector<std::string> const COLUMNS = {
  "l_orderkey",
  "l_extendedprice",
  "l_discount",
  "l_shipdate",
};

// ---- single-column metadata pruning ---------------------------------------
//
// Building a reader from a FileMetaData copies the struct VERBATIM and re-derives
// nothing (cudf hybrid_scan_helpers.hpp:38 — `static_cast<FileMetaData&>(*this) = other`).
// Neither InitSchema() nor WalkSchema() runs, so every derived field is a
// load-bearing input rather than an output: parent_idx, children_idx,
// max_*_level and ColumnChunk::schema_idx must all be rewritten by hand.
//
// The point of pruning is that the copy is O(row_groups x columns) ColumnChunks —
// 94 x 16 = 1504 for a TPC-H sf200 lineitem part.  A single-column view carries
// 94, so the per-reader copy shrinks by the column count.
//
// @p full MUST come from `hybrid_scan_reader{footer_bytes, opts}.parquet_metadata()`,
// which is the only path that populates the derived fields.
// Flat schemas only; throws rather than silently producing garbage on nested input.
[[nodiscard]] pq::FileMetaData prune_to_column(pq::FileMetaData const& full,
                                               std::string const& column_name)
{
  CUDF_EXPECTS(not full.schema.empty(), "prune_to_column: empty schema", std::invalid_argument);

  auto const& root      = full.schema.front();
  auto const num_fields = full.schema.size() - 1;

  CUDF_EXPECTS(std::cmp_equal(root.num_children, num_fields),
               "prune_to_column: schema is not flat (nested schemas unsupported)",
               std::invalid_argument);
  CUDF_EXPECTS(root.children_idx.size() == num_fields,
               "prune_to_column: root.children_idx unpopulated — the FileMetaData must come from "
               "hybrid_scan_reader::parquet_metadata(), not a hand-built struct",
               std::invalid_argument);
  for (std::size_t i = 1; i < full.schema.size(); ++i) {
    auto const& e = full.schema[i];
    CUDF_EXPECTS(e.num_children == 0 and e.parent_idx == 0 and e.max_repetition_level == 0,
                 "prune_to_column: schema is not flat (element '" + e.name + "')",
                 std::invalid_argument);
  }
  // select_columns() prepends the pandas index columns to the projection when
  // use_pandas_metadata is on (the default); a name missing from the pruned
  // schema would then throw deep inside the reader.
  CUDF_EXPECTS(std::none_of(full.key_value_metadata.begin(),
                            full.key_value_metadata.end(),
                            [](auto const& e) { return e.key == "pandas"; }),
               "prune_to_column: pandas metadata present; pruning could drop an index column",
               std::invalid_argument);

  auto const leaf_it = std::find_if(full.schema.begin() + 1, full.schema.end(), [&](auto const& e) {
    return e.name == column_name;
  });
  CUDF_EXPECTS(leaf_it != full.schema.end(),
               "prune_to_column: column '" + column_name + "' not found",
               std::invalid_argument);
  auto const leaf_idx = static_cast<cudf::size_type>(std::distance(full.schema.begin(), leaf_it));

  pq::FileMetaData out;
  out.version    = full.version;
  out.num_rows   = full.num_rows;
  out.created_by = full.created_by;
  // ARROW:schema describes all N columns; apply_arrow_schema() would fail its root
  // num_children check and log an error on every construction.
  std::copy_if(full.key_value_metadata.begin(),
               full.key_value_metadata.end(),
               std::back_inserter(out.key_value_metadata),
               [](auto const& e) { return e.key != "ARROW:schema"; });

  // Schema: [root, leaf].  num_children/children_idx/parent_idx are all load-bearing.
  out.schema.reserve(2);
  out.schema.push_back(root);
  out.schema.front().num_children = 1;
  out.schema.front().children_idx = {1};
  out.schema.front().parent_idx   = 0;
  out.schema.push_back(*leaf_it);  // verbatim: type/logical_type/max_*_level must survive
  out.schema.back().parent_idx = 0;
  out.schema.back().children_idx.clear();
  out.schema.back().num_children = 0;

  out.row_groups.reserve(full.row_groups.size());
  for (auto const& rg : full.row_groups) {
    auto const chunk_it = std::find_if(rg.columns.begin(), rg.columns.end(), [&](auto const& c) {
      return c.schema_idx == leaf_idx;
    });
    CUDF_EXPECTS(chunk_it != rg.columns.end(),
                 "prune_to_column: no chunk for '" + column_name + "' in a row group",
                 std::invalid_argument);

    pq::RowGroup out_rg;
    out_rg.num_rows              = rg.num_rows;
    out_rg.total_byte_size       = rg.total_byte_size;
    out_rg.total_compressed_size = rg.total_compressed_size;
    out_rg.ordinal               = rg.ordinal;
    // Byte-range row-group filtering falls back to columns.front() when file_offset is
    // unset; after pruning columns.front() is this column, not column 0 — so resolve the
    // full-metadata answer here and keep that filter bit-identical.
    out_rg.file_offset = [&]() -> std::optional<int64_t> {
      if (rg.file_offset.has_value()) { return rg.file_offset; }
      if (rg.columns.empty()) { return std::nullopt; }
      auto const& m = rg.columns.front().meta_data;
      return m.dictionary_page_offset != 0 ? std::min(m.dictionary_page_offset, m.data_page_offset)
                                           : m.data_page_offset;
    }();

    out_rg.columns.push_back(*chunk_it);
    // Load-bearing: chunks are resolved by schema_idx (find_colchunk_iter_offset), and
    // nothing re-derives it on the FileMetaData ctor path.
    out_rg.columns.front().schema_idx = 1;
    out.row_groups.push_back(std::move(out_rg));
  }
  return out;
}

// ---- file discovery --------------------------------------------------------

struct file_info {
  std::string path;
  std::unique_ptr<sirius::io::sirius_datasource> ds;
  // Column-chunk byte ranges grouped BY COLUMN — col_ranges[i] holds every
  // chunk of column i across all row groups.  This grouping is the whole point:
  // it is what lets each column be read and decoded independently.
  std::vector<std::vector<byte_range>> col_ranges;
  std::size_t range_bytes{0};
  // Footer metadata and row groups cached during discovery so the decode tasks
  // can rebuild a hybrid_scan_reader without a second footer fetch + Thrift
  // parse (mirrors src/op/scan/parquet_gpu_ingestible.cpp).
  cudf::io::parquet::FileMetaData metadata;
  // Per-column single-column views of `metadata`, built once here so the decode
  // side copies 1/n_columns of the chunk metadata per reader construction.
  std::vector<cudf::io::parquet::FileMetaData> col_metadata;
  std::vector<cudf::size_type> row_groups;
};

// Accumulated per-phase cost inside the timed region, so the reader-construction
// overhead can be separated from the decode it is attached to.
struct phase_timers {
  std::atomic<double> ctor_ms{0};
  std::atomic<double> materialize_ms{0};
  std::atomic<std::size_t> n_ctors{0};

  void add_ctor(double ms) noexcept
  {
    ctor_ms.fetch_add(ms, std::memory_order_relaxed);
    n_ctors.fetch_add(1, std::memory_order_relaxed);
  }
  void add_materialize(double ms) noexcept
  {
    materialize_ms.fetch_add(ms, std::memory_order_relaxed);
  }
};

std::vector<std::string> glob_parquet_files(std::string const& dir, std::size_t limit)
{
  static constexpr std::array<char const*, 2> kPatterns = {"part.*.parquet", "lineitem.*.parquet"};
  std::vector<std::string> paths;
  for (auto const* pat : kPatterns) {
    std::string const pattern = dir + "/" + pat;
    glob_t g{};
    if (::glob(pattern.c_str(), GLOB_TILDE, nullptr, &g) == 0) {
      for (std::size_t i = 0; i < g.gl_pathc; ++i) {
        paths.emplace_back(g.gl_pathv[i]);
      }
    }
    ::globfree(&g);
    if (!paths.empty()) break;
  }
  std::sort(paths.begin(), paths.end());
  if (limit > 0 && paths.size() > limit) { paths.resize(limit); }
  return paths;
}

// ---- baseline --------------------------------------------------------------

// Parse a single parquet file using the sirius_datasource already backed by
// the uring io_ctx.  source_info takes a raw (non-owning) datasource* so no
// shim or ownership transfer is needed.
std::unique_ptr<cudf::table> parse_parquet(sirius::io::sirius_datasource& ds,
                                           rmm::cuda_stream_view stream)
{
  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{&ds})
                .column_names(COLUMNS)
                .build();
  auto result = cudf::io::read_parquet(opts, stream);
  cudaStreamSynchronize(stream.value());
  return std::move(result.tbl);
}

void run_baseline(std::vector<file_info>& files,
                  sirius::exec::scoped_dispatcher& disp,
                  stream_pool_t& streams,
                  std::size_t total_bytes)
{
  std::atomic<std::size_t> total_rows{0};
  std::latch done(static_cast<std::ptrdiff_t>(files.size()));

  auto t0 = clock_type::now();
  for (auto& fi : files) {
    // `files` outlives every task — done.wait() below joins them all.
    disp.enqueue([&fi, &streams, &total_rows, &done] {
      auto stream = streams.acquire_stream(acquire_pol::GROW);
      auto tbl    = parse_parquet(*fi.ds, stream);
      total_rows.fetch_add(static_cast<std::size_t>(tbl->num_rows()), std::memory_order_relaxed);
      done.count_down();
    });
  }
  done.wait();
  double const elapsed = ms_since(t0);

  std::cout << "\n=== baseline results ===\n"
            << "  wall      : " << std::fixed << std::setprecision(1) << elapsed << " ms\n"
            << "  throughput: " << std::setprecision(2)
            << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) / (elapsed / 1000.0)
            << " GiB/s\n"
            << "  rows      : " << total_rows.load() << "\n";
}

// ---- columnar parser -------------------------------------------------------

/**
 * @brief Read and decode one parquet file one column at a time, in parallel.
 *
 * For each column i: allocate a device buffer per chunk, issue ONE vectored
 * device read for that column's chunks, and hand the completion to @p disp,
 * which materializes column i on its own stream.  A latch joins the N
 * single-column tables; their columns are released and reassembled into one
 * table in schema order.
 *
 * @param ds          datasource for the file (its ioctx serves the reads)
 * @param col_ranges  chunk byte ranges grouped by column; col_ranges[i] is column i
 * @param col_opts    per-column reader options; col_opts[i] selects exactly column i
 * @param metadata    parsed footer, so no reader re-parses Thrift
 * @param row_groups  row groups to materialize
 * @param streams     one stream is borrowed per column, for both its IO and its decode
 * @param mr          device resource for the output columns
 * @param disp        pool the per-column decodes run on
 *
 * @note Blocks the calling thread on the latch, so it must NOT be called from a
 *       @p disp worker — a blocked worker holds an inflight slot and the decode
 *       it is waiting for may never get one.  Call it from the main thread.
 */
std::unique_ptr<cudf::table> columnar_parquet_parser(
  sirius::io::sirius_datasource& ds,
  std::vector<std::vector<byte_range>> const& col_ranges,
  std::vector<cudf::io::parquet_reader_options> const& col_opts,
  std::vector<cudf::io::parquet::FileMetaData const*> const& decode_metadata,
  std::vector<cudf::size_type> const& row_groups,
  stream_pool_t& streams,
  rmm::device_async_resource_ref mr,
  sirius::exec::scoped_dispatcher& disp,
  phase_timers& timers)
{
  auto const n_cols = col_ranges.size();

  auto io_ctx           = ds.io_ctx();
  auto const& io_object = ds.get_io_object();

  // Written by the decode tasks, one distinct index each — no synchronisation
  // needed, and indexing (rather than push_back) is what keeps the columns in
  // schema order regardless of which column finishes first.
  std::vector<std::unique_ptr<cudf::column>> cols(n_cols);
  std::atomic<std::size_t> failures{0};
  std::latch done(static_cast<std::ptrdiff_t>(n_cols));

  for (std::size_t i = 0; i < n_cols; ++i) {
    auto const& ranges = col_ranges[i];
    if (ranges.empty()) {
      done.count_down();
      continue;
    }

    // One stream per column, borrowed here and carried all the way through: the
    // buffer allocations, the H2D copies and the decode all run on it, so the
    // decode is already ordered after the copies with no synchronisation in
    // between.  Two columns are two independent streams and can genuinely
    // overlap on the device.
    auto stream = streams.acquire_stream(acquire_pol::GROW);

    std::vector<rmm::device_buffer> buffers;
    buffers.reserve(ranges.size());
    for (auto const& r : ranges) {
      buffers.emplace_back(static_cast<std::size_t>(r.size()), stream.get(), mr);
    }

    std::vector<sirius::io::slice> reads;
    reads.reserve(ranges.size());
    for (std::size_t c = 0; c < ranges.size(); ++c) {
      reads.emplace_back(static_cast<std::size_t>(ranges[c].offset()),
                         static_cast<std::size_t>(ranges[c].size()),
                         static_cast<std::uint8_t*>(buffers[c].data()));
    }

    // The future resolves once this column's copies are enqueued on `stream`;
    // the decode is then chained onto the pool.  The main thread does not wait,
    // so column i+1's IO is issued while column i is decoding.
    //
    // then_try rather than then_value: a failed read must still count the latch
    // down, or the join below would hang.  `stream` is captured before
    // `buffers` on purpose — captures are destroyed in reverse declaration
    // order, so the buffers release their stream-ordered allocations before the
    // borrowed stream goes back to the pool.
    io_ctx->device_readv_async_io(io_object, reads, stream.get())
      .via(&disp)
      .then_try([i,
                 &decode_metadata,
                 &row_groups,
                 &col_opts,
                 &cols,
                 &failures,
                 &done,
                 &timers,
                 mr,
                 stream  = std::move(stream),
                 buffers = std::move(buffers)](sirius::exec::try_t<std::size_t>&& res) mutable {
        if (res.has_exception()) {
          failures.fetch_add(1, std::memory_order_relaxed);
          done.count_down();
          return;
        }

        std::vector<cudf::device_span<uint8_t const>> spans;
        spans.reserve(buffers.size());
        for (auto const& buf : buffers) {
          spans.emplace_back(static_cast<uint8_t const*>(buf.data()), buf.size());
        }

        auto const& opts = col_opts[i];
        auto const& meta = *decode_metadata[i];
        auto const rg_span =
          cudf::host_span<cudf::size_type const>(row_groups.data(), row_groups.size());
        auto const span_span =
          cudf::host_span<cudf::device_span<uint8_t const> const>(spans.data(), spans.size());

        auto const t_ctor = clock_type::now();
        hybrid_scan_reader reader(meta, opts);
        timers.add_ctor(ms_since(t_ctor));

        auto const t_mat = clock_type::now();
        auto result = reader.materialize_all_columns(rg_span, span_span, opts, stream.get(), mr);
        timers.add_materialize(ms_since(t_mat));

        // opts selects exactly one column, so the table has exactly one.  Drain
        // the stream before releasing it: the column outlives this task (and
        // this task's buffers), and the caller assembles it on another thread.
        cudaStreamSynchronize(stream.get().value());
        auto released = result.tbl->release();
        cols[i]       = std::move(released.front());
        done.count_down();
      });
  }

  done.wait();

  if (failures.load(std::memory_order_relaxed) != 0) {
    SIRIUS_LOG_ERROR("columnar_parquet_parser: {} column(s) failed to read",
                     failures.load(std::memory_order_relaxed));
    return nullptr;
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

void run_columnar(std::vector<file_info>& files,
                  std::vector<cudf::io::parquet_reader_options> const& col_opts,
                  sirius::exec::scoped_dispatcher& disp,
                  stream_pool_t& streams,
                  std::size_t total_bytes,
                  bool pruned,
                  char const* label)
{
  auto const mr_ref = cudf::get_current_device_resource_ref();
  phase_timers timers;

  std::size_t total_rows = 0;
  auto t0                = clock_type::now();
  for (auto& fi : files) {
    // Pointers, not copies: the unpruned arm must not pay an extra FileMetaData
    // copy per column just to build this table.
    std::vector<pq::FileMetaData const*> decode_metadata(col_opts.size());
    for (std::size_t i = 0; i < col_opts.size(); ++i) {
      decode_metadata[i] = pruned ? &fi.col_metadata[i] : &fi.metadata;
    }
    auto tbl = columnar_parquet_parser(*fi.ds,
                                       fi.col_ranges,
                                       col_opts,
                                       decode_metadata,
                                       fi.row_groups,
                                       streams,
                                       mr_ref,
                                       disp,
                                       timers);
    if (tbl) { total_rows += static_cast<std::size_t>(tbl->num_rows()); }
  }
  double const elapsed = ms_since(t0);

  auto const n_ctors = timers.n_ctors.load();
  auto const ctor_ms = timers.ctor_ms.load();
  auto const mat_ms  = timers.materialize_ms.load();

  std::cout << "\n=== " << label << " results ===\n"
            << "  wall           : " << std::fixed << std::setprecision(1) << elapsed << " ms\n"
            << "  throughput     : " << std::setprecision(2)
            << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) / (elapsed / 1000.0)
            << " GiB/s\n"
            << "  rows           : " << total_rows << "\n"
            << "  reader ctors   : " << n_ctors << "\n"
            << "  ctor total     : " << std::setprecision(1) << ctor_ms << " ms  ("
            << std::setprecision(2) << (elapsed > 0 ? 100.0 * ctor_ms / elapsed : 0.0)
            << "% of wall)\n"
            << "  ctor mean      : " << std::setprecision(3)
            << (n_ctors ? ctor_ms * 1000.0 / static_cast<double>(n_ctors) : 0.0) << " us\n"
            << "  materialize    : " << std::setprecision(1) << mat_ms << " ms  (sum over "
            << "concurrent columns)\n";
}

// ---- micro: isolate the construction cost ----------------------------------
//
// The end-to-end arms are IO-bound (O_DIRECT, ~5.5 GiB/s NVMe ceiling), so the
// reader-construction cost is invisible in wall clock.  This measures it directly,
// with no IO and no decode: construct + destroy a reader per iteration.
void run_micro(std::vector<file_info> const& files,
               std::vector<cudf::io::parquet_reader_options> const& col_opts,
               std::size_t iters)
{
  auto const bench = [&](char const* name, auto&& make_one) {
    auto const t0 = clock_type::now();
    for (std::size_t it = 0; it < iters; ++it) {
      for (auto const& fi : files) {
        for (std::size_t i = 0; i < col_opts.size(); ++i) {
          make_one(fi, i);
        }
      }
    }
    auto const total_us = us_since(t0);
    auto const n        = static_cast<double>(iters * files.size() * col_opts.size());
    std::cout << "  " << std::left << std::setw(42) << name << std::right << std::fixed
              << std::setprecision(1) << std::setw(9) << (total_us / n) << " us/ctor    "
              << std::setprecision(2) << std::setw(8) << (total_us / 1000.0) << " ms total\n";
  };

  std::cout << "\n=== micro: reader construction (no IO, no decode) ===\n"
            << "  " << iters << " iters x " << files.size() << " files x " << col_opts.size()
            << " columns = " << (iters * files.size() * col_opts.size())
            << " constructions each\n\n";

  bench("hybrid_scan_reader(full metadata)", [&](file_info const& fi, std::size_t i) {
    hybrid_scan_reader r(fi.metadata, col_opts[i]);
  });
  bench("hybrid_scan_reader(pruned metadata)", [&](file_info const& fi, std::size_t i) {
    hybrid_scan_reader r(fi.col_metadata[i], col_opts[i]);
  });
}

}  // namespace

// ---- main ------------------------------------------------------------------

int main(int argc, char** argv)
{
  std::string dir;
  std::size_t n_files = 0;
  std::string mode;  // "b" or "c"
  std::size_t nthreads = 4;

  std::size_t micro_iters = 20;

  auto const usage = [&] {
    std::cerr << "usage: " << argv[0]
              << " --dir DIR [--n_files N] --mode <b|c|o|m> [--nthreads N] [--micro_iters N]\n"
                 "  b  baseline  — one read_parquet per file, all columns\n"
                 "  c  columnar  — per-column read+decode, hybrid_scan_reader + full metadata\n"
                 "  o  optimized — same, but each reader is built from single-column metadata\n"
                 "  m  micro     — reader construction cost only (no IO, no decode)\n";
  };

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--dir" && i + 1 < argc) {
      dir = argv[++i];
    } else if (arg == "--n_files" && i + 1 < argc) {
      n_files = std::stoull(argv[++i]);
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--nthreads" && i + 1 < argc) {
      nthreads = std::stoull(argv[++i]);
    } else if (arg == "--micro_iters" && i + 1 < argc) {
      micro_iters = std::stoull(argv[++i]);
    } else {
      usage();
      return 1;
    }
  }
  if (dir.empty() || (mode != "b" && mode != "c" && mode != "o" && mode != "m")) {
    usage();
    return 1;
  }

  auto log_sink = sirius::log::make_spdlog_owning_sink({"log", std::nullopt});
  log_sink->set_level(sirius::log::level::warn);
  sirius::log::set_sink(std::move(log_sink));

  cudaFree(nullptr);

  // ---- memory setup --------------------------------------------------------
  // sirius_memory_reservation_manager installs the cucascade GPU allocator as
  // cudf's default device memory resource in its constructor.
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_usage_limit_ratio_per_gpu(0.5)  // GPU pool: 50% of total memory
    .set_reservation_fraction_per_gpu(0.9)
    .use_gpu_id_as_host_id()
    .set_per_numa_region_capacity(20ULL << 30)
    .set_reservation_fraction_per_numa_region(0.9)
    .set_host_pool_features(1ULL << 20,  // chunk_size  = 1 MiB
                            1024,        // pool_size   = blocks per slab (1 GiB each)
                            20);         // initial_pools

  auto mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());

  auto* host_space = mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  auto* bounce_mr  = host_space->get_memory_resource_of<cucascade::memory::Tier::HOST>();

  // ---- io context ----------------------------------------------------------
  auto ctx = std::make_shared<sirius::io::uring::uring_reactor::reactor_context>(
    sirius::io::uring::uring_reactor::reactor_config_type{}, bounce_mr);
  auto io_ctx = std::make_shared<sirius::io::uring::uring_ioctx>(1, std::move(ctx));
  io_ctx->start();

  // ---- thread pool + dispatcher --------------------------------------------
  sirius::exec::static_thread_pool pool(static_cast<int>(nthreads), "parse_pool");
  sirius::exec::scoped_dispatcher disp(pool);
  stream_pool_t streams(rmm::cuda_device_id{0}, nthreads);

  // ---- phase 1: discover files ---------------------------------------------
  auto paths = glob_parquet_files(dir, n_files);
  if (paths.empty()) {
    std::cerr << "no parquet files found in " << dir << "\n";
    return 1;
  }

  // One options object per column: each selects exactly one column, which is
  // what makes all_column_chunks_byte_ranges return that column's chunks and
  // materialize_all_columns produce a one-column table.
  auto disc_opts = cudf::io::parquet_reader_options::builder().column_names(COLUMNS).build();
  std::vector<cudf::io::parquet_reader_options> col_opts;
  col_opts.reserve(COLUMNS.size());
  for (auto const& name : COLUMNS) {
    col_opts.push_back(cudf::io::parquet_reader_options::builder()
                         .column_names(std::vector<std::string>{name})
                         .build());
  }

  std::cout << "found " << paths.size() << " file(s) in " << dir << "\n"
            << "mode=" << (mode == "b" ? "baseline" : "columnar") << "  nthreads=" << nthreads
            << "  n_columns=" << COLUMNS.size() << "\n\n"
            << "=== phase 1: per-column chunk discovery ===\n";

  std::vector<file_info> files;
  files.reserve(paths.size());
  std::size_t total_bytes = 0;
  double prune_ms         = 0.0;

  for (auto const& path : paths) {
    file_info fi;
    fi.path = path;
    fi.ds   = io_ctx->open_datasource(path);

    auto footer = cudf::io::parquet::fetch_footer_to_host(*fi.ds);
    hybrid_scan_reader reader(cudf::host_span<uint8_t const>(footer->data(), footer->size()),
                              disc_opts);

    fi.row_groups = reader.all_row_groups(disc_opts);
    fi.metadata   = reader.parquet_metadata();

    // The per-column grouping.  A reader's column selection is fixed at
    // construction — passing single-column options to a reader built over all
    // four columns does NOT re-select, and silently yields the wrong ranges.
    // So build one reader per column, each from the already-parsed metadata
    // (cheap: no footer fetch, no Thrift re-parse) and each carrying that
    // column's own options, exactly as the decode side does.
    auto const rg_span =
      cudf::host_span<cudf::size_type const>(fi.row_groups.data(), fi.row_groups.size());
    fi.col_ranges.reserve(col_opts.size());
    for (auto const& opts : col_opts) {
      hybrid_scan_reader col_reader(fi.metadata, opts);
      fi.col_ranges.push_back(col_reader.all_column_chunks_byte_ranges(rg_span, opts));
    }

    // Single-column metadata views, built once per (file, column).  Every later
    // reader construction copies one of these instead of the full 16-column
    // footer, so the copy shrinks with the projection.
    auto const t_prune = clock_type::now();
    fi.col_metadata.reserve(COLUMNS.size());
    for (auto const& name : COLUMNS) {
      fi.col_metadata.push_back(prune_to_column(fi.metadata, name));
    }
    prune_ms += ms_since(t_prune);

    std::size_t n_ranges = 0;
    for (auto const& ranges : fi.col_ranges) {
      n_ranges += ranges.size();
      for (auto const& r : ranges) {
        fi.range_bytes += static_cast<std::size_t>(r.size());
      }
    }
    total_bytes += fi.range_bytes;

    std::cout << std::filesystem::path(path).filename().string() << " { n_ranges=" << n_ranges
              << ", per_column=[";
    for (std::size_t i = 0; i < fi.col_ranges.size(); ++i) {
      std::cout << (i ? ", " : "") << fi.col_ranges[i].size();
    }
    std::cout << "], total_size=" << std::fixed << std::setprecision(2)
              << static_cast<double>(fi.range_bytes) / (1024.0 * 1024.0) << " MiB }\n";

    files.push_back(std::move(fi));
  }

  std::cout << "\ntotal: " << files.size() << " files, " << std::fixed << std::setprecision(2)
            << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB\n";

  std::cout << "metadata pruning: " << std::setprecision(2) << prune_ms << " ms for "
            << (files.size() * COLUMNS.size()) << " single-column views ("
            << (files.empty() ? 0.0 : prune_ms * 1000.0 / (files.size() * COLUMNS.size()))
            << " us each)\n";

  // ---- phase 2: benchmark --------------------------------------------------
  if (mode == "b") {
    run_baseline(files, disp, streams, total_bytes);
  } else if (mode == "c") {
    run_columnar(files, col_opts, disp, streams, total_bytes, false, "columnar");
  } else if (mode == "o") {
    run_columnar(files, col_opts, disp, streams, total_bytes, true, "columnar-pruned");
  } else {
    run_micro(files, col_opts, micro_iters);
  }

  disp.wait_for_all();
  pool.stop();
  return 0;
}
