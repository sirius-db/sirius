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

// s3_autotune_throughput_bench — does the request-shaping model in
// s3_autotune.hpp reach line rate on the column ranges a real TPC-H query
// actually needs?
//
// --sf picks the TPC-H scale-factor bucket and --query (1-22) picks which
// query's column projection to read, looked up from plan.json (query number ->
// {table: [columns]}, one entry per query, sitting at the repo root).  For
// every file under every table the query touches, a hybrid_scan_reader turns
// "these columns" into column-chunk byte ranges the same way a real scan would,
// and those real ranges -- not a synthetic layout -- are what the model's
// coalescer and GET-size solve run against.
//
// IO is gated by --npread: a counting semaphore starts with --npread permits,
// ONE FILE at a time is registered per acquired permit -- every one of that
// file's ranges (already coalesced and chunked by the model) goes out in a
// single batched host_readv_async_io call, and the permit is only
// released once every range of that file has completed.  So --npread is
// "how many files are being actively read at once", not a per-range knob:
// acquiring a permit for a file with 400 coalesced ranges puts all 400 on the
// wire together, and the next file only starts once that whole file is done.
//
// The reads bypass the prefetching cache and cuCascade staging entirely: the
// destinations are plain malloc'd host buffers, one per file, sliced per
// request -- same as the synthetic version of this benchmark, just now backed
// by real per-file ranges instead of a generated layout.
//
//   ./s3_autotune_throughput_bench \
//       --sf         1000            \
//       --query      9               \
//       --plan-json  plan.json       \
//       --npread     32              \  # files being actively read at once
//       --n-reactors 8                \
//       --conn-per-reactor 64         \
//       --rtt        0.03            \
//       --stream-throughput 400      \
//       --transfer-efficiency 0.85   \
//       --gap-efficiency      0.80   \
//       --repeat     3               \
//       --config     autotune.yml

#include "exec/scoped_dispatcher.hpp"
#include "exec/semi_future.hpp"
#include "exec/thread_pool.hpp"
#include "io/types.hpp"
#include "s3_autotune.hpp"
#include "s3_bench_common.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <latch>
#include <map>
#include <memory>
#include <optional>
#include <semaphore>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace sirius::bench;
namespace at = sirius::bench::autotune;

constexpr std::size_t mib_v = 1UL << 20;
constexpr std::size_t gib_v = 1UL << 30;

[[nodiscard]] std::size_t mib_to_bytes(double mib)
{
  return static_cast<std::size_t>(mib * static_cast<double>(mib_v));
}

[[nodiscard]] double as_mib(std::size_t bytes)
{
  return static_cast<double>(bytes) / static_cast<double>(mib_v);
}

[[nodiscard]] double as_gib(std::size_t bytes)
{
  return static_cast<double>(bytes) / static_cast<double>(gib_v);
}

// ---------------------------------------------------------------------------
// progress
// ---------------------------------------------------------------------------

/// Prints @p line()'s result to stderr every @p interval, overwriting the same
/// line, until the returned jthread is destroyed (its destructor requests stop
/// and joins, so a caller only has to let it go out of scope) -- a final call
/// to @p line() lands with a trailing newline so later output starts clean.
[[nodiscard]] std::jthread progress_reporter(
  std::function<std::string()> line,
  std::chrono::milliseconds interval = std::chrono::milliseconds{500})
{
  return std::jthread([line = std::move(line), interval](std::stop_token st) {
    while (!st.stop_requested()) {
      std::cerr << "\r" << line() << std::flush;
      // Sleep in small slices so a stop request is noticed promptly rather
      // than only after a full interval.
      constexpr auto slice = std::chrono::milliseconds{50};
      for (auto waited = std::chrono::milliseconds{0}; waited < interval && !st.stop_requested();
           waited += slice) {
        std::this_thread::sleep_for(slice);
      }
    }
    std::cerr << "\r" << line() << "\n";
  });
}

// ---------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------

struct query_bench_options {
  bench_options common;
  at::tuning_params params;

  std::size_t sf{1000};
  std::size_t query{0};  ///< 1-22; 0 means unset (checked after parsing)
  std::string plan_json_path{"plan.json"};
  std::string dataset_root{"s3://sirius-s3-test/datasets"};

  /// Files being actively read at once -- see the file comment.  Each permit
  /// covers a whole file's worth of (already coalesced) ranges, not one range.
  std::size_t npread{4};

  /// Threads used only during discovery (open + footer-fetch + range
  /// enumeration per file), not for the timed reads themselves.
  std::size_t discover_threads{20};

  /// Model knobs not part of tuning_params.
  std::size_t min_chunks_per_connection{2};
  double min_chunk_mib{8.0};

  /// Take the model's GET-size answer out of the loop for a manual sweep.
  /// Zero means "use the model".
  double override_chunk_mib{0.0};

  /// Bound a merge by the GET size cap as well -- see @c coalesce_and_chunk.
  bool merge_within_chunk{false};

  /// Plan and print, read nothing.
  bool dry_run{false};
};

bool parse_query_arg(arg_parser const& p, int& i, query_bench_options& o)
{
  return p.match(i, "--sf", o.sf) || p.match(i, "--query", o.query) ||
         p.match(i, "--plan-json", o.plan_json_path) ||
         p.match(i, "--dataset-root", o.dataset_root) || p.match(i, "--npread", o.npread) ||
         p.match(i, "--discover-threads", o.discover_threads) ||
         p.match(i, "--nic", o.params.nic_gbps) ||
         p.match(i, "--stream-throughput", o.params.stream_mbps) ||
         p.match(i, "--rtt", o.params.rtt_s) ||
         p.match(i, "--transfer-efficiency", o.params.transfer_efficiency) ||
         p.match(i, "--gap-efficiency", o.params.gap_efficiency) ||
         p.match(i, "--min-chunks-per-connection", o.min_chunks_per_connection) ||
         p.match(i, "--min-chunk", o.min_chunk_mib) ||
         // rest.max_connections is per reactor, which is exactly the quantity
         // the reactor count is derived from -- name it so at the CLI too.
         p.match(i, "--conn-per-reactor", o.common.max_nconnection) ||
         p.match(i, "--n-reactors", o.common.n_reactors) ||
         p.match(i, "--chunk-size", o.override_chunk_mib) ||
         p.toggle(i, "--merge-within-chunk", o.merge_within_chunk) ||
         p.toggle(i, "--dry-run", o.dry_run);
}

// ---------------------------------------------------------------------------
// plan.json — {"<query>": {"tables": {"<table>": ["<column>", ...], ...}}}
// ---------------------------------------------------------------------------
//
// A hand-rolled scanner rather than a JSON library: the file has exactly one
// shape (an object of objects of arrays of strings) and pulling in a
// dependency to parse it would outweigh the twenty lines below.

namespace plan_json {

void skip_ws(std::string_view s, std::size_t& i)
{
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
    ++i;
  }
}

std::string parse_string(std::string_view s, std::size_t& i)
{
  if (i >= s.size() || s[i] != '"') { throw std::runtime_error("plan.json: expected string"); }
  ++i;
  std::string out;
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\' && i + 1 < s.size()) {
      ++i;
      out.push_back(s[i]);
    } else {
      out.push_back(s[i]);
    }
    ++i;
  }
  if (i >= s.size()) { throw std::runtime_error("plan.json: unterminated string"); }
  ++i;  // closing quote
  return out;
}

/// tables -> columns for one query entry.
using table_columns = std::map<std::string, std::vector<std::string>>;

/// Parse {"tables": {"<table>": ["<col>", ...], ...}} for one query.
table_columns parse_query_entry(std::string_view s, std::size_t& i)
{
  table_columns tables;
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '{') { throw std::runtime_error("plan.json: expected object"); }
  ++i;
  skip_ws(s, i);
  if (i < s.size() && s[i] == '}') {
    ++i;
    return tables;
  }
  for (;;) {
    skip_ws(s, i);
    auto const key = parse_string(s, i);
    skip_ws(s, i);
    if (i >= s.size() || s[i] != ':') { throw std::runtime_error("plan.json: expected ':'"); }
    ++i;
    skip_ws(s, i);
    if (key == "tables") {
      if (i >= s.size() || s[i] != '{') {
        throw std::runtime_error("plan.json: \"tables\" must be an object");
      }
      ++i;
      skip_ws(s, i);
      if (!(i < s.size() && s[i] == '}')) {
        for (;;) {
          skip_ws(s, i);
          auto const table = parse_string(s, i);
          skip_ws(s, i);
          if (i >= s.size() || s[i] != ':') { throw std::runtime_error("plan.json: expected ':'"); }
          ++i;
          skip_ws(s, i);
          if (i >= s.size() || s[i] != '[') {
            throw std::runtime_error("plan.json: table value must be an array");
          }
          ++i;
          skip_ws(s, i);
          std::vector<std::string> cols;
          if (!(i < s.size() && s[i] == ']')) {
            for (;;) {
              skip_ws(s, i);
              cols.push_back(parse_string(s, i));
              skip_ws(s, i);
              if (i < s.size() && s[i] == ',') {
                ++i;
                continue;
              }
              break;
            }
          }
          skip_ws(s, i);
          if (i >= s.size() || s[i] != ']') { throw std::runtime_error("plan.json: expected ']'"); }
          ++i;
          tables.emplace(table, std::move(cols));
          skip_ws(s, i);
          if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
          }
          break;
        }
      }
      skip_ws(s, i);
      if (i >= s.size() || s[i] != '}') {
        throw std::runtime_error("plan.json: unterminated \"tables\" object");
      }
      ++i;
    } else {
      throw std::runtime_error("plan.json: unexpected key \"" + key + "\" in query entry");
    }
    skip_ws(s, i);
    if (i < s.size() && s[i] == ',') {
      ++i;
      continue;
    }
    break;
  }
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '}') { throw std::runtime_error("plan.json: unterminated object"); }
  ++i;
  return tables;
}

/// Find query @p query's entry in @p path and return its table->columns map.
table_columns load(std::string const& path, std::size_t query)
{
  std::ifstream f(path);
  if (!f) { throw std::runtime_error("plan.json: cannot open " + path); }
  std::stringstream buf;
  buf << f.rdbuf();
  std::string const text = buf.str();

  std::string_view s = text;
  std::size_t i      = 0;
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '{') {
    throw std::runtime_error(path + ": expected top-level object");
  }
  ++i;
  skip_ws(s, i);
  if (i < s.size() && s[i] == '}') { throw std::runtime_error(path + ": no queries found"); }
  std::string const target = std::to_string(query);
  for (;;) {
    skip_ws(s, i);
    auto const key = parse_string(s, i);
    skip_ws(s, i);
    if (i >= s.size() || s[i] != ':') { throw std::runtime_error(path + ": expected ':'"); }
    ++i;
    if (key == target) { return parse_query_entry(s, i); }
    // Not the query we want: skip its value by depth-tracking braces/brackets
    // across strings, since a naive scan would trip on '{' inside a name.
    skip_ws(s, i);
    int depth      = 0;
    bool in_string = false;
    for (; i < s.size(); ++i) {
      char const c = s[i];
      if (in_string) {
        if (c == '\\') {
          ++i;
        } else if (c == '"') {
          in_string = false;
        }
        continue;
      }
      if (c == '"') {
        in_string = true;
      } else if (c == '{' || c == '[') {
        ++depth;
      } else if (c == '}' || c == ']') {
        --depth;
        if (depth == 0) {
          ++i;
          break;
        }
      }
    }
    skip_ws(s, i);
    if (i < s.size() && s[i] == ',') {
      ++i;
      continue;
    }
    break;
  }
  throw std::runtime_error(path + ": no entry for query " + target);
}

}  // namespace plan_json

// ---------------------------------------------------------------------------
// query column-range discovery
// ---------------------------------------------------------------------------

/// One file's worth of real column-chunk ranges, shaped into GETs by the same
/// model that would shape a synthetic workload -- see s3_autotune.hpp.
struct file_plan {
  std::string path;
  std::string table;
  std::size_t n_segments_before_coalesce{0};  ///< column chunks, pre-coalesce
  at::request_plan io;
  /// One allocation per file, carved into a slice per request.  Value-init
  /// touches every page, so the timed loop never takes a first-touch fault.
  std::vector<std::uint8_t> buffer;
  std::vector<sirius::io::slice> slices;
  std::unique_ptr<sirius::io::sirius_datasource> ds;
};

/// Per-table rollup for the end-of-run report: how much of the workload each
/// table contributed and at what granularity.
struct table_stats {
  std::size_t n_files{0};
  std::size_t n_segments{0};  ///< column chunks before coalescing
  std::size_t n_requests{0};  ///< GETs after coalescing
  std::size_t useful_bytes{0};
  std::size_t wire_bytes{0};

  [[nodiscard]] double avg_segment_bytes() const noexcept
  {
    return n_segments == 0 ? 0.0
                           : static_cast<double>(useful_bytes) / static_cast<double>(n_segments);
  }
};

struct discovery {
  std::vector<file_plan> files;
  std::size_t n_requests{0};  ///< GETs across every file, after coalescing
  std::size_t n_segments{0};  ///< column chunks before coalescing
  std::size_t n_blocks{0};    ///< coalesced blocks
  std::size_t useful_bytes{0};
  std::size_t wire_bytes{0};
  std::size_t min_request{std::numeric_limits<std::size_t>::max()};
  std::size_t max_request{0};
  std::map<std::string, table_stats> per_table;
};

discovery discover_query(engine& eng,
                         query_bench_options const& opts,
                         plan_json::table_columns const& tables,
                         at::connection_plan const& plan)
{
  double const max_waste   = 1.0 - std::clamp(opts.params.gap_efficiency, 0.0, 1.0);
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

  discovery d;

  // Phase 1 (sequential, cheap): list every table's files and build one
  // parquet_reader_options per table -- index-stable, shared read-only by
  // every file task of that table in phase 2.
  struct table_entry {
    std::string name;
    cudf::io::parquet_reader_options reader_opts;
  };
  std::vector<table_entry> table_entries;
  table_entries.reserve(tables.size());

  struct file_task {
    std::size_t table_idx;
    s3_file file;
  };
  std::vector<file_task> tasks;

  for (auto const& [table, columns] : tables) {
    std::string const prefix =
      // Trailing slash matters: S3 prefix matching is a literal string prefix,
      // and "part" (no slash) also matches every key under "partsupp/".
      opts.dataset_root + "/tpch_sf" + std::to_string(opts.sf) + "/" + table + "/";
    // opts.common.n_files caps files PER TABLE here (list_prefix is called once
    // per table), unlike its "total across prefixes" meaning at other call sites
    // in this shared bench_options struct.
    auto const s3_files        = list_prefix(eng, prefix, opts.common.n_files);
    d.per_table[table].n_files = s3_files.size();

    std::size_t const table_idx = table_entries.size();
    table_entries.push_back(
      {table, cudf::io::parquet_reader_options::builder().column_names(columns).build()});
    tasks.reserve(tasks.size() + s3_files.size());
    for (auto const& f : s3_files) {
      tasks.push_back({table_idx, f});
    }
  }

  // Phase 2 (parallel): open + footer-fetch + range-discover each file on its
  // own thread pool slot.  Discovery is otherwise a blocking round trip per
  // file (open, then a suffix GET for the footer), and at production scale
  // that is dozens to low hundreds of files -- doing it one at a time would
  // make discovery itself the slow part of the benchmark.  Each task owns a
  // unique index into `slots`, so no locking is needed; the pool fully drains
  // (wait_for_all) before phase 3 reads any of them.
  std::vector<std::optional<file_plan>> slots(tasks.size());
  {
    std::size_t const nthreads = std::max<std::size_t>(
      1, std::min(opts.discover_threads, std::max<std::size_t>(1, tasks.size())));
    sirius::exec::static_thread_pool pool(static_cast<int>(nthreads), "discover");
    sirius::exec::scoped_dispatcher disp(pool);

    std::atomic<std::size_t> files_scanned{0};
    // Scoped so the reporter's destructor (stop + join) runs before this
    // block returns, right after wait_for_all -- not tied to `disp`/`pool`.
    auto reporter = progress_reporter([&files_scanned, total = tasks.size()] {
      return std::format(
        "[discover] {}/{} files ({:.0f}%)",
        files_scanned.load(std::memory_order_relaxed),
        total,
        total == 0 ? 100.0
                   : 100.0 * static_cast<double>(files_scanned.load(std::memory_order_relaxed)) /
                       static_cast<double>(total));
    });

    for (std::size_t i = 0; i < tasks.size(); ++i) {
      disp.enqueue([&, i] {
        auto const& task    = tasks[i];
        auto const& table_e = table_entries[task.table_idx];
        auto ds             = eng.io_ctx().open_datasource(task.file.path, task.file.size_bytes);

        auto footer = cudf::io::parquet::fetch_footer_to_host(*ds);
        hybrid_scan_reader reader(cudf::host_span<uint8_t const>(footer->data(), footer->size()),
                                  table_e.reader_opts);
        auto const row_groups = reader.all_row_groups(table_e.reader_opts);
        auto const ranges     = reader.all_column_chunks_byte_ranges(
          cudf::host_span<cudf::size_type const>(row_groups.data(), row_groups.size()),
          table_e.reader_opts);
        if (ranges.empty()) {
          files_scanned.fetch_add(1, std::memory_order_relaxed);
          return;
        }

        std::vector<at::byte_span> spans;
        spans.reserve(ranges.size());
        for (auto const& r : ranges) {
          spans.push_back(
            {static_cast<std::size_t>(r.offset()), static_cast<std::size_t>(r.size())});
        }

        file_plan fp;
        fp.path                       = task.file.path;
        fp.table                      = table_e.name;
        fp.n_segments_before_coalesce = spans.size();
        fp.io                         = at::coalesce_and_chunk(
          std::move(spans), plan.rtt_bytes, max_waste, plan.chunk_bytes, opts.merge_within_chunk);
        if (fp.io.requests.empty()) {
          files_scanned.fetch_add(1, std::memory_order_relaxed);
          return;
        }

        // One buffer per file, sliced per request -- the reactor writes the
        // body of each GET straight into its slice, no staging copy.
        fp.buffer.assign(fp.io.wire_bytes, std::uint8_t{0});
        fp.slices.reserve(fp.io.requests.size());
        std::size_t cursor = 0;
        for (auto const& r : fp.io.requests) {
          fp.slices.emplace_back(r.offset, r.length, fp.buffer.data() + cursor);
          cursor += r.length;
        }
        fp.ds    = std::move(ds);
        slots[i] = std::move(fp);
        files_scanned.fetch_add(1, std::memory_order_relaxed);
      });
    }
    disp.wait_for_all();
  }

  // Phase 3 (sequential, cheap): accumulate stats and move settled plans in.
  // Nothing needs flattening -- run_once submits each file's ranges as one
  // batched call, so d.files IS the submission unit.
  for (auto& slot : slots) {
    if (!slot) { continue; }
    file_plan& fp = *slot;
    d.n_segments += fp.n_segments_before_coalesce;
    d.n_requests += fp.io.requests.size();
    d.n_blocks += fp.io.n_blocks;
    d.useful_bytes += fp.io.useful_bytes;
    d.wire_bytes += fp.io.wire_bytes;
    for (auto const& r : fp.io.requests) {
      d.min_request = std::min(d.min_request, r.length);
      d.max_request = std::max(d.max_request, r.length);
    }

    auto& ts = d.per_table[fp.table];
    ts.n_segments += fp.n_segments_before_coalesce;
    ts.n_requests += fp.io.requests.size();
    ts.useful_bytes += fp.io.useful_bytes;
    ts.wire_bytes += fp.io.wire_bytes;

    d.files.push_back(std::move(*slot));
  }

  if (d.min_request > d.max_request) { d.min_request = 0; }
  return d;
}

// ---------------------------------------------------------------------------
// timed loop
// ---------------------------------------------------------------------------

/// Keep exactly `npread` FILES being actively read at once: acquire a permit,
/// submit every one of that file's already-coalesced ranges in a single
/// batched call, and only release the permit once the whole file's read has
/// completed.  The submission loop blocks on acquire() once `npread` files are
/// outstanding, so a file with hundreds of ranges is one unit of concurrency,
/// not hundreds -- the next file only starts once an entire file finishes.
iteration_result run_once(sirius::io::ioctx& io_ctx,
                          std::vector<file_plan>& files,
                          std::size_t npread)
{
  std::counting_semaphore<1 << 20> sem{
    static_cast<std::ptrdiff_t>(std::max<std::size_t>(1, npread))};
  std::atomic<std::size_t> bytes_read{0};
  std::atomic<std::size_t> files_completed{0};
  std::atomic<std::size_t> failures{0};
  std::latch done{static_cast<std::ptrdiff_t>(files.size())};

  auto const start = clock_type::now();

  // Scoped so the reporter's destructor (stop + join) runs before this
  // function returns the timed result -- it must not itself land inside the
  // timed window, only observe it.
  {
    auto reporter = progress_reporter([&] {
      auto const n      = files_completed.load(std::memory_order_relaxed);
      auto const bytes  = bytes_read.load(std::memory_order_relaxed);
      double const secs = ms_since(start) / 1e3;
      double const gbps = secs > 0 ? (static_cast<double>(bytes) / at::bytes_per_gb_v) / secs : 0.0;
      return std::format(
        "[read] {}/{} files ({:.0f}%), {:.2f} GiB, {:.2f} GB/s",
        n,
        files.size(),
        files.empty() ? 100.0 : 100.0 * static_cast<double>(n) / static_cast<double>(files.size()),
        as_gib(bytes),
        gbps);
    });

    for (auto& fp : files) {
      sem.acquire();
      // The whole file's ranges go out together in one call -- the reactor
      // gets fp.slices.size() logical ranges at once, not one at a time.
      auto fut = io_ctx.host_readv_async_io(fp.ds->get_io_object(), fp.slices);

      std::move(fut).install_callback([&sem, &bytes_read, &files_completed, &failures, &done](
                                        sirius::exec::try_t<std::size_t>&& t) mutable {
        if (t.has_exception()) {
          failures.fetch_add(1, std::memory_order_relaxed);
          try {
            std::rethrow_exception(t.exception());
          } catch (std::exception const& e) {
            std::cerr << "read failed: " << e.what() << "\n";
          } catch (...) {
            std::cerr << "read failed: unknown exception\n";
          }
        } else {
          bytes_read.fetch_add(std::move(t).get(), std::memory_order_relaxed);
        }
        files_completed.fetch_add(1, std::memory_order_relaxed);
        sem.release();
        done.count_down();
      });
    }

    done.wait();
  }

  iteration_result r;
  r.duration_ms = ms_since(start);
  r.bytes       = bytes_read.load();

  if (failures.load() != 0) {
    std::cerr << failures.load() << " file read(s) failed — throughput is not meaningful\n";
  }
  return r;
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

void report_plan(query_bench_options const& opts,
                 at::connection_plan const& plan,
                 plan_json::table_columns const& tables,
                 discovery const& d)
{
  double const nic_mbps  = opts.params.nic_gbps * 125.0;
  double const max_waste = 1.0 - std::clamp(opts.params.gap_efficiency, 0.0, 1.0);

  std::cout << "s3_autotune_throughput_bench\n"
            << std::format("  dataset              : {}/tpch_sf{}\n", opts.dataset_root, opts.sf)
            << std::format("  query                : {}\n", opts.query);
  for (auto const& [table, cols] : tables) {
    std::cout << std::format("    {:<10} : {} file(s), columns [{}]\n",
                             table,
                             d.per_table.count(table) ? d.per_table.at(table).n_files : 0,
                             [&] {
                               std::string s;
                               for (std::size_t i = 0; i < cols.size(); ++i) {
                                 if (i != 0) { s += ", "; }
                                 s += cols[i];
                               }
                               return s;
                             }());
  }

  std::cout << "  -- link model --\n"
            << std::format("  nic                  : {:.1f} Gbps ({:.0f} MB/s)\n",
                           opts.params.nic_gbps,
                           nic_mbps)
            << std::format("  per-stream ceiling   : {:.1f} MB/s\n", opts.params.stream_mbps)
            << std::format("  rtt (ttfb)           : {:.2f} ms\n", opts.params.rtt_s * 1e3)
            << std::format("  transfer efficiency  : {:.2f} -> GET {:.2f} MiB\n",
                           opts.params.transfer_efficiency,
                           as_mib(plan.chunk_bytes))
            << std::format(
                 "  gap efficiency       : {:.2f} -> waste <= {:.1f}%, read through "
                 "gaps <= {:.2f} MiB\n",
                 opts.params.gap_efficiency,
                 max_waste * 100.0,
                 as_mib(plan.rtt_bytes));

  std::cout << "  -- reactor pool (fixed, independent of the link model) --\n"
            << std::format("  reactors             : {}\n", opts.common.n_reactors)
            << std::format("  conn / reactor       : {}\n", opts.common.max_nconnection)
            << std::format("  ceiling              : {} concurrent GETs\n",
                           opts.common.n_reactors * opts.common.max_nconnection)
            << std::format(
                 "  nic wants (fyi)      : {} streams to fill the NIC at the assumed "
                 "per-stream rate\n",
                 plan.nic_connections)
            << std::format("  --npread             : {} files being actively read at once\n",
                           opts.npread);

  double const achieved_gap_eff =
    d.wire_bytes == 0 ? 0.0
                      : static_cast<double>(d.useful_bytes) / static_cast<double>(d.wire_bytes);

  std::cout << "  -- workload discovered from the query's real column ranges --\n"
            << std::format("  files                : {}\n", d.files.size())
            << std::format("  column chunks        : {} ({:.2f} GiB useful)\n",
                           d.n_segments,
                           as_gib(d.useful_bytes))
            << std::format("  coalesced blocks     : {}{}\n",
                           d.n_blocks,
                           opts.merge_within_chunk ? "  [merge capped at GET size]" : "")
            << std::format(
                 "  requests (GETs)      : {}  ({:.2f} MiB min / {:.2f} avg / {:.2f} "
                 "max)\n",
                 d.n_requests,
                 as_mib(d.min_request),
                 d.n_requests == 0 ? 0.0 : as_mib(d.wire_bytes) / static_cast<double>(d.n_requests),
                 as_mib(d.max_request))
            << std::format(
                 "  wire bytes           : {:.2f} GiB (gap efficiency achieved "
                 "{:.1f}%)\n",
                 as_gib(d.wire_bytes),
                 achieved_gap_eff * 100.0);

  std::cout << "  -- per table --\n"
            << std::format("  {:<10} {:>7} {:>10} {:>10} {:>12} {:>12}\n",
                           "table",
                           "files",
                           "segments",
                           "requests",
                           "avg seg",
                           "useful");
  for (auto const& [table, ts] : d.per_table) {
    std::cout << std::format("  {:<10} {:>7} {:>10} {:>10} {:>10.2f}K {:>10.2f}M\n",
                             table,
                             ts.n_files,
                             ts.n_segments,
                             ts.n_requests,
                             ts.avg_segment_bytes() / 1024.0,
                             static_cast<double>(ts.useful_bytes) / (1024.0 * 1024.0));
  }
  std::cout << "\n";
}

void report_runs(std::vector<iteration_result> const& runs,
                 std::size_t useful_bytes,
                 double nic_gbps)
{
  if (runs.empty()) { return; }

  std::vector<double> wire_gbps;
  wire_gbps.reserve(runs.size());
  for (auto const& r : runs) {
    wire_gbps.push_back(r.duration_ms > 0 ? (static_cast<double>(r.bytes) / at::bytes_per_gb_v) /
                                              (r.duration_ms / 1e3)
                                          : 0.0);
  }

  for (std::size_t i = 0; i < runs.size(); ++i) {
    double const secs = runs[i].duration_ms / 1e3;
    double const useful =
      secs > 0 ? (static_cast<double>(useful_bytes) / at::bytes_per_gb_v) / secs : 0.0;
    std::cout << std::format(
      "  [autotune] iter {}: wire {:.3f} GB/s, useful {:.3f} GB/s, {:.1f} msec ({:.1f}% of NIC)\n",
      i,
      wire_gbps[i],
      useful,
      runs[i].duration_ms,
      nic_gbps > 0 ? 100.0 * wire_gbps[i] * 8.0 / nic_gbps : 0.0);
  }

  auto sorted = wire_gbps;
  std::sort(sorted.begin(), sorted.end());
  double const median_wire = sorted[sorted.size() / 2];
  std::cout << std::format("  [autotune] median wire throughput: {:.3f} GB/s ({:.1f}% of NIC)\n",
                           median_wire,
                           nic_gbps > 0 ? 100.0 * median_wire * 8.0 / nic_gbps : 0.0);
}

}  // namespace

int main(int argc, char** argv)
{
  query_bench_options opts;
  opts.common.n_reactors = 8;    // production-shaped default; override with --n-reactors
  opts.common.n_files    = 100;  // per-table cap; override with --n-files

  try {
    arg_parser p{argc, argv};
    for (int i = 1; i < argc; ++i) {
      if (parse_query_arg(p, i, opts)) { continue; }
      if (parse_common_arg(p, i, opts.common)) { continue; }
      throw std::runtime_error(std::string{"unknown flag: "} + argv[i]);
    }
    if (opts.query < 1 || opts.query > 22) {
      throw std::runtime_error("--query is required and must be 1-22");
    }
    if (opts.sf == 0) { throw std::runtime_error("--sf must be > 0"); }

    auto const tables = plan_json::load(opts.plan_json_path, opts.query);
    if (tables.empty()) {
      throw std::runtime_error("plan.json: query " + std::to_string(opts.query) + " has no tables");
    }

    // ---- what the model says for GET size / merge threshold -----------------
    // Independent of the workload size (see s3_autotune.hpp) — computed before
    // discovery because discovery needs it to shape each file's logical requests.
    auto plan = at::plan_connections(opts.params,
                                     /*workload_bytes=*/0,
                                     opts.min_chunks_per_connection,
                                     mib_to_bytes(opts.min_chunk_mib));
    if (opts.override_chunk_mib > 0) { plan.chunk_bytes = mib_to_bytes(opts.override_chunk_mib); }
    if (plan.chunk_bytes == 0) { throw std::runtime_error("derived GET size is 0"); }

    // ---- turn that into an engine -------------------------------------------
    // Reactor pool is sized directly from --n-reactors/--conn-per-reactor, not
    // derived from the model: the workload's size is only known after
    // discovery, which needs the engine to already exist.
    opts.common.chunk_size_bytes = plan.chunk_bytes;
    // coalesce_and_chunk shapes logical requests. The reactor can still split
    // them dynamically to match its current connection availability and backlog.
    // Reads land in malloc'd buffers, so the pinned pool only has to cover the
    // reactors' bounce slots: one 1 MiB block per connection per reactor.
    opts.common.host_chunk_mib      = 1;
    std::size_t const bounce_blocks = opts.common.n_reactors * opts.common.max_nconnection;
    opts.common.host_initial_pools  = (bounce_blocks + host_pool_size_v - 1) / host_pool_size_v + 1;

    engine eng{opts.common};

    // ---- discover the query's real column-chunk ranges ----------------------
    auto d = discover_query(eng, opts, tables, plan);
    if (d.files.empty()) {
      throw std::runtime_error(
        "no readable column ranges: check --sf/--query/--dataset-root and that plan.json's "
        "columns exist in that dataset");
    }

    report_plan(opts, plan, tables, d);
    if (opts.dry_run) { return 0; }

    std::size_t const repeat = std::max<std::size_t>(1, opts.common.repeat);
    std::vector<iteration_result> runs;
    runs.reserve(repeat);
    for (std::size_t r = 0; r < repeat; ++r) {
      runs.push_back(run_once(eng.io_ctx(), d.files, opts.npread));
    }
    report_runs(runs, d.useful_bytes, opts.params.nic_gbps);
    return 0;

  } catch (std::exception const& e) {
    std::cerr << "s3_autotune_throughput_bench: " << e.what() << "\n";
    return 1;
  }
}
