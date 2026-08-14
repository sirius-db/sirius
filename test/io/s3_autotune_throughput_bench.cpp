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
// s3_autotune.hpp actually reach line rate?
//
// Where s3_throughput_test asks "what does N connections x M MiB GETs give me",
// this one asks the question the other way round: given a workload and a link,
// the model decides how many connections and how big a GET, and the benchmark
// then reads that workload and reports what it got.  Every number the model
// consumes is a flag, so a sweep tunes the model rather than the benchmark.
//
// The workload is synthetic but shaped like a scan: `--seg-size` pieces the
// query needs, `--gap-size` apart in runs of `--segs-per-group`, with the groups
// spread across the whole object.  Small gaps are the interesting case -- they
// are what @c coalesce_and_chunk has to decide about -- and the group spacing is
// what stops everything collapsing into one request.  Throughput is reported
// twice: over the bytes actually asked for (what a query would see) and over the
// bytes on the wire (what the NIC sees); their ratio is the achieved gap
// efficiency.
//
// The reads bypass the prefetching cache and cuCascade staging entirely: the
// destinations are plain malloc'd host buffers, one per file, and each file's
// requests go out as one (or a few) host_read_ranges_async_io calls from the
// main thread, each counting down a latch.  cuCascade is configured with only
// enough 1 MiB blocks to back the reactors' internal bounce slots, which nothing
// on this path touches.
//
//   ./s3_autotune_throughput_bench \
//       --bucket    s3://bucket/prefix \
//       --bucket    s3://other/prefix  \
//       --n-files   16                 \
//       --workload  16                 \  # GiB the "query" needs
//       --seg-size  1                  \  # MiB per wanted piece
//       --gap-size  0.25               \  # MiB of junk between pieces
//       --segs-per-group 8             \
//       --nic       100                \  # Gbps
//       --stream-throughput 100        \  # MB/s per connection
//       --rtt       0.015              \  # seconds to first byte
//       --transfer-efficiency 0.85     \
//       --gap-efficiency      0.80     \
//       --conn-per-reactor    128      \
//       --repeat    3                  \
//       --config    sirius_s3.yml

#include "exec/semi_future.hpp"
#include "io/types.hpp"
#include "s3_autotune.hpp"
#include "s3_bench_common.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
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
// options
// ---------------------------------------------------------------------------

struct autotune_options {
  bench_options common;      ///< buckets, credentials, --n-files, --repeat, ...
  at::tuning_params params;  ///< the link / endpoint model

  /// Bytes the "query" needs, in GiB.  Drives both the connection count and how
  /// many segments get laid out across the files.
  double workload_gib{8.0};

  /// Workload shape.  A group is `segs_per_group` segments `gap_size` apart --
  /// i.e. a run of pieces close enough that coalescing has a decision to make.
  /// Groups sit a whole stride apart so they never merge with each other.
  double seg_size_mib{1.0};
  double gap_size_mib{0.25};
  std::size_t segs_per_group{8};

  /// Model knobs that are not part of tuning_params.
  std::size_t min_chunks_per_connection{2};
  double min_chunk_mib{8.0};

  /// Overrides: take the model's answer out of the loop for a manual sweep.
  /// Zero means "use the model".
  double override_chunk_mib{0.0};
  std::size_t override_connections{0};
  std::size_t override_reactors{0};

  /// Range-read calls per file.  Each call lands whole on one reactor, so with
  /// fewer files than reactors some reactors would sit idle; zero means "split
  /// each file into just enough calls to feed them all".
  std::size_t calls_per_file{0};

  /// Bound a merge by the GET size cap as well -- see @c coalesce_and_chunk.
  bool merge_within_chunk{false};

  /// Plan and print, read nothing.
  bool dry_run{false};
};

bool parse_autotune_arg(arg_parser const& p, int& i, autotune_options& o)
{
  return p.match(i, "--workload", o.workload_gib) || p.match(i, "--seg-size", o.seg_size_mib) ||
         p.match(i, "--gap-size", o.gap_size_mib) ||
         p.match(i, "--segs-per-group", o.segs_per_group) ||
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
         p.match(i, "--calls-per-file", o.calls_per_file) ||
         // Intercepted before parse_common_arg: for this benchmark the GET size
         // and the reactor count are outputs, so the flags that name them are
         // overrides rather than settings.
         p.match(i, "--chunk-size", o.override_chunk_mib) ||
         p.match(i, "--n-reactors", o.override_reactors) ||
         p.match(i, "--connections", o.override_connections) ||
         p.toggle(i, "--merge-within-chunk", o.merge_within_chunk) ||
         p.toggle(i, "--dry-run", o.dry_run);
}

// ---------------------------------------------------------------------------
// workload layout
// ---------------------------------------------------------------------------

/// One file's share of the workload: what the query wants, what that turns into
/// on the wire, and the host memory those requests land in.
struct file_plan {
  s3_file file;
  std::vector<at::byte_span> useful;
  at::request_plan io;
  /// One allocation per file, carved into a slice per request.  Value-init
  /// touches every page, so the timed loop never takes a first-touch fault.
  std::vector<std::uint8_t> buffer;
  std::vector<sirius::io::io_object_segment> segments;
  std::unique_ptr<sirius::io::sirius_datasource> ds;
};

struct workload {
  std::vector<file_plan> files;
  std::size_t n_segments{0};
  std::size_t n_blocks{0};
  std::size_t n_requests{0};
  std::size_t useful_bytes{0};
  std::size_t wire_bytes{0};
  std::size_t min_request{std::numeric_limits<std::size_t>::max()};
  std::size_t max_request{0};
  /// Segments that did not fit: the files are too small for the workload asked
  /// for, and every reported byte count is the smaller one actually laid out.
  std::size_t unplaced_segments{0};
};

workload build_workload(engine& eng,
                        std::vector<s3_file> const& files,
                        autotune_options const& opts,
                        at::connection_plan const& plan,
                        std::size_t seg,
                        std::size_t gap,
                        std::size_t n_total_segments)
{
  double const max_waste = 1.0 - std::clamp(opts.params.gap_efficiency, 0.0, 1.0);

  // Spread the segments over the files as evenly as each file can take.
  std::size_t const base = n_total_segments / files.size();
  std::size_t const rem  = n_total_segments % files.size();

  workload w;
  w.files.reserve(files.size());

  for (std::size_t i = 0; i < files.size(); ++i) {
    std::size_t const want = base + (i < rem ? 1 : 0);
    std::size_t const fit =
      std::min(want, at::capacity_segments(files[i].size_bytes, seg, gap, opts.segs_per_group));
    w.unplaced_segments += want - fit;
    if (fit == 0) { continue; }

    file_plan fp;
    fp.file   = files[i];
    fp.useful = at::layout_file(files[i].size_bytes, fit, seg, gap, opts.segs_per_group);
    fp.io     = at::coalesce_and_chunk(
      fp.useful, plan.rtt_bytes, max_waste, plan.chunk_bytes, opts.merge_within_chunk);
    if (fp.io.requests.empty()) { continue; }

    // One buffer per file, sliced per request.  The reactor writes the body of
    // each GET straight into its slice -- no cuCascade block, no staging copy.
    fp.buffer.assign(fp.io.wire_bytes, std::uint8_t{0});
    fp.segments.reserve(fp.io.requests.size());
    std::size_t cursor = 0;
    for (auto const& r : fp.io.requests) {
      fp.segments.emplace_back(r.offset, r.length, fp.buffer.data() + cursor);
      cursor += r.length;
      w.min_request = std::min(w.min_request, r.length);
      w.max_request = std::max(w.max_request, r.length);
    }

    fp.ds = eng.io_ctx().open_datasource(fp.file.path, fp.file.size_bytes);

    w.n_segments += fp.useful.size();
    w.n_blocks += fp.io.n_blocks;
    w.n_requests += fp.io.requests.size();
    w.useful_bytes += fp.io.useful_bytes;
    w.wire_bytes += fp.io.wire_bytes;
    w.files.push_back(std::move(fp));
  }

  if (w.min_request > w.max_request) { w.min_request = 0; }
  return w;
}

// ---------------------------------------------------------------------------
// timed loop
// ---------------------------------------------------------------------------

/// Contiguous batch size that cuts @p n segments into at most @p parts calls.
[[nodiscard]] std::size_t batch_size(std::size_t n, std::size_t parts)
{
  parts = std::clamp<std::size_t>(parts, 1, std::max<std::size_t>(n, 1));
  return (n + parts - 1) / parts;
}

[[nodiscard]] std::size_t count_calls(std::vector<file_plan> const& files,
                                      std::size_t calls_per_file)
{
  std::size_t n = 0;
  for (auto const& f : files) {
    std::size_t const per = batch_size(f.segments.size(), calls_per_file);
    n += (f.segments.size() + per - 1) / per;
  }
  return n;
}

/// Issue every file's requests as one host_read_ranges_async_io per batch, each
/// future counting down a latch on completion, then wait.  Submission is on the
/// main thread; the callbacks fire on reactor threads.
iteration_result run_once(engine& eng, std::vector<file_plan>& files, std::size_t calls_per_file)
{
  std::atomic<std::size_t> bytes_read{0};
  std::atomic<std::size_t> failures{0};
  std::latch done{static_cast<std::ptrdiff_t>(count_calls(files, calls_per_file))};

  auto const start = clock_type::now();

  for (auto& f : files) {
    std::size_t const per = batch_size(f.segments.size(), calls_per_file);
    for (std::size_t begin = 0; begin < f.segments.size(); begin += per) {
      std::span<sirius::io::io_object_segment> batch{f.segments.data() + begin,
                                                     std::min(per, f.segments.size() - begin)};
      auto fut = eng.io_ctx().host_read_ranges_async_io(f.ds->get_io_object(), batch);

      std::move(fut).install_callback(
        [&bytes_read, &failures, &done](sirius::exec::try_t<std::size_t>&& t) mutable {
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
          done.count_down();
        });
    }
  }

  done.wait();

  iteration_result r;
  r.duration_ms = ms_since(start);
  r.bytes       = bytes_read.load();

  if (failures.load() != 0) {
    std::cerr << failures.load() << " range-read call(s) failed — throughput is not meaningful\n";
  }
  return r;
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

void report_plan(autotune_options const& opts,
                 at::connection_plan const& plan,
                 std::size_t workload_bytes,
                 std::size_t seg,
                 std::size_t gap,
                 std::size_t conn_per_reactor,
                 std::size_t n_reactors,
                 std::size_t calls_per_file,
                 workload const& w)
{
  double const nic_mbps  = opts.params.nic_gbps * 125.0;
  double const max_waste = 1.0 - std::clamp(opts.params.gap_efficiency, 0.0, 1.0);

  std::cout << "s3_autotune_throughput_bench\n";
  for (std::size_t i = 0; i < opts.common.buckets.size(); ++i) {
    std::cout << std::format("  bucket[{}]           : {}\n", i, opts.common.buckets[i]);
  }
  std::cout << std::format("  files                : {} of {} requested\n",
                           w.files.size(),
                           opts.common.n_files)
            << std::format("  workload             : {:.2f} GiB requested\n",
                           as_gib(workload_bytes))
            << std::format("  segment / gap        : {:.2f} / {:.2f} MiB, {} per group\n",
                           as_mib(seg),
                           as_mib(gap),
                           opts.segs_per_group);

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

  std::cout << "  -- plan --\n"
            << std::format("  connections (nic)    : {}\n", plan.nic_connections)
            << std::format("  connections (batch)  : {}{}\n",
                           plan.active_connections,
                           opts.override_connections != 0 ? "  [override]" : "")
            << std::format("  conn / reactor       : {}\n", conn_per_reactor)
            << std::format("  reactors             : {}{} -> ceiling {} concurrent GETs\n",
                           n_reactors,
                           opts.override_reactors != 0 ? "  [override]" : "",
                           n_reactors * conn_per_reactor);

  double const achieved_gap_eff =
    w.wire_bytes == 0 ? 0.0
                      : static_cast<double>(w.useful_bytes) / static_cast<double>(w.wire_bytes);

  std::cout << "  -- workload as laid out --\n"
            << std::format("  segments             : {} ({:.2f} GiB useful)\n",
                           w.n_segments,
                           as_gib(w.useful_bytes))
            << std::format("  coalesced blocks     : {}{}\n",
                           w.n_blocks,
                           opts.merge_within_chunk ? "  [merge capped at GET size]" : "")
            << std::format(
                 "  requests (GETs)      : {}  ({:.2f} MiB min / {:.2f} avg / {:.2f} "
                 "max)\n",
                 w.n_requests,
                 as_mib(w.min_request),
                 w.n_requests == 0 ? 0.0 : as_mib(w.wire_bytes) / static_cast<double>(w.n_requests),
                 as_mib(w.max_request))
            << std::format(
                 "  wire bytes           : {:.2f} GiB (gap efficiency achieved "
                 "{:.1f}%)\n",
                 as_gib(w.wire_bytes),
                 achieved_gap_eff * 100.0)
            << std::format("  range-read calls     : {} ({} per file)\n",
                           count_calls(w.files, calls_per_file),
                           calls_per_file);

  if (w.unplaced_segments != 0) {
    std::cerr << std::format(
      "  WARNING: {} segment(s) did not fit — the files are too small for --workload. "
      "Every number above and below is for the {:.2f} GiB actually laid out; raise --n-files "
      "or lower --workload.\n",
      w.unplaced_segments,
      as_gib(w.useful_bytes));
  }
  if (w.files.size() < n_reactors) {
    std::cerr << std::format(
      "  NOTE: {} file(s) over {} reactors — each range-read call lands whole on one reactor, "
      "so --calls-per-file splits them to keep every reactor fed.\n",
      w.files.size(),
      n_reactors);
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

/// Hand back what the run measured in the units the model consumes, so the next
/// invocation can be given real numbers instead of guesses.
void report_feedback(sirius::io::rest::rest_perf_snapshot const& s, iteration_result const& last)
{
  double const mean_inflight = s.inflight_samples == 0 ? 0.0
                                                       : static_cast<double>(s.inflight_sum) /
                                                           static_cast<double>(s.inflight_samples);
  double const mean_ttfb_s =
    s.curl_timed_count == 0
      ? 0.0
      : static_cast<double>(s.curl_ttfb_ns_total) / static_cast<double>(s.curl_timed_count) / 1e9;

  std::cout << "  -- measured, for the next run --\n"
            << std::format(
                 "  in-flight GETs       : mean {:.1f} / max {}\n", mean_inflight, s.inflight_max);
  if (mean_ttfb_s > 0) {
    std::cout << std::format(
      "  ttfb                 : {:.2f} ms -> --rtt {:.4f}\n", mean_ttfb_s * 1e3, mean_ttfb_s);
  }
  if (mean_inflight > 0 && last.duration_ms > 0) {
    double const mb_per_s =
      (static_cast<double>(last.bytes) / at::bytes_per_mb_v) / (last.duration_ms / 1e3);
    std::cout << std::format("  per-stream rate      : {:.1f} MB/s -> --stream-throughput {:.1f}\n",
                             mb_per_s / mean_inflight,
                             mb_per_s / mean_inflight);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  autotune_options opts;

  try {
    arg_parser p{argc, argv};
    for (int i = 1; i < argc; ++i) {
      if (parse_autotune_arg(p, i, opts)) { continue; }
      if (parse_common_arg(p, i, opts.common)) { continue; }
      throw std::runtime_error(std::string{"unknown flag: "} + argv[i]);
    }
    if (opts.common.buckets.empty()) { throw std::runtime_error("--bucket is required"); }

    std::size_t const seg = mib_to_bytes(opts.seg_size_mib);
    std::size_t const gap = mib_to_bytes(opts.gap_size_mib);
    if (seg == 0) { throw std::runtime_error("--seg-size must be > 0"); }
    if (opts.segs_per_group == 0) { throw std::runtime_error("--segs-per-group must be > 0"); }

    auto const workload_bytes =
      static_cast<std::size_t>(opts.workload_gib * static_cast<double>(gib_v));
    if (workload_bytes < seg) { throw std::runtime_error("--workload is smaller than --seg-size"); }

    // ---- what the model says -------------------------------------------------
    auto plan = at::plan_connections(opts.params,
                                     workload_bytes,
                                     opts.min_chunks_per_connection,
                                     mib_to_bytes(opts.min_chunk_mib));
    if (opts.override_chunk_mib > 0) { plan.chunk_bytes = mib_to_bytes(opts.override_chunk_mib); }
    if (opts.override_connections > 0) { plan.active_connections = opts.override_connections; }
    if (plan.chunk_bytes == 0) { throw std::runtime_error("derived GET size is 0"); }

    std::size_t const conn_per_reactor = std::max<std::size_t>(1, opts.common.max_nconnection);
    std::size_t const n_reactors =
      opts.override_reactors > 0
        ? opts.override_reactors
        : std::max<std::size_t>(
            1, (plan.active_connections + conn_per_reactor - 1) / conn_per_reactor);

    // ---- turn that into an engine -------------------------------------------
    opts.common.n_reactors       = n_reactors;
    opts.common.max_nconnection  = conn_per_reactor;
    opts.common.chunk_size_bytes = plan.chunk_bytes;
    // The benchmark has already decided what each GET is; stop the reactor from
    // deciding again.  max_n_chunks 1 disables fusing file-adjacent requests
    // back together, and rest.chunk_size == our cap means none is split, so the
    // requests map one-to-one onto GETs.
    opts.common.max_n_chunks = 1;
    // Reads land in malloc'd buffers, so the pinned pool only has to cover the
    // reactors' bounce slots: one 1 MiB block per connection per reactor.
    opts.common.host_chunk_mib      = 1;
    std::size_t const bounce_blocks = n_reactors * conn_per_reactor;
    opts.common.host_initial_pools  = (bounce_blocks + host_pool_size_v - 1) / host_pool_size_v + 1;
    opts.common.perf_instrumentation = true;

    engine eng{opts.common};

    auto const files = get_files_from_prefixes(eng, opts.common.buckets, opts.common.n_files);

    std::size_t const n_total_segments = (workload_bytes + seg - 1) / seg;
    auto w = build_workload(eng, files, opts, plan, seg, gap, n_total_segments);
    if (w.files.empty()) {
      throw std::runtime_error(
        "no readable segments: the listed files are smaller than one group "
        "(--segs-per-group x --seg-size)");
    }

    // A range-read call is dispatched whole to one reactor, so fewer calls than
    // reactors leaves reactors idle and the measured concurrency below the plan.
    std::size_t const calls_per_file =
      opts.calls_per_file != 0
        ? opts.calls_per_file
        : std::max<std::size_t>(1, (n_reactors + w.files.size() - 1) / w.files.size());

    report_plan(
      opts, plan, workload_bytes, seg, gap, conn_per_reactor, n_reactors, calls_per_file, w);
    if (opts.dry_run) { return 0; }

    std::size_t const repeat = std::max<std::size_t>(1, opts.common.repeat);
    std::vector<iteration_result> runs;
    runs.reserve(repeat);
    std::string perf;
    for (std::size_t r = 0; r < repeat; ++r) {
      runs.push_back(run_once(eng, w.files, calls_per_file));
      // Snapshot before the report resets the counters, so both describe the
      // iteration that just finished rather than every iteration so far.
      auto const snapshot = eng.rest().perf_snapshot();
      perf                = eng.rest().perf_report_and_reset();
      if (r + 1 == repeat) {
        report_runs(runs, w.useful_bytes, opts.params.nic_gbps);
        if (!perf.empty()) { std::cout << perf; }
        report_feedback(snapshot, runs.back());
      }
    }
    return 0;

  } catch (std::exception const& e) {
    std::cerr << "s3_autotune_throughput_bench: " << e.what() << "\n";
    return 1;
  }
}
