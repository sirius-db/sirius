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

// s3_throughput_test — raw read throughput from S3 through sirius_datasource.
//
// No parquet decoding: a hybrid_scan_reader is used only to learn WHICH bytes a
// real scan would read (every column chunk of every row group), and then those
// byte ranges are pulled into pinned host memory as fast as the REST reactor
// manages.  What is measured is the object store, the reactor and the link.
//
//   ./s3_throughput_test --bucket s3://bucket/prefix --n-files 12 --repeat 1 \
//       --n_threads 2 --max_segment 32 --max_nconnection 128 --chunk_size 1 \
//       --mode [grouped|chunked] --config sirius.yml
//
// Modes:
//   grouped   one vectored read per file: every segment handed to
//             host_read_ranges_async_io in a single dispatch, which is what
//             lets the reactor fuse file-adjacent segments into scatter GETs.
//   chunked   one host_read_async per segment, so each is its own request.
//
// The two differ only in how the SAME segments are submitted, so the delta is
// the value of batching at the reactor boundary.

#include "s3_bench_common.hpp"

#include "exec/scoped_dispatcher.hpp"
#include "io/types.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <latch>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace sirius::bench;

enum class read_mode { grouped, chunked };

read_mode parse_mode(std::string const& s)
{
  if (s == "grouped") { return read_mode::grouped; }
  if (s == "chunked") { return read_mode::chunked; }
  throw std::runtime_error("--mode must be grouped or chunked, got: " + s);
}

// ---------------------------------------------------------------------------
// staging
// ---------------------------------------------------------------------------

/// One pinned allocation covering every chunk of every file, carved into
/// chunk_size blocks.  Allocated once and reused across iterations so the
/// measurement is the read, not the allocator.
class pinned_staging {
 public:
  pinned_staging(std::size_t n_chunks, std::size_t chunk_bytes) : _chunk_bytes(chunk_bytes)
  {
    constexpr std::size_t blocks_per_slab = 128;
    const std::size_t slabs     = (n_chunks + blocks_per_slab - 1) / blocks_per_slab;
    const std::size_t capacity  = slabs * blocks_per_slab * chunk_bytes;

    _upstream =
      std::make_unique<cucascade::memory::numa_region_pinned_host_memory_resource>(0, true);
    _mr = std::make_unique<cucascade::memory::fixed_size_host_memory_resource>(
      0, *_upstream, capacity, capacity, chunk_bytes, blocks_per_slab, slabs);

    _alloc  = _mr->allocate_multiple_blocks(n_chunks * chunk_bytes);
    auto bs = _alloc->get_blocks();
    _blocks.assign(bs.begin(), bs.end());
    if (_blocks.size() < n_chunks) {
      throw std::runtime_error("pinned pool returned fewer blocks than requested");
    }
  }

  [[nodiscard]] std::uint8_t* block(std::size_t i) const
  {
    return reinterpret_cast<std::uint8_t*>(_blocks[i]);
  }

  [[nodiscard]] std::size_t chunk_bytes() const noexcept { return _chunk_bytes; }

 private:
  std::size_t _chunk_bytes;
  std::unique_ptr<cucascade::memory::numa_region_pinned_host_memory_resource> _upstream;
  std::unique_ptr<cucascade::memory::fixed_size_host_memory_resource> _mr;
  cucascade::memory::fixed_multiple_blocks_allocation _alloc;
  std::vector<std::byte*> _blocks;
};

/// A file's worth of IO: its io_object plus every segment to read, each already
/// pointing at its own pinned destination.
struct io_work {
  const sirius::io::io_object* obj{nullptr};
  std::vector<sirius::io::io_object_segment> segments;
  std::size_t bytes{0};
};

/// Split each column-chunk range into chunk_size pieces and bind every piece to
/// a pinned block.  Done once, outside the timed region.
std::vector<io_work> prepare_for_benchmark(std::vector<file_ranges> const& files,
                                           pinned_staging& staging)
{
  std::vector<io_work> work;
  work.reserve(files.size());

  std::size_t next_block = 0;
  for (auto const& f : files) {
    io_work w;
    w.obj = &f.ds->get_io_object();
    for (auto const& r : f.ranges) {
      const auto offset = static_cast<std::size_t>(r.offset());
      const auto size   = static_cast<std::size_t>(r.size());
      for (std::size_t done = 0; done < size; done += staging.chunk_bytes()) {
        const std::size_t piece = std::min(staging.chunk_bytes(), size - done);
        w.segments.emplace_back(offset + done, piece, staging.block(next_block++));
        w.bytes += piece;
      }
    }
    work.push_back(std::move(w));
  }
  return work;
}

/// Total number of chunk_size pieces the ranges split into.
std::size_t count_chunks(std::vector<file_ranges> const& files, std::size_t chunk_bytes)
{
  std::size_t n = 0;
  for (auto const& f : files) {
    for (auto const& r : f.ranges) {
      const auto size = static_cast<std::size_t>(r.size());
      n += (size + chunk_bytes - 1) / chunk_bytes;
    }
  }
  return n;
}

// ---------------------------------------------------------------------------
// the timed loop
// ---------------------------------------------------------------------------

iteration_result run_once(engine& eng, std::vector<io_work>& work, read_mode mode)
{
  sirius::exec::scoped_dispatcher disp(eng.pool());

  std::atomic<std::size_t> bytes_read{0};
  std::atomic<std::size_t> failures{0};
  std::latch done{static_cast<std::ptrdiff_t>(work.size())};

  const auto start = clock_type::now();

  for (auto& w : work) {
    disp.enqueue([&eng, &w, &bytes_read, &failures, &done, mode]() {
      try {
        if (mode == read_mode::grouped) {
          // One dispatch for the whole file: the reactor fuses file-adjacent
          // segments into scatter GETs up to rest.chunk_size / max_n_chunks.
          auto fut = eng.io_ctx().host_read_ranges_async_io(*w.obj, w.segments);
          bytes_read.fetch_add(std::move(fut).get(), std::memory_order_relaxed);
        } else {
          // One request per segment.  Issued up front, then joined, so the
          // difference from `grouped` is batching and not concurrency.
          std::vector<sirius::exec::semi_future<std::size_t>> futs;
          futs.reserve(w.segments.size());
          for (auto const& seg : w.segments) {
            futs.push_back(eng.io_ctx().host_read_async_io(
              *w.obj, seg.offset, seg.size, seg.data()));
          }
          for (auto& f : futs) {
            bytes_read.fetch_add(std::move(f).get(), std::memory_order_relaxed);
          }
        }
      } catch (std::exception const& e) {
        failures.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "read failed: " << e.what() << "\n";
      }
      done.count_down();
    });
  }

  done.wait();
  disp.wait_for_all();

  iteration_result r;
  r.duration_ms = ms_since(start);
  r.bytes       = bytes_read.load();

  if (failures.load() != 0) {
    std::cerr << failures.load() << " file(s) failed -- throughput below is not meaningful\n";
  }
  return r;
}

}  // namespace

int main(int argc, char** argv)
{
  bench_options opts;
  opts.mode = "grouped";

  try {
    arg_parser p{argc, argv};
    for (int i = 1; i < argc; ++i) {
      if (parse_common_arg(p, i, opts)) { continue; }
      throw std::runtime_error(std::string{"unknown flag: "} + argv[i]);
    }
    if (opts.bucket.empty()) { throw std::runtime_error("--bucket is required"); }
    const auto mode = parse_mode(opts.mode);

    engine eng{opts};

    auto const paths = get_files_from_bucket(eng, opts.bucket, opts.n_files);
    auto const files = use_hybrid_scan_to_get_column_chunks(eng, paths);

    const std::size_t n_chunks = count_chunks(files, opts.chunk_bytes());
    std::size_t total_bytes    = 0;
    for (auto const& f : files) {
      total_bytes += f.total_bytes();
    }

    std::cout << "s3_throughput_test\n"
              << "  bucket          : " << opts.bucket << "\n"
              << "  files           : " << files.size() << "\n"
              << "  mode            : " << opts.mode << "\n"
              << "  threads         : " << opts.n_threads << "\n"
              << "  max_segment     : " << opts.max_segment << " (rest.max_n_chunks)\n"
              << "  max_nconnection : " << opts.max_nconnection << " (rest.max_connections)\n"
              << "  chunk_size      : " << opts.chunk_size_mib << " MiB\n"
              << "  column chunks   : " << n_chunks << " pieces, "
              << static_cast<double>(total_bytes) / 1e9 << " GB\n\n";

    pinned_staging staging{n_chunks, opts.chunk_bytes()};
    auto work = prepare_for_benchmark(files, staging);

    std::vector<iteration_result> runs;
    runs.reserve(opts.repeat);
    for (std::size_t r = 0; r < opts.repeat; ++r) {
      runs.push_back(run_once(eng, work, mode));
    }
    report(opts.mode, runs);
    return 0;

  } catch (std::exception const& e) {
    std::cerr << "s3_throughput_test: " << e.what() << "\n";
    return 1;
  }
}
