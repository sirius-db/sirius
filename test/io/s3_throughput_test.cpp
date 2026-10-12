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

// s3_throughput_test — raw S3 read throughput via the Sirius REST reactor.
//
// For each selected file the benchmark picks random non-overlapping aligned
// slices of --chunk-size to satisfy the --per-file read budget. One batched
// host_readv_async_io call submits all logical slices for a file; the REST
// worker chooses physical GET sizes dynamically from backlog and free handles.
// Each future has an inline callback that counts down a latch, so no extra
// thread pool is required.
//
//   ./s3_throughput_test \
//       --bucket       s3://bucket/prefix   \
//       --n-files      4                    \
//       --per-file     1                    \  # GB per file
//       --chunk-size   1                    \  # MiB per GET
//       --max-connection 128                \
//       --n-reactors   1                    \
//       --repeat       3                    \
//       --config       sirius_s3.yml

#include "exec/semi_future.hpp"
#include "io/types.hpp"
#include "s3_bench_common.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <latch>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace sirius::bench;

// ---------------------------------------------------------------------------
// pinned staging
// ---------------------------------------------------------------------------

/// One pinned allocation carved into chunk_size blocks — one block per segment.
/// Allocated once before the timed loop and reused across iterations.
class pinned_staging {
 public:
  pinned_staging(std::size_t n_chunks, std::size_t chunk_bytes) : _chunk_bytes(chunk_bytes)
  {
    constexpr std::size_t blocks_per_slab = 128;
    const std::size_t slabs               = (n_chunks + blocks_per_slab - 1) / blocks_per_slab;
    const std::size_t capacity            = slabs * blocks_per_slab * chunk_bytes;

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

// ---------------------------------------------------------------------------
// segment selection
// ---------------------------------------------------------------------------

/// Pick @p n_chunks random non-overlapping chunk_bytes-aligned offsets from a
/// file of @p file_size bytes. Returns sorted offsets (better IO locality).
std::vector<std::size_t> pick_offsets(std::size_t file_size,
                                      std::size_t chunk_bytes,
                                      std::size_t n_chunks,
                                      std::mt19937& rng)
{
  const std::size_t available = file_size / chunk_bytes;
  n_chunks                    = std::min(n_chunks, available);
  if (n_chunks == 0) { return {}; }

  std::vector<std::size_t> indices(available);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);
  indices.resize(n_chunks);
  std::sort(indices.begin(), indices.end());

  std::vector<std::size_t> offsets;
  offsets.reserve(n_chunks);
  for (auto idx : indices) {
    offsets.push_back(idx * chunk_bytes);
  }
  return offsets;
}

// ---------------------------------------------------------------------------
// work list
// ---------------------------------------------------------------------------

struct file_work {
  std::unique_ptr<sirius::io::sirius_datasource> ds;
  std::vector<sirius::io::slice> slices;
  std::size_t total_bytes{0};
};

/// Open datasources and assign pinned staging blocks to each segment.
/// Fixed RNG seed so repeated runs read the same byte ranges.
std::vector<file_work> prepare_work(engine& eng,
                                    std::vector<s3_file> const& files,
                                    std::size_t per_file_bytes,
                                    pinned_staging& staging)
{
  std::mt19937 rng{42};
  const std::size_t chunk_bytes = staging.chunk_bytes();

  std::vector<file_work> work;
  work.reserve(files.size());

  std::size_t next_block = 0;
  for (auto const& f : files) {
    file_work w;
    w.ds = eng.io_ctx().open_datasource(f.path);

    const std::size_t n_chunks = per_file_bytes / chunk_bytes;
    auto offsets               = pick_offsets(f.size_bytes, chunk_bytes, n_chunks, rng);

    for (auto offset : offsets) {
      const std::size_t sz = std::min(chunk_bytes, f.size_bytes - offset);
      w.slices.emplace_back(offset, sz, staging.block(next_block++));
      w.total_bytes += sz;
    }
    if (!w.slices.empty()) { work.push_back(std::move(w)); }
  }
  return work;
}

std::size_t count_segments(std::vector<file_work> const& work)
{
  std::size_t n = 0;
  for (auto const& w : work) {
    n += w.slices.size();
  }
  return n;
}

// ---------------------------------------------------------------------------
// timed loop
// ---------------------------------------------------------------------------

/// Issue each file's logical slices in one host_readv_async_io call, attach an
/// inline callback to each file future that counts down a latch, then wait.
/// Reads are submitted from the main thread; callbacks fire on reactor threads.
iteration_result run_once(engine& eng, std::vector<file_work>& work)
{
  std::atomic<std::size_t> bytes_read{0};
  std::atomic<std::size_t> failures{0};
  std::latch done{static_cast<std::ptrdiff_t>(work.size())};

  const auto start = clock_type::now();

  for (auto& w : work) {
    auto fut = eng.io_ctx().host_readv_async_io(w.ds->get_io_object(), w.slices);

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

  done.wait();

  iteration_result r;
  r.duration_ms = ms_since(start);
  r.bytes       = bytes_read.load();

  if (failures.load() != 0) {
    std::cerr << failures.load() << " file batch(es) failed — throughput is not meaningful\n";
  }
  return r;
}

}  // namespace

int main(int argc, char** argv)
{
  bench_options opts;

  try {
    arg_parser p{argc, argv};
    for (int i = 1; i < argc; ++i) {
      if (parse_common_arg(p, i, opts)) { continue; }
      throw std::runtime_error(std::string{"unknown flag: "} + argv[i]);
    }
    if (opts.buckets.empty()) { throw std::runtime_error("--bucket is required"); }

    engine eng{opts};

    const auto files = get_files_from_prefixes(eng, opts.buckets, opts.n_files);

    const std::size_t per_file_bytes = opts.per_file_bytes();
    const std::size_t chunk_bytes    = opts.chunk_bytes();

    std::size_t total_chunks = 0;
    for (auto const& f : files) {
      total_chunks += std::min(per_file_bytes / chunk_bytes, f.size_bytes / chunk_bytes);
    }
    if (total_chunks == 0) {
      throw std::runtime_error(
        "no chunks to read: files are smaller than --chunk-size or --per-file is 0");
    }

    pinned_staging staging{total_chunks, chunk_bytes};
    auto work = prepare_work(eng, files, per_file_bytes, staging);

    std::size_t total_bytes = 0;
    for (auto const& w : work) {
      total_bytes += w.total_bytes;
    }

    std::cout << "s3_throughput_test\n";
    for (std::size_t i = 0; i < opts.buckets.size(); ++i) {
      std::cout << "  bucket[" << i << "]     : " << opts.buckets[i] << "\n";
    }
    std::cout << "  files         : " << files.size() << "\n"
              << "  per-file      : " << opts.per_file_gib << " GB\n"
              << "  chunk-size    : " << opts.chunk_size_mib << " MiB logical slices\n"
              << "  max-connection: " << opts.max_nconnection << "\n"
              << "  n-reactors    : " << opts.n_reactors << "\n"
              << "  total chunks  : " << count_segments(work) << "\n"
              << "  total data    : " << static_cast<double>(total_bytes) / 1e9 << " GB\n\n";

    std::vector<iteration_result> runs;
    runs.reserve(opts.repeat);
    for (std::size_t r = 0; r < opts.repeat; ++r) {
      runs.push_back(run_once(eng, work));
    }
    report("s3", runs);
    return 0;

  } catch (std::exception const& e) {
    std::cerr << "s3_throughput_test: " << e.what() << "\n";
    return 1;
  }
}
