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

// Stream-lineage validation for the integrated stream-safety fixes ("item 5 is
// unnecessary" claim):
//
//   - gpu_table_representation::release_table(stream) rebinds converter-produced
//     buffers to the caller's stream (cucascade item 4), so the Sirius steal path
//     (scan_operator_input::prepare_for_processing) frees them in the consumer
//     stream's order instead of on the idle conversion-pool stream.
//   - The data_batch consumer-event API (record_consumer_event / await_consumers /
//     consumers_done) orders batch-managed reclaims (install_converted_representation,
//     set_data) after cross-stream reads recorded by consumers.
//
// These tests exercise the exact hazard convert_host_fast_to_gpu used to expose:
// converted GPU buffers are allocated on a memory-space pool stream; if their
// eventual free retires on that idle foreign stream while a consumer stream still
// has in-flight reads, the cudaMallocAsync pool can hand the block to the next
// allocation mid-read (use-after-free / corruption). The "sensitivity" test
// re-creates the pre-fix binding deliberately and proves the harness detects the
// corruption; the fixed paths must then be deterministically clean.

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace {

//===----------------------------------------------------------------------===//
// Fixture: single-GPU memory manager + converter registry (modeled on
// test/cpp/pipeline/test_batch_lock_utils.cpp's batch_lock_utils_fixture).
//===----------------------------------------------------------------------===//
struct stream_lineage_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu0  = nullptr;
  cucascade::memory::memory_space* host0 = nullptr;

  bool setup()
  {
    sirius::converter_registry::reset_for_testing();
    try {
      cucascade::memory::reservation_manager_configurator builder;
      builder.set_number_of_gpus(1)
        .set_gpu_usage_limit(512ULL << 20)
        .set_reservation_fraction_per_gpu(0.75)
        .set_per_numa_region_capacity(1ULL << 30)
        .use_numa_id_as_host_id()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_numa_region(0.75);
      auto space_configs = builder.build();
      manager            = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
        std::move(space_configs));
    } catch (const std::exception&) {
      return false;
    }
    sirius::converter_registry::initialize();

    auto gpu_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    if (gpu_spaces.empty()) { return false; }
    gpu0             = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
    auto host_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (host_spaces.empty()) { return false; }
    host0 = const_cast<cucascade::memory::memory_space*>(host_spaces[0]);
    return true;
  }

  ~stream_lineage_fixture()
  {
    if (manager) {
      manager->shutdown();
      sirius::converter_registry::shutdown();
    }
  }
};

/// INT64 column holding value(i) = seed + i, allocated from `space` on `stream`.
/// When `with_nulls` is set, every 5th row is null (payload still written).
std::unique_ptr<cudf::column> make_patterned_column(std::size_t num_rows,
                                                    std::int64_t seed,
                                                    bool with_nulls,
                                                    cucascade::memory::memory_space& space,
                                                    rmm::cuda_stream_view stream)
{
  auto mr = space.get_default_allocator();

  std::vector<std::int64_t> host_vals(num_rows);
  for (std::size_t i = 0; i < num_rows; ++i) {
    host_vals[i] = seed + static_cast<std::int64_t>(i);
  }
  rmm::device_buffer data{num_rows * sizeof(std::int64_t), stream, mr};
  cudaMemcpyAsync(data.data(),
                  host_vals.data(),
                  num_rows * sizeof(std::int64_t),
                  cudaMemcpyHostToDevice,
                  stream.value());

  rmm::device_buffer mask{};
  cudf::size_type null_count = 0;
  if (with_nulls) {
    std::size_t const num_words = (num_rows + 31) / 32;
    std::vector<std::uint32_t> words(num_words, ~std::uint32_t{0});
    for (std::size_t i = 0; i < num_rows; i += 5) {
      words[i / 32] &= ~(std::uint32_t{1} << (i % 32));
      ++null_count;
    }
    mask = rmm::device_buffer{num_words * sizeof(std::uint32_t), stream, mr};
    cudaMemcpyAsync(mask.data(),
                    words.data(),
                    num_words * sizeof(std::uint32_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
  }
  stream.synchronize();
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT64},
                                        static_cast<cudf::size_type>(num_rows),
                                        std::move(data),
                                        std::move(mask),
                                        null_count);
}

/// STRING column of `num_rows` four-byte strings, allocated from `space` on `stream`.
std::unique_ptr<cudf::column> make_strings_column_patterned(
  std::size_t num_rows, cucascade::memory::memory_space& space, rmm::cuda_stream_view stream)
{
  auto mr = space.get_default_allocator();

  std::vector<std::int32_t> offsets(num_rows + 1);
  std::vector<char> chars(num_rows * 4);
  for (std::size_t i = 0; i <= num_rows; ++i) {
    offsets[i] = static_cast<std::int32_t>(4 * i);
  }
  for (std::size_t i = 0; i < num_rows; ++i) {
    char const c = static_cast<char>('a' + (i % 26));
    chars[4 * i] = chars[4 * i + 1] = chars[4 * i + 2] = chars[4 * i + 3] = c;
  }

  rmm::device_buffer offsets_buf{offsets.size() * sizeof(std::int32_t), stream, mr};
  cudaMemcpyAsync(offsets_buf.data(),
                  offsets.data(),
                  offsets.size() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    static_cast<cudf::size_type>(num_rows + 1),
                                                    std::move(offsets_buf),
                                                    rmm::device_buffer{},
                                                    0);

  rmm::device_buffer chars_buf{chars.size(), stream, mr};
  cudaMemcpyAsync(
    chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice, stream.value());
  stream.synchronize();

  return cudf::make_strings_column(static_cast<cudf::size_type>(num_rows),
                                   std::move(offsets_col),
                                   std::move(chars_buf),
                                   0,
                                   rmm::device_buffer{});
}

/// GPU batch spilled in place to host_data_representation (spilled-cache shape).
/// Converting it back to gpu_table_representation goes through
/// convert_host_fast_to_gpu, which allocates the reconstructed buffers on one of
/// the GPU memory space's pool streams — the exact binding item 5 targeted.
std::shared_ptr<cucascade::data_batch> make_host_batch(stream_lineage_fixture& f,
                                                       std::size_t num_rows,
                                                       std::int64_t seed,
                                                       bool with_nulls,
                                                       bool with_strings,
                                                       rmm::cuda_stream_view stream)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_patterned_column(num_rows, seed, with_nulls, *f.gpu0, stream));
  if (with_strings) { cols.push_back(make_strings_column_patterned(num_rows, *f.gpu0, stream)); }
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto batch = sirius::make_data_batch(
    std::move(table), *f.gpu0, stream, sirius::telemetry::batch_telemetry_info{});
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), f.host0, stream);
  }
  stream.synchronize();
  return batch;
}

/// Convert a host batch back to the GPU tier in place (convert_host_fast_to_gpu).
void upload_to_gpu(stream_lineage_fixture& f,
                   const std::shared_ptr<cucascade::data_batch>& batch,
                   rmm::cuda_stream_view stream)
{
  auto mut = batch->to_mutable();
  mut.convert_to<cucascade::gpu_table_representation>(
    sirius::converter_registry::get(), f.gpu0, stream);
}

/// Mirror of the Sirius steal (sirius_gpu_scan_operator_data.cpp): release the
/// owned table to `stream` and leave a valid empty placeholder in the batch.
std::unique_ptr<cudf::table> steal_table(const std::shared_ptr<cucascade::data_batch>& batch,
                                         cucascade::memory::memory_space& space,
                                         rmm::cuda_stream_view stream)
{
  auto mut      = batch->to_mutable();
  auto* gpu_rep = dynamic_cast<cucascade::gpu_table_representation*>(mut.get_data());
  REQUIRE(gpu_rep != nullptr);
  auto stolen = gpu_rep->release_table(stream);
  mut.set_data(std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(), space, rmm::cuda_stream_view{}));
  return stolen;
}

/// Enqueue ~2 GB of memset traffic on `stream` so work enqueued after it is
/// guaranteed to still be pending when the host regains control.
void enqueue_blockers(void* scratch, std::size_t scratch_bytes, rmm::cuda_stream_view stream)
{
  for (int i = 0; i < 32; ++i) {
    cudaMemsetAsync(scratch, 0, scratch_bytes, stream.value());
  }
}

/// Force conversion-pool churn: build small host batches and convert them to the
/// GPU (each conversion acquires a pool stream and allocates from the async pool,
/// grabbing any block whose free has already retired).
void hammer_conversions(stream_lineage_fixture& f,
                        rmm::cuda_stream_view stream,
                        int count,
                        std::size_t rows)
{
  for (int i = 0; i < count; ++i) {
    auto hb = make_host_batch(f, rows, /*seed=*/0x5A5A5A5A + i, false, false, stream);
    upload_to_gpu(f, hb, stream);
    // Dropping `hb` here destroys the freshly converted representation and
    // returns its blocks to the pool, maximizing recycling pressure.
  }
}

constexpr std::size_t kRows       = 1u << 20;               // 1M rows -> 8 MB INT64 payload
constexpr std::size_t kBytes      = kRows * sizeof(std::int64_t);
constexpr std::size_t kScratchMiB = 64;

}  // namespace

//===----------------------------------------------------------------------===//
// 1. release_table must rebind converter-produced buffers to the caller's stream
//===----------------------------------------------------------------------===//
TEST_CASE("release_table rebinds converted buffers (data, null mask, nested) to caller stream",
          "[stream_lineage]")
{
  stream_lineage_fixture f;
  REQUIRE(f.setup());

  rmm::cuda_stream setup_stream;
  auto batch =
    make_host_batch(f, 4096, /*seed=*/7, /*with_nulls=*/true, /*with_strings=*/true, setup_stream);
  upload_to_gpu(f, batch, setup_stream);
  setup_stream.synchronize();

  // Created AFTER the conversion allocated the GPU buffers, so no buffer can be
  // "accidentally" born bound to it: any binding observed below must come from
  // release_table's rebind.
  rmm::cuda_stream task_stream;
  auto stolen = steal_table(batch, *f.gpu0, task_stream.view());
  REQUIRE(stolen != nullptr);
  REQUIRE(stolen->num_columns() == 2);

  auto cols = stolen->release();

  // INT64 column: data buffer and null mask must deallocate on task_stream.
  {
    auto contents = cols[0]->release();
    REQUIRE(contents.data != nullptr);
    CHECK(contents.data->stream().value() == task_stream.value());
    REQUIRE(contents.null_mask != nullptr);
    REQUIRE(contents.null_mask->size() > 0);
    CHECK(contents.null_mask->stream().value() == task_stream.value());
  }

  // STRING column (nested): chars payload and the offsets child's data buffer
  // must both deallocate on task_stream (cudf::rebind_stream recurses).
  {
    auto contents = cols[1]->release();
    REQUIRE(contents.data != nullptr);
    REQUIRE(contents.data->size() > 0);
    CHECK(contents.data->stream().value() == task_stream.value());
    REQUIRE(!contents.children.empty());
    auto offsets_contents = contents.children[0]->release();
    REQUIRE(offsets_contents.data != nullptr);
    CHECK(offsets_contents.data->stream().value() == task_stream.value());
  }
}

//===----------------------------------------------------------------------===//
// 2. Steal-path UAF stress: mid-flight column destruction under pool churn
//===----------------------------------------------------------------------===//
TEST_CASE("stolen table tolerates mid-flight column destruction under conversion pressure",
          "[stream_lineage]")
{
  stream_lineage_fixture f;
  REQUIRE(f.setup());

  void* scratch    = nullptr;
  void* result_dev = nullptr;
  REQUIRE(cudaMalloc(&scratch, kScratchMiB << 20) == cudaSuccess);
  REQUIRE(cudaMalloc(&result_dev, kBytes) == cudaSuccess);

  rmm::cuda_stream hammer_stream;
  std::vector<std::int64_t> host_result(kRows);

  for (int iter = 0; iter < 20; ++iter) {
    std::int64_t const seed = 1000000LL * (iter + 1);
    rmm::cuda_stream task_stream;

    auto batch = make_host_batch(f, kRows, seed, false, false, task_stream.view());
    upload_to_gpu(f, batch, task_stream.view());
    auto stolen = steal_table(batch, *f.gpu0, task_stream.view());
    REQUIRE(stolen != nullptr);

    // Enqueue a long blocker, then the read: the D2D copy of the stolen column
    // is guaranteed to still be pending when the host destroys the column.
    enqueue_blockers(scratch, kScratchMiB << 20, task_stream.view());
    void const* src = stolen->view().column(0).head();
    cudaMemcpyAsync(result_dev, src, kBytes, cudaMemcpyDeviceToDevice, task_stream.value());

    // Destroy the column mid-flight. With release_table's rebind the free is
    // enqueued on task_stream BEHIND the read; without it (pre-fix) it would
    // retire instantly on the idle conversion-pool stream and the block could
    // be recycled by the conversions below while the read is still pending.
    {
      auto cols = stolen->release();
      cols[0].reset();
    }

    // Hammer the conversion pool while the read is still blocked.
    hammer_conversions(f, hammer_stream.view(), 3, kRows / 4);

    cudaMemcpyAsync(
      host_result.data(), result_dev, kBytes, cudaMemcpyDeviceToHost, task_stream.value());
    task_stream.synchronize();

    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < kRows; ++i) {
      if (host_result[i] != seed + static_cast<std::int64_t>(i)) { ++mismatches; }
    }
    INFO("iteration " << iter << ": " << mismatches << " corrupted elements of " << kRows);
    REQUIRE(mismatches == 0);
  }

  cudaFree(scratch);
  cudaFree(result_dev);
}

//===----------------------------------------------------------------------===//
// 3. Sensitivity check: the pre-fix binding (free on an idle foreign stream)
//    corrupts the very read the fixed path protects.
//===----------------------------------------------------------------------===//
TEST_CASE("rebinding a stolen column back to an idle foreign stream reproduces pre-fix corruption",
          "[stream_lineage]")
{
  stream_lineage_fixture f;
  REQUIRE(f.setup());

  void* scratch    = nullptr;
  void* result_dev = nullptr;
  REQUIRE(cudaMalloc(&scratch, kScratchMiB << 20) == cudaSuccess);
  REQUIRE(cudaMalloc(&result_dev, kBytes) == cudaSuccess);

  std::int64_t const seed = 42;
  rmm::cuda_stream task_stream;
  rmm::cuda_stream foreign_stream;  // stands in for the idle conversion-pool stream

  auto batch = make_host_batch(f, kRows, seed, false, false, task_stream.view());
  upload_to_gpu(f, batch, task_stream.view());
  auto stolen = steal_table(batch, *f.gpu0, task_stream.view());

  // Undo item 4's rebind: bind the column's buffers back to an idle foreign
  // stream, exactly the binding release_table produced before the fix.
  auto cols = stolen->release();
  cols[0]   = cudf::rebind_stream(std::move(*cols[0]), foreign_stream.view());

  void const* src = cols[0]->view().head();
  enqueue_blockers(scratch, kScratchMiB << 20, task_stream.view());
  cudaMemcpyAsync(result_dev, src, kBytes, cudaMemcpyDeviceToDevice, task_stream.value());

  // Destroy mid-flight: the free retires immediately on the idle foreign stream.
  cols[0].reset();

  // The next same-stream allocation of the same size grabs the freed block and
  // poisons it while the read above is still stuck behind the blocker.
  rmm::device_buffer reuse{kBytes, foreign_stream.view(), f.gpu0->get_default_allocator()};
  bool const block_reused = reuse.data() == src;
  if (block_reused) {
    cudaMemsetAsync(reuse.data(), 0xFF, kBytes, foreign_stream.value());
    foreign_stream.synchronize();
  }

  std::vector<std::int64_t> host_result(kRows);
  cudaMemcpyAsync(
    host_result.data(), result_dev, kBytes, cudaMemcpyDeviceToHost, task_stream.value());
  task_stream.synchronize();

  std::size_t corrupted = 0;
  for (std::size_t i = 0; i < kRows; ++i) {
    if (host_result[i] != seed + static_cast<std::int64_t>(i)) { ++corrupted; }
  }

  if (block_reused) {
    // The harness must be able to see the pre-fix failure mode, otherwise the
    // clean runs in the other tests prove nothing.
    INFO("corrupted elements with pre-fix binding: " << corrupted << " of " << kRows);
    REQUIRE(corrupted > 0);
  } else {
    WARN("async pool did not recycle the freed block in place (reuse="
         << reuse.data() << " src=" << src << ") — sensitivity check inconclusive this run");
  }

  cudaFree(scratch);
  cudaFree(result_dev);
}

//===----------------------------------------------------------------------===//
// 4. Consumer events order a convert_to reclaim after in-flight reads
//===----------------------------------------------------------------------===//
TEST_CASE("consumer events order install_converted_representation after in-flight reads",
          "[stream_lineage]")
{
  stream_lineage_fixture f;
  REQUIRE(f.setup());

  void* scratch    = nullptr;
  void* result_dev = nullptr;
  REQUIRE(cudaMalloc(&scratch, kScratchMiB << 20) == cudaSuccess);
  REQUIRE(cudaMalloc(&result_dev, kBytes) == cudaSuccess);

  std::int64_t const seed = 77;
  rmm::cuda_stream setup_stream;
  rmm::cuda_stream reader_stream;
  rmm::cuda_stream reclaim_stream;

  auto batch = make_host_batch(f, kRows, seed, false, false, setup_stream.view());
  upload_to_gpu(f, batch, setup_stream.view());
  setup_stream.synchronize();

  // Reader: enqueue a slow read of the converted buffers on its own stream and
  // record a consumer event AFTER the reads are enqueued (the API contract).
  {
    auto ro         = batch->to_read_only();
    auto view       = sirius::get_cudf_table_view(ro);
    void const* src = view.column(0).head();
    enqueue_blockers(scratch, kScratchMiB << 20, reader_stream.view());
    cudaMemcpyAsync(result_dev, src, kBytes, cudaMemcpyDeviceToDevice, reader_stream.value());
    ro.record_consumer_event(reader_stream.view());
    REQUIRE_FALSE(batch->consumers_done());
  }  // shared lock dropped with the read still in flight — the pre-fix hazard window

  // Reclaimer: mirror the Sirius downgrade discipline (convertible_data_batch.hpp):
  // rebind to the reclaim stream, then convert GPU -> host in place. With the
  // consumer-event hooks, install_converted_representation must not destroy the
  // old GPU representation until the recorded reader work has completed.
  {
    auto mut = batch->to_mutable();
    mut.rebind_stream(reclaim_stream.view());
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), f.host0, reclaim_stream.view());
    // The reclaim host-synced behind await_consumers: the reader's recorded
    // reads must be complete by now.
    CHECK(cudaStreamQuery(reader_stream.value()) == cudaSuccess);
    REQUIRE(batch->consumers_done());
  }

  // Churn the pool: if the reclaim had freed the GPU buffers early these
  // allocations would recycle them under the (by then already finished) read.
  hammer_conversions(f, reclaim_stream.view(), 3, kRows / 4);

  std::vector<std::int64_t> host_result(kRows);
  cudaMemcpyAsync(
    host_result.data(), result_dev, kBytes, cudaMemcpyDeviceToHost, reader_stream.value());
  reader_stream.synchronize();

  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < kRows; ++i) {
    if (host_result[i] != seed + static_cast<std::int64_t>(i)) { ++mismatches; }
  }
  INFO("corrupted elements: " << mismatches << " of " << kRows);
  REQUIRE(mismatches == 0);

  cudaFree(scratch);
  cudaFree(result_dev);
}

//===----------------------------------------------------------------------===//
// 5. set_data blocks on recorded consumer events before dropping the old data
//===----------------------------------------------------------------------===//
TEST_CASE("set_data waits for recorded consumer reads before replacing the representation",
          "[stream_lineage]")
{
  stream_lineage_fixture f;
  REQUIRE(f.setup());

  void* scratch    = nullptr;
  void* result_dev = nullptr;
  REQUIRE(cudaMalloc(&scratch, kScratchMiB << 20) == cudaSuccess);
  REQUIRE(cudaMalloc(&result_dev, kBytes) == cudaSuccess);

  std::int64_t const seed = 99;
  rmm::cuda_stream setup_stream;
  rmm::cuda_stream reader_stream;

  auto batch = make_host_batch(f, kRows, seed, false, false, setup_stream.view());
  upload_to_gpu(f, batch, setup_stream.view());
  setup_stream.synchronize();

  {
    auto ro         = batch->to_read_only();
    auto view       = sirius::get_cudf_table_view(ro);
    void const* src = view.column(0).head();
    enqueue_blockers(scratch, kScratchMiB << 20, reader_stream.view());
    cudaMemcpyAsync(result_dev, src, kBytes, cudaMemcpyDeviceToDevice, reader_stream.value());
    ro.record_consumer_event(reader_stream.view());
    REQUIRE_FALSE(batch->consumers_done());
  }

  // set_data destroys the old GPU representation (buffers still bound to the
  // conversion-pool stream — no rebind happens on this path). The consumer-event
  // hook must host-block until the reader's recorded reads complete.
  {
    auto mut = batch->to_mutable();
    mut.set_data(std::make_unique<cucascade::gpu_table_representation>(
      std::make_unique<cudf::table>(), *f.gpu0, rmm::cuda_stream_view{}));
    CHECK(cudaStreamQuery(reader_stream.value()) == cudaSuccess);
    REQUIRE(batch->consumers_done());
  }

  hammer_conversions(f, setup_stream.view(), 3, kRows / 4);

  std::vector<std::int64_t> host_result(kRows);
  cudaMemcpyAsync(
    host_result.data(), result_dev, kBytes, cudaMemcpyDeviceToHost, reader_stream.value());
  reader_stream.synchronize();

  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < kRows; ++i) {
    if (host_result[i] != seed + static_cast<std::int64_t>(i)) { ++mismatches; }
  }
  INFO("corrupted elements: " << mismatches << " of " << kRows);
  REQUIRE(mismatches == 0);

  cudaFree(scratch);
  cudaFree(result_dev);
}
