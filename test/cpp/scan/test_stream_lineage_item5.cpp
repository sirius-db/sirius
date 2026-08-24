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

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/owning_table_view.hpp"

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
        .set_per_host_capacity(1ULL << 30)
        .use_host_per_numa()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_host(0.75);
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

std::unique_ptr<cudf::column> make_strings_column_patterned(std::size_t num_rows,
                                                            cucascade::memory::memory_space& space,
                                                            rmm::cuda_stream_view stream)
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

/// Host-resident batch; converting it back to GPU tier allocates the rebuilt
/// buffers on a memory-space pool stream — the foreign-stream binding under test.
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

void upload_to_gpu(stream_lineage_fixture& f,
                   const std::shared_ptr<cucascade::data_batch>& batch,
                   rmm::cuda_stream_view stream)
{
  auto mut = batch->to_mutable();
  mut.convert_to<cucascade::gpu_table_representation>(
    sirius::converter_registry::get(), f.gpu0, stream);
}

/// Mirrors the production steal path (sirius_gpu_scan_operator_data.cpp).
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

/// Enough traffic that work enqueued after it is still pending when the host regains control.
void enqueue_blockers(void* scratch, std::size_t scratch_bytes, rmm::cuda_stream_view stream)
{
  for (int i = 0; i < 32; ++i) {
    cudaMemsetAsync(scratch, 0, scratch_bytes, stream.value());
  }
}

/// Conversion-pool churn: each conversion allocates from the async pool,
/// recycling any block whose free has already retired.
void hammer_conversions(stream_lineage_fixture& f,
                        rmm::cuda_stream_view stream,
                        int count,
                        std::size_t rows)
{
  for (int i = 0; i < count; ++i) {
    auto hb = make_host_batch(f, rows, /*seed=*/0x5A5A5A5A + i, false, false, stream);
    upload_to_gpu(f, hb, stream);
  }
}

constexpr std::size_t kRows       = 1u << 20;
constexpr std::size_t kBytes      = kRows * sizeof(std::int64_t);
constexpr std::size_t kScratchMiB = 64;

}  // namespace

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

    enqueue_blockers(scratch, kScratchMiB << 20, task_stream.view());
    void const* src = stolen->view().column(0).head();
    cudaMemcpyAsync(result_dev, src, kBytes, cudaMemcpyDeviceToDevice, task_stream.value());

    // Pre-fix, this free retired on the idle conversion-pool stream and the
    // block could be recycled under the still-pending read.
    {
      auto cols = stolen->release();
      cols[0].reset();
    }

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
  rmm::cuda_stream foreign_stream;

  auto batch = make_host_batch(f, kRows, seed, false, false, task_stream.view());
  upload_to_gpu(f, batch, task_stream.view());
  auto stolen = steal_table(batch, *f.gpu0, task_stream.view());

  // Recreate the pre-fix binding: buffers bound back to an idle foreign stream.
  auto cols = stolen->release();
  cols[0]   = cudf::rebind_stream(std::move(*cols[0]), foreign_stream.view());

  void const* src = cols[0]->view().head();
  enqueue_blockers(scratch, kScratchMiB << 20, task_stream.view());
  cudaMemcpyAsync(result_dev, src, kBytes, cudaMemcpyDeviceToDevice, task_stream.value());

  cols[0].reset();

  // A same-stream allocation of the same size recycles the freed block; poison
  // it while the read above is still stuck behind the blocker.
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
    // Anti-vacuity: the harness must detect the pre-fix failure mode, or the
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

TEST_CASE("reader events order a convert reclaim's mutable acquisition after in-flight reads",
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

  {
    auto mut        = batch->to_mutable();
    auto view       = sirius::get_cudf_table_view(*batch);
    void const* src = view.column(0).head();
    enqueue_blockers(scratch, kScratchMiB << 20, reader_stream.view());
    cudaMemcpyAsync(result_dev, src, kBytes, cudaMemcpyDeviceToDevice, reader_stream.value());
    mut.get_data()->record_writer_event(reader_stream.view());
  }  // read lock dropped with the read still in flight

  // Anti-vacuity: the lock is already gone, so only a pending reader event can
  // make the non-blocking mutable path decline.
  REQUIRE_FALSE(batch->try_to_mutable().has_value());

  // The production downgrade discipline (convertible_data_batch.hpp).
  {
    auto mut = batch->to_mutable();
    CHECK(cudaStreamQuery(reader_stream.value()) == cudaSuccess);
    mut.rebind_stream(reclaim_stream.view());
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), f.host0, reclaim_stream.view());
  }
  REQUIRE(batch->try_to_mutable().has_value());

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

// Driving the real ingestibles would need full scan plans and bind data, so the
// test replays the filtered serving sequence against the same APIs they use.
TEST_CASE("filtered serving records reader events through owning_table_view before owner drop",
          "[stream_lineage]")
{
  stream_lineage_fixture f;
  REQUIRE(f.setup());

  void* scratch    = nullptr;
  void* result_dev = nullptr;
  REQUIRE(cudaMalloc(&scratch, kScratchMiB << 20) == cudaSuccess);
  REQUIRE(cudaMalloc(&result_dev, kBytes) == cudaSuccess);

  std::int64_t const seed = 91;
  rmm::cuda_stream setup_stream;
  rmm::cuda_stream task_stream;
  rmm::cuda_stream reclaim_stream;

  auto batch = make_host_batch(f, kRows, seed, false, false, setup_stream.view());
  upload_to_gpu(f, batch, setup_stream.view());
  setup_stream.synchronize();

  {
    auto rbatch = batch->to_read_only();
    auto view   = sirius::get_cudf_table_view(rbatch);
    sirius::op::scan::owning_table_view served{std::move(rbatch), view};

    enqueue_blockers(scratch, kScratchMiB << 20, task_stream.view());
    void const* src = served.view().column(0).head();
    cudaMemcpyAsync(result_dev, src, kBytes, cudaMemcpyDeviceToDevice, task_stream.value());
    served.record_reader_event(task_stream.view());

    served.drop();
  }

  // Anti-vacuity: the read lock is already gone, so only a reader event
  // recorded through the type-erased owner can make this decline.
  REQUIRE_FALSE(batch->try_to_mutable().has_value());

  {
    auto mut = batch->to_mutable();
    CHECK(cudaStreamQuery(task_stream.value()) == cudaSuccess);
    mut.rebind_stream(reclaim_stream.view());
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), f.host0, reclaim_stream.view());
  }
  REQUIRE(batch->try_to_mutable().has_value());

  hammer_conversions(f, reclaim_stream.view(), 3, kRows / 4);

  std::vector<std::int64_t> host_result(kRows);
  cudaMemcpyAsync(
    host_result.data(), result_dev, kBytes, cudaMemcpyDeviceToHost, task_stream.value());
  task_stream.synchronize();

  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < kRows; ++i) {
    if (host_result[i] != seed + static_cast<std::int64_t>(i)) { ++mismatches; }
  }
  INFO("corrupted elements: " << mismatches << " of " << kRows);
  REQUIRE(mismatches == 0);

  cudaFree(scratch);
  cudaFree(result_dev);
}
