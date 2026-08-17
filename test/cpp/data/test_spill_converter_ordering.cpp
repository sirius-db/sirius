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

// Regression guard for the residual freed-while-read corruption (SF1000
// RF2-window strikes): a GPU->HOST conversion must order itself after the
// source representation's WRITER EVENT before reading any source buffer.
// Batch handoff in the engine is event-ordered, not host-synced — a parked
// batch can reach the downgrade converter while the stream that produced it
// still has writes in flight. Without the writer-event wait the D2H copies
// read the buffer's PRE-WRITE bytes (recycled pool content in production;
// a sentinel here), the torn host image re-uploads later, and its scribbled
// string geometry detonates in whatever consumes it (string gathers, LIKE
// walks, batched memcpys — the five 2026-08-14 coredumps).
//
// The test enqueues a delayed final write on a producer stream, converts on a
// DIFFERENT stream while the producer is still blocked, round-trips the host
// image back to the GPU, and requires the FINAL bytes. Fails on the pre-fix
// converters (they read the sentinel), passes with the writer-event wait
// (cucascade ff9c5e5).

#include "catch.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/builtin_converters.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

struct ordering_test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;

  ordering_test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0))
  {
  }
};

ordering_test_env& env()
{
  static ordering_test_env e;
  return e;
}

constexpr std::size_t kRows      = 1 << 16;
constexpr std::int32_t kStale    = 0x2AAAAAAA;  // pre-write sentinel
constexpr std::int32_t kExpected = 0x1BBBBBBB;  // the producer's final value

struct delay_state {
  std::atomic<bool> release{false};
};

// Host function that parks the producer stream until the test releases it —
// deterministic "producer still writing" window, no sleeps on the sanity path.
void CUDART_CB block_until_released(void* userData)
{
  auto* state = static_cast<delay_state*>(userData);
  while (!state->release.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

/// Round-trip the produced HOST representation back to a GPU table on
/// `stream` and return the first column's bytes.
std::vector<std::int32_t> read_back(cucascade::representation_converter_registry& registry,
                                    cucascade::idata_representation& host_rep,
                                    rmm::cuda_stream_view stream)
{
  auto gpu_rep =
    registry.convert<cucascade::gpu_table_representation>(host_rep, env().gpu_space, stream);
  auto& gpu = gpu_rep->cast<cucascade::gpu_table_representation>();
  auto view = gpu.get_table_view();
  std::vector<std::int32_t> out(static_cast<std::size_t>(view.column(0).size()));
  REQUIRE(cudaMemcpyAsync(out.data(),
                          view.column(0).head<std::int32_t>(),
                          out.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
  return out;
}

/// Warm every first-call cost on the conversion path (pinned-block pool,
/// batch-memcpy driver state, converter internals) so the parked run's tear
/// window is not consumed by host-side allocation latency.
void warm_conversion_path(cucascade::representation_converter_registry& registry)
{
  rmm::cuda_stream stream;
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(kRows),
                                       cudf::mask_state::UNALLOCATED,
                                       stream.view());
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto rep = std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(cols)), *env().gpu_space, stream.view());
  auto host_rep =
    registry.convert<cucascade::host_data_representation>(*rep, env().host_space, stream.view());
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
}

/// Shared scenario: produce a column whose FINAL bytes land behind a parked
/// host function on the producer stream, convert GPU->HOST on another stream
/// mid-flight, then verify the host image carries the final bytes.
void run_ordering_scenario(cucascade::representation_converter_registry& registry)
{
  warm_conversion_path(registry);

  rmm::cuda_stream producer_stream;
  rmm::cuda_stream conversion_stream;

  // Build the column and settle the STALE sentinel everywhere.
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(kRows),
                                       cudf::mask_state::UNALLOCATED,
                                       producer_stream.view());
  {
    std::vector<std::int32_t> stale(kRows, kStale);
    REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                            stale.data(),
                            kRows * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice,
                            producer_stream.value()) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(producer_stream.value()) == cudaSuccess);
  }

  // Park the producer stream, then enqueue the FINAL write behind the park.
  // The source buffer is PINNED (cudaHostRegister) so this enqueue is truly
  // asynchronous — a pageable H2D behind a parked stream can block the
  // calling host thread, which would serialize the test instead of opening
  // the tear window.
  delay_state gate;
  std::vector<std::int32_t> final_bytes(kRows, kExpected);  // outlives the sync below
  REQUIRE(cudaHostRegister(final_bytes.data(), kRows * sizeof(std::int32_t), 0) == cudaSuccess);
  REQUIRE(cudaLaunchHostFunc(producer_stream.value(), block_until_released, &gate) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                          final_bytes.data(),
                          kRows * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice,
                          producer_stream.value()) == cudaSuccess);

  // The representation records its writer event on the producer stream —
  // AFTER the parked write, exactly like an operator emitting a batch whose
  // stream work is still pending.
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto table   = std::make_unique<cudf::table>(std::move(cols));
  auto gpu_rep = std::make_unique<cucascade::gpu_table_representation>(
    std::move(table), *env().gpu_space, producer_stream.view());

  // Convert on ANOTHER stream while the producer is provably still parked.
  // A FIXED converter blocks (its device work is ordered after the parked
  // producer), so the gate opens from a delayed releaser thread: the tear
  // window stays open long enough for a pre-fix converter to read stale
  // bytes, and a post-fix converter simply waits the extra 200 ms.
  std::thread releaser([&gate] {
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    gate.release.store(true, std::memory_order_release);
  });
  auto host_rep = registry.convert<cucascade::host_data_representation>(
    *gpu_rep, env().host_space, conversion_stream.view());
  releaser.join();

  // Settle everything before inspecting.
  REQUIRE(cudaStreamSynchronize(producer_stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(conversion_stream.value()) == cudaSuccess);
  REQUIRE(cudaHostUnregister(final_bytes.data()) == cudaSuccess);

  auto const out = read_back(registry, *host_rep, conversion_stream.view());
  REQUIRE(out.size() == kRows);
  std::size_t stale_count = 0;
  for (auto v : out) {
    if (v != kExpected) { ++stale_count; }
  }
  INFO("host image carries " << stale_count << " stale (torn) values of " << kRows);
  REQUIRE(stale_count == 0);
}

}  // namespace

TEST_CASE("builtin fast GPU->HOST conversion orders after the producer's writer event",
          "[spill_converter_ordering]")
{
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  run_ordering_scenario(registry);
}

// Regression guard for the reader-side seam of the freed-while-read class: a
// downgrade grabs a parked batch whose consumer already ENQUEUED reads on the
// producer stream and returned (read lock dropped, kernels in flight — the
// writer event is long signaled, so the converter-side producer-ordering wait
// does not block). rebind_stream() then moves the deallocation binding onto the
// idle downgrade stream, and the conversion's free executes there while the
// reader still runs: the async pool unmaps/rebinds the VA under the reader
// (in production, a row-hash distinct sketch MMU-faulting on scribbled string
// offsets). The fix (cucascade e19f5d3) makes rebind_stream() itself record an
// event on the previously-bound stream and fence the adopting stream behind it.
TEST_CASE("rebind_stream orders the adopting stream after a straggler reader",
          "[spill_converter_ordering]")
{
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  warm_conversion_path(registry);

  rmm::cuda_stream reader_stream;     // producer AND straggler-reader stream
  rmm::cuda_stream downgrade_stream;  // the stream adopting the dealloc binding

  // Build the column, settle the EXPECTED bytes, and record the writer event on
  // the reader stream with NO work pending — the writer event must be signaled
  // so this guard cannot be satisfied by the converter's writer-event wait.
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(kRows),
                                       cudf::mask_state::UNALLOCATED,
                                       reader_stream.view());
  {
    std::vector<std::int32_t> expected(kRows, kExpected);
    REQUIRE(cudaMemcpyAsync(col->mutable_view().head<void>(),
                            expected.data(),
                            kRows * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice,
                            reader_stream.value()) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(reader_stream.value()) == cudaSuccess);
  }
  auto const* batch_bytes = col->view().head<std::int32_t>();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto gpu_rep = std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(cols)), *env().gpu_space, reader_stream.view());

  // Park the reader stream, then enqueue the straggler READ behind the park —
  // exactly a consumer that enqueued its kernel and dropped its read lock.
  delay_state gate;
  std::vector<std::int32_t> reader_out(kRows, 0);
  REQUIRE(cudaHostRegister(reader_out.data(), kRows * sizeof(std::int32_t), 0) == cudaSuccess);
  REQUIRE(cudaLaunchHostFunc(reader_stream.value(), block_until_released, &gate) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(reader_out.data(),
                          batch_bytes,
                          kRows * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost,
                          reader_stream.value()) == cudaSuccess);

  // The downgrade-side rebind. FIXED: the adopting stream is fenced behind the
  // parked reader stream — probe it directly. PRE-FIX: the probe event completes
  // immediately (no edge), and the free below would land on the idle stream.
  gpu_rep->rebind_stream(downgrade_stream.view());
  cudaEvent_t probe{nullptr};
  REQUIRE(cudaEventCreateWithFlags(&probe, cudaEventDisableTiming) == cudaSuccess);
  REQUIRE(cudaEventRecord(probe, downgrade_stream.value()) == cudaSuccess);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(cudaEventQuery(probe) == cudaErrorNotReady);  // the reader-ordering edge

  // Full downgrade shape end-to-end: convert on the adopting stream, free the
  // source, and force VA reuse with a poison fill — all while the reader is
  // parked (pre-fix this scribbles the reader's input; post-fix every step is
  // ordered after the reader). The releaser opens the gate mid-flight because
  // a FIXED convert host-blocks on its internal stream sync.
  std::thread releaser([&gate] {
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    gate.release.store(true, std::memory_order_release);
  });
  auto host_rep = registry.convert<cucascade::host_data_representation>(
    *gpu_rep, env().host_space, downgrade_stream.view());
  gpu_rep.reset();  // stream-ordered free on the rebound (downgrade) stream
  {
    rmm::device_buffer poison(kRows * sizeof(std::int32_t), downgrade_stream.view());
    REQUIRE(cudaMemsetAsync(
              poison.data(), 0xEE, kRows * sizeof(std::int32_t), downgrade_stream.value()) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(downgrade_stream.value()) == cudaSuccess);
  }
  releaser.join();
  REQUIRE(cudaStreamSynchronize(reader_stream.value()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(downgrade_stream.value()) == cudaSuccess);
  REQUIRE(cudaEventDestroy(probe) == cudaSuccess);
  REQUIRE(cudaHostUnregister(reader_out.data()) == cudaSuccess);

  std::size_t scribbled = 0;
  for (auto v : reader_out) {
    if (v != kExpected) { ++scribbled; }
  }
  INFO("straggler reader observed " << scribbled << " scribbled values of " << kRows);
  REQUIRE(scribbled == 0);

  // The host image must also carry the expected bytes (the conversion read a
  // fully-written, still-live source).
  auto const out = read_back(registry, *host_rep, downgrade_stream.view());
  REQUIRE(out.size() == kRows);
  std::size_t torn = 0;
  for (auto v : out) {
    if (v != kExpected) { ++torn; }
  }
  INFO("host image carries " << torn << " torn values of " << kRows);
  REQUIRE(torn == 0);
}
