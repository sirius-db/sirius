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

// A batch through the exchange's two ends in one process: described and copied frame by frame
// into a device buffer standing in for a send frame, carried to a receive frame as a NIXL WRITE
// would, copied into its landing, and sealed.

#include "catch.hpp"
#include "exchange/batch_io.hpp"
#include "exchange/exchange_test_utils.hpp"
#include "sirius/exception.hpp"

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace sirius::test::exchange_utils;
using sirius::exchange::decode_header;
using sirius::exchange::encode_header;
using sirius::exchange::incoming_batch;
using sirius::exchange::outgoing_batch;

/// Sends @p batch to a landing through @p frame_bytes frames, as the protocol would: the header
/// decoded from its bytes, each frame staged, written to a receive frame, copied in, and the
/// receive frame overwritten once its copy is done.
std::shared_ptr<cucascade::data_batch> ship(std::shared_ptr<cucascade::data_batch> batch,
                                            std::size_t frame_bytes,
                                            Tier expect_tier)
{
  rmm::cuda_stream send_stream;
  rmm::cuda_stream recv_stream;
  auto out = outgoing_batch::make(std::move(batch), send_stream.view());
  REQUIRE(out.has_value());
  auto landing = incoming_batch::land(
    decode_header(encode_header(out->header())), *env().gpu, env().host, recv_stream.view());
  REQUIRE(landing != nullptr);
  CHECK(landing->tier() == expect_tier);

  rmm::device_buffer send_frame(frame_bytes, send_stream.view());
  rmm::device_buffer recv_frame(frame_bytes, recv_stream.view());
  auto const total = out->stream_bytes();
  for (std::size_t at = 0; at < total; at += frame_bytes) {
    auto const len = std::min(frame_bytes, total - at);
    out->copy_range(at, at + len, send_frame.data(), send_stream.view());
    send_stream.synchronize();
    // The peer's WRITE.
    REQUIRE(cudaMemcpy(recv_frame.data(), send_frame.data(), len, cudaMemcpyDeviceToDevice) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    landing->copy_in(at, recv_frame.data(), len, recv_stream.view());
    recv_stream.synchronize();
    // Once copied in, the frame goes back to its sender: a later WRITE must not change the batch.
    REQUIRE(cudaMemset(recv_frame.data(), 0xab, frame_bytes) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  }
  out.reset();
  return landing->seal(recv_stream.view());
}

std::vector<std::uint8_t> fixed_width_bytes(cudf::column_view const& column)
{
  auto const width = cudf::size_of(column.type());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(column.size()) * width);
  REQUIRE(cudaMemcpy(bytes.data(),
                     static_cast<std::uint8_t const*>(column.head()) + column.offset() * width,
                     bytes.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return bytes;
}

/// A strings column's rows on the host; the offsets may be INT32 or INT64, so they are read as
/// INT64.
std::vector<std::string> strings_to_host(cudf::column_view const& column)
{
  cudf::strings_column_view const strings(column);
  auto const offsets_column = cudf::cast(strings.offsets(), cudf::data_type{cudf::type_id::INT64});
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(column.size()) + 1);
  REQUIRE(cudaMemcpy(offsets.data(),
                     offsets_column->view().data<std::int64_t>() + column.offset(),
                     offsets.size() * sizeof(std::int64_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  std::vector<char> chars(static_cast<std::size_t>(offsets.back()));
  REQUIRE(cudaMemcpy(chars.data(),
                     strings.chars_begin(cudf::get_default_stream()),
                     chars.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  std::vector<std::string> rows;
  rows.reserve(column.size());
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    rows.emplace_back(chars.data() + offsets[i], chars.data() + offsets[i + 1]);
  }
  return rows;
}

/// Requires @p actual to hold @p expected's values and NULL positions, row for row.
void check_same_rows(cudf::table_view const& expected, cudf::table_view const& actual)
{
  REQUIRE(actual.num_columns() == expected.num_columns());
  REQUIRE(actual.num_rows() == expected.num_rows());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  for (cudf::size_type i = 0; i < expected.num_columns(); ++i) {
    INFO("column " << i);
    auto const& want = expected.column(i);
    auto const& got  = actual.column(i);
    REQUIRE(got.type() == want.type());
    CHECK(got.null_count() == want.null_count());
    // Compared as bools: Catch would print every row of a mismatch.
    bool const same_nulls = utils::copy_validity_to_host(got) == utils::copy_validity_to_host(want);
    CHECK(same_nulls);
    bool const same_values = want.type().id() == cudf::type_id::STRING
                               ? strings_to_host(got) == strings_to_host(want)
                               : fixed_width_bytes(got) == fixed_width_bytes(want);
    CHECK(same_values);
  }
}

/// The gate of test_spill_converter_ordering.cpp: parks a stream until the test releases it.
struct delay_state {
  std::atomic<bool> release{false};
  std::atomic<bool> entered{false};
  std::atomic<bool> timed_out{false};

  ~delay_state() { release.store(true, std::memory_order_release); }
};

void CUDART_CB block_until_released(void* user_data)
{
  auto* state = static_cast<delay_state*>(user_data);
  state->entered.store(true, std::memory_order_release);
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  while (!state->release.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      state->timed_out.store(true, std::memory_order_release);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

bool wait_for_flag(std::atomic<bool> const& flag,
                   std::chrono::milliseconds timeout = std::chrono::seconds{5})
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return true;
}

}  // namespace

TEST_CASE("a GPU batch lands on the GPU frame by frame", "[exchange][batch_io]")
{
  auto batch          = gpu_batch(sample_table(20'000));
  auto const expected = digest(batch);
  // A frame size that is not a multiple of the stream alignment cuts buffers anywhere.
  for (std::size_t frame : {64 * KiB, 100'000ul, MiB}) {
    auto received = ship(batch, frame, Tier::GPU);
    CHECK(received->to_read_only().get_current_tier() == Tier::GPU);
    CHECK(digest(received) == expected);
  }
}

TEST_CASE("a batch lands on host when the GPU has no room", "[exchange][batch_io]")
{
  auto batch          = gpu_batch(sample_table(20'000));
  auto const expected = digest(batch);
  std::shared_ptr<cucascade::data_batch> received;
  {
    // Hold every free GPU byte, so the landing cannot reserve any.
    auto const hold = hold_gpu();
    received        = ship(batch, 64 * KiB, Tier::HOST);
  }
  {
    auto ro = received->to_read_only();
    CHECK(dynamic_cast<cucascade::host_data_representation const*>(ro.get_data()) != nullptr);
  }
  CHECK(digest(received) == expected);
}

TEST_CASE("a host landing spread over several host blocks keeps every value and NULL",
          "[exchange][batch_io]")
{
  auto const* host_mr =
    env().host->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  REQUIRE(host_mr != nullptr);
  auto const block = host_mr->get_block_size();
  // About 3.2 MB; frames that do not divide the block size straddle block edges.
  constexpr std::size_t frame = 100'000;
  REQUIRE(block % frame != 0);
  auto batch = gpu_batch(sample_table(80'000));
  {
    rmm::cuda_stream stream;
    auto out = outgoing_batch::make(batch, stream.view());
    REQUIRE(out.has_value());
    REQUIRE(out->stream_bytes() > 2 * block);
  }
  std::shared_ptr<cucascade::data_batch> received;
  {
    auto const hold = hold_gpu();
    received        = ship(batch, frame, Tier::HOST);
  }
  {
    auto ro = received->to_read_only();
    REQUIRE(ro.get_current_tier() == Tier::HOST);
    auto const& host = ro.get_data()->cast<cucascade::host_data_representation>();
    CHECK(host.get_host_table()->allocation->size() > 2);
  }
  CHECK(digest(received) == digest(batch));  // brings the received batch to the GPU
  auto const want = batch->to_read_only();
  auto const got  = received->to_read_only();
  check_same_rows(sirius::get_cudf_table_view(want), sirius::get_cudf_table_view(got));
}

TEST_CASE("a send waits for the batch's pending writer, known or not", "[exchange][batch_io]")
{
  // A batch names its writer by an event; one built on a null writer stream has none.
  bool const known_writer = GENERATE(true, false);
  CAPTURE(known_writer);
  constexpr cudf::size_type rows = 50'000;
  std::vector<std::int32_t> values(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    values[i] = i * 3 + 1;
  }
  auto const final_values = numeric_column(cudf::data_type{cudf::type_id::INT32}, values);
  auto stale =
    numeric_column(cudf::data_type{cudf::type_id::INT32}, std::vector<std::int32_t>(rows, -1));
  auto* const target = stale->mutable_view().data<std::int32_t>();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(stale));
  auto table = std::make_unique<cudf::table>(std::move(columns));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // Non-blocking, so nothing on the legacy default stream waits behind the gate.
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream send_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::cuda_stream recv_stream;
  rmm::device_buffer send_frame(rows * sizeof(std::int32_t), send_stream.view());
  send_stream.synchronize();
  std::shared_ptr<cucascade::data_batch> batch;
  std::optional<outgoing_batch> out;
  std::unique_ptr<incoming_batch> landing;
  delay_state gate;
  absl::Cleanup release_producer = [&] {
    gate.release.store(true, std::memory_order_release);
    producer.synchronize_no_throw();
    send_stream.synchronize_no_throw();
  };
  REQUIRE(cudaLaunchHostFunc(producer.value(), block_until_released, &gate) == cudaSuccess);
  // The batch's last write, queued behind the gate.
  REQUIRE(cudaMemcpyAsync(target,
                          final_values->view().data<std::int32_t>(),
                          rows * sizeof(std::int32_t),
                          cudaMemcpyDeviceToDevice,
                          producer.value()) == cudaSuccess);
  if (known_writer) {
    batch = sirius::make_data_batch(
      std::move(table), *env().gpu, producer.view(), sirius::telemetry::batch_telemetry_info{});
  } else {
    batch = cucascade::data_batch::make(
      sirius::get_next_batch_id(),
      std::make_unique<cucascade::gpu_table_representation>(
        std::move(table), *env().gpu, ::cuda::stream_ref{cudaStream_t{nullptr}}));
  }
  REQUIRE(wait_for_flag(gate.entered));

  // The send takes the caller's owner: nothing else keeps the batch alive while it is sent.
  auto planning = std::async(std::launch::async, [&, sent = std::move(batch)]() mutable {
    return outgoing_batch::make(std::move(sent), send_stream.view());
  });
  if (known_writer) {
    // The send stream waits for the writer's event; the caller does not.
    REQUIRE(planning.wait_for(std::chrono::seconds{5}) == std::future_status::ready);
  } else {
    // No event to order the send stream after: the caller waits for the device, so for the writer.
    CHECK(planning.wait_for(std::chrono::milliseconds{200}) == std::future_status::timeout);
    CHECK_FALSE(gate.timed_out.load());
    gate.release.store(true, std::memory_order_release);
  }
  out = planning.get();
  REQUIRE(out.has_value());
  auto const bytes = out->stream_bytes();
  REQUIRE(bytes <= send_frame.size());
  landing = incoming_batch::land(
    decode_header(encode_header(out->header())), *env().gpu, env().host, recv_stream.view());
  REQUIRE(landing != nullptr);
  CHECK(landing->tier() == Tier::GPU);
  recv_stream.synchronize();
  // The copy out of the batch and its delivery, both on the send stream.
  out->copy_range(0, bytes, send_frame.data(), send_stream.view());
  landing->copy_in(0, send_frame.data(), bytes, send_stream.view());
  if (known_writer) {
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    CHECK(cudaStreamQuery(send_stream.value()) == cudaErrorNotReady);
    CHECK_FALSE(gate.timed_out.load());
    gate.release.store(true, std::memory_order_release);
  }
  send_stream.synchronize();
  REQUIRE_FALSE(gate.timed_out.load());
  out.reset();
  auto const received = landing->seal(recv_stream.view());
  auto const ro       = received->to_read_only();
  if (cudaEvent_t writer = ro.get_writer_event()) {
    REQUIRE(cudaEventSynchronize(writer) == cudaSuccess);
  }
  auto const column = sirius::get_cudf_table_view(ro).column(0);
  CHECK(column.null_count() == 0);
  bool const every_value_final = utils::copy_column_to_host<std::int32_t>(column) == values;
  CHECK(every_value_final);
}

TEST_CASE("a landing with no room anywhere is refused, not waited for", "[exchange][batch_io]")
{
  rmm::cuda_stream stream;
  auto batch = gpu_batch(sample_table(1000));
  auto out   = outgoing_batch::make(batch, stream.view());
  REQUIRE(out.has_value());
  auto const hold = hold_gpu();
  CHECK(incoming_batch::land(out->header(), *env().gpu, nullptr, stream.view()) == nullptr);
}

TEST_CASE("a batch no tier could ever hold fails at once, naming its size", "[exchange][batch_io]")
{
  rmm::cuda_stream stream;
  auto out = outgoing_batch::make(gpu_batch(sample_table(1000)), stream.view());
  REQUIRE(out.has_value());
  auto header = out->header();
  auto column = std::find_if(
    header.columns.begin(), header.columns.end(), [](auto const& c) { return c.has_data; });
  REQUIRE(column != header.columns.end());
  auto const too_big  = env().gpu->get_max_memory() + 1;
  column->data_size   = too_big;
  header.stream_bytes = too_big;
  CHECK_THROWS_WITH(incoming_batch::land(header, *env().gpu, nullptr, stream.view()),
                    Catch::Matchers::ContainsSubstring("can land neither"));
}

TEST_CASE("a sliced batch is sent as its rows only", "[exchange][batch_io]")
{
  auto table          = sample_table(1000);
  auto slices         = cudf::slice(table->view(), {100, 350});
  auto sliced         = std::make_unique<cudf::table>(slices.front());
  auto const expected = digest(sliced->view());
  // A view batch over the parent's memory keeps its offsets.
  auto batch = sirius::make_data_batch_from_view(slices.front(),
                                                 std::shared_ptr<cudf::table>(std::move(table)),
                                                 0,
                                                 *env().gpu,
                                                 utils::default_stream(),
                                                 sirius::telemetry::batch_telemetry_info{});
  CHECK(digest(ship(batch, 64 * KiB, Tier::GPU)) == expected);
}

TEST_CASE("an empty batch is not sent; a range outside the stream is refused",
          "[exchange][batch_io]")
{
  rmm::cuda_stream stream;
  CHECK_FALSE(outgoing_batch::make(gpu_batch(sample_table(0)), stream.view()).has_value());

  auto out = outgoing_batch::make(gpu_batch(sample_table(100)), stream.view());
  REQUIRE(out.has_value());
  rmm::device_buffer frame(out->stream_bytes() + 1, stream.view());
  CHECK_THROWS_AS(out->copy_range(0, out->stream_bytes() + 1, frame.data(), stream.view()),
                  sirius::invalid_input_exception);
  auto landing = incoming_batch::land(out->header(), *env().gpu, env().host, stream.view());
  REQUIRE(landing != nullptr);
  CHECK_THROWS_AS(landing->copy_in(out->stream_bytes(), frame.data(), 1, stream.view()),
                  sirius::invalid_input_exception);
  stream.synchronize();
}
