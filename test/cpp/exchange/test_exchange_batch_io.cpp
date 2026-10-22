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

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <cucascade/cudf/host_data_representation.hpp>

#include <memory>

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

}  // namespace

TEST_CASE("a GPU batch lands on the GPU frame by frame", "[exchange][batch_io]")
{
  auto batch          = gpu_batch(sample_table(20'000));
  auto const expected = digest(batch);
  // A frame size that is not a multiple of the stream alignment cuts buffers anywhere.
  for (std::size_t frame : {64 * KiB, 100'000ul, 1ul << 20}) {
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

TEST_CASE("a landing with no room anywhere is refused, not waited for", "[exchange][batch_io]")
{
  rmm::cuda_stream stream;
  auto batch = gpu_batch(sample_table(1000));
  auto out   = outgoing_batch::make(batch, stream.view());
  REQUIRE(out.has_value());
  auto const hold = hold_gpu();
  CHECK(incoming_batch::land(out->header(), *env().gpu, nullptr, stream.view()) == nullptr);
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
