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

// The exchange transport's protocol, with every rank in one process over a loopback fabric:
// credits, landing, end of stream, notifications in any order, and failure.

#include "catch.hpp"
#include "exchange/loopback_fabric.hpp"
#include "exchange/messages.hpp"
#include "exchange/transport_test_utils.hpp"
#include "sirius/exception.hpp"

#include <chrono>
#include <memory>
#include <vector>

namespace {

using namespace sirius::test::exchange_utils;

/// World-size ranks, each with its rings, fabric and transport.
struct cluster {
  struct rank_state {
    std::unique_ptr<frame_ring> send;
    std::unique_ptr<frame_ring> recv;
    std::unique_ptr<fabric> fab;
    std::unique_ptr<received> got = std::make_unique<received>();
    std::unique_ptr<transport> t;
  };
  loopback_hub hub;
  std::vector<rank_state> ranks;

  cluster(rank_t world, std::uint32_t send_frames, std::uint32_t recv_frames)
    : cluster(world, send_frames, recv_frames, loopback_hub::options{}, env().host)
  {
  }

  cluster(rank_t world,
          std::uint32_t send_frames,
          std::uint32_t recv_frames,
          loopback_hub::options opts,
          cucascade::memory::memory_space* host)
    : hub(world, opts), ranks(world)
  {
    for (rank_t r = 0; r < world; ++r) {
      auto& s = ranks[r];
      s.send  = std::make_unique<frame_ring>(send_frames, 64 * KiB);
      s.recv  = std::make_unique<frame_ring>(recv_frames, 64 * KiB);
      s.fab   = hub.attach(r, *s.send, *s.recv);
      s.t =
        std::make_unique<transport>(*s.fab, *s.send, *s.recv, *env().gpu, host, recording(*s.got));
    }
  }

  ~cluster()
  {
    // Transports stop before the fabrics and rings they use.
    for (auto& s : ranks) {
      s.t.reset();
    }
  }
};

/// Every rank sends to every rank, itself included; checks what each received.
void all_to_all(cluster& c, std::vector<stream_id> const& streams, int per_stream, std::size_t rows)
{
  for (auto& r : c.ranks) {
    send_all(*r.t, streams, per_stream, rows);
  }
  auto const world = static_cast<rank_t>(c.ranks.size());
  for (rank_t to = 0; to < world; ++to) {
    check_all(*c.ranks[to].got, to, world, streams, per_stream, rows);
  }
}

}  // namespace

TEST_CASE("every rank sends to every rank, itself included", "[exchange][transport]")
{
  cluster c(3, /*send_frames=*/4, /*recv_frames=*/8);
  all_to_all(c, {7, 9}, 3, 5'000);
}

TEST_CASE("one credit per peer and one send frame still deliver many-frame batches",
          "[exchange][transport]")
{
  // About 1 MiB a batch: 16 frames of 64 KiB through one frame at a time.
  cluster c(3, /*send_frames=*/1, /*recv_frames=*/2);
  all_to_all(c, {1}, 2, 30'000);
}

TEST_CASE("notifications may arrive in any order", "[exchange][transport]")
{
  // Several notifications arrive together and shuffled: frames before their header, ends
  // before the batches they count, credits late.
  cluster c(4, 3, 6, {.shuffle = true, .deliver_every = 5}, env().host);
  all_to_all(c, {1, 2}, 2, 8'000);
}

TEST_CASE("a receiver with no room holds the frames until room appears", "[exchange][transport]")
{
  cluster c(2, 4, 4, loopback_hub::options{}, /*host=*/nullptr);
  auto& got = *c.ranks[1].got;
  {
    auto const hold = hold_gpu();
    c.ranks[0].t->send(5, 1, gpu_batch(sample_table(5'000)));
    c.ranks[0].t->close(5, 1);
    CHECK_FALSE(got.wait_for([&] { return !got.batches.empty(); }, std::chrono::seconds{1}));
  }
  REQUIRE(got.wait_for([&] { return got.ended.size() == 1; }));
  std::lock_guard lock{got.mutex};
  CHECK(got.batches[{0, 5}].size() == 1);
  CHECK_FALSE(got.error);
}

TEST_CASE("a sender holds a batch of a query until its last frame is staged",
          "[exchange][transport]")
{
  // About 1 MiB: 16 frames of 64 KiB, against 4 receive frames a receiver with no room keeps.
  cluster c(2, 4, 4, loopback_hub::options{}, /*host=*/nullptr);
  auto& sender           = *c.ranks[0].t;
  auto& got              = *c.ranks[1].got;
  stream_id const stream = (stream_id{5} << sirius::exchange::query_stream_bits) | 1;
  {
    auto const hold = hold_gpu();
    sender.send(stream, 1, gpu_batch(sample_table(30'000)));
    sender.close(stream, 1);
    CHECK_FALSE(sender.wait_sent(5, std::chrono::milliseconds{500}));
    CHECK(sender.wait_sent(6, std::chrono::milliseconds{0}));
  }
  CHECK(sender.wait_sent(5, std::chrono::seconds{30}));
  REQUIRE(got.wait_for([&] { return got.ended.size() == 1; }));
}

TEST_CASE("a failed write fails the sender and its peer", "[exchange][transport]")
{
  cluster c(2, 4, 4, {.fail_after_writes = 0}, env().host);
  c.ranks[0].t->send(3, 1, gpu_batch(sample_table(1'000)));
  for (auto& r : c.ranks) {
    auto& got = *r.got;
    REQUIRE(got.wait_for([&] { return got.error != nullptr; }));
  }
  CHECK_THROWS(c.ranks[0].t->send(3, 1, gpu_batch(sample_table(1'000))));
  CHECK_THROWS(c.ranks[1].t->close(3, 0));
}

TEST_CASE("exchange messages round-trip and refuse malformed bytes", "[exchange][transport]")
{
  using namespace sirius::exchange;
  std::vector<message> const all{frame_message{7, 3, 0, 5, 1024, {1, 2, 3}},
                                 frame_message{7, 3, 4, 6, 99, {}},
                                 credit_message{{1, 2, 3}},
                                 credit_message{},
                                 end_message{7, 12},
                                 error_message{"out of memory"},
                                 abort_message{3, "rank 1: out of memory"}};
  for (auto const& m : all) {
    CHECK(decode(encode(m)) == m);
  }
  CHECK_THROWS_AS(decode(""), sirius::invalid_input_exception);
  CHECK_THROWS_AS(decode(std::string(1, '\x09')), sirius::invalid_input_exception);
  auto truncated = encode(end_message{1, 2});
  truncated.pop_back();
  CHECK_THROWS_AS(decode(truncated), sirius::invalid_input_exception);
  // Only frame 0 carries a header.
  CHECK_THROWS_AS(decode(encode(frame_message{1, 1, 2, 0, 8, {9}})),
                  sirius::invalid_input_exception);
  CHECK_THROWS_AS(decode(encode(frame_message{1, 1, 0, 0, 8, {}})),
                  sirius::invalid_input_exception);
}
