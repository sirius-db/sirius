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
#include "exchange/batch_io.hpp"
#include "exchange/exchange_runtime.hpp"
#include "exchange/loopback_fabric.hpp"
#include "exchange/messages.hpp"
#include "exchange/transport_test_utils.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_stream.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
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
          cucascade::memory::memory_space* host,
          transport::options transport_opts = {})
    : hub(world, opts), ranks(world)
  {
    for (rank_t r = 0; r < world; ++r) {
      auto& s = ranks[r];
      s.send  = std::make_unique<frame_ring>(send_frames, 64 * KiB);
      s.recv  = std::make_unique<frame_ring>(recv_frames, 64 * KiB);
      s.fab   = hub.attach(r, *s.send, *s.recv);
      s.t     = std::make_unique<transport>(
        *s.fab, *s.send, *s.recv, *env().gpu, host, recording(*s.got), transport_opts);
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

/// Rank 0's transport, and rank 1 as a bare fabric endpoint that notifies rank 0 what it likes.
struct forged_peer {
  loopback_hub hub{2};
  frame_ring send{4, 64 * KiB};
  frame_ring recv{4, 64 * KiB};
  frame_ring peer_send{4, 64 * KiB};
  frame_ring peer_recv{4, 64 * KiB};
  std::unique_ptr<fabric> fab  = hub.attach(0, send, recv);
  std::unique_ptr<fabric> peer = hub.attach(1, peer_send, peer_recv);
  received got;
  std::unique_ptr<transport> t;

  explicit forged_peer(transport::options opts = {})
    : t(std::make_unique<transport>(*fab, send, recv, *env().gpu, env().host, recording(got), opts))
  {
  }
  ~forged_peer() { t.reset(); }

  void forge(sirius::exchange::message const& m) { peer->notify(0, sirius::exchange::encode(m)); }

  /// Waits for rank 0 to fail, and checks its error says @p why.
  void fails_with(std::string const& why)
  {
    REQUIRE(got.wait_for([&] { return got.error != nullptr; }, std::chrono::seconds{30}));
    CHECK_THROWS_WITH(std::rethrow_exception(got.error), Catch::Matchers::ContainsSubstring(why));
  }
};

/// The encoded header of a batch of about 1.2 MB: its frame 0 alone does not complete it, so a
/// receiver keeps it landing.
std::vector<std::uint8_t> many_frame_header()
{
  rmm::cuda_stream stream;
  auto const out =
    sirius::exchange::outgoing_batch::make(gpu_batch(sample_table(30'000)), stream.view());
  REQUIRE(out.has_value());
  REQUIRE(out->stream_bytes() > 64 * KiB);
  return sirius::exchange::encode_header(out->header());
}

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

TEST_CASE("every rank sends to every rank, then the same streams again once they ended",
          "[exchange][transport]")
{
  // The same ids carry a second round only if both ends forgot the first: its end and its counts.
  cluster c(3, /*send_frames=*/4, /*recv_frames=*/8);
  all_to_all(c, {7, 9}, 2, 5'000);
  for (auto& r : c.ranks) {
    std::lock_guard lock{r.got->mutex};
    r.got->batches.clear();
    r.got->ended.clear();
  }
  all_to_all(c, {7, 9}, 2, 5'000);
}

TEST_CASE("one credit per peer and one send frame still deliver many-frame batches",
          "[exchange][transport]")
{
  // About 1.2 MB a batch: 19 frames of 64 KiB through one frame at a time.
  cluster c(3, /*send_frames=*/1, /*recv_frames=*/2);
  all_to_all(c, {1}, 2, 30'000);
}

TEST_CASE("one write covers consecutive frames, up to half a peer's credits",
          "[exchange][transport]")
{
  // 8 credits per peer: writes of up to 4 frames, so two can be in flight to a peer. About
  // 1.2 MB a batch: 19 frames of 64 KiB.
  cluster c(2, /*send_frames=*/16, /*recv_frames=*/8);
  all_to_all(c, {1, 2}, 3, 30'000);
  CHECK(c.hub.largest_write() == 4 * 64 * KiB);
}

TEST_CASE("one write per frame when a peer has one credit", "[exchange][transport]")
{
  cluster c(2, /*send_frames=*/16, /*recv_frames=*/1);
  all_to_all(c, {1}, 2, 30'000);
  CHECK(c.hub.largest_write() == 64 * KiB);
}

TEST_CASE("one write per frame to a peer whose fabric takes a frame at a time",
          "[exchange][transport]")
{
  // As over TCP lanes: 8 credits per peer would allow writes of 4 frames.
  cluster c(2, 16, 8, {.max_write_bytes = 64 * KiB}, env().host);
  all_to_all(c, {1, 2}, 3, 30'000);
  CHECK(c.hub.largest_write() == 64 * KiB);
}

TEST_CASE("a write past the frames a peer holds fails the receiver", "[exchange][transport]")
{
  // Rank 1 holds rank 0's receive frames 0 to 3; frames 2 to 4 are not all its own.
  forged_peer f;
  sirius::exchange::frame_message m{1, 0, 0, 2, 3 * 64 * KiB, many_frame_header()};
  m.frames = 3;
  f.forge(m);
  f.fails_with("which it does not hold");
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
  // About 1.2 MB: 19 frames of 64 KiB, against 4 receive frames a receiver with no room keeps.
  cluster c(2, 4, 4, loopback_hub::options{}, /*host=*/nullptr);
  auto& sender           = *c.ranks[0].t;
  auto& got              = *c.ranks[1].got;
  stream_id const stream = sirius::exchange::make_stream_id(5, 1);
  auto const query       = sirius::exchange::query_of(stream);
  {
    auto const hold = hold_gpu();
    sender.send(stream, 1, gpu_batch(sample_table(30'000)));
    sender.close(stream, 1);
    CHECK_FALSE(sender.wait_sent(query, std::chrono::milliseconds{500}));
    CHECK(sender.wait_sent(query + 1, std::chrono::milliseconds{0}));
  }
  CHECK(sender.wait_sent(query, std::chrono::seconds{30}));
  REQUIRE(got.wait_for([&] { return got.ended.size() == 1; }));
}

TEST_CASE("send blocks while queued bytes exceed the cap, then delivers everything",
          "[exchange][transport]")
{
  // About 1.2 MB a batch against a 64 KiB cap: the first goes alone, the second waits for it.
  cluster c(2, 4, 4, {}, /*host=*/nullptr, {.max_queued_bytes = 64 * KiB});
  auto& sender = *c.ranks[0].t;
  auto& got    = *c.ranks[1].got;
  auto first   = gpu_batch(sample_table(30'000));
  auto second  = gpu_batch(sample_table(30'001));
  std::future<void> blocked;
  {
    auto const hold = hold_gpu();
    sender.send(5, 1, std::move(first));
    blocked = std::async(std::launch::async, [&] { sender.send(5, 1, std::move(second)); });
    CHECK(blocked.wait_for(std::chrono::milliseconds{500}) == std::future_status::timeout);
  }
  REQUIRE(blocked.wait_for(std::chrono::seconds{30}) == std::future_status::ready);
  blocked.get();
  sender.close(5, 1);
  REQUIRE(got.wait_for([&] { return got.ended.size() == 1; }));
  std::lock_guard lock{got.mutex};
  CHECK(got.batches[{0, 5}].size() == 2);
}

TEST_CASE("a cancelled query's sender stops waiting for room, and the transport carries on",
          "[exchange][transport]")
{
  // About 1.2 MB a batch against a 64 KiB cap and a receiver with no room: the second send waits
  // for room that the first never frees.
  cluster c(2, 4, 4, {}, /*host=*/nullptr, {.max_queued_bytes = 64 * KiB});
  auto& sender           = *c.ranks[0].t;
  auto& got              = *c.ranks[1].got;
  stream_id const stream = sirius::exchange::make_stream_id(5, 1);
  auto const query       = sirius::exchange::query_of(stream);
  auto first             = gpu_batch(sample_table(30'000));
  auto second            = gpu_batch(sample_table(30'001));
  {
    auto const hold = hold_gpu();
    sender.send(stream, 1, std::move(first));
    auto blocked =
      std::async(std::launch::async, [&] { sender.send(stream, 1, std::move(second)); });
    CHECK(blocked.wait_for(std::chrono::milliseconds{500}) == std::future_status::timeout);
    // The receiver still has no room: only the cancellation can wake the sender.
    sender.cancel(query);
    REQUIRE(blocked.wait_for(std::chrono::seconds{30}) == std::future_status::ready);
    CHECK_NOTHROW(blocked.get());
    // Nothing more is queued for the query, with or without room in the queue.
    CHECK_NOTHROW(sender.send(stream, 1, gpu_batch(sample_table(1'000))));
  }
  // What was queued before the cancellation still goes, and its end counts only that.
  sender.close(stream, 1);
  REQUIRE(got.wait_for([&] { return got.error != nullptr || got.ended.size() == 1; }));
  {
    std::lock_guard lock{got.mutex};
    REQUIRE_FALSE(got.error);
    CHECK(got.batches[{0, stream}].size() == 1);
  }
  CHECK(sender.wait_sent(query, std::chrono::seconds{30}));

  stream_id const next = sirius::exchange::make_stream_id(6, 1);
  sender.send(next, 1, gpu_batch(sample_table(5'000)));
  sender.close(next, 1);
  REQUIRE(got.wait_for([&] { return got.error != nullptr || got.ended.size() == 2; }));
  {
    std::lock_guard lock{got.mutex};
    REQUIRE_FALSE(got.error);
    CHECK(got.batches[{0, next}].size() == 1);
  }

  // Once forgotten, the number carries batches again.
  sender.forget(query, query);
  stream_id const again = sirius::exchange::make_stream_id(5, 2);
  sender.send(again, 1, gpu_batch(sample_table(5'000)));
  sender.close(again, 1);
  REQUIRE(got.wait_for([&] { return got.error != nullptr || got.ended.size() == 3; }));
  std::lock_guard lock{got.mutex};
  REQUIRE_FALSE(got.error);
  CHECK(got.batches[{0, again}].size() == 1);
}

TEST_CASE("a stopped transport wakes a sender waiting for room, which drops its batch",
          "[exchange][transport]")
{
  // As above: the second send waits for room that the first never frees.
  cluster c(2, 4, 4, {}, /*host=*/nullptr, {.max_queued_bytes = 64 * KiB});
  auto& sender           = *c.ranks[0].t;
  stream_id const stream = sirius::exchange::make_stream_id(5, 1);
  auto const query       = sirius::exchange::query_of(stream);
  auto first             = gpu_batch(sample_table(30'000));
  auto second            = gpu_batch(sample_table(30'001));
  auto const hold        = hold_gpu();
  sender.send(stream, 1, std::move(first));
  auto blocked = std::async(std::launch::async, [&] { sender.send(stream, 1, std::move(second)); });
  CHECK(blocked.wait_for(std::chrono::milliseconds{500}) == std::future_status::timeout);
  // The owner shuts down while the sender is still inside the transport.
  sender.stop();
  REQUIRE(blocked.wait_for(std::chrono::seconds{30}) == std::future_status::ready);
  CHECK_NOTHROW(blocked.get());
  CHECK_NOTHROW(sender.send(stream, 1, gpu_batch(sample_table(1'000))));
  // The first batch is still queued, and nothing will send it: the wait returns at once.
  auto const before = std::chrono::steady_clock::now();
  CHECK_FALSE(sender.wait_sent(query, std::chrono::seconds{30}));
  CHECK(std::chrono::steady_clock::now() - before < std::chrono::seconds{5});
}

TEST_CASE("a batch with no room asks for it once, and lands once it has it",
          "[exchange][transport]")
{
  // About 1.2 MB against a GPU held full and no host tier: it lands only after the hook frees room.
  std::optional<std::vector<std::unique_ptr<cucascade::memory::reservation>>> hold;
  std::atomic<int> asked{0};
  transport::options opts;
  opts.request_room = [&](std::size_t) {
    ++asked;
    hold.reset();
  };
  cluster c(2, 4, 4, {}, /*host=*/nullptr, opts);
  auto& got = *c.ranks[1].got;
  hold.emplace(hold_gpu());
  c.ranks[0].t->send(5, 1, gpu_batch(sample_table(30'000)));
  c.ranks[0].t->close(5, 1);
  REQUIRE(got.wait_for([&] { return got.error != nullptr || got.ended.size() == 1; }));
  std::lock_guard lock{got.mutex};
  CHECK_FALSE(got.error);
  CHECK(got.batches[{0, 5}].size() == 1);
  CHECK(asked == 1);
}

TEST_CASE("a transport that makes no progress for the stall timeout fails both ranks",
          "[exchange][transport]")
{
  // About 1.2 MB: more frames than credits, so the sender still holds the batch, and its memory,
  // while the receiver waits for room.
  cluster c(2, 4, 4, {}, /*host=*/nullptr, {.stall_timeout = std::chrono::milliseconds{300}});
  auto const hold = hold_gpu();
  c.ranks[0].t->send(5, 1, gpu_batch(sample_table(30'000)));
  for (auto& r : c.ranks) {
    auto& got = *r.got;
    REQUIRE(got.wait_for([&] { return got.error != nullptr; }, std::chrono::seconds{30}));
  }
  CHECK_THROWS_WITH(std::rethrow_exception(c.ranks[1].got->error),
                    Catch::Matchers::ContainsSubstring("no progress"));
}

TEST_CASE("an end whose promised batches never arrive fails the receiver at the stall timeout",
          "[exchange][transport]")
{
  using namespace sirius::exchange;
  transport::options const quick{.stall_timeout = std::chrono::milliseconds{300}};
  constexpr stream_id stream = 5;
  SECTION("one batch promised, none sent")
  {
    forged_peer c(quick);
    c.forge(end_message{stream, 1});
    c.fails_with("no progress");
  }
  SECTION("no batch promised: the stream ends, and an idle transport does not stall")
  {
    forged_peer c(quick);
    c.forge(end_message{stream, 0});
    REQUIRE(c.got.wait_for([&] { return c.got.ended.size() == 1; }, std::chrono::seconds{30}));
    CHECK_FALSE(c.got.wait_for([&] { return c.got.error != nullptr; }, std::chrono::seconds{1}));
  }
  SECTION("ends that overtake their batches are not a stall")
  {
    cluster c(3,
              3,
              6,
              {.shuffle = true, .deliver_every = 5},
              env().host,
              {.stall_timeout = std::chrono::seconds{2}});
    all_to_all(c, {1, 2}, 2, 8'000);
  }
}

TEST_CASE("a failed write fails both ranks, and frees the batches the sender held",
          "[exchange][transport]")
{
  // About 1.2 MB: 19 frames of 64 KiB, more than the 4 send frames, so the batch is still held
  // when the first write fails.
  cluster c(2, 4, 4, {.fail_after_writes = 0}, env().host);
  auto& sender           = *c.ranks[0].t;
  stream_id const stream = sirius::exchange::make_stream_id(5, 1);
  sender.send(stream, 1, gpu_batch(sample_table(30'000)));
  for (auto& r : c.ranks) {
    auto& got = *r.got;
    REQUIRE(got.wait_for([&] { return got.error != nullptr; }));
  }
  CHECK(sender.wait_sent(sirius::exchange::query_of(stream), std::chrono::seconds{5}));
  CHECK_THROWS(sender.send(stream, 1, gpu_batch(sample_table(1'000))));
  CHECK_THROWS(c.ranks[1].t->close(stream, 0));
}

TEST_CASE("a failed transport frees the batches it was still receiving", "[exchange][transport]")
{
  using namespace sirius::exchange;
  forged_peer c;
  auto const header = many_frame_header();
  auto const before = env().gpu->get_available_memory();
  // Frame 0 of about 19: the batch lands, taking its whole size, and stays incomplete.
  c.forge(frame_message{5, 0, 0, 0, 64 * KiB, header});
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
  while (env().gpu->get_available_memory() >= before) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  c.forge(error_message{"forged failure"});
  c.fails_with("forged failure");
  // The transport is still alive: its failure freed the landing, not its destructor.
  CHECK(env().gpu->get_available_memory() == before);
}

TEST_CASE("a transport destroyed right after close and abort still sends them",
          "[exchange][transport]")
{
  // A rank that already holds its peers' ends may tear down the moment it closes its own, as a
  // barrier's slowest rank does. Many ends make it likely some are still queued at the teardown.
  cluster c(2, 4, 4);
  int const streams = 100;
  for (int s = 0; s < streams; ++s) {
    c.ranks[0].t->close(static_cast<stream_id>(s), 1);
  }
  c.ranks[0].t->abort(7, "stopped");
  c.ranks[0].t.reset();
  auto& got = *c.ranks[1].got;
  REQUIRE(got.wait_for(
    [&] { return got.error != nullptr && got.ended.size() == static_cast<std::size_t>(streams); },
    std::chrono::seconds{10}));
  CHECK_THROWS_WITH(std::rethrow_exception(got.error),
                    Catch::Matchers::ContainsSubstring("stopped"));
}

TEST_CASE("exchange messages round-trip and refuse malformed bytes", "[exchange][transport]")
{
  using namespace sirius::exchange;
  frame_message several{7, 3, 4, 2, 3 * 1024, {}};
  several.frames = 3;
  std::vector<message> const all{frame_message{7, 3, 0, 5, 1024, {1, 2, 3}},
                                 frame_message{7, 3, 4, 6, 99, {}},
                                 several,
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
  // A write of no frames.
  frame_message none{1, 1, 2, 0, 8, {}};
  none.frames = 0;
  CHECK_THROWS_AS(decode(encode(none)), sirius::invalid_input_exception);
  // A frame and an end with a byte too many, a credit with one too few.
  auto frame = encode(frame_message{7, 3, 0, 5, 1024, {1, 2, 3}});
  frame.push_back('\0');
  CHECK_THROWS_AS(decode(frame), sirius::invalid_input_exception);
  auto credit = encode(credit_message{{1, 2, 3}});
  credit.pop_back();
  CHECK_THROWS_AS(decode(credit), sirius::invalid_input_exception);
  auto ended = encode(end_message{7, 12});
  ended.push_back('\0');
  CHECK_THROWS_AS(decode(ended), sirius::invalid_input_exception);
  // The kind and 7 of the query's 8 bytes.
  auto const aborted = encode(abort_message{3, ""}).substr(0, 8);
  CHECK_THROWS_AS(decode(aborted), sirius::invalid_input_exception);
}

TEST_CASE(
  "a frame ring refuses no frames, frames of no or unaligned size, and a size that "
  "overflows",
  "[exchange][transport]")
{
  CHECK_THROWS_AS(frame_ring(0, 256), sirius::invalid_input_exception);
  CHECK_THROWS_AS(frame_ring(1, 100), sirius::invalid_input_exception);
  CHECK_THROWS_AS(frame_ring(1, 0), sirius::invalid_input_exception);
  auto const huge_frame = std::numeric_limits<std::size_t>::max() & ~std::size_t{255};
  CHECK_THROWS_AS(frame_ring(std::numeric_limits<std::uint32_t>::max(), huge_frame),
                  sirius::invalid_input_exception);
  CHECK_THROWS_WITH(frame_ring(std::numeric_limits<std::uint32_t>::max(), huge_frame),
                    Catch::Matchers::ContainsSubstring("overflow"));
}

TEST_CASE(
  "a transport refuses fewer receive frames than peers, frames of two sizes, and an unset "
  "callback",
  "[exchange][transport]")
{
  loopback_hub hub(3);
  frame_ring send(4, 64 * KiB);
  received got;
  SECTION("one receive frame for two peers")
  {
    frame_ring recv(1, 64 * KiB);
    auto fab = hub.attach(0, send, recv);
    CHECK_THROWS_AS(transport(*fab, send, recv, *env().gpu, env().host, recording(got)),
                    sirius::invalid_input_exception);
  }
  SECTION("receive frames larger than send frames")
  {
    frame_ring recv(8, 128 * KiB);
    auto fab = hub.attach(0, send, recv);
    CHECK_THROWS_AS(transport(*fab, send, recv, *env().gpu, env().host, recording(got)),
                    sirius::invalid_input_exception);
  }
  SECTION("a callback left unset")
  {
    frame_ring recv(8, 64 * KiB);
    auto fab = hub.attach(0, send, recv);
    std::vector<std::function<void(transport::callbacks&)>> const unset{
      [](transport::callbacks& cb) { cb.on_batch = nullptr; },
      [](transport::callbacks& cb) { cb.on_end = nullptr; },
      [](transport::callbacks& cb) { cb.on_error = nullptr; },
      [](transport::callbacks& cb) { cb.on_abort = nullptr; }};
    for (auto const& clear : unset) {
      auto cb = recording(got);
      clear(cb);
      CHECK_THROWS_WITH(transport(*fab, send, recv, *env().gpu, env().host, std::move(cb)),
                        Catch::Matchers::ContainsSubstring("every transport callback must be set"));
    }
  }
}

TEST_CASE("a malformed notification from a peer fails the receiver", "[exchange][transport]")
{
  using namespace sirius::exchange;
  forged_peer c;
  constexpr stream_id stream = 5;
  // Rank 1 holds rank 0's receive frames 0 to 3 as credits, and rank 0 holds rank 1's 0 to 3.
  SECTION("bytes that are not a message")
  {
    c.peer->notify(0, std::string(1, '\x09'));
    c.fails_with("unknown kind");
  }
  SECTION("a write into a receive frame the sender does not hold")
  {
    c.forge(frame_message{stream, 0, 1, 4, 8, {}});
    c.fails_with("which it does not hold");
  }
  SECTION("a credit for a frame that is not the receiver's")
  {
    c.forge(credit_message{{4}});
    c.fails_with("not one of ours");
  }
  SECTION("a stream ended twice")
  {
    // One batch expected, none sent: the first end stays open.
    c.forge(end_message{stream, 1});
    c.forge(end_message{stream, 1});
    c.fails_with("ended stream 5 twice");
  }
  SECTION("the peer's own failure")
  {
    c.forge(error_message{"forged failure"});
    c.fails_with("rank 1 failed: forged failure");
  }
  SECTION("a batch's frame 0 twice")
  {
    auto const header = many_frame_header();
    c.forge(frame_message{stream, 0, 0, 0, 64 * KiB, header});
    c.forge(frame_message{stream, 0, 0, 1, 64 * KiB, header});
    c.fails_with("sent batch 0 of stream 5 twice");
  }
  SECTION("a frame of more bytes than its place in the batch")
  {
    c.forge(frame_message{stream, 0, 0, 0, 64 * KiB + 1, many_frame_header()});
    c.fails_with("does not fit");
  }
}

TEST_CASE("sirius_exchange_host_staging takes auto, always, never or remote",
          "[exchange][transport]")
{
  using sirius::exchange::host_staging;
  using sirius::exchange::parse_host_staging;
  CHECK(parse_host_staging("auto") == host_staging::automatic);
  CHECK(parse_host_staging("always") == host_staging::always);
  CHECK(parse_host_staging("never") == host_staging::never);
  CHECK(parse_host_staging("remote") == host_staging::remote);
  CHECK_THROWS_AS(parse_host_staging(""), sirius::invalid_input_exception);
  CHECK_THROWS_AS(parse_host_staging("Always"), sirius::invalid_input_exception);
}
