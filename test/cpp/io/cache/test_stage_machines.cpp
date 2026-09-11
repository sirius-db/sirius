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
#include "io/cache/prefetching_cache.hpp"
#include "io/cache/types.hpp"

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using sirius::io::cache::consumer_stage;
using sirius::io::cache::producer_stage;

namespace {

constexpr auto WAIT_TIMEOUT = std::chrono::seconds(10);

constexpr producer_stage::value PRODUCER_STATES[] = {producer_stage::initialized,
                                                     producer_stage::queued,
                                                     producer_stage::preparing,
                                                     producer_stage::prepared,
                                                     producer_stage::loading,
                                                     producer_stage::ready,
                                                     producer_stage::abandoned};

constexpr consumer_stage::value CONSUMER_STATES[] = {consumer_stage::initialized,
                                                     consumer_stage::queued,
                                                     consumer_stage::preparing,
                                                     consumer_stage::reading,
                                                     consumer_stage::disposed};

void advance_to(producer_stage& stage, producer_stage::value target)
{
  if (target == producer_stage::abandoned) {
    stage.mark_abandoned();
    return;
  }
  if (target == producer_stage::initialized) return;
  REQUIRE(stage.mark_queued());
  if (target == producer_stage::queued) return;
  REQUIRE(stage.mark_preparing());
  if (target == producer_stage::preparing) return;
  REQUIRE(stage.mark_prepared());
  if (target == producer_stage::prepared) return;
  REQUIRE(stage.mark_loading());
  if (target == producer_stage::loading) return;
  REQUIRE(stage.mark_ready());
}

void advance_to(consumer_stage& stage, consumer_stage::value target)
{
  if (target == consumer_stage::initialized) return;
  REQUIRE(stage.mark_queued());
  if (target == consumer_stage::queued) return;
  REQUIRE(stage.mark_preparing());
  if (target == consumer_stage::preparing) return;
  REQUIRE(stage.mark_reading());
  if (target == consumer_stage::reading) return;
  stage.mark_disposed();
}

// Runs `park` on a helper thread and requires that it finishes within
// WAIT_TIMEOUT once `release` has run, so a missed notify fails instead of
// hanging the suite.  Returns what `park` returned.
template <class Park, class Release>
bool wait_is_released_by(Park&& park, Release&& release)
{
  auto started = std::make_shared<std::promise<void>>();
  auto entered = started->get_future();

  auto done = std::async(std::launch::async, [park = std::forward<Park>(park), started]() mutable {
    started->set_value();
    return park();
  });

  REQUIRE(entered.wait_for(WAIT_TIMEOUT) == std::future_status::ready);
  release();
  REQUIRE(done.wait_for(WAIT_TIMEOUT) == std::future_status::ready);
  return done.get();
}

}  // namespace

TEST_CASE("producer_stage walks the full legal path", "[stage]")
{
  producer_stage stage;
  CHECK(stage.get() == producer_stage::initialized);
  CHECK(stage.mark_queued());
  CHECK(stage.get() == producer_stage::queued);
  CHECK(stage.mark_preparing());
  CHECK(stage.get() == producer_stage::preparing);
  CHECK(stage.mark_prepared());
  CHECK(stage.get() == producer_stage::prepared);
  CHECK(stage.mark_loading());
  CHECK(stage.get() == producer_stage::loading);
  CHECK(stage.mark_ready());
  CHECK(stage.get() == producer_stage::ready);
}

TEST_CASE("producer_stage forward transitions are monotone-max", "[stage]")
{
  auto check = [](producer_stage::value from, producer_stage::value target, auto mark) {
    producer_stage stage;
    advance_to(stage, from);
    bool const expected = from < target;
    CHECK(mark(stage) == expected);
    CHECK(stage.get() == (expected ? target : from));
  };

  for (auto from : PRODUCER_STATES) {
    check(from, producer_stage::queued, [](producer_stage& s) { return s.mark_queued(); });
    check(from, producer_stage::preparing, [](producer_stage& s) { return s.mark_preparing(); });
    check(from, producer_stage::prepared, [](producer_stage& s) { return s.mark_prepared(); });
    check(from, producer_stage::loading, [](producer_stage& s) { return s.mark_loading(); });
    check(from, producer_stage::ready, [](producer_stage& s) { return s.mark_ready(); });
  }
}

TEST_CASE("producer_stage skips straight to the target stage", "[stage]")
{
  producer_stage stage;
  CHECK(stage.mark_loading());
  CHECK(stage.get() == producer_stage::loading);
  CHECK_FALSE(stage.mark_queued());
  CHECK_FALSE(stage.mark_preparing());
  CHECK_FALSE(stage.mark_prepared());
  CHECK_FALSE(stage.mark_loading());
  CHECK(stage.get() == producer_stage::loading);

  producer_stage jumped;
  CHECK(jumped.mark_ready());
  CHECK(jumped.get() == producer_stage::ready);
}

TEST_CASE("producer_stage mark_load_failed only fires from loading", "[stage]")
{
  for (auto from : PRODUCER_STATES) {
    producer_stage stage;
    advance_to(stage, from);
    bool const expected = from == producer_stage::loading;
    CHECK(stage.mark_load_failed() == expected);
    CHECK(stage.get() == (expected ? producer_stage::prepared : from));
  }
}

TEST_CASE("producer_stage can be abandoned from any state", "[stage]")
{
  for (auto from : PRODUCER_STATES) {
    producer_stage stage;
    advance_to(stage, from);
    stage.mark_abandoned();
    CHECK(stage.get() == producer_stage::abandoned);
    CHECK_FALSE(stage.mark_queued());
    CHECK_FALSE(stage.mark_prepared());
    CHECK_FALSE(stage.mark_ready());
    CHECK_FALSE(stage.mark_load_failed());
    CHECK(stage.get() == producer_stage::abandoned);
  }
}

TEST_CASE("producer_stage load failure reverts to prepared and can retry", "[stage]")
{
  producer_stage stage;
  advance_to(stage, producer_stage::loading);
  CHECK(stage.mark_load_failed());
  CHECK(stage.get() == producer_stage::prepared);
  CHECK(stage.mark_loading());
  CHECK(stage.mark_ready());
  CHECK(stage.get() == producer_stage::ready);
}

TEST_CASE("producer_stage wait_for_prepared is released by mark_prepared", "[stage]")
{
  auto stage = std::make_shared<producer_stage>();
  advance_to(*stage, producer_stage::preparing);

  bool const reached = wait_is_released_by([stage] { return stage->wait_for_prepared(); },
                                           [stage] { CHECK(stage->mark_prepared()); });

  CHECK(reached);
  CHECK(stage->get() == producer_stage::prepared);
}

TEST_CASE("producer_stage wait_for_prepared is released by mark_abandoned", "[stage]")
{
  auto stage = std::make_shared<producer_stage>();
  advance_to(*stage, producer_stage::preparing);

  bool const reached = wait_is_released_by([stage] { return stage->wait_for_prepared(); },
                                           [stage] { stage->mark_abandoned(); });

  CHECK_FALSE(reached);
  CHECK(stage->get() == producer_stage::abandoned);
}

TEST_CASE("producer_stage wait_till_not_loading is released by mark_ready", "[stage]")
{
  auto stage = std::make_shared<producer_stage>();
  advance_to(*stage, producer_stage::loading);

  bool const reached = wait_is_released_by([stage] { return stage->wait_till_not_loading(); },
                                           [stage] { CHECK(stage->mark_ready()); });

  CHECK(reached);
  CHECK(stage->get() == producer_stage::ready);
}

TEST_CASE("producer_stage wait_till_not_loading is released by mark_load_failed", "[stage]")
{
  auto stage = std::make_shared<producer_stage>();
  advance_to(*stage, producer_stage::loading);

  bool const reached = wait_is_released_by([stage] { return stage->wait_till_not_loading(); },
                                           [stage] { CHECK(stage->mark_load_failed()); });

  CHECK_FALSE(reached);
  CHECK(stage->get() == producer_stage::prepared);
}

TEST_CASE("producer_stage wait_till_not_loading is released by mark_abandoned", "[stage]")
{
  auto stage = std::make_shared<producer_stage>();
  advance_to(*stage, producer_stage::loading);

  bool const reached = wait_is_released_by([stage] { return stage->wait_till_not_loading(); },
                                           [stage] { stage->mark_abandoned(); });

  CHECK_FALSE(reached);
  CHECK(stage->get() == producer_stage::abandoned);
}

TEST_CASE("producer_stage waits return immediately when the state already moved on", "[stage]")
{
  producer_stage prepared;
  advance_to(prepared, producer_stage::prepared);
  CHECK(prepared.wait_for_prepared());
  CHECK(prepared.get() == producer_stage::prepared);

  producer_stage ready;
  advance_to(ready, producer_stage::ready);
  CHECK(ready.wait_for_prepared());
  // wait_till_not_loading reports on a load it witnessed finish; a request already
  // past `loading` has nothing to wait out, so it declines rather than
  // claiming credit for a load this call never saw.
  CHECK_FALSE(ready.wait_till_not_loading());
  CHECK(ready.get() == producer_stage::ready);

  producer_stage abandoned;
  advance_to(abandoned, producer_stage::abandoned);
  CHECK_FALSE(abandoned.wait_for_prepared());
  CHECK_FALSE(abandoned.wait_till_not_loading());
}

TEST_CASE("consumer_stage walks the full legal path", "[stage]")
{
  consumer_stage stage;
  CHECK(stage.get() == consumer_stage::initialized);
  CHECK(stage.mark_queued());
  CHECK(stage.get() == consumer_stage::queued);
  CHECK(stage.mark_preparing());
  CHECK(stage.get() == consumer_stage::preparing);
  CHECK(stage.mark_reading());
  CHECK(stage.get() == consumer_stage::reading);
  stage.mark_disposed();
  CHECK(stage.get() == consumer_stage::disposed);
}

TEST_CASE("consumer_stage forward transitions are monotone-max", "[stage]")
{
  auto check = [](consumer_stage::value from, consumer_stage::value target, auto mark) {
    consumer_stage stage;
    advance_to(stage, from);
    bool const expected = from < target;
    CHECK(mark(stage) == expected);
    CHECK(stage.get() == (expected ? target : from));
  };

  for (auto from : CONSUMER_STATES) {
    check(from, consumer_stage::queued, [](consumer_stage& s) { return s.mark_queued(); });
    check(from, consumer_stage::preparing, [](consumer_stage& s) { return s.mark_preparing(); });
    check(from, consumer_stage::reading, [](consumer_stage& s) { return s.mark_reading(); });
  }
}

TEST_CASE("consumer_stage skips forward but never goes backwards", "[stage]")
{
  consumer_stage stage;
  CHECK(stage.mark_reading());
  CHECK(stage.get() == consumer_stage::reading);
  CHECK_FALSE(stage.mark_queued());
  CHECK_FALSE(stage.mark_preparing());
  CHECK_FALSE(stage.mark_reading());
  CHECK(stage.get() == consumer_stage::reading);
}

TEST_CASE("consumer_stage can be disposed from any state", "[stage]")
{
  for (auto from : CONSUMER_STATES) {
    consumer_stage stage;
    advance_to(stage, from);
    stage.mark_disposed();
    CHECK(stage.get() == consumer_stage::disposed);
    CHECK_FALSE(stage.mark_queued());
    CHECK_FALSE(stage.mark_reading());
    CHECK(stage.get() == consumer_stage::disposed);
  }
}

namespace {

class fake_io_object : public sirius::io::io_object {
 public:
  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return _path; }
  [[nodiscard]] size_t size() const noexcept override { return 1024; }

 private:
  std::string _path{"memory://stage-machines"};
};

sirius::io::cache::prefetch_request make_request(const std::shared_ptr<sirius::io::io_object>& obj,
                                                 const std::shared_ptr<consumer_stage>& consumer)
{
  sirius::io::cache::prefetch_request req;
  req.obj      = obj;
  req.producer = std::make_shared<sirius::io::cache::producer_stage>();
  req.consumer = consumer;
  return req;
}

}  // namespace

TEST_CASE("prefetch_request predicates follow the consumer stage", "[stage][cache]")
{
  auto obj      = std::make_shared<fake_io_object>();
  auto consumer = std::make_shared<consumer_stage>();
  auto req      = make_request(obj, consumer);

  CHECK_FALSE(req.is_active());
  CHECK_FALSE(req.is_cancelled());

  REQUIRE(consumer->mark_queued());
  CHECK(req.is_active());
  CHECK_FALSE(req.is_cancelled());

  REQUIRE(consumer->mark_preparing());
  CHECK(req.is_active());

  REQUIRE(consumer->mark_reading());
  CHECK(req.is_active());

  consumer->mark_disposed();
  CHECK_FALSE(req.is_active());
  CHECK(req.is_cancelled());
}

TEST_CASE("prefetch_request predicates hold when stages are skipped", "[stage][cache]")
{
  auto obj      = std::make_shared<fake_io_object>();
  auto consumer = std::make_shared<consumer_stage>();
  auto req      = make_request(obj, consumer);

  CHECK_FALSE(req.is_active());

  REQUIRE(consumer->mark_reading());
  CHECK(req.is_active());
  CHECK_FALSE(req.is_cancelled());

  consumer->mark_disposed();
  CHECK_FALSE(req.is_active());
  CHECK(req.is_cancelled());
}

TEST_CASE("has_fallen_behind turns a prefetch away once the reader passed it", "[stage][cache]")
{
  auto obj      = std::make_shared<fake_io_object>();
  auto consumer = std::make_shared<consumer_stage>();
  auto req      = make_request(obj, consumer);

  // Consumer still behind the readahead: the prefetch is worth starting.
  CHECK_FALSE(req.has_fallen_behind());
  REQUIRE(consumer->mark_queued());
  CHECK_FALSE(req.has_fallen_behind());

  // The executor reached `preparing` before any IO was issued -- it is pulling
  // the bytes itself now, so starting a prefetch would duplicate that read.
  REQUIRE(consumer->mark_preparing());
  CHECK(req.has_fallen_behind());

  // Once the IO is in flight there is nothing left to call off, so the answer
  // flips back regardless of how far the consumer has run ahead.
  REQUIRE(req.producer->mark_loading());
  CHECK_FALSE(req.has_fallen_behind());
  REQUIRE(consumer->mark_reading());
  CHECK_FALSE(req.has_fallen_behind());

  // ...and disposal is is_cancelled's question from here on, not this one.
  consumer->mark_disposed();
  CHECK_FALSE(req.has_fallen_behind());
  CHECK(req.is_cancelled());
}

TEST_CASE("a prefetch_request with no consumer has nothing to run ahead of", "[stage][cache]")
{
  auto obj = std::make_shared<fake_io_object>();
  auto req = make_request(obj, nullptr);

  CHECK(req.has_fallen_behind());
}

TEST_CASE("prefetch_request without a consumer is cancelled", "[stage][cache]")
{
  auto obj = std::make_shared<fake_io_object>();
  auto req = make_request(obj, nullptr);

  CHECK_FALSE(req.is_active());
  CHECK(req.is_cancelled());
}

TEST_CASE("prepared requires every chunk to own a buffer", "[stage][cache]")
{
  using sirius::io::cache::all_chunks_have_buffers;
  using sirius::io::cache::cached_chunk;

  auto first  = std::make_unique<cached_chunk>(0);
  auto second = std::make_unique<cached_chunk>(4096);
  std::vector<cached_chunk*> chunks{first.get(), second.get()};

  REQUIRE(first->state.mark_queued());
  REQUIRE(first->state.mark_allocated());

  CHECK_FALSE(all_chunks_have_buffers(chunks));

  REQUIRE(second->state.mark_queued());
  CHECK_FALSE(all_chunks_have_buffers(chunks));

  REQUIRE(second->state.mark_allocated());
  CHECK(all_chunks_have_buffers(chunks));

  REQUIRE(second->state.mark_evicting());
  CHECK_FALSE(all_chunks_have_buffers(chunks));
}

TEST_CASE("unbacked chunks abandon the producer instead of wedging it", "[stage][cache]")
{
  using sirius::io::cache::all_chunks_have_buffers;
  using sirius::io::cache::cached_chunk;

  auto chunk = std::make_unique<cached_chunk>(0);
  std::vector<cached_chunk*> chunks{chunk.get()};

  producer_stage producer;
  REQUIRE(producer.mark_queued());
  REQUIRE(producer.mark_preparing());

  if (!all_chunks_have_buffers(chunks)) { producer.mark_abandoned(); }
  CHECK(producer.get() == producer_stage::abandoned);
  CHECK_FALSE(producer.mark_loading());
  CHECK_FALSE(producer.wait_for_prepared());
}

TEST_CASE("prepared fires once the chunks are backed", "[stage][cache]")
{
  using sirius::io::cache::all_chunks_have_buffers;
  using sirius::io::cache::cached_chunk;

  auto chunk = std::make_unique<cached_chunk>(0);
  std::vector<cached_chunk*> chunks{chunk.get()};

  producer_stage producer;
  REQUIRE(producer.mark_queued());
  REQUIRE(producer.mark_preparing());

  REQUIRE(chunk->state.mark_queued());
  REQUIRE(chunk->state.mark_allocated());
  REQUIRE(all_chunks_have_buffers(chunks));
  CHECK(producer.mark_prepared());
  CHECK(producer.get() == producer_stage::prepared);
  CHECK(producer.mark_loading());
}

TEST_CASE("to_consumer_stage maps every scan_stage", "[stage][cache]")
{
  using sirius::io::cache::scan_stage;
  using sirius::io::cache::to_consumer_stage;

  CHECK_FALSE(to_consumer_stage(scan_stage::none).has_value());
  CHECK(to_consumer_stage(scan_stage::initialized) == consumer_stage::initialized);
  CHECK(to_consumer_stage(scan_stage::queued) == consumer_stage::queued);
  CHECK(to_consumer_stage(scan_stage::preparing) == consumer_stage::preparing);
  CHECK(to_consumer_stage(scan_stage::reading) == consumer_stage::reading);
  CHECK(to_consumer_stage(scan_stage::disposed) == consumer_stage::disposed);
}

TEST_CASE("to_consumer_stage results drive the consumer machine forward", "[stage][cache]")
{
  using sirius::io::cache::scan_stage;
  using sirius::io::cache::to_consumer_stage;

  consumer_stage consumer;
  CHECK(consumer.mark(*to_consumer_stage(scan_stage::queued)));
  CHECK(consumer.get() == consumer_stage::queued);
  CHECK_FALSE(consumer.mark(*to_consumer_stage(scan_stage::initialized)));
  CHECK(consumer.mark(*to_consumer_stage(scan_stage::disposed)));
  CHECK(consumer.get() == consumer_stage::disposed);
}
