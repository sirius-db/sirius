/*
 * Copyright 2025, Sirius Contributors.
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

// The batch_stream primitive on its own — no operator, no pipeline, no task protocol. The batches
// are real (a repository stores data_batches), so these need a GPU.

#include "operator/operator_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <exec/batch_stream.hpp>
#include <sirius/exception.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace cucascade;
using namespace cucascade::memory;
using namespace std::chrono_literals;

using availability = batch_stream::availability;

namespace {

using namespace sirius::test::operator_utils;

/// The single sender used by the ordinary (non-fan-in) tests.
constexpr sender_id_t SOLE_SENDER = 0;

/// A stream over a fresh repository, plus the repository — held separately on purpose: the caller
/// owns and registers it, the stream only borrows it.
auto make_stream(std::set<sender_id_t> expected = {SOLE_SENDER})
{
  auto repo   = std::make_shared<cucascade::shared_data_repository>();
  auto stream = std::make_shared<batch_stream>(repo, std::move(expected));
  return std::make_tuple(std::move(stream), std::move(repo));
}

std::exception_ptr producer_failure(const std::string& what = "producer failed")
{
  return std::make_exception_ptr(std::runtime_error(what));
}

}  // namespace

// ============================================================================
// BSTR-1: what goes in comes out — same objects, in order
// ============================================================================

TEST_CASE("batch_stream BSTR-1: batches are pulled FIFO with identity", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [stream, repo] = make_stream();

  std::vector<std::shared_ptr<cucascade::data_batch>> pushed;
  for (int i = 0; i < 4; ++i) {
    auto batch = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    pushed.push_back(batch);
    REQUIRE(stream->push(batch));
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> pulled;
  while (auto batch = stream->try_pull()) {
    pulled.push_back(std::move(batch));
  }
  // Pointer identity, not just batch ids: the stream never copies or rewraps a batch.
  REQUIRE(pulled == pushed);
}

// ============================================================================
// BSTR-2: a queued batch lives in the repository, which is what keeps it spillable.
//         A private queue inside the stream would not be.
// ============================================================================

TEST_CASE("batch_stream BSTR-2: queued batches are held by the repository", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [stream, repo] = make_stream();
  auto batch          = make_numeric_batch<int32_t>(*gpu_space, {1, 2}, cudf::type_id::INT32);
  auto batch_id       = batch->get_batch_id();
  REQUIRE(stream->push(batch));

  REQUIRE(repo->total_size() == 1);
  REQUIRE(repo->get_batch_ids() == std::vector<uint64_t>{batch_id});
  REQUIRE(stream->repository() == repo);

  // And it leaves the repository when the consumer takes it.
  REQUIRE(stream->try_pull() != nullptr);
  REQUIRE(repo->total_size() == 0);
}

// ============================================================================
// BSTR-3: push is admission — refused once terminal, announced when accepted (S1)
// ============================================================================

TEST_CASE("batch_stream BSTR-3: push is refused after the stream ends", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [stream, repo] = make_stream();

  int woken = 0;
  stream->set_on_data([&] { ++woken; });

  REQUIRE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  REQUIRE(woken == 1);
  REQUIRE(repo->total_size() == 1);

  // Every push announces, not just the first: the hook is not one-shot, so nothing had to be
  // re-armed in between and no push can race past an un-armed notification.
  REQUIRE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32)));
  REQUIRE(woken == 2);

  stream->close(SOLE_SENDER);

  // A late batch must not appear behind a consumer that already saw the end — and a refused
  // push is not announced.
  REQUIRE_FALSE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {3}, cudf::type_id::INT32)));
  REQUIRE(repo->total_size() == 2);
  REQUIRE(woken == 2);
}

// ============================================================================
// BSTR-4: classify() truth table with no error in play
// ============================================================================

TEST_CASE("batch_stream BSTR-4: classify truth table", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [stream, repo] = make_stream();

  // Open + empty: more may still arrive.
  REQUIRE(stream->classify() == availability::WAITING);
  REQUIRE_FALSE(stream->drained());

  REQUIRE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  REQUIRE(stream->classify() == availability::HAS_DATA);
  REQUIRE_FALSE(stream->drained());  // open + data

  // Terminal but still holding data: data wins, so nothing queued is lost to an early EOS.
  stream->close(SOLE_SENDER);
  REQUIRE(stream->classify() == availability::HAS_DATA);
  REQUIRE_FALSE(stream->drained());

  REQUIRE(stream->try_pull() != nullptr);
  REQUIRE(stream->classify() == availability::END_OF_STREAM);
  REQUIRE(stream->drained());
}

// ============================================================================
// BSTR-5: fan-in ends by sender identity, not by close count
// ============================================================================

TEST_CASE("batch_stream BSTR-5: a repeated close cannot end a fan-in stream", "[batch_stream]")
{
  auto [stream, repo] = make_stream({0, 1});

  stream->close(0);
  stream->close(0);
  REQUIRE(stream->classify() == availability::WAITING);
  REQUIRE_FALSE(stream->drained());

  // An unexpected sender is a wiring bug, not something to silently count — and the rejected
  // close leaves no trace: EOS still needs the real second sender.
  REQUIRE_THROWS_AS(stream->close(7), sirius::invalid_input_exception);
  REQUIRE(stream->classify() == availability::WAITING);

  stream->close(1);
  REQUIRE(stream->classify() == availability::END_OF_STREAM);
  REQUIRE(stream->drained());
}

// ============================================================================
// BSTR-6: the blocking consumer surface — wait() and the end-of-stream hook
// ============================================================================

TEST_CASE("batch_stream BSTR-6: wait unblocks on a push and on the final close", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [stream, repo] = make_stream({0, 1});

  int ended = 0;
  stream->set_on_end_of_stream([&] { ++ended; });

  // (a) a push wakes a blocked consumer
  std::atomic<bool> returned{false};
  std::thread consumer([&] {
    stream->wait();
    returned = true;
  });
  std::this_thread::sleep_for(20ms);
  REQUIRE_FALSE(returned.load());
  REQUIRE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  consumer.join();
  REQUIRE(returned.load());
  REQUIRE(ended == 0);

  REQUIRE(stream->try_pull() != nullptr);

  // (b) the final close wakes it too, and ends the stream exactly once
  returned = false;
  std::thread waiter([&] {
    stream->wait();
    returned = true;
  });
  std::this_thread::sleep_for(20ms);
  stream->close(0);
  std::this_thread::sleep_for(20ms);
  REQUIRE_FALSE(returned.load());  // fan-in incomplete
  stream->close(1);
  waiter.join();
  REQUIRE(returned.load());
  REQUIRE(ended == 1);

  // Repeat closes after terminal must not re-fire it.
  stream->close(0);
  stream->close(1);
  REQUIRE(ended == 1);

  // (c) a hook wired after the fact is never lost
  int late = 0;
  stream->set_on_end_of_stream([&] { ++late; });
  REQUIRE(late == 1);
}

// ============================================================================
// BSTR-7: fail() is immediate, fail-fast terminal, and first-wins (S2 / P1–P3)
// ============================================================================

TEST_CASE("batch_stream BSTR-7: fail() poisons the stream at once", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [stream, repo] = make_stream({0, 1});
  REQUIRE(stream->pending_error() == nullptr);

  stream->fail(producer_failure("first"));

  // P1 / P3: visible now, and sender 1 is not waited for.
  REQUIRE(stream->pending_error() != nullptr);
  REQUIRE_FALSE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  REQUIRE(repo->total_size() == 0);

  // P2: neither a second failure nor a clean close displaces the original cause.
  stream->fail(producer_failure("second"));
  stream->close(1);
  try {
    std::rethrow_exception(stream->pending_error());
  } catch (const std::runtime_error& e) {
    REQUIRE(std::string(e.what()) == "first");
  }
}

// ============================================================================
// BSTR-8: try_pull rethrows before it pops — error beats data (S4)
// ============================================================================

TEST_CASE("batch_stream BSTR-8: try_pull rethrows ahead of queued batches", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [stream, repo] = make_stream();
  REQUIRE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  REQUIRE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32)));

  stream->fail(producer_failure());

  // Batches queued behind the failure are dropped: a consumer must not keep processing data
  // produced before an error it has not seen.
  REQUIRE_THROWS_AS(stream->try_pull(), std::runtime_error);
  REQUIRE(repo->total_size() == 2);
  // Every call rethrows — the error is never consumed away by asking again.
  REQUIRE_THROWS_AS(stream->try_pull(), std::runtime_error);
}

// ============================================================================
// BSTR-9: the poison announces itself like data (S2 / P4)
// ============================================================================

TEST_CASE("batch_stream BSTR-9: fail() wakes the consumer and reads as HAS_DATA", "[batch_stream]")
{
  auto [stream, repo] = make_stream({0, 1});

  int woken = 0;
  int ended = 0;
  stream->set_on_data([&] { ++woken; });
  stream->set_on_end_of_stream([&] { ++ended; });

  stream->fail(producer_failure());

  // Only on_data reschedules a consumer parked on WAITING; without it the rethrow sits unread.
  // on_end_of_stream fires too, because the failure is what made the stream terminal.
  REQUIRE(woken == 1);
  REQUIRE(ended == 1);
  REQUIRE(repo->total_size() == 0);
  REQUIRE(stream->classify() == availability::HAS_DATA);

  // A later failure changes nothing: an error is already waiting, and the stream ends once.
  stream->fail(producer_failure("later"));
  REQUIRE(woken == 1);
  REQUIRE(ended == 1);
}

// ============================================================================
// BSTR-10: an errored stream never reports a clean end (S3)
// ============================================================================

TEST_CASE("batch_stream BSTR-10: an errored stream is never drained or END_OF_STREAM",
          "[batch_stream]")
{
  auto [stream, repo] = make_stream();
  stream->fail(producer_failure());

  // Terminal with an empty queue. The clean-close answer here would be END_OF_STREAM — the quiet
  // success that would let a failed query finish as if it had worked.
  REQUIRE(stream->classify() == availability::HAS_DATA);
  REQUIRE_FALSE(stream->drained());

  // And it stays that way: the only way out is the rethrow.
  REQUIRE_THROWS_AS(stream->try_pull(), std::runtime_error);
  REQUIRE_FALSE(stream->drained());
  REQUIRE(stream->classify() != availability::END_OF_STREAM);
}

// ============================================================================
// BSTR-11: two consumers over one stream. wait-then-pop is not atomic, so a blocking
//          loop has to re-check; no batch is lost, duplicated, or reported as a clean
//          end while still queued (S5).
// ============================================================================

TEST_CASE("batch_stream BSTR-11: concurrent consumers deliver every batch exactly once",
          "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int K     = 60;
  auto [stream, repo] = make_stream();

  std::mutex mutex;
  std::vector<uint64_t> pulled;
  std::atomic<int> premature_eos{0};

  auto consumer = [&] {
    while (true) {
      // The blocking loop shape a session would write: wait, then re-check, because another
      // consumer may have taken the batch that woke us.
      stream->wait();
      auto batch = stream->try_pull();
      if (!batch) {
        if (stream->drained()) { break; }
        continue;
      }
      std::lock_guard<std::mutex> lock(mutex);
      pulled.push_back(batch->get_batch_id());
    }
    // A consumer only gets here on a clean end; anything still queued means a batch was
    // reported as end-of-stream.
    if (!repo->all_empty()) { premature_eos.fetch_add(1, std::memory_order_relaxed); }
  };

  std::thread c0(consumer);
  std::thread c1(consumer);

  for (int i = 0; i < K; ++i) {
    REQUIRE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32)));
  }
  stream->close(SOLE_SENDER);

  c0.join();
  c1.join();

  const std::set<uint64_t> distinct(pulled.begin(), pulled.end());
  REQUIRE(static_cast<int>(pulled.size()) == K);
  REQUIRE(distinct.size() == pulled.size());
  REQUIRE(premature_eos.load() == 0);
}

// ============================================================================
// BSTR-12: fail() releases a blocked consumer (the session's wait loop)
// ============================================================================

TEST_CASE("batch_stream BSTR-12: wait unblocks on fail()", "[batch_stream]")
{
  auto [stream, repo] = make_stream({0, 1});
  std::atomic<bool> returned{false};

  std::thread consumer([&] {
    stream->wait();
    returned = true;
  });

  std::this_thread::sleep_for(20ms);
  REQUIRE_FALSE(returned.load());

  // One sender's failure is enough; waiting for sender 1 to close would hang forever.
  stream->fail(producer_failure());
  consumer.join();
  REQUIRE(returned.load());
  REQUIRE_THROWS_AS(stream->try_pull(), std::runtime_error);
}

// ============================================================================
// BSTR-13: invalid arguments are defined errors — a null repository, a null failure
// ============================================================================

TEST_CASE("batch_stream BSTR-13: null repository and null failure are rejected", "[batch_stream]")
{
  REQUIRE_THROWS_AS((batch_stream(nullptr, std::set<sender_id_t>{SOLE_SENDER})),
                    sirius::invalid_input_exception);

  // A null failure is a bug at the call site, not a clean close.
  auto [stream, repo] = make_stream();
  REQUIRE_THROWS_AS(stream->fail(nullptr), sirius::invalid_input_exception);
  REQUIRE_FALSE(stream->terminal());
}

// ============================================================================
// BSTR-14: an empty expected set is terminal from construction
// ============================================================================

TEST_CASE("batch_stream BSTR-14: no expected senders means immediate EOS", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [stream, repo] = make_stream({});

  REQUIRE(stream->terminal());
  REQUIRE(stream->classify() == availability::END_OF_STREAM);
  REQUIRE(stream->drained());
  REQUIRE_FALSE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
}

// ============================================================================
// BSTR-15: the batch is in the repository before on_data announces it
// ============================================================================

TEST_CASE("batch_stream BSTR-15: push registers before it wakes", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [stream, repo] = make_stream();

  bool seen_registered = false;
  stream->set_on_data([&, repo = repo] { seen_registered = repo->total_size() > 0; });

  // S1 — register-then-announce. An on_data hook that ran first would schedule a task for a
  // batch not yet in the repository, and the task's pop would come back empty.
  REQUIRE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  REQUIRE(seen_registered);
}

// ============================================================================
// BSTR-16: concurrent producers and racing closes — every accepted push is in the
//          repository, and nothing lands after the stream ends
// ============================================================================

TEST_CASE("batch_stream BSTR-16: concurrent pushes and a racing close", "[batch_stream]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr int kPerThread = 50;
  auto [stream, repo]      = make_stream({0, 1});

  std::atomic<int> accepted{0};

  auto producer = [&, s = stream](sender_id_t id) {
    for (int i = 0; i < kPerThread; ++i) {
      auto batch = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
      if (s->push(std::move(batch))) { accepted.fetch_add(1, std::memory_order_relaxed); }
    }
    s->close(id);
  };

  std::thread t0(producer, 0);
  std::thread t1(producer, 1);
  t0.join();
  t1.join();

  // Every accepted push is queued exactly once, and no push landed without acceptance.
  REQUIRE(repo->total_size() == static_cast<std::size_t>(accepted.load()));
  REQUIRE(stream->terminal());
  REQUIRE_FALSE(stream->push(make_numeric_batch<int32_t>(*gpu_space, {0}, cudf::type_id::INT32)));
  REQUIRE(repo->total_size() == static_cast<std::size_t>(accepted.load()));
}

// ============================================================================
// BSTR-17: per-sender progress is observable — and a failed sender's peers are
//          not waited for (P3's observable face)
// ============================================================================

TEST_CASE("batch_stream BSTR-17: sender_closed tracks individual senders", "[batch_stream]")
{
  auto [stream, repo] = make_stream({0, 1, 2});

  REQUIRE_FALSE(stream->sender_closed(0));
  stream->close(1);
  REQUIRE(stream->sender_closed(1));
  REQUIRE_FALSE(stream->sender_closed(0));
  REQUIRE_FALSE(stream->sender_closed(2));
  REQUIRE_FALSE(stream->terminal());

  // A failure takes the stream terminal while its peers are still open (P3) — and marks no
  // sender closed: failure is stream-wide, it has no identity.
  stream->fail(producer_failure());
  REQUIRE(stream->terminal());
  REQUIRE_FALSE(stream->sender_closed(0));
  REQUIRE_FALSE(stream->sender_closed(2));
}
