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
#include "event/downgraded_batch.hpp"
#include "memory/device_preload_budget.hpp"
#include "preload/candidate_store.hpp"
#include "preload/config.hpp"
#include "preload/preload_policy.hpp"
#include "preload/preload_stats.hpp"
#include "query_id.hpp"

#include <cucascade/memory/common.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using cucascade::memory::memory_space_id;
using cucascade::memory::Tier;
using sirius::make_query_id;
using sirius::query_id_t;
using sirius::event::downgrade_source;
using sirius::event::downgraded_batch;
using namespace sirius::preload;

namespace {

memory_space_id const gpu0{Tier::GPU, 0};
memory_space_id const gpu1{Tier::GPU, 1};
memory_space_id const host0{Tier::HOST, 0};
memory_space_id const disk0{Tier::DISK, 0};

target const gpu_t0{Tier::GPU, 0};
target const gpu_t1{Tier::GPU, 1};
target const host_t0{Tier::HOST, 0};

std::uintptr_t const repo_a = 0xA0;
std::uintptr_t const repo_b = 0xB0;

/**
 * A stand-in for a live batch: the store only ever checks whether its weak reference expired, so
 * an aliasing pointer that shares ownership with a plain object is enough and needs no GPU.
 */
struct fake_batch {
  std::shared_ptr<int> owner = std::make_shared<int>(0);

  [[nodiscard]] std::shared_ptr<cucascade::data_batch> ptr() const
  {
    return {owner, static_cast<cucascade::data_batch*>(nullptr)};
  }
  void destroy() { owner.reset(); }
};

downgraded_batch record(fake_batch const& b,
                        std::uint64_t id,
                        std::uint64_t seq,
                        memory_space_id from,
                        memory_space_id to,
                        std::size_t bytes,
                        std::uintptr_t repo   = repo_a,
                        std::size_t partition = 0)
{
  downgraded_batch r;
  r.batch          = b.ptr();
  r.batch_id       = id;
  r.conversion_seq = seq;
  r.from           = from;
  r.to             = to;
  r.bytes          = bytes;
  r.source         = downgrade_source::repository;
  r.repository     = repo;
  r.partition      = partition;
  return r;
}

promotion_result outcome(promotion_outcome o, std::size_t bytes = 0)
{
  promotion_result r;
  r.outcome = o;
  r.bytes   = bytes;
  return r;
}

repo_info repo(std::uintptr_t token, sirius::exec::queue_priority dest_priority)
{
  repo_info r;
  r.token         = token;
  r.dest_priority = dest_priority;
  return r;
}

struct store_fixture {
  preload_executor_config cfg{};
  preload_counters counters{0, 2};
  std::vector<int> gpu_devices{0, 1};
  sirius::memory::device_preload_budget budget{gpu_devices};
  candidate_store store{cfg, {0, 1}, {0}, counters, &budget};
  query_id_t const q1 = make_query_id(1);

  /// Bytes the board charges when a GPU job of a candidate with @p est_bytes is taken.
  static std::size_t charged(std::size_t est_bytes) { return reserve_estimate(est_bytes); }
  query_id_t const q2 = make_query_id(2);

  store_fixture()
  {
    REQUIRE(store.add_query(q1, {0}, {repo(repo_b, 20), repo(repo_a, 10)}));
    REQUIRE(store.add_query(q2, {0}, {}));
  }

  /// Take a job for @p id toward @p t, which must succeed.
  promotion_job take(std::uint64_t id, target t, klass k = klass::A, std::size_t ceiling = 0)
  {
    auto job = store.take({id, t, k, ceiling});
    REQUIRE(job.has_value());
    return *job;
  }

  /// Promote @p id to @p t end to end.
  void promote(std::uint64_t id, target t, klass k, std::size_t bytes)
  {
    auto const job = take(id, t, k);
    auto r         = outcome(promotion_outcome::promoted, bytes);
    r.kind         = t.tier == Tier::GPU ? promotion_kind::h2g : promotion_kind::d2h;
    store.apply_outcome(job, r);
  }
};

}  // namespace

// ---- intake ------------------------------------------------------------------------------------

TEST_CASE("candidate_store dedups by batch id", "[preload]")
{
  store_fixture f;
  fake_batch b;
  CHECK(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::inserted);
  CHECK(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::merged);
  CHECK(f.store.size() == 1);
  CHECK(f.counters.get(counter::events_received) == 2);
  CHECK(f.counters.get(counter::merged) == 1);
}

TEST_CASE("candidate_store merges records by conversion sequence, not arrival order", "[preload]")
{
  store_fixture f;
  fake_batch b;
  // HOST->DISK (seq 2) arrives before GPU->HOST (seq 1).
  REQUIRE(f.store.intake(f.q1, record(b, 7, 2, host0, disk0, 80)) == intake_status::inserted);
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::merged);

  auto const* c = f.store.find(7);
  REQUIRE(c != nullptr);
  CHECK(c->location == disk0);
  CHECK(c->last_seq == 2);
  CHECK(c->origin_gpu == 0);
  CHECK(c->origin_host == 0);
  CHECK(c->est_bytes == 100);  // the GPU size is the best estimate

  SECTION("a newer record moves the candidate and refiles its repository")
  {
    auto newer = record(b, 7, 3, disk0, disk0, 50, repo_b, 1);
    REQUIRE(f.store.intake(f.q1, newer) == intake_status::merged);
    CHECK(c->last_seq == 3);
    CHECK(c->repo_token == repo_b);
    CHECK(c->partition == 1);
    auto const snap = f.store.snapshot_phase0();
    REQUIRE(snap.queries.size() == 2);
    REQUIRE(snap.queries[0].repos.size() == 1);
    CHECK(snap.queries[0].repos[0].info.token == repo_b);
  }

  SECTION("task hints are filled once known")
  {
    auto hinted             = record(b, 7, 1, gpu0, host0, 100);
    hinted.source           = downgrade_source::task_queue;
    hinted.task_priority    = 42;
    hinted.preferred_device = 1;
    REQUIRE(f.store.intake(f.q1, hinted) == intake_status::merged);
    CHECK(c->task_priority == 42);
    CHECK(c->preferred_device == 1);
    CHECK(c->source == downgrade_source::repository);  // older record: location fields stay
  }
}

TEST_CASE("candidate_store drops records of unknown or retired queries", "[preload]")
{
  store_fixture f;
  fake_batch b;
  auto const q9 = make_query_id(9);
  CHECK(f.store.intake(q9, record(b, 1, 1, gpu0, host0, 10)) ==
        intake_status::dropped_unknown_query);
  CHECK_FALSE(f.store.knows_query(q9));
  CHECK(f.store.size() == 0);

  f.store.retire_query(f.q1);
  CHECK(f.store.intake(f.q1, record(b, 1, 1, gpu0, host0, 10)) ==
        intake_status::dropped_unknown_query);
  CHECK_FALSE(f.store.knows_query(f.q1));
  CHECK(f.counters.get(counter::dropped_unknown_query) == 2);

  SECTION("records of expired batches or moves to a GPU are dropped")
  {
    fake_batch gone;
    auto r = record(gone, 2, 1, gpu0, host0, 10);
    gone.destroy();
    CHECK(f.store.intake(f.q2, r) == intake_status::dropped_expired);
    CHECK(f.store.intake(f.q2, record(b, 3, 1, host0, gpu0, 10)) ==
          intake_status::dropped_unsupported);
    CHECK(f.store.size() == 0);
  }
}

TEST_CASE("candidate_store evicts the newest query's most recent candidate on overflow",
          "[preload]")
{
  store_fixture f;
  f.cfg.max_candidates = 3;
  candidate_store store{f.cfg, {0}, {0}, f.counters};
  REQUIRE(store.add_query(f.q1, {0}, {}));
  REQUIRE(store.add_query(f.q2, {0}, {}));
  fake_batch b;

  REQUIRE(store.intake(f.q2, record(b, 20, 5, gpu0, host0, 1)) == intake_status::inserted);
  REQUIRE(store.intake(f.q2, record(b, 21, 6, gpu0, host0, 1)) == intake_status::inserted);
  REQUIRE(store.intake(f.q1, record(b, 10, 7, gpu0, host0, 1)) == intake_status::inserted);

  SECTION("an older query's record evicts the newest query's highest sequence")
  {
    CHECK(store.intake(f.q1, record(b, 11, 8, gpu0, host0, 1)) == intake_status::inserted);
    CHECK(store.size() == 3);
    CHECK(store.find(21) == nullptr);
    CHECK(store.find(20) != nullptr);
  }

  SECTION("a candidate in flight is never evicted")
  {
    REQUIRE(store.take({21, gpu_t0, klass::A, 0}).has_value());
    CHECK(store.intake(f.q1, record(b, 11, 8, gpu0, host0, 1)) == intake_status::inserted);
    CHECK(store.find(21) != nullptr);
    CHECK(store.find(20) == nullptr);
  }

  SECTION("a record of the newest query that would itself be evicted is dropped")
  {
    CHECK(store.intake(f.q2, record(b, 22, 9, gpu0, host0, 1)) == intake_status::dropped_overflow);
    CHECK(store.find(22) == nullptr);
    CHECK(store.size() == 3);
  }
  CHECK(f.counters.get(counter::dropped_overflow) == 1);
}

// ---- outcomes ----------------------------------------------------------------------------------

TEST_CASE("candidate_store promotes into the ledger and counts the promotion", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::inserted);
  f.promote(7, gpu_t0, klass::B, 128);

  CHECK(f.store.find(7) == nullptr);
  auto const* e = f.store.find_ledger(7);
  REQUIRE(e != nullptr);
  CHECK(e->gpu == 0);
  CHECK(e->k == klass::B);
  CHECK(e->bytes == 128);
  // Outstanding is the budget charged when the job was taken (the candidate's est_bytes), not the
  // ledger byte count (the copy's exact need).
  CHECK(f.store.outstanding_bytes(0) == store_fixture::charged(100));
  CHECK(f.store.outstanding_b_bytes(0) == store_fixture::charged(100));
  CHECK(f.counters.get(counter::promoted_h2g) == 1);
  CHECK(f.counters.get(counter::bytes_h2g) == 128);
  CHECK(f.store.find_query(f.q1)->stats[counter::promoted_h2g] == 1);

  SECTION("a promotion off HOST advances that host's memory epoch")
  {
    CHECK(f.store.mem_epoch(host_t0) == 1);
    CHECK(f.store.mem_epoch(gpu_t0) == 0);
  }
}

TEST_CASE("candidate_store: an outcome after retire_query only updates statistics", "[preload]")
{
  store_fixture f;
  fake_batch b;
  fake_batch other;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::inserted);
  REQUIRE(f.store.intake(f.q1, record(other, 8, 2, gpu0, host0, 50)) == intake_status::inserted);
  f.promote(8, gpu_t0, klass::A, 64);
  REQUIRE(f.store.outstanding_bytes(0) == store_fixture::charged(50));

  auto const job = f.take(7, gpu_t0);
  f.store.retire_query(f.q1);
  CHECK_FALSE(f.store.knows_query(f.q1));
  CHECK(f.store.size() == 0);
  CHECK(f.store.ledger_size() == 0);
  CHECK(f.store.outstanding_bytes(0) == 0);
  CHECK(f.counters.get(counter::wasted_unconsumed) == 1);

  auto const effect = f.store.apply_outcome(job, outcome(promotion_outcome::promoted, 100));
  CHECK(effect.late);
  CHECK_FALSE(f.store.knows_query(f.q1));
  CHECK(f.store.find(7) == nullptr);
  CHECK(f.store.find_ledger(7) == nullptr);
  CHECK(f.store.outstanding_bytes(0) == 0);
  CHECK(f.counters.get(counter::late_outcome_after_retire) == 1);

  auto const epoch = f.store.mem_epoch(gpu_t0);
  auto const stats = f.store.forget_query(f.q1);
  REQUIRE(stats.has_value());
  CHECK((*stats)[counter::promoted_h2g] == 2);
  CHECK((*stats)[counter::wasted_unconsumed] == 1);
  CHECK((*stats)[counter::late_outcome_after_retire] == 1);
  CHECK(f.store.mem_epoch(gpu_t0) == epoch + 1);

  // Idempotent: nothing left to report, and still nothing recreated.
  CHECK_FALSE(f.store.forget_query(f.q1).has_value());
  CHECK_FALSE(f.store.knows_query(f.q1));
  f.store.retire_query(f.q1);
  CHECK_FALSE(f.store.knows_query(f.q1));
}

TEST_CASE("candidate_store stamps refusal gates with the epoch seen at dispatch", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::inserted);

  SECTION("a refusal with no memory event during the attempt gates the space")
  {
    auto const job    = f.take(7, gpu_t0);
    auto const effect = f.store.apply_outcome(job, outcome(promotion_outcome::no_memory));
    CHECK(effect.purge == purge_scope::target);
    CHECK(f.store.gated(gpu_t0));
    CHECK_FALSE(f.store.gated(gpu_t1));
    f.store.advance_mem_epoch(gpu_t0);
    CHECK_FALSE(f.store.gated(gpu_t0));
  }

  SECTION("a memory event during the attempt leaves the space ungated")
  {
    auto const job = f.take(7, gpu_t0);
    f.store.advance_mem_epoch(gpu_t0);
    f.store.apply_outcome(job, outcome(promotion_outcome::gated));
    CHECK(f.store.gate_epoch(gpu_t0) == job.mem_epoch_at_take);
    CHECK_FALSE(f.store.gated(gpu_t0));
  }

  SECTION("a refusal for a retired query still gates the space")
  {
    auto const job = f.take(7, gpu_t0);
    f.store.retire_query(f.q1);
    f.store.apply_outcome(job, outcome(promotion_outcome::no_memory));
    CHECK(f.store.gated(gpu_t0));
  }

  SECTION("a hold-off records no gate")
  {
    auto const job    = f.take(7, gpu_t0);
    auto const effect = f.store.apply_outcome(job, outcome(promotion_outcome::held_off));
    CHECK(effect.purge == purge_scope::held_off);
    CHECK_FALSE(f.store.gated(gpu_t0));
    CHECK(f.store.plannable(*f.store.find(7)));
  }

  SECTION("a contended candidate is blocked until the queue moves past the dispatch epoch")
  {
    auto const job = f.take(7, gpu_t0);
    CHECK_FALSE(f.store.plannable(*f.store.find(7)));  // in flight
    f.store.apply_outcome(job, outcome(promotion_outcome::contended));
    CHECK_FALSE(f.store.plannable(*f.store.find(7)));
    CHECK_FALSE(f.store.take({7, gpu_t0, klass::A, 0}).has_value());
    f.store.advance_queue_epoch();
    CHECK(f.store.plannable(*f.store.find(7)));
  }

  SECTION("a contended outcome uses the queue epoch seen at dispatch")
  {
    auto const job = f.take(7, gpu_t0);
    f.store.advance_queue_epoch();  // the queue moved during the attempt
    f.store.apply_outcome(job, outcome(promotion_outcome::contended));
    CHECK(f.store.find(7)->blocked_queue_epoch == job.queue_epoch_at_take);
    CHECK(f.store.plannable(*f.store.find(7)));
  }

  SECTION("a candidate is dropped after max_lock_attempts lock refusals")
  {
    for (int i = 0; i < f.cfg.max_lock_attempts; ++i) {
      REQUIRE(f.store.find(7) != nullptr);
      auto const job = f.take(7, gpu_t0);
      f.store.apply_outcome(job, outcome(promotion_outcome::contended));
      f.store.advance_queue_epoch();
    }
    CHECK(f.store.find(7) == nullptr);
    CHECK(f.counters.get(counter::dropped_lock_attempts) == 1);
  }
}

TEST_CASE("candidate_store detects a bounce during the attempt", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::inserted);
  auto const job = f.take(7, gpu_t0, klass::B);

  // The promotion lands, then downgrade moves the batch again before the outcome is applied.
  REQUIRE(f.store.intake(f.q1, record(b, 7, 5, gpu0, disk0, 100)) == intake_status::merged);
  f.store.apply_outcome(job, outcome(promotion_outcome::promoted, 100));

  auto const* c = f.store.find(7);
  REQUIRE(c != nullptr);
  CHECK(c->location == disk0);
  CHECK(c->speculative_banned);
  CHECK_FALSE(c->in_flight);
  CHECK(f.store.find_ledger(7) == nullptr);
  CHECK(f.store.outstanding_bytes(0) == 0);
  CHECK(f.counters.get(counter::bounce) == 1);
}

TEST_CASE("candidate_store: ledger bounce, then the promotion cap", "[preload]")
{
  store_fixture f;  // max_promotions_per_batch = 2
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::inserted);
  f.promote(7, gpu_t0, klass::A, 100);
  REQUIRE(f.store.find_ledger(7) != nullptr);

  // Downgraded again while preloaded: a bounce.
  CHECK(f.store.intake(f.q1, record(b, 7, 2, gpu0, host0, 100)) == intake_status::bounced);
  CHECK(f.store.find_ledger(7) == nullptr);
  CHECK(f.store.outstanding_bytes(0) == 0);
  REQUIRE(f.store.find(7) != nullptr);
  CHECK(f.store.find(7)->speculative_banned);
  CHECK(f.counters.get(counter::bounce) == 1);

  f.promote(7, gpu_t0, klass::A, 100);
  CHECK(f.store.intake(f.q1, record(b, 7, 3, gpu0, host0, 100)) ==
        intake_status::dropped_promotion_cap);
  CHECK(f.store.find(7) == nullptr);
  CHECK(f.store.find_ledger(7) == nullptr);
  CHECK(f.store.outstanding_bytes(0) == 0);
  CHECK(f.counters.get(counter::bounce) == 2);
  CHECK(f.counters.get(counter::dropped_promotion_cap) == 1);
}

TEST_CASE("candidate_store: the first OOM bans the GPU for the candidate's lifetime", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, disk0, 64 << 20)) == intake_status::inserted);

  auto job = f.take(7, gpu_t0);
  // No doubled-estimate retry: the reserve estimate is never inflated.
  CHECK(job.est_reserve == reserve_estimate(64 << 20));
  f.store.apply_outcome(job, outcome(promotion_outcome::oom));
  CHECK_FALSE(f.store.gated(gpu_t0));  // local to the candidate
  CHECK(f.store.find(7)->gpu_banned);  // banned on the first OOM
  CHECK(f.counters.get(counter::gpu_banned) == 1);

  // Staging is not inflated by the GPU's OOM history either.
  job = f.take(7, host_t0);
  CHECK(job.est_reserve == reserve_estimate(64 << 20));
}

TEST_CASE("candidate_store: a duplicate or stale downgrade record never undoes a claim",
          "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 5, gpu0, host0, 100)) == intake_status::inserted);
  // A task claims the watched id before preload ever takes it.
  std::array<std::uint64_t, 1> const ids{7};
  f.budget.claim(ids);

  SECTION("an equal or older record only fills hints and leaves the claim standing")
  {
    REQUIRE(f.store.intake(f.q1, record(b, 7, 5, gpu0, host0, 100)) == intake_status::merged);
    REQUIRE(f.store.intake(f.q1, record(b, 7, 3, gpu0, host0, 100)) == intake_status::merged);
    // A worker TAKE is refused (the charge finds the id claimed) and the candidate is dropped.
    CHECK_FALSE(f.store.take({7, gpu_t0, klass::A, 0}).has_value());
    CHECK(f.counters.get(counter::claimed) == 1);
    CHECK(f.store.find(7) == nullptr);
    CHECK(f.store.outstanding_bytes(0) == 0);
  }
  SECTION("a strictly newer record starts a new generation and reopens the id")
  {
    REQUIRE(f.store.intake(f.q1, record(b, 7, 6, gpu0, host0, 100)) == intake_status::merged);
    auto const job = f.store.take({7, gpu_t0, klass::A, 0});
    REQUIRE(job.has_value());  // the new generation cleared the claim, so the charge succeeds
    CHECK(f.store.outstanding_bytes(0) == store_fixture::charged(100));
  }
}

TEST_CASE("candidate_store: a claim survives the candidate's removal via a tombstone", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 5, gpu0, host0, 100)) == intake_status::inserted);
  // A task claims the watched id before preload takes it.
  std::array<std::uint64_t, 1> const ids{7};
  f.budget.claim(ids);
  // The refused TAKE erases the candidate but keeps a claimed tombstone (seq 5).
  CHECK_FALSE(f.store.take({7, gpu_t0, klass::A, 0}).has_value());
  CHECK(f.counters.get(counter::claimed) == 1);
  REQUIRE(f.store.find(7) == nullptr);

  SECTION("a duplicate or older record after removal is rejected, not recreated unclaimed")
  {
    CHECK(f.store.intake(f.q1, record(b, 7, 5, gpu0, host0, 100)) ==
          intake_status::dropped_stale_record);  // duplicate
    CHECK(f.store.intake(f.q1, record(b, 7, 3, gpu0, host0, 100)) ==
          intake_status::dropped_stale_record);  // older
    CHECK(f.store.find(7) == nullptr);
    CHECK(f.store.outstanding_bytes(0) == 0);
  }
  SECTION("a strictly newer record reopens it as a fresh eligible candidate")
  {
    REQUIRE(f.store.intake(f.q1, record(b, 7, 6, gpu0, host0, 100)) == intake_status::inserted);
    auto const job = f.store.take({7, gpu_t0, klass::A, 0});
    REQUIRE(job.has_value());  // the new generation cleared the claim, so the charge succeeds
    CHECK(f.store.outstanding_bytes(0) == store_fixture::charged(100));
  }
  SECTION("retire releases the tombstone's board watch")
  {
    f.store.retire_query(f.q1);
    CHECK(f.budget.watched_count() == 0);
  }
}

TEST_CASE("candidate_store: too_big bans the GPU at once", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, disk0, 100)) == intake_status::inserted);
  f.store.apply_outcome(f.take(7, gpu_t0), outcome(promotion_outcome::too_big));
  CHECK(f.store.find(7)->gpu_banned);
  CHECK(f.counters.get(counter::gpu_banned) == 1);
}

TEST_CASE("candidate_store applies the remaining outcomes", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, disk0, 100)) == intake_status::inserted);

  SECTION("stale moves the location hint")
  {
    auto r     = outcome(promotion_outcome::stale);
    r.observed = host0;
    f.store.apply_outcome(f.take(7, gpu_t0), r);
    CHECK(f.store.find(7)->location == host0);
    CHECK(f.store.plannable(*f.store.find(7)));
  }

  SECTION("unsupported is never planned again")
  {
    f.store.apply_outcome(f.take(7, gpu_t0), outcome(promotion_outcome::unsupported));
    CHECK_FALSE(f.store.plannable(*f.store.find(7)));
  }

  SECTION("retired, expired and already_resident erase the candidate")
  {
    for (auto const o : {promotion_outcome::retired,
                         promotion_outcome::expired,
                         promotion_outcome::already_resident}) {
      if (f.store.find(7) == nullptr) {
        REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, disk0, 100)) == intake_status::inserted);
      }
      f.store.apply_outcome(f.take(7, gpu_t0), outcome(o));
      CHECK(f.store.find(7) == nullptr);
    }
  }

  SECTION("failed erases and reports once per query")
  {
    auto const first = f.store.apply_outcome(f.take(7, gpu_t0), outcome(promotion_outcome::failed));
    CHECK(first.first_failure_in_query);
    CHECK(f.store.find(7) == nullptr);
    REQUIRE(f.store.intake(f.q1, record(b, 8, 2, gpu0, disk0, 100)) == intake_status::inserted);
    auto const second =
      f.store.apply_outcome(f.take(8, gpu_t0), outcome(promotion_outcome::failed));
    CHECK_FALSE(second.first_failure_in_query);
    CHECK(f.counters.get(counter::failed) == 2);
  }

  SECTION("a job can be taken once until its outcome is applied")
  {
    auto const job = f.take(7, gpu_t0);
    CHECK(job.expected_source == disk0);
    CHECK_FALSE(f.store.take({7, gpu_t0, klass::A, 0}).has_value());
    CHECK_FALSE(f.store.take({7, target{Tier::GPU, 9}, klass::A, 0}).has_value());
  }
}

// ---- ledger ------------------------------------------------------------------------------------

TEST_CASE("candidate_store erases expired ledger entries without a lease", "[preload]")
{
  store_fixture f;
  fake_batch kept;
  fake_batch consumed;
  REQUIRE(f.store.intake(f.q1, record(kept, 1, 1, gpu0, host0, 10)) == intake_status::inserted);
  REQUIRE(f.store.intake(f.q1, record(consumed, 2, 2, gpu0, host0, 20)) == intake_status::inserted);
  f.promote(1, gpu_t0, klass::A, 10);
  f.promote(2, gpu_t0, klass::A, 20);
  REQUIRE(f.store.outstanding_bytes(0) == store_fixture::charged(10) + store_fixture::charged(20));

  consumed.destroy();
  auto const snap = f.store.snapshot_phase0();
  REQUIRE(snap.ledger.size() == 1);  // only live entries need classifying
  CHECK(snap.ledger[0].batch_id == 1);

  CHECK(f.store.sweep_expired_ledger() == 1);
  CHECK(f.store.find_ledger(2) == nullptr);
  CHECK(f.store.find_ledger(1) != nullptr);
  CHECK(f.store.outstanding_bytes(0) == store_fixture::charged(10));
  CHECK(f.counters.get(counter::hit) == 1);
}

TEST_CASE("candidate_store applies the planner's ledger classification", "[preload]")
{
  store_fixture f;
  fake_batch b1;
  fake_batch b2;
  fake_batch b3;
  REQUIRE(f.store.intake(f.q1, record(b1, 1, 1, gpu0, host0, 10)) == intake_status::inserted);
  REQUIRE(f.store.intake(f.q1, record(b2, 2, 2, gpu0, host0, 20)) == intake_status::inserted);
  REQUIRE(f.store.intake(f.q1, record(b3, 3, 3, gpu1, host0, 40)) == intake_status::inserted);
  f.promote(1, gpu_t0, klass::A, 10);
  f.promote(2, gpu_t0, klass::B, 20);
  f.promote(3, gpu_t1, klass::B, 40);
  REQUIRE(f.store.outstanding_bytes(0) == store_fixture::charged(10) + store_fixture::charged(20));
  REQUIRE(f.store.outstanding_b_bytes(0) == store_fixture::charged(20));
  REQUIRE(f.store.outstanding_b_bytes(1) == store_fixture::charged(40));

  std::vector<std::pair<std::uint64_t, ledger_verdict>> const verdicts{
    {1, ledger_verdict::hit},           // class A whose task ran
    {2, ledger_verdict::reclassify_a},  // class B a task claimed
    {3, ledger_verdict::outstanding},
    {99, ledger_verdict::hit},  // gone already: skipped
  };
  f.store.apply_ledger_verdicts(verdicts);

  CHECK(f.store.find_ledger(1) == nullptr);
  REQUIRE(f.store.find_ledger(2) != nullptr);
  CHECK(f.store.find_ledger(2)->k == klass::A);
  CHECK(f.store.outstanding_bytes(0) == store_fixture::charged(20));
  CHECK(f.store.outstanding_b_bytes(0) == 0);
  CHECK(f.store.outstanding_bytes(1) == store_fixture::charged(40));
  CHECK(f.store.outstanding_b_bytes(1) == store_fixture::charged(40));
  CHECK(f.counters.get(counter::hit) == 1);

  SECTION("a consumer on another GPU counts a misprediction")
  {
    f.store.note_consumer_device(3, 0);
    f.store.note_consumer_device(3, 1);
    CHECK(f.counters.get(counter::mispredicted_device) == 1);
    CHECK(f.counters.get(counter::mispredicted_device_b) == 1);
    f.store.note_consumer_device(3, 0);  // the same entry, seen again: counted once
    CHECK(f.counters.get(counter::mispredicted_device) == 1);
    CHECK(f.counters.get(counter::mispredicted_device_b) == 1);
  }

  SECTION("a class-A misprediction is not counted as class B")
  {
    f.store.note_consumer_device(2, 1);  // reclassified to A above
    CHECK(f.counters.get(counter::mispredicted_device) == 1);
    CHECK(f.counters.get(counter::mispredicted_device_b) == 0);
  }
}

// ---- snapshot and pruning ----------------------------------------------------------------------

TEST_CASE("candidate_store snapshots queries, repositories with candidates and epochs", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 5, 1, gpu0, host0, 1, repo_b)) == intake_status::inserted);
  REQUIRE(f.store.intake(f.q1, record(b, 4, 2, gpu0, host0, 1, repo_b)) == intake_status::inserted);
  REQUIRE(f.store.intake(f.q1, record(b, 6, 3, gpu0, host0, 1, repo_a)) == intake_status::inserted);
  auto queued   = record(b, 9, 4, gpu0, host0, 1, 0);
  queued.source = downgrade_source::task_queue;
  REQUIRE(f.store.intake(f.q2, queued) == intake_status::inserted);
  f.store.advance_queue_epoch();
  f.store.advance_mem_epoch(gpu_t1);

  auto const snap = f.store.snapshot_phase0();
  CHECK(snap.has_candidates);
  CHECK(snap.queue_epoch == 1);
  CHECK(snap.mem_epoch == std::vector<std::uint64_t>{0, 1, 0});
  CHECK(snap.inflight_bytes.size() == 3);
  REQUIRE(snap.queries.size() == 2);
  CHECK(snap.queries[0].query == f.q1);  // oldest first
  CHECK(snap.queries[0].readiness_dirty);
  REQUIRE(snap.queries[0].repos.size() == 2);
  CHECK(snap.queries[0].repos[0].info.token == repo_a);  // ascending destination priority
  CHECK(snap.queries[0].repos[0].candidate_ids == std::vector<std::uint64_t>{6});
  CHECK(snap.queries[0].repos[1].candidate_ids == std::vector<std::uint64_t>{4, 5});
  CHECK(snap.queries[1].repos.empty());  // tier-2 candidates are not filed by repository

  auto const again = f.store.snapshot_phase0();
  CHECK_FALSE(again.queries[0].readiness_dirty);
  f.store.mark_readiness_dirty(f.q1);
  CHECK(f.store.snapshot_phase0().queries[0].readiness_dirty);
}

TEST_CASE("candidate_store prunes expired candidates a bounded number at a time", "[preload]")
{
  store_fixture f;
  std::vector<fake_batch> batches(6);
  for (std::uint64_t i = 0; i < batches.size(); ++i) {
    REQUIRE(
      f.store.intake(i % 2 == 0 ? f.q1 : f.q2, record(batches[i], i, i + 1, gpu0, host0, 1)) ==
      intake_status::inserted);
  }
  for (std::size_t i : {0, 1, 2, 4}) {
    batches[i].destroy();
  }
  REQUIRE(f.store.take({4, gpu_t0, klass::A, 0}).has_value());  // in flight: kept

  std::size_t pruned = 0;
  for (int pass = 0; pass < 4; ++pass) {
    pruned += f.store.prune_expired_candidates(2);
  }
  CHECK(pruned == 3);
  CHECK(f.store.size() == 3);
  CHECK(f.store.find(4) != nullptr);
  CHECK(f.store.find(3) != nullptr);
  CHECK(f.store.find(5) != nullptr);
  CHECK(f.counters.get(counter::dropped_expired) == 3);
}

TEST_CASE("candidate_store keeps in-flight accounting per space", "[preload]")
{
  store_fixture f;
  f.store.add_inflight(gpu_t0, 10);
  f.store.add_inflight(host_t0, 5);
  f.store.remove_inflight(gpu_t0, 4);
  CHECK(f.store.inflight_bytes(gpu_t0) == 6);
  CHECK(f.store.inflight_bytes(host_t0) == 5);
  f.store.remove_inflight(host_t0, 50);
  CHECK(f.store.inflight_bytes(host_t0) == 0);

  f.store.acquire_h2g_token(1);
  f.store.acquire_h2g_token(1);
  f.store.release_h2g_token(1);
  CHECK(f.store.h2g_tokens(1) == 1);
  f.store.release_h2g_token(0);
  CHECK(f.store.h2g_tokens(0) == 0);

  CHECK(f.store.gpu_index_of(1) == 1);
  CHECK_FALSE(f.store.gpu_index_of(7).has_value());
  CHECK(f.store.host_index_of(0) == 0);

  f.store.advance_host_mem_epochs();
  CHECK(f.store.mem_epoch(host_t0) == 1);
  CHECK(f.store.mem_epoch(gpu_t0) == 0);
  f.store.advance_all_mem_epochs();
  CHECK(f.store.mem_epoch(gpu_t1) == 1);
  CHECK(f.store.mem_epoch(host_t0) == 2);
}

// ---- ordering and staging ----------------------------------------------------------------------

TEST_CASE("candidate_store: refusal gates only move forward", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::inserted);
  REQUIRE(f.store.intake(f.q1, record(b, 8, 2, gpu0, host0, 100)) == intake_status::inserted);
  auto const older = f.take(7, gpu_t0);
  f.store.advance_mem_epoch(gpu_t0);
  auto const newer = f.take(8, gpu_t0);
  REQUIRE(older.mem_epoch_at_take + 1 == newer.mem_epoch_at_take);

  SECTION("the newer refusal applied first is not undone by the older one")
  {
    f.store.apply_outcome(newer, outcome(promotion_outcome::no_memory));
    CHECK(f.store.gated(gpu_t0));
    f.store.apply_outcome(older, outcome(promotion_outcome::gated));
    CHECK(f.store.gate_epoch(gpu_t0) == newer.mem_epoch_at_take);
    CHECK(f.store.gated(gpu_t0));
  }

  SECTION("in order, the newer refusal gates the space")
  {
    f.store.apply_outcome(older, outcome(promotion_outcome::no_memory));
    CHECK_FALSE(f.store.gated(gpu_t0));  // a memory event arrived during that attempt
    f.store.apply_outcome(newer, outcome(promotion_outcome::no_memory));
    CHECK(f.store.gated(gpu_t0));
  }

  f.store.advance_mem_epoch(gpu_t0);
  CHECK_FALSE(f.store.gated(gpu_t0));
}

TEST_CASE("candidate_store ignores a record older than the promotion in the ledger", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::inserted);

  SECTION("ordered by the sequence the worker took under the batch lock")
  {
    auto const job   = f.take(7, gpu_t0);
    auto r           = outcome(promotion_outcome::promoted, 100);
    r.kind           = promotion_kind::h2g;
    r.conversion_seq = 10;
    f.store.apply_outcome(job, r);
    REQUIRE(f.store.find_ledger(7) != nullptr);
    CHECK(f.store.find_ledger(7)->seq == 10);

    // A HOST->DISK move from before the promotion, delivered late.
    CHECK(f.store.intake(f.q1, record(b, 7, 6, host0, disk0, 100)) ==
          intake_status::dropped_stale_record);
    CHECK(f.store.find_ledger(7) != nullptr);
    CHECK(f.store.outstanding_bytes(0) == store_fixture::charged(100));
    CHECK(f.store.find(7) == nullptr);
    CHECK(f.counters.get(counter::bounce) == 0);
    CHECK(f.counters.get(counter::dropped_stale_record) == 1);

    CHECK(f.store.intake(f.q1, record(b, 7, 11, gpu0, host0, 100)) == intake_status::bounced);
    CHECK(f.store.find_ledger(7) == nullptr);
    CHECK(f.counters.get(counter::bounce) == 1);
  }

  SECTION("without a worker sequence, the sequence at take time")
  {
    f.promote(7, gpu_t0, klass::A, 100);
    REQUIRE(f.store.find_ledger(7) != nullptr);
    CHECK(f.store.find_ledger(7)->seq == 1);
    CHECK(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) ==
          intake_status::dropped_stale_record);
    CHECK(f.store.intake(f.q1, record(b, 7, 2, gpu0, host0, 100)) == intake_status::bounced);
  }
}

TEST_CASE("candidate_store: a stale outcome never overrides a newer record", "[preload]")
{
  store_fixture f;
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, gpu0, host0, 100)) == intake_status::inserted);

  SECTION("a record merged during the attempt keeps its location")
  {
    auto const job = f.take(7, gpu_t0);
    REQUIRE(f.store.intake(f.q1, record(b, 7, 5, host0, disk0, 100)) == intake_status::merged);
    auto r     = outcome(promotion_outcome::stale);
    r.observed = gpu0;
    f.store.apply_outcome(job, r);
    CHECK(f.store.find(7)->location == disk0);
    CHECK(f.store.plannable(*f.store.find(7)));
  }

  SECTION("a stale outcome where nothing moved waits for the queue, then gives up")
  {
    for (int i = 0; i < f.cfg.max_lock_attempts; ++i) {
      REQUIRE(f.store.find(7) != nullptr);
      auto const job = f.take(7, gpu_t0, klass::B);
      auto r         = outcome(promotion_outcome::stale);
      r.observed     = host0;  // where the job expected it: a task claimed it
      f.store.apply_outcome(job, r);
      if (f.store.find(7) == nullptr) { break; }
      CHECK(f.store.find(7)->location == host0);
      CHECK_FALSE(f.store.plannable(*f.store.find(7)));
      f.store.advance_queue_epoch();
      CHECK(f.store.plannable(*f.store.find(7)));
    }
    CHECK(f.store.find(7) == nullptr);
    CHECK(f.counters.get(counter::dropped_lock_attempts) == 1);
  }
}

TEST_CASE("candidate_store: add_query keeps the first entry of a repeated token", "[preload]")
{
  store_fixture f;
  auto const q3 = make_query_id(3);
  REQUIRE(f.store.add_query(q3, {0}, {repo(repo_a, 30), repo(repo_b, 20), repo(repo_a, 10)}));
  auto const* qs = f.store.find_query(q3);
  REQUIRE(qs != nullptr);
  REQUIRE(qs->repos.size() == 2);
  CHECK(qs->repos[0].token == repo_b);
  CHECK(qs->repos[1].token == repo_a);
  CHECK(qs->repos[1].dest_priority == 30);
  CHECK(qs->repo_index.at(repo_b) == 0);
  CHECK(qs->repo_index.at(repo_a) == 1);
  CHECK_FALSE(f.store.add_query(q3, {}, {}));
}

TEST_CASE("candidate_store: forget_query of an unknown query leaves the epochs alone", "[preload]")
{
  store_fixture f;
  auto const before = f.store.snapshot_phase0().mem_epoch;
  CHECK_FALSE(f.store.forget_query(make_query_id(9)).has_value());
  CHECK(f.store.snapshot_phase0().mem_epoch == before);

  REQUIRE(f.store.forget_query(f.q2).has_value());  // known: its memory was freed
  auto const after = f.store.snapshot_phase0().mem_epoch;
  CHECK(after != before);
  CHECK_FALSE(f.store.forget_query(f.q2).has_value());  // already forgotten
  CHECK(f.store.snapshot_phase0().mem_epoch == after);
}

TEST_CASE("candidate_store keeps a staged candidate for its HOST->GPU hop", "[preload]")
{
  store_fixture f;  // max_promotions_per_batch = 2
  fake_batch b;
  REQUIRE(f.store.intake(f.q1, record(b, 7, 1, host0, disk0, 100)) == intake_status::inserted);
  auto staging     = f.take(7, host_t0);
  auto r           = outcome(promotion_outcome::promoted, 100);
  r.kind           = promotion_kind::d2h;
  r.conversion_seq = 4;
  f.store.apply_outcome(staging, r);

  auto const* c = f.store.find(7);
  REQUIRE(c != nullptr);
  CHECK(c->location == host0);
  CHECK(c->last_seq == 4);
  CHECK(f.store.plannable(*c));
  auto const* e = f.store.find_ledger(7);
  REQUIRE(e != nullptr);
  CHECK(e->gpu == -1);
  CHECK(f.store.outstanding_bytes(0) == 0);
  CHECK(f.counters.get(counter::promoted_d2h) == 1);
  CHECK(f.store.find_query(f.q1)->promotions.at(7) == 1);

  SECTION("the GPU hop moves the ledger entry to the GPU")
  {
    auto const hop = f.take(7, gpu_t0);
    CHECK(hop.expected_source == host0);
    CHECK(hop.seq_at_take == 4);
    f.store.apply_outcome(hop, outcome(promotion_outcome::promoted, 100));
    CHECK(f.store.find(7) == nullptr);
    REQUIRE(f.store.find_ledger(7) != nullptr);
    CHECK(f.store.find_ledger(7)->gpu == 0);
    CHECK(f.store.outstanding_bytes(0) == store_fixture::charged(100));
    CHECK(f.counters.get(counter::promoted_h2g) == 1);
    // Both moves count toward the cap: a bounce now drops the batch.
    CHECK(f.store.intake(f.q1, record(b, 7, 9, gpu0, host0, 100)) ==
          intake_status::dropped_promotion_cap);
    CHECK(f.store.outstanding_bytes(0) == 0);
  }

  SECTION("a move back to DISK is a bounce of the staged batch")
  {
    CHECK(f.store.intake(f.q1, record(b, 7, 3, host0, disk0, 100)) ==
          intake_status::dropped_stale_record);
    CHECK(f.store.find(7)->location == host0);
    CHECK(f.store.intake(f.q1, record(b, 7, 5, host0, disk0, 100)) == intake_status::merged);
    CHECK(f.store.find_ledger(7) == nullptr);
    CHECK(f.store.find(7)->location == disk0);
    CHECK(f.store.find(7)->speculative_banned);
    CHECK(f.counters.get(counter::bounce) == 1);
  }

  SECTION("retire purges the staged candidate and its HOST ledger entry, nothing else")
  {
    fake_batch other;
    REQUIRE(f.store.intake(f.q2, record(other, 8, 6, gpu0, host0, 50)) == intake_status::inserted);
    f.promote(8, gpu_t0, klass::A, 50);
    f.store.retire_query(f.q1);
    CHECK(f.store.find(7) == nullptr);
    CHECK(f.store.find_ledger(7) == nullptr);
    CHECK(f.counters.get(counter::wasted_unconsumed) == 1);
    REQUIRE(f.store.find_ledger(8) != nullptr);
    CHECK(f.store.outstanding_bytes(0) == store_fixture::charged(50));
    CHECK(f.store.find_query(f.q2)->ledger_ids.contains(8));
  }
}

TEST_CASE("candidate_store drops a staged candidate that reached the promotion cap", "[preload]")
{
  store_fixture f;
  f.cfg.max_promotions_per_batch = 1;
  candidate_store store{f.cfg, {0}, {0}, f.counters};
  REQUIRE(store.add_query(f.q1, {0}, {}));
  fake_batch b;
  REQUIRE(store.intake(f.q1, record(b, 7, 1, host0, disk0, 100)) == intake_status::inserted);
  auto const job = store.take({7, host_t0, klass::A, 0});
  REQUIRE(job.has_value());
  store.apply_outcome(*job, outcome(promotion_outcome::promoted, 100));
  CHECK(store.find(7) == nullptr);
  REQUIRE(store.find_ledger(7) != nullptr);
  CHECK(store.find_ledger(7)->gpu == -1);
}

TEST_CASE("candidate_store overflow skips a newest query that is entirely in flight", "[preload]")
{
  store_fixture f;
  f.cfg.max_candidates = 3;
  candidate_store store{f.cfg, {0}, {0}, f.counters};
  REQUIRE(store.add_query(f.q1, {0}, {}));
  REQUIRE(store.add_query(f.q2, {0}, {}));
  fake_batch b;
  REQUIRE(store.intake(f.q2, record(b, 20, 5, gpu0, host0, 1)) == intake_status::inserted);
  REQUIRE(store.intake(f.q2, record(b, 21, 6, gpu0, host0, 1)) == intake_status::inserted);
  REQUIRE(store.intake(f.q1, record(b, 10, 7, gpu0, host0, 1)) == intake_status::inserted);
  REQUIRE(store.take({20, gpu_t0, klass::A, 0}).has_value());
  REQUIRE(store.take({21, gpu_t0, klass::A, 0}).has_value());

  // The older query's own newest candidate goes: here, the record itself.
  CHECK(store.intake(f.q1, record(b, 11, 8, gpu0, host0, 1)) == intake_status::dropped_overflow);
  CHECK(store.find(11) == nullptr);
  // A late record with an older sequence evicts the older query's newest candidate instead.
  CHECK(store.intake(f.q1, record(b, 12, 3, gpu0, host0, 1)) == intake_status::inserted);
  CHECK(store.find(10) == nullptr);
  CHECK(store.find(12) != nullptr);
  CHECK(store.find(20) != nullptr);
  CHECK(store.find(21) != nullptr);
  CHECK(store.size() == 3);
}

TEST_CASE("candidate_store pruning resumes after its cursor and wraps around", "[preload]")
{
  store_fixture f;
  std::vector<fake_batch> batches(5);
  for (std::uint64_t i = 0; i < 4; ++i) {
    REQUIRE(f.store.intake(f.q1, record(batches[i], i, i + 1, gpu0, host0, 1)) ==
            intake_status::inserted);
  }
  REQUIRE(f.store.intake(f.q2, record(batches[4], 4, 5, gpu0, host0, 1)) ==
          intake_status::inserted);

  CHECK(f.store.prune_expired_candidates(3) == 0);  // visits q1's seq 1..3
  batches[0].destroy();                             // behind the cursor
  batches[4].destroy();                             // ahead of it, in the next query
  CHECK(f.store.prune_expired_candidates(3) == 2);  // seq 4, q2's seq 5, then wraps to seq 1
  CHECK(f.store.find(0) == nullptr);
  CHECK(f.store.find(4) == nullptr);
  CHECK(f.store.size() == 3);
  // A budget above the store size visits each candidate once.
  CHECK(f.store.prune_expired_candidates(100) == 0);
}
