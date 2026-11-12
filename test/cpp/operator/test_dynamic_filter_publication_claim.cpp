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

/*
 * Tests a hash join's publication session through the calls its build PARTITION makes: one
 * certified build batch, `begin`, a contribution, and the final contributor's publication. They pin
 * the session-level outcomes the join depends on: publication in any join mode and in BUILD_PROBE
 * mode, replica narrowing, fail-open memory exhaustion, cancellation, drained targets, unexpected
 * errors, and the rights a session acquires or abandons.
 */

#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "utils/dynamic_filter_publication_fixture.hpp"

#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

using namespace sirius::test::publication;

TEST_CASE("a hash join's session constructs and publishes in any join mode",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  claim_fixture fixture;
  REQUIRE_FALSE(fixture.hash_join->is_build_probe_mode());
  fixture.construct();
  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 1);
  CHECK_FALSE(fixture.channel->snapshot().empty());
  CHECK(fixture.channel->snapshot().terminal());
}

TEST_CASE("a publication counts incompatible probe bindings in the session stats",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  // No integer key domain reads a FLOAT64 probe, so the only binding gets no filter, and no
  // membership filter is built for it.
  claim_fixture fixture{duckdb::JoinType::INNER, 1, 0, cudf::data_type{cudf::type_id::FLOAT64}};
  fixture.construct();
  auto const counters = fixture.stats.snapshot();
  CHECK(counters.publication_attempts == 1);
  CHECK(counters.membership_filters_built == 0);
  CHECK(counters.bindings_skipped_incompatible_probe == 1);
  CHECK(counters.filters_pushed == 0);
  CHECK(fixture.channel->snapshot().empty());
  CHECK(fixture.channel->snapshot().terminal());
}

TEST_CASE("a hash join in BUILD_PROBE mode still publishes",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  claim_fixture fixture;
  // Sizing from a small foldable build side selects BUILD_PROBE.
  auto const strategy = fixture.hash_join->get_partition_strategy(
    sirius::op::partition_sizing_input{.total_bytes    = 1024,
                                       .is_build_side  = true,
                                       .build_foldable = true,
                                       .measured_rows  = 64,
                                       .resident_bytes = 1024});
  REQUIRE(strategy.build_probe);
  REQUIRE(fixture.hash_join->is_build_probe_mode());
  fixture.construct();
  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 1);
  CHECK_FALSE(fixture.channel->snapshot().empty());
}

TEST_CASE("a join whose replica restriction removed every GPU never attempts",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  claim_fixture fixture;
  // The pipeline converter's per-query GPU restriction admitted none of the plan's replica
  // devices, disabling publication for this join before execution.
  fixture.hash_join->restrict_dynamic_filter_replicas({kDeviceId + 1});
  fixture.hash_join->seal_dynamic_filter_plan();
  CHECK_FALSE(fixture.hash_join->publishes_dynamic_filters());
  CHECK(fixture.channel->snapshot().terminal());
  CHECK_NOTHROW(fixture.construct());
  CHECK(fixture.stats.publication_attempts.load() == 0);
  CHECK(fixture.channel->snapshot().empty());
}

TEST_CASE("device memory exhaustion during construction fails open",
          "[dynamic_filter][publication_claim][gpu_execution]")
{
  claim_fixture fixture;
  // Past the small-list gate, so construction must lease device memory.
  constexpr std::size_t kLargeBuildRows = 100000;
  auto batch = claim_fixture::make_build_batch_on(*fixture.gpu_space, kLargeBuildRows);
  {
    // Leave under 64 KiB of the space's accounting capacity, so every lease is refused.
    constexpr std::size_t kHeadroomBytes = 64ull << 10;
    auto const stream                    = fixture.gpu_space->acquire_stream();
    auto const available                 = fixture.gpu_space->get_available_memory();
    REQUIRE(available > kHeadroomBytes);
    rmm::device_buffer const ballast{
      available - kHeadroomBytes, stream, fixture.gpu_space->get_default_allocator()};
    CHECK_NOTHROW(fixture.construct_from(batch));
  }
  auto const counters = fixture.stats.snapshot();
  CHECK(counters.publication_attempts == 1);
  CHECK(counters.accumulations_skipped_admission == 1);
  CHECK(counters.membership_filters_built == 0);
  CHECK(fixture.channel->snapshot().empty());
  CHECK(fixture.channel->snapshot().terminal());
  // The attempt is over: with the ballast freed, another build does not reattempt.
  fixture.construct();
  CHECK(fixture.stats.publication_attempts.load() == 1);
}

TEST_CASE("cancellation before publication accounts the attempt as failed",
          "[dynamic_filter][publication_claim][publication_lifecycle][gpu_execution]")
{
  claim_fixture fixture;
  auto& session = fixture.hash_join->dynamic_filter_session();
  auto batch    = fixture.make_build_batch();
  sirius::op::build_arrival_ledger ledger;
  ledger.record(*batch);
  auto certified = ledger.certify({.repository_batches = 1});
  REQUIRE(session.begin(std::move(std::get<sirius::op::complete_build_inventory>(certified))));
  auto const stream = fixture.gpu_space->acquire_stream();
  {
    auto source = batch->to_read_only();
    session.contribute(batch->get_batch_id(), source, stream);
    stream.sync();
  }
  // No operation is active, so the cancelled attempt completes at once and owes no publication.
  fixture.hash_join->cancel_dynamic_filter_publication();
  CHECK(fixture.channel->snapshot().terminal());
  session.publish_if_final(batch->get_batch_id(), *fixture.gpu_space, stream);
  auto const after = fixture.channel->snapshot();
  CHECK(after.terminal());
  CHECK(after.empty());
  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_failed.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 0);
}

TEST_CASE("cancelling an open session does not invent an attempt",
          "[dynamic_filter][publication_claim][publication_lifecycle][gpu_execution]")
{
  claim_fixture fixture;
  fixture.hash_join->seal_dynamic_filter_plan();
  fixture.hash_join->cancel_dynamic_filter_publication();
  fixture.construct();
  CHECK(fixture.channel->snapshot().terminal());
  CHECK(fixture.channel->snapshot().empty());
  CHECK(fixture.stats.publication_attempts.load() == 0);
  CHECK(fixture.stats.publications_finished.load() == 0);
  CHECK(fixture.stats.publications_failed.load() == 0);
}

TEST_CASE("a drained target completes a successful zero-filter attempt",
          "[dynamic_filter][publication_claim][publication_lifecycle][gpu_execution]")
{
  claim_fixture fixture;
  fixture.channel->close_for_new_filters();
  fixture.construct();
  CHECK(fixture.channel->snapshot().terminal());
  CHECK(fixture.channel->snapshot().empty());
  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_finished.load() == 1);
  CHECK(fixture.stats.publications_failed.load() == 0);
  CHECK(fixture.stats.publications_skipped_targets_drained.load() == 1);
}

TEST_CASE("unexpected construction errors resolve completion and propagate",
          "[dynamic_filter][publication_claim][publication_lifecycle][gpu_execution]")
{
  // The admitted key's build ordinal lies outside the build schema: planning rejects it.
  claim_fixture fixture{duckdb::JoinType::INNER, 1, 1};
  CHECK_THROWS_AS(fixture.construct(), std::logic_error);
  CHECK(fixture.channel->snapshot().terminal());
  CHECK(fixture.channel->snapshot().empty());
  CHECK(fixture.stats.publication_attempts.load() == 1);
  CHECK(fixture.stats.publications_failed.load() == 1);
}

TEST_CASE("session construction failure resolves rights already acquired",
          "[dynamic_filter][publication_claim][publication_lifecycle][gpu_execution]")
{
  claim_fixture fixture;
  auto first  = std::make_shared<sirius_dynamic_filter_set>();
  auto sealed = std::make_shared<sirius_dynamic_filter_set>();
  sealed->freeze_registration();
  auto const& original       = fixture.hash_join->dynamic_filter_plan();
  auto targets               = original.probe_targets();
  targets.front().filter_set = first;
  targets.push_back(targets.front());
  targets.back().filter_set = sealed;
  dynamic_filter_publish_plan plan{
    original.admitted_keys(), std::move(targets), original.replica_spaces()};

  CHECK_THROWS_AS(sirius::op::dynamic_filter_publication_session{std::move(plan)},
                  std::logic_error);
  first->freeze_registration();
  CHECK(first->snapshot().terminal());
  CHECK(first->snapshot().empty());
}

TEST_CASE("abandoning a planned session does not freeze sibling registrations",
          "[dynamic_filter][publication_claim][publication_lifecycle][gpu_execution]")
{
  claim_fixture fixture;
  {
    sirius::op::dynamic_filter_publication_session abandoned{
      fixture.hash_join->dynamic_filter_plan()};
  }
  CHECK_FALSE(fixture.channel->snapshot().terminal());
  CHECK_NOTHROW(
    sirius::op::dynamic_filter_publication_session{fixture.hash_join->dynamic_filter_plan()});

  fixture.hash_join->seal_dynamic_filter_plan();
  CHECK_FALSE(fixture.channel->snapshot().terminal());
  fixture.hash_join->cancel_dynamic_filter_publication();
  CHECK(fixture.channel->snapshot().terminal());
  CHECK(fixture.stats.publication_attempts.load() == 0);
}
