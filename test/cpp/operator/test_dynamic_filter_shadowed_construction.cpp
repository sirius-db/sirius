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

/**
 * Tests of a Publication Session whose keys are Shadowed Keys: two joins publish to one Channel,
 * and the second join's Bloom filters lose to the Small IN-list the first one published. Both
 * sessions, their Constructions, and the Channel are real.
 */

#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/dynamic_filter/static_filter_selection.hpp"
#include "operator/dynamic_filter_accumulation_test_utils.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace op  = sirius::op;
namespace acc = sirius::test::accumulation;
using op::membership_filter_kind;
using plan_type = op::dynamic_filter_publish_plan;

constexpr cudf::size_type k_rows     = 500;
constexpr std::size_t k_value_domain = 1000;  ///< Each judged key keeps half of its value domain
constexpr cudf::size_type k_strong_rows =
  8;                                       ///< The other join's keys: a Small IN-list keeping 0.8%
constexpr std::size_t k_other_column = 7;  ///< The Channel column the other join publishes to
constexpr std::size_t k_other_condition =
  9;  ///< Marks the other join's `[dynamic_filter_key]` line
constexpr std::uint64_t k_bloom_bytes = 256 * acc::mib;

struct plan_options {
  std::vector<bool> proven_unique   = {true, true};  ///< Per key: unique by proof
  std::size_t decode_membership_cap = 1;
  double inlist_max_l2_fraction     = 0.0;  ///< 0 plans a Bloom filter for every key
};

/**
 * @brief The fixture's two-key plan (INT32 and INT64 on Channel columns 0 and 1) with a known value
 * domain for both keys.
 */
plan_type shadow_plan(acc::fixture const& fixture, plan_options options = {})
{
  std::vector const types{cudf::data_type{cudf::type_id::INT32},
                          cudf::data_type{cudf::type_id::INT64}};
  std::vector<plan_type::admitted_key> keys;
  std::vector<plan_type::key_binding> bindings;
  for (std::size_t index = 0; index < types.size(); ++index) {
    keys.push_back({.planner_condition_index = index,
                    .build_key_ordinal       = static_cast<cudf::size_type>(index),
                    .storage_type            = types[index],
                    .build_key_value_domain  = k_value_domain,
                    .build_key_proven_unique = options.proven_unique.at(index)});
    bindings.push_back({index, index, types[index]});
  }
  return plan_type{std::move(keys),
                   {{fixture.channel, op::dynamic_filter_route_class::scan, false, bindings}},
                   fixture.spaces,
                   {.inlist_max_l2_fraction  = options.inlist_max_l2_fraction,
                    .max_bloom_bytes_per_gpu = k_bloom_bytes,
                    .decode_membership_cap   = options.decode_membership_cap}};
}

/**
 * @brief Certifies @p batches and begins @p session with them.
 */
bool begin(op::dynamic_filter_publication_session& session,
           std::vector<acc::batch_ptr> const& batches)
{
  auto certified = acc::fixture::certify(batches);
  REQUIRE(std::holds_alternative<op::complete_build_inventory>(certified));
  return session.begin(std::move(std::get<op::complete_build_inventory>(certified)));
}

/**
 * @brief The other join: its own session on the fixture's Channel, which publishes a Small IN-list
 * of `k_strong_rows` unique keys to column `k_other_column`.
 *
 * Both joins register their Producers when their sessions are created, before either seals the
 * Channel, so the fixture's session must exist before this one publishes.
 */
struct other_join {
  op::dynamic_filter_stats stats;
  std::unique_ptr<op::dynamic_filter_publication_session> session;

  /**
   * @param proven Whether the published estimate is a Proven Estimate
   * @param replica_devices The GPUs the filter gets a Replica on; empty means all of the fixture's
   */
  explicit other_join(acc::fixture const& fixture,
                      bool proven                      = true,
                      std::vector<int> replica_devices = {})
  {
    std::vector<op::dynamic_filter_replica_space> replicas;
    for (std::size_t device = 0; device < fixture.devices; ++device) {
      if (replica_devices.empty() ||
          std::ranges::find(replica_devices, static_cast<int>(device)) != replica_devices.end()) {
        replicas.push_back(fixture.spaces[device]);
      }
    }
    auto const type = cudf::data_type{cudf::type_id::INT64};
    session         = std::make_unique<op::dynamic_filter_publication_session>(
      plan_type{{{.planner_condition_index = k_other_condition,
                          .build_key_ordinal       = 1,
                          .storage_type            = type,
                          .build_key_value_domain  = k_value_domain,
                          .build_key_proven_unique = proven}},
                        {{fixture.channel,
                          op::dynamic_filter_route_class::scan,
                          false,
                          {{0, k_other_column, type}}}},
                std::move(replicas),
                        {.inlist_max_l2_fraction = 1.0}},
      &stats);
  }

  /**
   * @brief Constructs and publishes the join's filter from one batch on GPU 0.
   */
  void publish(acc::fixture const& fixture) const
  {
    auto const batch = fixture.make_batch(0, 0, k_strong_rows);
    REQUIRE(begin(*session, {batch}));
    auto source = batch->to_read_only();
    session->contribute(batch->get_batch_id(), source, fixture.task_stream(0));
    fixture.task_stream(0).sync();
    session->publish_if_final(batch->get_batch_id(), fixture.gpu(0), fixture.publication_stream(0));
    session->finalize_input();
    REQUIRE(stats.snapshot().membership_filters_built == 1);
  }
};

/**
 * @brief Creates the fixture's session without beginning it, which registers its Producer.
 */
void open_session(acc::fixture& fixture, plan_options options = {})
{
  fixture.session = std::make_unique<op::dynamic_filter_publication_session>(
    shadow_plan(fixture, std::move(options)), &fixture.stats);
}

/**
 * @brief Contributes every batch and publishes from the last one's task, as the build PARTITION
 * does once `begin` returned true.
 */
void construct(acc::fixture const& fixture, std::vector<acc::batch_ptr> const& batches)
{
  for (auto const& batch : batches) {
    fixture.contribute(batch);
  }
  auto const& last = batches.back();
  fixture.publish(last->to_read_only().get_memory_space()->get_device_id(), last->get_batch_id());
}

/**
 * @brief The `reason=` token of each `[dynamic_filter_key]` line of the fixture's session that the
 * sink recorded, in order.
 */
std::vector<std::string> key_reasons(sirius::test::scoped_recording_log_sink const& sink)
{
  auto const other = std::format(" condition={} ", k_other_condition);
  std::vector<std::string> reasons;
  for (auto const& record : sink.records()) {
    auto const& message = record.message;
    if (!message.starts_with("[dynamic_filter_key]") || message.find(other) != std::string::npos) {
      continue;
    }
    reasons.push_back(message.substr(message.rfind("reason=") + 7));
  }
  return reasons;
}

/**
 * @brief The filters on one column of the fixture's Channel.
 */
std::vector<std::shared_ptr<op::sirius_dynamic_filter const>> built_on(acc::fixture const& fixture,
                                                                       std::size_t column)
{
  return acc::filters_on_column(*fixture.channel, column);
}

}  // namespace

TEST_CASE("a session whose keys are all shadowed constructs nothing",
          "[dynamic_filter][construction][shadow]")
{
  acc::fixture fixture;
  sirius::test::scoped_recording_log_sink sink{"debug"};
  auto const batches = std::vector{fixture.make_batch(0, 0, 300), fixture.make_batch(0, 1000, 200)};
  auto const baseline = fixture.allocated_bytes();
  {
    other_join const other{fixture};
    open_session(fixture);
    other.publish(fixture);
  }

  REQUIRE_FALSE(begin(*fixture.session, batches));

  // The attempt ended at `begin`: no lease was taken and the Channel has nothing left to wait for.
  CHECK(fixture.session->terminal());
  CHECK(fixture.channel->snapshot().terminal());
  CHECK(fixture.channel->snapshot().entries().size() == 1);
  auto const counts = fixture.stats.snapshot();
  CHECK(counts.keys_skipped_shadowed == 2);
  CHECK(counts.accumulations_started == 0);
  CHECK(counts.membership_filters_built == 0);
  CHECK(counts.publication_attempts == 1);
  CHECK(counts.publications_finished == 1);
  CHECK(key_reasons(sink) == std::vector<std::string>{"shadowed", "shadowed"});
  fixture.session->finalize_input();
  fixture.require_storage_returned(baseline);
}

TEST_CASE("a shadowed key leaves the other key of its join to publish",
          "[dynamic_filter][construction][shadow]")
{
  acc::fixture fixture;
  sirius::test::scoped_recording_log_sink sink{"debug"};
  other_join const other{fixture};
  auto const batches = std::vector{fixture.make_batch(0, 0, k_rows)};
  // Only the first key has a Proven Estimate, so only it can be judged.
  open_session(fixture, {.proven_unique = {true, false}});
  other.publish(fixture);

  REQUIRE(begin(*fixture.session, batches));
  construct(fixture, batches);
  fixture.session->finalize_input();

  CHECK(built_on(fixture, 0).empty());
  REQUIRE(built_on(fixture, 1).size() == 1);
  CHECK(fixture.channel->snapshot().terminal());
  auto const counts = fixture.stats.snapshot();
  CHECK(counts.keys_skipped_shadowed == 1);
  CHECK(counts.membership_filters_built == 1);
  CHECK(counts.accumulation_publications_finished == 1);
  CHECK(key_reasons(sink) == std::vector<std::string>{"shadowed", "-"});
}

TEST_CASE("a key is constructed when a Consumer could still select its filter",
          "[dynamic_filter][construction][shadow]")
{
  acc::fixture fixture;
  auto const batches = std::vector{fixture.make_batch(0, 0, k_rows)};
  plan_options options;
  bool other_proven    = true;
  bool other_publishes = true;

  SECTION("nothing is published yet") { other_publishes = false; }
  SECTION("a second Filtered Decode slot carries it") { options.decode_membership_cap = 2; }
  SECTION("the published estimate is not proven") { other_proven = false; }
  SECTION("the keys have no Proven Estimate") { options.proven_unique = {false, false}; }

  other_join const other{fixture, other_proven};
  open_session(fixture, options);
  if (other_publishes) { other.publish(fixture); }
  REQUIRE(begin(*fixture.session, batches));
  construct(fixture, batches);
  fixture.session->finalize_input();
  CHECK(built_on(fixture, 0).size() == 1);
  CHECK(built_on(fixture, 1).size() == 1);
  CHECK(fixture.stats.snapshot().keys_skipped_shadowed == 0);
  CHECK(fixture.stats.snapshot().membership_filters_built == 2);
}

TEST_CASE("a published filter carries the estimate its key was planned with",
          "[dynamic_filter][construction][shadow]")
{
  acc::fixture fixture;
  auto const expected_keep = static_cast<double>(k_rows) / static_cast<double>(k_value_domain);

  SECTION("a Bloom filter over a unique key: a Proven Estimate and no key count")
  {
    auto const batches = std::vector{fixture.make_batch(0, 0, k_rows)};
    open_session(fixture);
    REQUIRE(begin(*fixture.session, batches));
    construct(fixture, batches);
    REQUIRE(built_on(fixture, 1).size() == 1);
    CHECK(op::facts_of(*built_on(fixture, 1).front()) ==
          op::static_filter_facts{.kind            = membership_filter_kind::bloom,
                                  .keys            = std::numeric_limits<std::uint64_t>::max(),
                                  .estimated_keep  = expected_keep,
                                  .estimate_proven = true});
  }
  SECTION("a key that is not unique: the estimate is not proven")
  {
    auto const batches = std::vector{fixture.make_batch(0, 0, k_rows)};
    open_session(fixture, {.proven_unique = {false, false}});
    REQUIRE(begin(*fixture.session, batches));
    construct(fixture, batches);
    REQUIRE(built_on(fixture, 0).size() == 1);
    auto const facts = op::facts_of(*built_on(fixture, 0).front());
    CHECK(facts.estimated_keep == expected_keep);
    CHECK_FALSE(facts.estimate_proven);
  }
  SECTION("a hash IN-list: the key count is the build's valid rows")
  {
    auto const batches = std::vector{fixture.make_batch(0, 0, k_rows)};
    open_session(fixture, {.inlist_max_l2_fraction = 1.0});
    REQUIRE(begin(*fixture.session, batches));
    construct(fixture, batches);
    REQUIRE(built_on(fixture, 1).size() == 1);
    CHECK(op::facts_of(*built_on(fixture, 1).front()) ==
          op::static_filter_facts{.kind            = membership_filter_kind::hash_in_list,
                                  .keys            = k_rows,
                                  .estimated_keep  = expected_keep,
                                  .estimate_proven = true});
  }
  SECTION("a small IN-list")
  {
    constexpr cudf::size_type few = 8;
    auto const batches            = std::vector{fixture.make_batch(0, 0, few)};
    open_session(fixture, {.inlist_max_l2_fraction = 1.0});
    REQUIRE(begin(*fixture.session, batches));
    construct(fixture, batches);
    REQUIRE(built_on(fixture, 1).size() == 1);
    CHECK(op::facts_of(*built_on(fixture, 1).front()) ==
          op::static_filter_facts{.kind            = membership_filter_kind::small_in_list,
                                  .keys            = few,
                                  .estimated_keep  = static_cast<double>(few) / k_value_domain,
                                  .estimate_proven = true});
  }
  fixture.session->finalize_input();
}

TEST_CASE("a session on two GPUs is shadowed only by a filter with a Replica on both",
          "[dynamic_filter][construction][shadow][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  acc::fixture fixture{{.devices = 2}};
  auto const batches = std::vector{fixture.make_batch(0, 0, 300), fixture.make_batch(1, 1000, 200)};

  SECTION("a Replica on both GPUs shadows every key")
  {
    other_join const other{fixture};
    open_session(fixture);
    other.publish(fixture);
    REQUIRE_FALSE(begin(*fixture.session, batches));
    CHECK(fixture.session->terminal());
    CHECK(fixture.channel->snapshot().terminal());
    CHECK(fixture.stats.snapshot().keys_skipped_shadowed == 2);
    CHECK(fixture.stats.snapshot().accumulations_started == 0);
  }
  SECTION("a Replica on one GPU leaves the other GPU's Consumers to the new filters")
  {
    other_join const other{fixture, true, {0}};
    open_session(fixture);
    other.publish(fixture);
    REQUIRE(begin(*fixture.session, batches));
    construct(fixture, batches);
    CHECK(built_on(fixture, 0).size() == 1);
    CHECK(built_on(fixture, 1).size() == 1);
    CHECK(fixture.stats.snapshot().keys_skipped_shadowed == 0);
  }
  fixture.session->finalize_input();
}

TEST_CASE("a construction shadowed before it finishes is abandoned",
          "[dynamic_filter][construction][shadow]")
{
  acc::fixture fixture;
  sirius::test::scoped_recording_log_sink sink{"debug"};
  auto const batches = std::vector{fixture.make_batch(0, 0, 300), fixture.make_batch(0, 1000, 200)};
  auto const baseline = fixture.allocated_bytes();
  auto other          = std::make_unique<other_join>(fixture);
  open_session(fixture);
  REQUIRE(begin(*fixture.session, batches));

  SECTION("the stronger filter is published while the build is partitioned")
  {
    fixture.contribute(batches[0]);
    other->publish(fixture);
    fixture.contribute(batches[1]);
    fixture.publish(0, batches[1]->get_batch_id());

    // The Publishing Task ended the attempt before finishing: nothing was merged or replicated.
    CHECK(fixture.session->terminal());
    CHECK(fixture.channel->snapshot().terminal());
    CHECK(fixture.channel->snapshot().entries().size() == 1);
    auto const counts = fixture.stats.snapshot();
    CHECK(counts.keys_skipped_shadowed_late == 2);
    CHECK(counts.keys_skipped_shadowed == 0);
    CHECK(counts.accumulations_started == 1);
    CHECK(counts.membership_filters_built == 0);
    CHECK(counts.accumulation_publications_finished == 0);
    CHECK(counts.publications_finished == 1);
    CHECK(key_reasons(sink) == std::vector<std::string>{"shadowed_late", "shadowed_late"});
    fixture.session->finalize_input();
    other.reset();
    fixture.require_storage_returned(baseline);
  }
  SECTION("a stronger filter published after the session published leaves both on the Channel")
  {
    construct(fixture, batches);
    other->publish(fixture);
    fixture.session->finalize_input();
    CHECK(built_on(fixture, 0).size() == 1);
    CHECK(built_on(fixture, 1).size() == 1);
    CHECK(fixture.channel->snapshot().entries().size() == 3);
    CHECK(fixture.stats.snapshot().keys_skipped_shadowed_late == 0);
    CHECK(key_reasons(sink) == std::vector<std::string>{"-", "-"});
  }
}

TEST_CASE("a construction with one key a Consumer could select finishes every key",
          "[dynamic_filter][construction][shadow]")
{
  acc::fixture fixture;
  other_join const other{fixture};
  auto const batches = std::vector{fixture.make_batch(0, 0, k_rows)};
  // Only the first key has a Proven Estimate, so the second is never shadowed.
  open_session(fixture, {.proven_unique = {true, false}});
  REQUIRE(begin(*fixture.session, batches));
  other.publish(fixture);
  construct(fixture, batches);
  fixture.session->finalize_input();

  CHECK(built_on(fixture, 0).size() == 1);
  CHECK(built_on(fixture, 1).size() == 1);
  CHECK(fixture.stats.snapshot().keys_skipped_shadowed_late == 0);
  CHECK(fixture.stats.snapshot().membership_filters_built == 2);
}
