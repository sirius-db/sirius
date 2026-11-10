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
 * Tests the `[dynamic_filter_key]` outcome line: its spelling, the structural join identity, the
 * deferred `post_unlock_actions` that carries it, and the one line per bound key that each legacy
 * publication path (whole-build and accumulated) emits when its session becomes terminal.
 */

#include "dynamic_filter_accumulation_test_utils.hpp"
#include "op/dynamic_filter/dynamic_filter_key_outcome.hpp"
#include "op/dynamic_filter/post_unlock_actions.hpp"
#include "utils/dynamic_filter_publication_fixture.hpp"
#include "utils/log_test_utils.hpp"

#include <catch.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

namespace acc = sirius::test::accumulation;
namespace op  = sirius::op;
using op::dynamic_filter_key_line;
using op::dynamic_filter_key_reason;
using op::membership_filter_kind;

constexpr std::string_view k_prefix = "[dynamic_filter_key] ";

/// The `[dynamic_filter_key]` messages @p sink recorded, in order.
std::vector<std::string> key_lines(sirius::test::scoped_recording_log_sink const& sink)
{
  std::vector<std::string> lines;
  for (auto const& record : sink.records()) {
    if (record.message.starts_with(k_prefix)) { lines.push_back(record.message); }
  }
  return lines;
}

dynamic_filter_key_line sample_line()
{
  return {.join = {.signature = 0x0123456789abcdefULL, .occurrence = 2, .operator_id = 17},
          .admitted_key_index      = 4,
          .planner_condition_index = 1,
          .build_key_ordinal       = 3,
          .outcome                 = {.rows       = 100,
                                      .valid_rows = 90,
                                      .kind       = membership_filter_kind::hash_in_list,
                                      .size       = 90,
                                      .zone_map   = true,
                                      .pushed     = 2,
                                      .reason     = dynamic_filter_key_reason::published}};
}

}  // namespace

TEST_CASE("a key line spells every field in a fixed order", "[dynamic_filter][key_outcome]")
{
  CHECK(op::to_string(sample_line()) ==
        "[dynamic_filter_key] v=1 sig=0123456789abcdef occ=2 join=17 condition=1 build_ordinal=3 "
        "rows=100 valid=90 kind=hash_in_list size=90 zone_map=1 pushed=2 reason=-");
}

TEST_CASE("a key line spells unknown values and attempt reasons", "[dynamic_filter][key_outcome]")
{
  dynamic_filter_key_line line{.planner_condition_index = 0,
                               .build_key_ordinal       = 0,
                               .outcome = {.reason        = dynamic_filter_key_reason::attempt_skip,
                                           .reason_detail = "accumulations_incomplete"}};
  CHECK(op::to_string(line) ==
        "[dynamic_filter_key] v=1 sig=0000000000000000 occ=0 join=- condition=0 build_ordinal=0 "
        "rows=0 valid=unknown kind=none size=0 zone_map=0 pushed=0 "
        "reason=attempt_skip_accumulations_incomplete");

  using enum dynamic_filter_key_reason;
  auto const reason_of = [&](dynamic_filter_key_reason reason, std::string_view counter = {}) {
    line.outcome.reason        = reason;
    line.outcome.reason_detail = counter;
    auto const text            = op::to_string(line);
    return text.substr(text.rfind("reason=") + 7);
  };
  CHECK(reason_of(domain_gate) == "domain_gate");
  CHECK(reason_of(type_mismatch) == "type_mismatch");
  CHECK(reason_of(unsupported_type) == "unsupported_type");
  CHECK(reason_of(unreadable_probe) == "unreadable_probe");
  CHECK(reason_of(bloom_cap) == "bloom_cap");
  CHECK(reason_of(decimal128_unfit) == "decimal128_unfit");
  CHECK(reason_of(not_constructed) == "not_constructed");
  CHECK(reason_of(attempt_fail, "publications_failed") == "attempt_fail_publications_failed");
  CHECK(reason_of(attempt_cancel, "publications_failed") == "attempt_cancel_publications_failed");

  line.outcome.kind = membership_filter_kind::small_in_list;
  CHECK(op::to_string(line).find(" kind=small_in_list ") != std::string::npos);
  line.outcome.kind = membership_filter_kind::bloom;
  CHECK(op::to_string(line).find(" kind=bloom ") != std::string::npos);
}

TEST_CASE("the join signature is FNV-1a over the join type and ordered conditions",
          "[dynamic_filter][key_outcome]")
{
  // Expected values come from an independent FNV-1a over the same byte stream.
  std::array<std::string, 1> const one{"l_orderkey = o_orderkey"};
  CHECK(op::dynamic_filter_join_signature("INNER", one) == 0x6b85214e8d4b623aULL);
  CHECK(op::dynamic_filter_join_signature("", {}) == 0xcbf29ce484222325ULL);

  std::array<std::string, 2> const ab{"a = b", "c = d"};
  std::array<std::string, 2> const ba{"c = d", "a = b"};
  CHECK(op::dynamic_filter_join_signature("INNER", ab) == 0x0002db22499b13c7ULL);
  CHECK(op::dynamic_filter_join_signature("INNER", ba) == 0xde8475fcd9790c0fULL);
  CHECK(op::dynamic_filter_join_signature("LEFT", ab) == 0x2567866319e72cf8ULL);

  // The separator keeps a condition boundary from being moved without changing the hash.
  std::array<std::string, 1> const merged{"a = bc = d"};
  CHECK(op::dynamic_filter_join_signature("INNER", merged) !=
        op::dynamic_filter_join_signature("INNER", ab));
}

TEST_CASE("join occurrences count earlier joins with the same signature",
          "[dynamic_filter][key_outcome]")
{
  std::array<std::uint64_t, 5> const signatures{7, 9, 7, 7, 9};
  CHECK(op::dynamic_filter_join_occurrences(signatures) == std::vector<std::size_t>{0, 0, 1, 2, 1});
  CHECK(op::dynamic_filter_join_occurrences({}).empty());

  // Renumbering operators while keeping their relative order (for example inserting a
  // BUILD_FILTER) leaves the signature order, and therefore every occurrence, unchanged.
  std::array<std::uint64_t, 6> const with_unrelated_join{7, 9, 3, 7, 7, 9};
  CHECK(op::dynamic_filter_join_occurrences(with_unrelated_join) ==
        std::vector<std::size_t>{0, 0, 0, 1, 2, 1});
}

TEST_CASE("post-unlock actions are move-only and log their lines once when dispatched",
          "[dynamic_filter][key_outcome]")
{
  static_assert(!std::is_copy_constructible_v<op::post_unlock_actions>);
  static_assert(!std::is_copy_assignable_v<op::post_unlock_actions>);
  static_assert(!std::is_move_assignable_v<op::post_unlock_actions>);
  static_assert(std::is_nothrow_move_constructible_v<op::post_unlock_actions>);

  sirius::test::scoped_recording_log_sink sink{"debug"};
  op::post_unlock_actions actions;
  CHECK(actions.empty());

  auto first                     = sample_line();
  auto second                    = sample_line();
  second.planner_condition_index = 5;
  actions.append(op::post_unlock_actions{{first}});
  actions.append(op::post_unlock_actions{});
  actions.append(op::post_unlock_actions{{second}});
  CHECK_FALSE(actions.empty());
  CHECK(key_lines(sink).empty());

  op::post_unlock_actions moved{std::move(actions)};
  CHECK(actions.empty());  // NOLINT(bugprone-use-after-move): a moved-from value is empty
  std::move(moved).dispatch();
  CHECK(moved.empty());  // NOLINT(bugprone-use-after-move): dispatch clears the value
  CHECK(key_lines(sink) == std::vector<std::string>{op::to_string(first), op::to_string(second)});

  std::move(moved).dispatch();
  CHECK(key_lines(sink).size() == 2);
}

namespace {

/// The one key line of a single-key whole-build session after @p act, with its identity set.
std::vector<std::string> whole_build_lines(sirius::test::publication::claim_fixture& fixture,
                                           auto const& act)
{
  sirius::test::scoped_recording_log_sink sink{"debug"};
  fixture.hash_join->dynamic_filter_session().set_join_identity(
    {.signature = 0x2a, .occurrence = 1, .operator_id = 7});
  act();
  fixture.hash_join->dynamic_filter_session().finalize_input();
  fixture.hash_join->dynamic_filter_session().cancel();
  return key_lines(sink);
}

constexpr std::string_view k_whole_build_identity =
  "[dynamic_filter_key] v=1 sig=000000000000002a occ=1 join=7 condition=0 build_ordinal=0 ";

}  // namespace

TEST_CASE("a whole-build publication logs its key once, after terminal completion",
          "[dynamic_filter][key_outcome][publication_claim]")
{
  sirius::test::publication::claim_fixture fixture;
  auto const lines = whole_build_lines(fixture, [&] {
    fixture.hash_join->set_build_arrives_whole(true);
    fixture.hash_join->push_data_batch_partitioned(
      "build",
      sirius::test::publication::claim_fixture::make_build_batch_on(*fixture.gpu_space, 5),
      /*partition_idx=*/0);
    REQUIRE(fixture.channel->snapshot().terminal());
  });
  CHECK(lines == std::vector<std::string>{std::string{k_whole_build_identity} +
                                          "rows=5 valid=5 kind=small_in_list size=5 zone_map=0 "
                                          "pushed=1 reason=-"});
}

TEST_CASE("a whole-build membership filter no binding can read is logged as absent",
          "[dynamic_filter][key_outcome][publication_claim]")
{
  // No integer key domain reads a FLOAT64 probe: the filter is built but never pushed.
  sirius::test::publication::claim_fixture fixture{
    duckdb::JoinType::INNER, 1, 0, cudf::data_type{cudf::type_id::FLOAT64}};
  auto const lines = whole_build_lines(fixture, [&] {
    fixture.hash_join->set_build_arrives_whole(true);
    fixture.hash_join->push_data_batch_partitioned(
      "build",
      sirius::test::publication::claim_fixture::make_build_batch_on(*fixture.gpu_space, 5),
      /*partition_idx=*/0);
  });
  REQUIRE(fixture.stats.snapshot().membership_filters_built == 1);
  CHECK(lines == std::vector<std::string>{std::string{k_whole_build_identity} +
                                          "rows=5 valid=5 kind=none size=0 zone_map=0 pushed=0 "
                                          "reason=unreadable_probe"});
}

TEST_CASE("a whole-build session that never claims logs its key at finalization",
          "[dynamic_filter][key_outcome][publication_claim]")
{
  sirius::test::publication::claim_fixture fixture;
  auto const lines = whole_build_lines(fixture, [&] { fixture.push_build_batch(); });
  CHECK(lines == std::vector<std::string>{std::string{k_whole_build_identity} +
                                          "rows=0 valid=unknown kind=none size=0 zone_map=0 "
                                          "pushed=0 reason=not_constructed"});
}

TEST_CASE("a whole-build attempt that ends early names the counter that ended it",
          "[dynamic_filter][key_outcome][publication_claim]")
{
  sirius::test::publication::claim_fixture fixture;
  SECTION("drained targets")
  {
    auto const lines = whole_build_lines(fixture, [&] {
      fixture.channel->close_for_new_filters();
      fixture.hash_join->set_build_arrives_whole(true);
      fixture.push_build_batch();
    });
    CHECK(lines ==
          std::vector<std::string>{std::string{k_whole_build_identity} +
                                   "rows=0 valid=unknown kind=none size=0 zone_map=0 pushed=0 "
                                   "reason=attempt_skip_publications_skipped_targets_drained"});
  }
  SECTION("a source outside the replica set, then closed input")
  {
    auto const lines = whole_build_lines(fixture, [&] {
      fixture.hash_join->set_build_arrives_whole(true);
      fixture.hash_join->push_data_batch_partitioned(
        "build", fixture.make_host_build_batch(), /*partition_idx=*/0);
    });
    CHECK(lines ==
          std::vector<std::string>{std::string{k_whole_build_identity} +
                                   "rows=0 valid=unknown kind=none size=0 zone_map=0 pushed=0 "
                                   "reason=attempt_skip_publications_skipped_source_not_resident"});
  }
}

TEST_CASE("a session narrowed to no replica logs no key line",
          "[dynamic_filter][key_outcome][publication_claim]")
{
  sirius::test::publication::claim_fixture fixture;
  sirius::test::scoped_recording_log_sink sink{"debug"};
  fixture.hash_join->restrict_dynamic_filter_replicas({sirius::test::publication::kDeviceId + 1});
  fixture.hash_join->seal_dynamic_filter_plan();
  REQUIRE_FALSE(fixture.hash_join->publishes_dynamic_filters());
  REQUIRE(fixture.channel->snapshot().terminal());
  fixture.hash_join->dynamic_filter_session().finalize_input();
  CHECK(key_lines(sink).empty());
}

TEST_CASE("an accumulated publication logs every bound key once",
          "[dynamic_filter][key_outcome][multi_partition]")
{
  acc::fixture fixture;
  sirius::test::scoped_recording_log_sink sink{"debug"};
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(0, 1000, 200);
  REQUIRE(fixture.begin({first, second}));
  fixture.contribute(first);
  fixture.contribute(second);
  CHECK(key_lines(sink).empty());
  fixture.publish(0, second->get_batch_id());
  REQUIRE(fixture.channel->snapshot().terminal());
  fixture.session->finalize_input();

  // Each key's Bloom filter is sized on its own valid rows, which the inventory records.
  auto const bytes = op::sirius_dynamic_bloom_filter::estimated_bytes(500);
  auto const line  = [&](std::size_t key) {
    return std::format(
      "[dynamic_filter_key] v=1 sig=0000000000000000 occ=0 join=- condition={0} build_ordinal={0} "
       "rows=500 valid=500 kind=bloom size={1} zone_map=0 pushed=1 reason=-",
      key,
      bytes);
  };
  CHECK(key_lines(sink) == std::vector<std::string>{line(0), line(1)});
}

TEST_CASE("an accumulation that ends without publishing logs why per key",
          "[dynamic_filter][key_outcome][multi_partition]")
{
  acc::fixture fixture;
  sirius::test::scoped_recording_log_sink sink{"debug"};
  auto const first  = fixture.make_batch(0, 0, 300);
  auto const second = fixture.make_batch(0, 1000, 200);
  std::string reason;
  std::string rows  = "500";
  std::string valid = "500";
  SECTION("input closed before every batch contributed")
  {
    REQUIRE(fixture.begin({first, second}));
    fixture.contribute(first);
    // An attempt that is still collecting has logged nothing yet.
    CHECK(key_lines(sink).empty());
    reason = "attempt_skip_accumulations_incomplete";
  }
  SECTION("a cap below the arrays")
  {
    REQUIRE_FALSE(fixture.begin({first, second}, 256));
    reason = "bloom_cap";
  }
  SECTION("keys whose recorded types the build does not carry")
  {
    REQUIRE_FALSE(fixture.begin({first, second},
                                fixture.make_plan(256 * acc::mib,
                                                  {cudf::data_type{cudf::type_id::INT16},
                                                   cudf::data_type{cudf::type_id::INT16}})));
    reason = "type_mismatch";
    valid  = "unknown";
  }
  SECTION("a ledger that disagrees with its repository")
  {
    fixture.session = std::make_unique<op::dynamic_filter_publication_session>(
      fixture.make_plan(256 * acc::mib), &fixture.stats);
    fixture.session->decline(op::construction_decline::inventory_mismatch);
    reason = "declined_inventory_mismatch";
    rows   = "0";
    valid  = "unknown";
  }
  fixture.session->finalize_input();
  fixture.session->cancel();

  auto const line = [&](std::size_t key) {
    return std::format(
      "[dynamic_filter_key] v=1 sig=0000000000000000 occ=0 join=- condition={0} build_ordinal={0} "
      "rows={1} valid={3} kind=none size=0 zone_map=0 pushed=0 reason={2}",
      key,
      rows,
      reason,
      valid);
  };
  CHECK(key_lines(sink) == std::vector<std::string>{line(0), line(1)});
}
