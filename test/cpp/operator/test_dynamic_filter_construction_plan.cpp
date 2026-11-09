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
 * Host-only tests of `plan_construction`, the per-key policy that chooses each admitted key's
 * filters from the facts of a certified build inventory. The membership kind is checked against
 * `choose_membership_filter` called with the inputs the whole-build (one-shot) publisher passes.
 */

#include "helper/numeric_narrowing.hpp"
#include "op/dynamic_filter/bloom_sizing.hpp"
#include "op/dynamic_filter/construction_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_key_domain.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cudf/types.hpp>

#include <rmm/aligned.hpp>

#include <catch.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

namespace op = sirius::op;
using op::dynamic_filter_key_reason;
using op::membership_filter_kind;
using admitted_key = op::dynamic_filter_publish_plan::admitted_key;

bool is_decimal(cudf::type_id id)
{
  return id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64 ||
         id == cudf::type_id::DECIMAL128;
}

cudf::data_type type(cudf::type_id id, std::int32_t scale = 0)
{
  return is_decimal(id) ? cudf::data_type{id, scale} : cudf::data_type{id};
}

auto const k_int16  = type(cudf::type_id::INT16);
auto const k_int32  = type(cudf::type_id::INT32);
auto const k_int64  = type(cudf::type_id::INT64);
auto const k_date   = type(cudf::type_id::TIMESTAMP_DAYS);
auto const k_float  = type(cudf::type_id::FLOAT64);
auto const k_bool   = type(cudf::type_id::BOOL8);
auto const k_string = type(cudf::type_id::STRING);
auto const k_empty  = type(cudf::type_id::EMPTY);

constexpr std::size_t k_l2 = std::size_t{4} << 20;

/// One key of a test plan: its recorded type, its column's facts, and its bindings.
struct key_case {
  cudf::data_type recorded;
  op::key_column_facts column;
  std::vector<op::binding_facts> bindings;
  bool preserve_nulls = false;
  op::dynamic_filter_domain_evidence domain{};
  bool proven_unique = false;
};

/// A binding whose probe reads the recorded type through a scan that accepts zone maps.
op::binding_facts scan_binding(cudf::data_type probe)
{
  return {.probe_type = probe, .accepts_zone_map = true};
}

key_case plain_key(cudf::data_type recorded, std::optional<std::size_t> null_rows = 0)
{
  return {.recorded = recorded,
          .column   = {.carriers = {recorded}, .null_rows = null_rows},
          .bindings = {scan_binding(recorded)}};
}

struct planned {
  op::construction_plan plan;
  std::vector<admitted_key> keys;
};

op::construction_limits limits(std::size_t l2 = k_l2)
{
  return {.l2_bytes                  = l2,
          .inlist_max_l2_fraction    = 0.125,
          .emit_zone_map_filters     = false,
          .domain_coverage_threshold = 0.9,
          .max_bloom_bytes_per_gpu   = std::uint64_t{256} << 20};
}

/// Plans @p cases, key `i` reading build ordinal `i`, over @p rows rows in one batch.
planned plan_for(std::vector<key_case> const& cases,
                 std::size_t rows,
                 op::construction_limits const& limit = limits())
{
  op::build_input_facts facts{.total_rows = rows, .max_batch_rows = rows, .columns = {}};
  std::vector<admitted_key> keys;
  std::vector<std::vector<op::binding_facts>> bindings;
  for (std::size_t index = 0; index < cases.size(); ++index) {
    auto const& c = cases[index];
    facts.columns.push_back(c.column);
    keys.push_back({.planner_condition_index = index + 10,
                    .build_key_ordinal       = static_cast<cudf::size_type>(index),
                    .storage_type            = c.recorded,
                    .build_key_domain        = c.domain,
                    .build_key_proven_unique = c.proven_unique,
                    .preserve_nulls          = c.preserve_nulls});
    bindings.push_back(c.bindings);
  }
  return {op::plan_construction(facts, keys, bindings, limit), std::move(keys)};
}

/// The kind the one-shot publisher chooses for @p valid_rows valid keys of @p type.
membership_filter_kind one_shot_kind(cudf::data_type type, std::size_t valid_rows, std::size_t l2)
{
  return op::choose_membership_filter(
    {.build_rows     = valid_rows,
     .l2_cache_bytes = l2,
     .estimated_hash_set_bytes =
       op::sirius_dynamic_in_list_filter::estimated_set_bytes(valid_rows, type),
     .inlist_max_l2_fraction = 0.125,
     .supports_small_in_list =
       valid_rows >= 1 && valid_rows <= 12 && op::membership_key_supported(type),
     .supports_hash_in_list = op::membership_key_supported(type),
     .supports_bloom        = op::membership_key_supported(type)});
}

/// The planned membership kind of admitted key @p index, or none when it has no membership.
membership_filter_kind planned_kind(op::construction_plan const& plan, std::size_t index)
{
  for (auto const& key : plan.keys) {
    if (key.admitted_key_index != index) { continue; }
    if (auto const* exact = std::get_if<op::exact_membership>(&key.membership)) {
      return exact->kind;
    }
    if (std::holds_alternative<op::bloom_membership>(key.membership)) {
      return membership_filter_kind::bloom;
    }
  }
  return membership_filter_kind::none;
}

op::key_plan const* key_of(op::construction_plan const& plan, std::size_t index)
{
  for (auto const& key : plan.keys) {
    if (key.admitted_key_index == index) { return &key; }
  }
  return nullptr;
}

op::dynamic_filter_key_outcome const& outcome_of(op::construction_plan const& plan,
                                                 std::size_t index)
{
  for (auto const& line : plan.outcomes) {
    if (line.admitted_key_index == index) { return line.outcome; }
  }
  FAIL("no outcome for admitted key " << index);
  throw std::logic_error{"unreachable"};
}

}  // namespace

TEST_CASE("each key's membership kind equals the one-shot choice",
          "[dynamic_filter][construction_plan]")
{
  auto const recorded  = GENERATE(type(cudf::type_id::INT32),
                                 type(cudf::type_id::INT64),
                                 type(cudf::type_id::UINT16),
                                 type(cudf::type_id::TIMESTAMP_DAYS),
                                 type(cudf::type_id::TIMESTAMP_MILLISECONDS),
                                 type(cudf::type_id::DECIMAL64, -2),
                                 type(cudf::type_id::STRING));
  auto const rows      = GENERATE(std::size_t{1}, 12, 13, 5000, 10'000'000);
  auto const null_rows = GENERATE(std::size_t{0}, 1);
  auto const l2        = GENERATE(std::size_t{0}, k_l2);
  CAPTURE(static_cast<int>(recorded.id()), rows, null_rows, l2);

  auto const result = plan_for({plain_key(recorded, null_rows)}, rows, limits(l2)).plan;
  auto const valid  = rows - null_rows;
  CHECK(planned_kind(result, 0) == one_shot_kind(recorded, valid, l2));
  auto const* key = key_of(result, 0);
  REQUIRE(key != nullptr);
  CHECK(key->construction_type == recorded);
  if (auto const* exact = std::get_if<op::exact_membership>(&key->membership)) {
    CHECK(exact->valid_rows == valid);
  }
  if (auto const* bloom = std::get_if<op::bloom_membership>(&key->membership)) {
    CHECK(bloom->sized_rows == valid);
    CHECK(bloom->blocks == op::bloom_blocks_for(valid));
  }
  auto const& outcome = outcome_of(result, 0);
  CHECK(outcome.rows == rows);
  CHECK(outcome.valid_rows == valid);
  CHECK(outcome.reason == dynamic_filter_key_reason::not_constructed);
}

TEST_CASE("the hash IN-list stops at the configured L2 fraction",
          "[dynamic_filter][construction_plan]")
{
  // An INT64 set takes 8-byte slots at a load factor of one half: 16 bytes per key. With 1600
  // bytes of L2 and a fraction of one half, 50 keys fit and 51 do not.
  auto limit                   = limits(1600);
  limit.inlist_max_l2_fraction = 0.5;
  CHECK(planned_kind(plan_for({plain_key(k_int64)}, 50, limit).plan, 0) ==
        membership_filter_kind::hash_in_list);
  CHECK(planned_kind(plan_for({plain_key(k_int64)}, 51, limit).plan, 0) ==
        membership_filter_kind::bloom);
}

TEST_CASE("an all-null key plans the one-shot empty set", "[dynamic_filter][construction_plan]")
{
  auto const known  = plan_for({plain_key(k_int32, 40)}, 40).plan;
  auto const* exact = std::get_if<op::exact_membership>(&key_of(known, 0)->membership);
  REQUIRE(exact != nullptr);
  CHECK(exact->kind == membership_filter_kind::hash_in_list);
  CHECK(exact->valid_rows == 0);

  auto const unknown_l2 = plan_for({plain_key(k_int32, 40)}, 40, limits(0)).plan;
  auto const* bloom     = std::get_if<op::bloom_membership>(&key_of(unknown_l2, 0)->membership);
  REQUIRE(bloom != nullptr);
  CHECK(bloom->blocks == 1);
}

TEST_CASE("an unknown null count plans a Bloom filter sized on every row",
          "[dynamic_filter][construction_plan]")
{
  auto const result = plan_for({plain_key(k_int32, std::nullopt)}, 5).plan;
  auto const* bloom = std::get_if<op::bloom_membership>(&key_of(result, 0)->membership);
  REQUIRE(bloom != nullptr);
  CHECK(bloom->sized_rows == 5);
  CHECK(bloom->blocks == op::bloom_blocks_for(5));
  CHECK(result.selection.keys_sized_on_total_rows == 1);
  CHECK_FALSE(outcome_of(result, 0).valid_rows);
}

TEST_CASE("the carrier rule picks each key's construction type",
          "[dynamic_filter][construction_plan]")
{
  auto narrowed        = plain_key(k_int64);
  narrowed.column      = {.carriers = {k_int16}, .null_rows = 0};
  auto date_as_int     = plain_key(k_date);
  date_as_int.column   = {.carriers = {k_int16}, .null_rows = 0};
  auto heterogeneous   = plain_key(k_int64);
  heterogeneous.column = {.carriers = {k_int16, k_int32}, .null_rows = 0};
  auto mixed_equal     = plain_key(k_int64);
  mixed_equal.column   = {.carriers = {k_int64, k_int16}, .null_rows = 0};
  auto const result    = plan_for({narrowed, date_as_int, heterogeneous, mixed_equal}, 100).plan;

  CHECK(key_of(result, 0)->construction_type == k_int16);  // Same family: build at the carrier.
  CHECK(key_of(result, 1)->construction_type == k_date);   // Another family: restore first.
  CHECK(key_of(result, 2)->construction_type == k_int64);  // Several carriers: the recorded type.
  CHECK(key_of(result, 3)->construction_type == k_int64);  // Equality is exempt from restoration.
  CHECK(result.selection.keys_skipped_type_mismatch == 0);
}

TEST_CASE("a carrier that cannot restore skips only its key", "[dynamic_filter][construction_plan]")
{
  auto widened      = plain_key(k_int32);
  widened.column    = {.carriers = {k_int64}, .null_rows = 0};
  auto absent       = plain_key(k_int32);
  absent.column     = {.carriers = {}, .null_rows = std::nullopt};
  auto const result = plan_for({widened, absent, plain_key(k_int64)}, 100).plan;
  CHECK(result.selection.keys_skipped_type_mismatch == 2);
  CHECK(result.selection.keys_considered == 3);
  CHECK(key_of(result, 0) == nullptr);
  CHECK(key_of(result, 1) == nullptr);
  CHECK(key_of(result, 2) != nullptr);
  CHECK(outcome_of(result, 0).reason == dynamic_filter_key_reason::type_mismatch);
  CHECK(outcome_of(result, 1).reason == dynamic_filter_key_reason::type_mismatch);
  CHECK_FALSE(outcome_of(result, 0).valid_rows);
}

TEST_CASE("an ordinal outside every batch's schema is a plan bug",
          "[dynamic_filter][construction_plan]")
{
  op::build_input_facts const facts{
    .total_rows = 1, .max_batch_rows = 1, .columns = {{.carriers = {k_int32}, .null_rows = 0}}};
  std::vector<admitted_key> const keys{{.build_key_ordinal = 1, .storage_type = k_int32}};
  std::vector<std::vector<op::binding_facts>> const bindings{{scan_binding(k_int32)}};
  CHECK_THROWS_AS(op::plan_construction(facts, keys, bindings, limits()), std::logic_error);
}

TEST_CASE("types no membership filter supports plan only a zone map",
          "[dynamic_filter][construction_plan]")
{
  auto limit                  = limits();
  limit.emit_zone_map_filters = true;
  auto unaccepted             = plain_key(k_bool);
  unaccepted.bindings         = {{.probe_type = k_bool, .accepts_zone_map = false}};
  auto const result =
    plan_for({plain_key(k_float), plain_key(k_bool), unaccepted}, 100, limit).plan;

  CHECK(key_of(result, 0) == nullptr);
  CHECK(outcome_of(result, 0).reason == dynamic_filter_key_reason::unsupported_type);
  auto const* zone_only = key_of(result, 1);
  REQUIRE(zone_only != nullptr);
  CHECK(std::holds_alternative<std::monostate>(zone_only->membership));
  CHECK(zone_only->zone_map);
  CHECK(zone_only->zone_map_deliverable);
  CHECK(outcome_of(result, 1).reason == dynamic_filter_key_reason::not_constructed);
  // A zone map no binding accepts explains nothing, so the key reports its type.
  CHECK(outcome_of(result, 2).reason == dynamic_filter_key_reason::unsupported_type);
  CHECK(result.selection.bindings_skipped_incompatible_probe == 0);
}

TEST_CASE("a key no binding can read plans no membership", "[dynamic_filter][construction_plan]")
{
  auto unreadable        = plain_key(k_int32);
  unreadable.bindings    = {scan_binding(k_float), scan_binding(k_string)};
  auto unknown_probe     = plain_key(k_int32);
  unknown_probe.bindings = {scan_binding(k_empty)};
  auto const result      = plan_for({unreadable, unknown_probe}, 100).plan;
  CHECK(key_of(result, 0) == nullptr);
  CHECK(outcome_of(result, 0).reason == dynamic_filter_key_reason::unreadable_probe);
  CHECK(result.selection.bindings_skipped_incompatible_probe == 2);
  CHECK(planned_kind(result, 1) !=
        membership_filter_kind::none);  // An unknown probe type is readable.
}

TEST_CASE("the domain gate judges only publishable keys", "[dynamic_filter][construction_plan]")
{
  auto covered          = plain_key(k_int32);
  covered.proven_unique = true;
  covered.domain        = {.cardinality = 100};
  auto mismatched       = covered;
  mismatched.column     = {.carriers = {k_int64}, .null_rows = 0};
  auto const result     = plan_for({covered, mismatched}, 100).plan;
  CHECK(key_of(result, 0) == nullptr);
  CHECK(outcome_of(result, 0).reason == dynamic_filter_key_reason::domain_gate);
  CHECK(outcome_of(result, 1).reason == dynamic_filter_key_reason::type_mismatch);
  CHECK(result.selection.keys_skipped_domain_gate == 1);
  CHECK(result.selection.keys_skipped_type_mismatch == 1);
  CHECK(result.selection.keys_considered == 2);
  CHECK(result.selection.keys_with_known_domain == 2);
}

TEST_CASE("zone maps follow the option, the key type, and NULL preservation",
          "[dynamic_filter][construction_plan]")
{
  auto limit                  = limits();
  limit.emit_zone_map_filters = GENERATE(true, false);
  auto null_safe              = plain_key(k_int32);
  null_safe.preserve_nulls    = true;
  auto decimal128             = plain_key(type(cudf::type_id::DECIMAL128, -2));
  auto const result =
    plan_for({plain_key(k_int32), null_safe, plain_key(k_float), decimal128}, 100, limit).plan;

  CHECK(key_of(result, 0)->zone_map == limit.emit_zone_map_filters);
  CHECK_FALSE(key_of(result, 1)->zone_map);
  CHECK(key_of(result, 2) == nullptr);
  auto const* wide = key_of(result, 3);
  REQUIRE(wide != nullptr);
  CHECK(wide->fit_pending);
  CHECK(wide->zone_map == limit.emit_zone_map_filters);
  CHECK_FALSE(key_of(result, 0)->fit_pending);
}

TEST_CASE("the Bloom cap applies to the sum of the keys' aligned arrays",
          "[dynamic_filter][construction_plan]")
{
  constexpr std::size_t rows = 100'000;
  auto const array_bytes     = rmm::align_up(op::bloom_blocks_for(rows) * op::bloom_bytes_per_block,
                                         rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto limit                 = limits(0);  // No L2: every key plans a Bloom filter.
  limit.emit_zone_map_filters = true;
  auto exact                  = plain_key(k_int32, rows - 5);  // Five valid keys: a small IN-list.
  std::vector<key_case> const cases{plain_key(k_int32), plain_key(k_int64), exact};

  SECTION("at the cap")
  {
    limit.max_bloom_bytes_per_gpu = 2 * array_bytes;
    auto const result             = plan_for(cases, rows, limit).plan;
    CHECK(planned_kind(result, 0) == membership_filter_kind::bloom);
    CHECK(planned_kind(result, 1) == membership_filter_kind::bloom);
    CHECK(result.selection.keys_skipped_bloom_size_gate == 0);
  }
  SECTION("below the cap")
  {
    limit.max_bloom_bytes_per_gpu = 2 * array_bytes - 1;
    auto const result             = plan_for(cases, rows, limit).plan;
    for (std::size_t index : {0, 1}) {
      CHECK(planned_kind(result, index) == membership_filter_kind::none);
      CHECK(key_of(result, index)->zone_map);  // The zone map survives the cap.
      CHECK(outcome_of(result, index).reason == dynamic_filter_key_reason::bloom_cap);
    }
    CHECK(planned_kind(result, 2) == membership_filter_kind::small_in_list);
    CHECK(result.selection.keys_skipped_bloom_size_gate == 2);
  }
}

TEST_CASE("only keys with a binding are planned and reported",
          "[dynamic_filter][construction_plan]")
{
  auto unbound      = plain_key(k_int32);
  unbound.bindings  = {};
  auto const result = plan_for({unbound, plain_key(k_int64)}, 7).plan;
  REQUIRE(result.outcomes.size() == 1);
  CHECK(result.outcomes.front().admitted_key_index == 1);
  CHECK(result.outcomes.front().planner_condition_index == 11);
  CHECK(result.outcomes.front().build_key_ordinal == 1);
  REQUIRE(result.keys.size() == 1);
  CHECK(result.keys.front().admitted_key_index == 1);
  CHECK(result.selection.keys_considered == 1);
}

TEST_CASE("count-based IN-list support matches the column gates",
          "[dynamic_filter][construction_plan]")
{
  CHECK_FALSE(op::sirius_dynamic_small_in_list_filter::supports(k_int32, 0));
  CHECK(op::sirius_dynamic_small_in_list_filter::supports(k_int32, 1));
  CHECK(op::sirius_dynamic_small_in_list_filter::supports(k_int32, 12));
  CHECK_FALSE(op::sirius_dynamic_small_in_list_filter::supports(k_int32, 13));
  CHECK_FALSE(op::sirius_dynamic_small_in_list_filter::supports(k_float, 1));
  CHECK(op::sirius_dynamic_in_list_filter::supports(k_string, 0));
  CHECK_FALSE(op::sirius_dynamic_in_list_filter::supports(k_float, 10));
}

TEST_CASE("every restorable carrier is a probe carrier of its recorded key",
          "[dynamic_filter][construction_plan]")
{
  // Per-element restoration through the construction domain's adapter relies on this implication.
  // Pairs that fail would need a materialized cast; there are none.
  std::vector<cudf::data_type> types;
  for (auto id = 0; id < static_cast<int>(cudf::type_id::NUM_TYPE_IDS); ++id) {
    auto const type_id = static_cast<cudf::type_id>(id);
    if (is_decimal(type_id)) {
      for (std::int32_t scale : {0, -2, -4}) {
        types.emplace_back(type_id, scale);
      }
    } else {
      types.emplace_back(type_id);
    }
  }
  std::vector<std::string> failures;
  std::size_t restorable = 0;
  for (auto const source : types) {
    for (auto const target : types) {
      if (!sirius::can_restore_to(source, target)) { continue; }
      ++restorable;
      auto const domain = op::classify_membership_key(target);
      if (domain && !op::membership_probe_compatible(*domain, source)) {
        failures.push_back(std::to_string(static_cast<int>(source.id())) + "->" +
                           std::to_string(static_cast<int>(target.id())));
      }
    }
  }
  CHECK(restorable > 0);
  CHECK(failures.empty());
  CAPTURE(failures);
}
