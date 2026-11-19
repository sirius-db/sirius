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
 * Tests of `read_published_filters`: which of a Channel's filters count when the Construction Plan
 * judges a Shadowed Key. The filters are stand-ins that report a kind, an estimate, and the GPUs
 * they have a Replica on; the Channels and the plan are real.
 */

#include "op/dynamic_filter/construction_plan.hpp"
#include "op/dynamic_filter/cross_rank_filter_merge.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/types.hpp>

#include <cuda/stream>

#include <catch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace op = sirius::op;
using op::membership_filter_kind;
using plan_type = op::dynamic_filter_publish_plan;

constexpr auto k_int64 = cudf::data_type{cudf::type_id::INT64};

/**
 * @brief A filter of one kind with a Replica on every GPU, or on none.
 */
class stub_filter final : public op::sirius_dynamic_filter {
 public:
  explicit stub_filter(op::sirius_dynamic_filter_kind kind, bool replicated = true)
    : _kind{kind}, _replicated{replicated}
  {
  }
  [[nodiscard]] op::sirius_dynamic_filter_kind kind() const override { return _kind; }
  [[nodiscard]] bool is_available_on_device(int /*device_id*/) const noexcept override
  {
    return _replicated;
  }

 private:
  op::sirius_dynamic_filter_kind _kind;
  bool _replicated;
};

std::shared_ptr<stub_filter> bloom(std::optional<double> keep, bool proven, bool replicated = true)
{
  auto filter = std::make_shared<stub_filter>(op::sirius_dynamic_filter_kind::BLOOM, replicated);
  if (keep) { filter->set_estimated_selectivity(*keep, proven); }
  return filter;
}

/**
 * @brief A Channel with one Producer that has room on columns 0 and 1.
 */
struct channel {
  std::shared_ptr<op::sirius_dynamic_filter_set> set =
    std::make_shared<op::sirius_dynamic_filter_set>();
  op::sirius_dynamic_filter_set::producer producer = set->register_producer({0, 1});

  channel() { set->freeze_registration(); }

  void publish(std::size_t column, std::shared_ptr<op::sirius_dynamic_filter const> filter) const
  {
    REQUIRE(producer.push_filter(column, std::move(filter)));
  }
};

plan_type::probe_target target_on(channel const& to, std::vector<std::size_t> admitted_keys)
{
  plan_type::probe_target target{.filter_set               = to.set,
                                 .route_class              = op::dynamic_filter_route_class::scan,
                                 .accepts_zone_map_filters = true,
                                 .key_bindings             = {}};
  for (auto const key : admitted_keys) {
    target.key_bindings.push_back(
      {.admitted_key_index = key, .channel_push_ordinal = 0, .probe_storage_type = k_int64});
  }
  return target;
}

/**
 * @brief GPU 0's memory space and a host space, which every enabled plan needs as a replica space.
 */
struct replica_fixture {
  decltype(sirius::test::operator_utils::initialize_memory_manager(1)) memory_manager =
    sirius::test::operator_utils::initialize_memory_manager(1);

  [[nodiscard]] std::vector<op::dynamic_filter_replica_space> replicas() const
  {
    auto* const gpu_space = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    REQUIRE(gpu_space != nullptr);
    auto const host_spaces =
      memory_manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    REQUIRE_FALSE(host_spaces.empty());
    return {{*gpu_space, *host_spaces.front()}};
  }
};

plan_type plan_of(std::size_t keys,
                  std::vector<plan_type::probe_target> targets,
                  std::vector<op::dynamic_filter_replica_space> replicas)
{
  std::vector<plan_type::admitted_key> admitted(keys);
  for (std::size_t index = 0; index < keys; ++index) {
    admitted[index].planner_condition_index = index;
    admitted[index].storage_type            = k_int64;
    admitted[index].probe_storage_type      = k_int64;
  }
  return plan_type{std::move(admitted), std::move(targets), std::move(replicas)};
}

std::vector<std::vector<op::binding_facts>> published_of(plan_type const& plan)
{
  auto bindings = op::binding_facts_of(plan);
  op::read_published_filters(plan, bindings);
  return bindings;
}

struct idle_merge final : op::cross_rank_filter_merge {
  void merge(std::span<std::shared_ptr<op::sirius_dynamic_filter>> /*membership*/,
             ::cuda::stream_ref /*stream*/) override
  {
  }
  void leave() noexcept override {}
};

}  // namespace

TEST_CASE("only published Membership Filters with a Proven Estimate can shadow",
          "[dynamic_filter][construction_plan][published_filters]")
{
  replica_fixture const spaces;
  channel scan;
  auto const plan = plan_of(1, {target_on(scan, {0})}, spaces.replicas());

  SECTION("an empty Channel reports nothing") { CHECK(published_of(plan)[0][0].published.empty()); }
  SECTION("a zone map does not count")
  {
    scan.publish(0, std::make_shared<stub_filter>(op::sirius_dynamic_filter_kind::ZONE_MAP));
    CHECK(published_of(plan)[0][0].published.empty());
  }
  SECTION("every Membership Filter is reported, with its estimate only when proven")
  {
    scan.publish(0, bloom(0.25, true));
    scan.publish(1, bloom(0.5, false));
    scan.publish(0, bloom(std::nullopt, false));
    auto const published = published_of(plan)[0][0].published;
    REQUIRE(published.size() == 3);
    // An estimate that is not proven is dropped: its filter can still take a decode slot.
    auto const bloom_without_estimate =
      op::static_filter_facts{.kind = membership_filter_kind::bloom};
    auto proven_bloom            = bloom_without_estimate;
    proven_bloom.estimated_keep  = 0.25;
    proven_bloom.estimate_proven = true;
    CHECK(std::ranges::count(published, proven_bloom) == 1);
    CHECK(std::ranges::count(published, bloom_without_estimate) == 2);
  }
  SECTION("one filter published on two columns counts once")
  {
    auto const filter = bloom(0.25, true);
    scan.publish(0, filter);
    scan.publish(1, filter);
    CHECK(published_of(plan)[0][0].published.size() == 1);
  }
  SECTION("a later call sees what was published since")
  {
    auto bindings = op::binding_facts_of(plan);
    op::read_published_filters(plan, bindings);
    REQUIRE(bindings[0][0].published.empty());
    scan.publish(0, bloom(0.25, true));
    op::read_published_filters(plan, bindings);
    CHECK(bindings[0][0].published.size() == 1);
  }
}

TEST_CASE("each binding reads its own Channel",
          "[dynamic_filter][construction_plan][published_filters]")
{
  replica_fixture const spaces;
  channel first;
  channel second;
  first.publish(0, bloom(0.25, true));
  // Key 1 is bound to both Channels, key 0 to the second only, key 2 to none.
  auto const plan =
    plan_of(3, {target_on(first, {1}), target_on(second, {0, 1})}, spaces.replicas());
  auto const bindings = published_of(plan);
  REQUIRE(bindings.size() == 3);
  REQUIRE(bindings[0].size() == 1);
  REQUIRE(bindings[1].size() == 2);
  CHECK(bindings[2].empty());
  CHECK(bindings[0][0].published.empty());
  CHECK(bindings[1][0].published.size() == 1);
  CHECK(bindings[1][1].published.empty());

  SECTION("bindings of another plan are refused")
  {
    auto const other = plan_of(3, {target_on(first, {1})}, spaces.replicas());
    auto mismatched  = op::binding_facts_of(other);
    CHECK_THROWS_AS(op::read_published_filters(plan, mismatched), std::logic_error);
    std::vector<std::vector<op::binding_facts>> short_list(2);
    CHECK_THROWS_AS(op::read_published_filters(plan, short_list), std::logic_error);
  }
}

TEST_CASE("a cross-rank merge counts no published filter",
          "[dynamic_filter][construction_plan][published_filters]")
{
  replica_fixture const spaces;
  channel scan;
  scan.publish(0, bloom(0.25, true));
  auto plan     = plan_of(1, {target_on(scan, {0})}, spaces.replicas());
  auto bindings = published_of(plan);
  REQUIRE(bindings[0][0].published.size() == 1);

  plan.set_cross_rank_merge(std::make_shared<idle_merge>(), plan.policy());
  op::read_published_filters(plan, bindings);
  CHECK(bindings[0][0].published.empty());
}

TEST_CASE("a filter without a Replica on every GPU of the plan does not count",
          "[dynamic_filter][construction_plan][published_filters]")
{
  replica_fixture const spaces;
  channel scan;
  scan.publish(0, bloom(0.25, true, false));
  scan.publish(1, bloom(0.5, true));
  auto const plan      = plan_of(1, {target_on(scan, {0})}, spaces.replicas());
  auto const published = published_of(plan)[0][0].published;
  REQUIRE(published.size() == 1);
  CHECK(published[0].estimated_keep == 0.5);
}
