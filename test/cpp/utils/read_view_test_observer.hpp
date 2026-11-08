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

#pragma once

#include "op/scan/table_scan/bound_read_view.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace sirius::test {

struct observed_read_view_map {
  bool internal;
  uint64_t generation;
  std::vector<sirius::op::scan::logical_bound_read_view> views;
};

struct read_view_observer_info : duckdb::OptimizerExtensionInfo {
  std::atomic<duckdb::ClientContext*> target{nullptr};
  bool discard_copy = false;
  std::mutex mutex;
  std::vector<observed_read_view_map> maps;
  std::function<void()> after_original;
  std::function<void()> after_internal;
};

inline void observe_read_view_map(duckdb::OptimizerExtensionInput& input,
                                  duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  auto& info = static_cast<read_view_observer_info&>(*input.info);
  if (info.target.load() != &input.context) return;
  auto state = duckdb::get_sirius_connection_state(input.context);
  if (!state) return;
  auto views = sirius::op::scan::capture_bound_read_views(*plan, input.context);
  if (views.empty()) return;
  auto const internal = state->is_internal_query_active();
  std::function<void()> publish;
  {
    std::lock_guard lock(info.mutex);
    info.maps.push_back({internal, state->planning_generation(), std::move(views)});
    publish = std::exchange(internal ? info.after_internal : info.after_original, {});
  }
  if (info.discard_copy && !internal) {
    // Exercise finalize's SQL replan without discarding the original binding evidence.
    auto original = state->take_captured_original_views_if_current();
    state->clear_captured_plan();
    if (original) state->set_captured_original_views(std::move(original->views));
  }
  // Callbacks may publish fixture metadata or throw, but never make Catch assertions.
  if (publish) publish();
}

class read_view_map_observer {
 public:
  explicit read_view_map_observer(duckdb::Connection& con,
                                  bool discard_copy                    = false,
                                  std::function<void()> after_original = {},
                                  std::function<void()> after_internal = {})
    : info(duckdb::make_shared_ptr<read_view_observer_info>())
  {
    info->target         = con.context.get();
    info->discard_copy   = discard_copy;
    info->after_original = std::move(after_original);
    info->after_internal = std::move(after_internal);
    duckdb::OptimizerExtension extension;
    extension.optimizer_info    = info;
    extension.optimize_function = observe_read_view_map;
    duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context),
                                         std::move(extension));
  }

  ~read_view_map_observer()
  {
    info->target = nullptr;
    std::lock_guard lock(info->mutex);
    info->after_original = {};
    info->after_internal = {};
    info->maps.clear();
  }

  std::vector<observed_read_view_map> snapshot() const
  {
    std::lock_guard lock(info->mutex);
    return info->maps;
  }

  void require_equal_replan_map(std::size_t scans) const
  {
    auto const maps = snapshot();
    REQUIRE(maps.size() == 2);
    auto const& original  = maps[0];
    auto const& candidate = maps[1];
    REQUIRE_FALSE(original.internal);
    REQUIRE(candidate.internal);
    CHECK(original.generation == candidate.generation);
    REQUIRE(original.views.size() == scans);
    REQUIRE(candidate.views.size() == scans);
    std::set<duckdb::idx_t> matched;
    for (auto const& rebound : candidate.views) {
      CAPTURE(rebound.table_index);
      REQUIRE(matched.insert(rebound.table_index).second);
      auto const found =
        std::find_if(original.views.begin(), original.views.end(), [&](auto const& bound) {
          return bound.table_index == rebound.table_index;
        });
      REQUIRE(found != original.views.end());
      REQUIRE(found->view.identity);
      REQUIRE(rebound.view.identity);
      CHECK(found->view.identity->fingerprint.canonical ==
            rebound.view.identity->fingerprint.canonical);
      CHECK(found->view.selector_evidence_required == rebound.view.selector_evidence_required);
      CHECK(found->view.logical_selector_evidence == rebound.view.logical_selector_evidence);
    }
  }

 private:
  duckdb::shared_ptr<read_view_observer_info> info;
};
}  // namespace sirius::test
