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

#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"
#include "op/dynamic_filter/exact_host_scalar.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace sirius::op {

/**
 * @brief Which layer a target receives
 *
 * `FIRST_KEY` carries a bound on key zero alone and travels as far upstream as key zero traces;
 * `LEX` carries the full-tuple predicate and applies only where every key coexists. The LEX
 * predicate implies the first-key bound, so one site never receives both.
 *
 * The first-key bound is inclusive for a multi-key producer, because later keys may still order
 * rows that tie key zero. A single-key producer has no later key, so its `FIRST_KEY` target
 * carries the producer's whole *strict* predicate -- `LEX_RANGE` requires two components, so
 * RANGE's strict form is what expresses it. `top_n_first_key_filters_pushed` therefore counts
 * that strict whole-predicate install too, and `top_n_lex_filters_pushed` stays zero for a
 * single-key producer.
 */
enum class top_n_filter_layer { FIRST_KEY, LEX };

/**
 * @brief What kind of producer owns this plan
 *
 * `ROW` is the Top-N sink: the boundary is the Kth retained row's key tuple. `GROUP_KEY` is the
 * Stage-5 aggregate producer, which follows the distinct-key discipline instead.
 */
enum class top_n_producer_kind { ROW, GROUP_KEY };

/**
 * @brief Immutable routing and placement for one Top-N producer
 *
 * Frozen at plan time; owns no boundary and no mutable revision -- those live in
 * `top_n_threshold_coordinator`, which receives this plan and is the sole policy owner of every
 * publisher inside it. Reuses @ref dynamic_filter_replica_space but not the join-specific
 * admitted-key or source-policy plan. `replica_spaces` covers every planned consumer device,
 * including endpoint devices. A `FIRST_KEY` target subsumed by a `LEX` target at the same site is
 * never planned.
 *
 * Move-only, because each target owns a move-only refinement publisher: the planner builds the
 * plan once and hands it over; consumers hold it `const`.
 */
class top_n_dynamic_filter_publish_plan final {
 public:
  /**
   * @brief One ORDER BY key, in key order
   */
  struct key {
    std::size_t child_ordinal;  ///< At the Top-N child; traces remap it per site
    top_n_key_semantics semantics;
    bool type_admitted;  ///< Per-key allowlist verdict
  };

  /**
   * @brief One channel this producer publishes into, and the layer it receives
   *
   * `component_ordinals` are this site's consumer ordinals, primary first, and match the slot's
   * declared (primary, referenced) ordinals. They are per target because each site remaps the
   * keys through its own trace: a filter carries the ordinals it references, so two sites with
   * different mappings need separately constructed filters even under one revision.
   */
  struct target {
    dynamic_filter_refinement_publisher publisher;  ///< Bound to its (primary) consumer ordinal
    top_n_filter_layer layer;
    std::vector<std::size_t> component_ordinals;
  };

  top_n_dynamic_filter_publish_plan() = default;

  top_n_dynamic_filter_publish_plan(top_n_dynamic_filter_publish_plan&&) noexcept = default;
  top_n_dynamic_filter_publish_plan& operator=(top_n_dynamic_filter_publish_plan&&) noexcept =
    default;
  top_n_dynamic_filter_publish_plan(top_n_dynamic_filter_publish_plan const&)            = delete;
  top_n_dynamic_filter_publish_plan& operator=(top_n_dynamic_filter_publish_plan const&) = delete;

  std::size_t k            = 0;  ///< Checked limit + offset
  top_n_producer_kind kind = top_n_producer_kind::ROW;
  std::vector<key> keys;        ///< Complete ORDER BY, in order; keys[0] feeds FIRST_KEY
  std::vector<target> targets;  ///< Scan channels and endpoint channels, uniformly
  std::vector<dynamic_filter_replica_space> replica_spaces;
};

}  // namespace sirius::op
