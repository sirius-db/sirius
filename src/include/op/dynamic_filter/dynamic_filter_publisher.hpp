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

#pragma once

#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>

namespace sirius::op {

//===----------------------------------------------------------------------===//
// publish_dynamic_filters
//===----------------------------------------------------------------------===//

/**
 * @brief What one publication attempt did
 *
 * Purely descriptive counts, returned to the caller rather than written to a sink, so the
 * publisher stays free of any context and every unit test asserts on a value. The producing join
 * decides where the observation goes.
 */
struct dynamic_filter_publication_outcome {
  std::size_t keys_considered            = 0;  ///< Admitted keys the attempt walked
  std::size_t keys_skipped_domain_gate   = 0;  ///< Skipped: build too dense a sample of the domain
  std::size_t keys_skipped_type_mismatch = 0;  ///< Skipped: plan type disagreed with the column
  std::size_t membership_filters_built   = 0;
  std::size_t zone_map_filters_built     = 0;
  std::size_t active_targets             = 0;  ///< Targets still accepting filters
  std::size_t filters_pushed             = 0;  ///< Accepted pushes across every target
};

/**
 * @brief Build, replicate, and fan out one immutable dynamic-filter snapshot from a complete
 * hash-join build table
 *
 * The immutable @ref dynamic_filter_publish_plan is the only key/target input: filters are
 * constructed densely over its admitted keys and fanned out sparsely along each target's key
 * bindings, pushing at each binding's channel push ordinal. All inputs are read only for the
 * duration of the call; nothing is retained.
 *
 * The caller -- @ref sirius_physical_hash_join::publish_dynamic_filters -- owns source readiness
 * and the exactly-once publication arbitration and calls this at most once per query execution.
 *
 * A key whose recorded storage type disagrees with the runtime build column is skipped and
 * counted rather than published: the check proves only that plan-time and runtime type derivation
 * agree, and cannot detect the wrong-column case that would actually remove valid rows. A build
 * ordinal outside the build table is different in kind -- it proves the plan is incoherent, and
 * `cudf::table_view::column()` does not bounds-check -- so it fails the attempt.
 *
 * @throw std::logic_error if an admitted key's build ordinal lies outside `build_view`
 *
 * @param[in] plan The join's enabled publication plan (admitted keys, targets, policy, replica
 * placements)
 * @param[in] build_view The complete build table to reduce / build membership over; admitted build
 * key ordinals index its columns
 * @param[in] stream Durable build-memory-space stream used for filter construction
 * @return Counts describing what the attempt constructed, skipped, and pushed
 */
[[nodiscard]] dynamic_filter_publication_outcome publish_dynamic_filters(
  dynamic_filter_publish_plan const& plan,
  cudf::table_view const& build_view,
  rmm::cuda_stream_view stream);

}  // namespace sirius::op
