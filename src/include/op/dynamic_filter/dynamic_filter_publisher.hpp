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

/**
 * @brief What one publication attempt did
 *
 * The publisher only returns these counts; the calling join folds them into its optional
 * `dynamic_filter_stats` sink.
 */
struct dynamic_filter_publication_outcome {
  std::size_t keys_considered            = 0;  ///< Bound admitted keys the attempt walked
  std::size_t keys_with_known_domain     = 0;  ///< Keys whose domain cardinality was nonzero
  std::size_t keys_build_exceeded_domain = 0;  ///< Build rows exceeded the domain bound
  std::size_t skipped_targets_drained    = 0;  ///< 1 when the attempt hit the all-drained return
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
 * Constructs filters only for admitted keys with at least one binding, completes device
 * replication before publishing, and pushes each filter at its binding's channel push ordinal.
 * Retains none of its inputs. A key whose recorded storage type disagrees with its runtime build
 * column is skipped and counted. The caller owns source readiness and exactly-once arbitration.
 *
 * @pre @p plan is enabled
 * @throw std::runtime_error if the source GPU cannot be identified
 * @throw std::logic_error if an admitted key's build ordinal lies outside `build_view`, if the
 * source GPU is absent from the plan's replica spaces, or if a constructed filter does not
 * implement `sirius_device_replicable`
 *
 * @param[in] plan The join's enabled publication plan
 * @param[in] build_view The complete build table; admitted keys' build ordinals index its columns
 * @param[in] stream Stream used for filter construction
 * @return Counts describing what the attempt constructed, skipped, and pushed
 */
[[nodiscard]] dynamic_filter_publication_outcome publish_dynamic_filters(
  dynamic_filter_publish_plan const& plan,
  cudf::table_view const& build_view,
  rmm::cuda_stream_view stream);

}  // namespace sirius::op
