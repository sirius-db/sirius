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
 * @file prepared_build_keys.hpp
 * @brief Prepares build keys for the membership filters: the valid rows of a build column,
 * converted to their key domain's rep.
 *
 * A prepared column has the type `membership_rep_type(domain)` and no nulls.
 * `sirius_dynamic_small_in_list_filter`, `sirius_dynamic_in_list_filter`, and
 * `sirius_dynamic_bloom_filter` each construct from a `(membership_key_domain, prepared column)`
 * pair, and the IN-list column constructors are `prepare_build_column` followed by that
 * construction. Preparation enqueues its work on the given stream and never waits on the host, so
 * batches of one build can be prepared into slices of one buffer as they arrive.
 */

#pragma once

#include "op/dynamic_filter/dynamic_filter_key_domain.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <string_view>

namespace sirius::op {

/**
 * @brief Writes the valid rows of @p keys, in row order and converted to @p domain's rep, into @p
 * out
 *
 * Each element is converted from its carrier to the rep on the device (`with_build_key_iterator`);
 * STRING keys become the 64-bit fingerprints the filters' probes compute. A valid row's destination
 * is its rank among the valid rows. A row whose rank is at or beyond `out.size()` is not written;
 * instead the word at @p overflow is set to a nonzero value, so a caller that sized @p out from
 * metadata detects a disagreement after its next synchronization without any neighbor being
 * overwritten.
 *
 * @pre @p keys is a carrier of @p domain whose values fit the rep (a DECIMAL128 column has passed
 * `membership_build_fits_rep`), and its storage stays valid until work enqueued on @p stream
 * completes
 * @throw std::invalid_argument if @p out is not a non-nullable column of
 * `membership_rep_type(domain)`
 * @throw std::logic_error if @p keys is not a carrier of @p domain
 * @param overflow Device word set when a valid row does not fit, or null when @p out is sized
 * exactly from @p keys
 * @param scratch Resource for temporaries, which free stream-ordered on @p stream
 */
void prepare_build_keys(membership_key_domain const& domain,
                        cudf::column_view const& keys,
                        cudf::mutable_column_view out,
                        std::uint32_t* overflow,
                        ::cuda::stream_ref stream,
                        rmm::device_async_resource_ref scratch);

/**
 * @brief A build column prepared for the membership filters
 */
struct prepared_build_column {
  membership_key_domain domain;        ///< The column's key domain
  std::unique_ptr<cudf::column> keys;  ///< Its valid rows at the rep, freed stream-ordered
};

/**
 * @brief Classifies @p keys, checks that its values fit the rep, and prepares its `size() -
 * null_count()` valid rows into a new column
 *
 * Only the DECIMAL128 fit check waits on the host; preparation itself does not.
 *
 * @param filter_name Bracketed filter name used as the exception message prefix
 * @throw std::invalid_argument if @p keys is unsupported or its values do not fit the key rep
 */
[[nodiscard]] prepared_build_column prepare_build_column(std::string_view filter_name,
                                                         cudf::column_view const& keys,
                                                         ::cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr);

}  // namespace sirius::op
