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

// cudf
#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sirius {

/**
 * @brief SWAR fast path for multi-literal LIKE patterns of the form `%lit1%lit2%...%litN%`.
 *
 * cudf's `like_fn` is thread-per-row and byte-at-a-time; on wide comment-style columns it is
 * load-instruction-bound. For the restricted-but-common pattern shape `%lit%...%lit%` an exact
 * match can be computed by scanning each row's characters as aligned `u64` words, detecting
 * candidate positions with a SWAR digram (first-two-bytes) equality mask, and verifying the
 * full literal only at candidate hits (measured ~6x over `cudf::strings::like` on TPC-H q13's
 * `o_comment NOT LIKE '%special%requests%'`).
 *
 * The classifier is deliberately conservative: anything that is not plainly
 * `%lit1%lit2%...%litN%` with ASCII literals (no `_` wildcard, no escape character, no
 * non-ASCII bytes) is refused so the caller falls back to `cudf::strings::like`.
 */

/// Maximum number of literals the fast-path kernel supports.
inline constexpr int like_multiliteral_max_literals = 4;
/// Maximum byte length of each literal the fast-path kernel supports.
inline constexpr int like_multiliteral_max_literal_bytes = 32;

/// Parsed form of an eligible `%lit1%lit2%...%litN%` pattern.
struct like_multiliteral_pattern {
  /// The literals, in pattern order. Each is 1..max_literal_bytes ASCII bytes.
  std::vector<std::string> literals;
};

/**
 * @brief Classify a LIKE pattern for the multi-literal SWAR fast path.
 *
 * Accepts exactly the shape `%lit1%lit2%...%litN%`:
 * - the pattern starts and ends with `%` (consecutive `%` collapse, per SQL semantics);
 * - it contains no `_` wildcard;
 * - `escape` is empty (no escape semantics in play);
 * - every literal byte is ASCII in [0x01, 0x7F];
 * - 1..like_multiliteral_max_literals literals, each of at most
 *   like_multiliteral_max_literal_bytes bytes.
 *
 * @return The parsed literal sequence, or std::nullopt when the pattern must take the
 *         `cudf::strings::like` path.
 */
std::optional<like_multiliteral_pattern> classify_like_multiliteral(std::string_view pattern,
                                                                    std::string_view escape);

/**
 * @brief Evaluate a classified multi-literal LIKE pattern over a strings column.
 *
 * Produces a BOOL8 column identical to
 * `cudf::strings::like(input, pattern)` (negated when @p invert is true, i.e. NOT LIKE is
 * fused): row i is true iff the literals occur in order, non-overlapping, within row i.
 * The null mask of @p input is copied to the result; null rows carry value `invert`
 * (matching the cudf like → cudf NOT composition, where null rows carry false pre-NOT).
 *
 * @param input   Strings column to match. Rows must be valid UTF-8 (the same precondition
 *                cudf's strings APIs document); for ASCII literals, byte-level search is then
 *                exactly character-level search.
 * @param pattern A pattern accepted by classify_like_multiliteral().
 * @param invert  When true, computes NOT LIKE.
 * @param stream  CUDA stream to run on.
 * @param mr      Memory resource for the result (and nothing else; no temporaries are made).
 * @return The BOOL8 match column, or nullptr when the column layout is ineligible
 *         (chars base not 8-byte aligned or unexpected offsets type) and the caller must
 *         fall back to `cudf::strings::like`.
 */
std::unique_ptr<cudf::column> like_multiliteral(cudf::strings_column_view const& input,
                                                like_multiliteral_pattern const& pattern,
                                                bool invert,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace sirius
