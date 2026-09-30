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

// sirius
#include "expression/ast/constant.hpp"
#include "helper/numeric_narrowing.hpp"

// cudf
#include <cudf/types.hpp>

// standard library
#include <initializer_list>
#include <optional>

namespace sirius::ast {

struct node;

/**
 * @file
 * @brief Numeric-range views over `sirius::ast::constant` literals.
 *
 * `helper/numeric_narrowing` owns the carrier-fitting machinery (`sirius::numeric_range`,
 * `numeric_range_fits`) without any AST dependency; this header adapts AST constant nodes to it.
 * `narrow_domain_carrier_eligible` (a carried column against literals) and
 * `narrow_domain_reference_pair_eligible` (a carried column against another carried column) are the
 * complete narrow-domain eligibility predicates, shared by the expression evaluator's
 * `narrow_domain_carrier` and `narrow_domain_reference_pair_carrier` checks and the planner's
 * `apply_tier_narrowing_policy` classifier, so plan-time and evaluation-time decisions about the
 * same comparison shape cannot drift; only the reference lookup and the carrier retrieval (batch
 * column vs. planned target) remain caller-local.
 */

/**
 * @brief Host-exact value of a numeric constant as a degenerate [v, v] range.
 *
 * Returns `std::nullopt` when the constant is not a supported narrowable-domain literal or its
 * payload alternative disagrees with its declared logical type. Typed NULLs return `std::nullopt`
 * here; `constant_representable_in_carrier` treats them as representable in any carrier because
 * only their validity bit is materialized.
 *
 * @param expr Constant to inspect.
 * @return Exact single-value range, or `std::nullopt` when no supported range is available.
 */
[[nodiscard]] std::optional<sirius::numeric_range> constant_numeric_range(constant const& expr);

/**
 * @brief Whether @p expr can be materialized in @p carrier without changing its value.
 *
 * A typed NULL always fits. Any other literal fits when it is a supported numeric literal whose
 * exact value passes `numeric_range_fits` against @p carrier (same numeric family, and for
 * decimals the same scale); unsupported literal kinds are never representable.
 *
 * @param carrier Candidate physical carrier.
 * @param expr Constant to inspect.
 * @return `true` when materializing @p expr in @p carrier preserves its value.
 */
[[nodiscard]] bool constant_representable_in_carrier(cudf::data_type carrier, constant const& expr);

/**
 * @brief Whether a comparison over a column of declared type @p logical carried as @p carrier may
 * evaluate directly at the narrow width.
 *
 * True when @p logical is a narrowable numeric type, @p carrier is a strict same-family narrowing
 * of its native cuDF carrier, and every entry of @p constant_operands is a constant representable
 * in @p carrier. The evaluator probes this with the batch's actual carrier; the narrowing policy
 * probes it with the planned carrier.
 *
 * A literal must also come from the column's own narrowing domain (`sirius::narrow_domain_of`),
 * because epoch days and plain integers are indistinguishable once both sit in the same signed
 * carrier. A typed NULL is exempt: only its validity bit is materialized, so no width can misread
 * it.
 *
 * @param logical Declared logical type of the carried column.
 * @param carrier Materialized or planned physical carrier.
 * @param constant_operands Operands that must all be representable constants.
 * @return `true` when the comparison may execute without restoring the column.
 */
[[nodiscard]] bool narrow_domain_carrier_eligible(
  sirius::logical_type const& logical,
  cudf::data_type carrier,
  std::initializer_list<node const*> constant_operands);

/**
 * @brief Whether a comparison between two carried references may evaluate at the narrow width.
 *
 * True when both sides are strict narrowings of their own native carrier and both landed on the
 * *same* physical carrier. Narrowing is value-preserving and applies no bias or offset, so an
 * identical-carrier pair orders exactly as the native pair does; restoring either side would widen
 * both back and buy nothing. Carrier equality also settles signedness and decimal scale, which is
 * why neither is checked separately.
 *
 * The one thing carrier equality does not settle is domain: epoch days and plain integers both
 * land in INT16, so DATE is admitted only against DATE.
 *
 * @param lhs_logical Declared logical type of the left reference.
 * @param lhs_carrier Physical carrier of the left reference.
 * @param rhs_logical Declared logical type of the right reference.
 * @param rhs_carrier Physical carrier of the right reference.
 * @return `true` when the pair may execute without restoring either reference.
 */
[[nodiscard]] bool narrow_domain_reference_pair_eligible(sirius::logical_type const& lhs_logical,
                                                         cudf::data_type lhs_carrier,
                                                         sirius::logical_type const& rhs_logical,
                                                         cudf::data_type rhs_carrier);

}  // namespace sirius::ast
