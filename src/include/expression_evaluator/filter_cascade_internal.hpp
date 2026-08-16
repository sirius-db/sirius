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

// Internal declaration shared between filter_cascade.cpp and its unit tests, following the
// gpu_expression_translator_internal.hpp precedent: the classifier is an implementation detail
// of sirius::expression_evaluator::try_select_cascade, exposed here so tests can exercise each
// classification arm directly.

// sirius
#include <expression/ast/node.hpp>  // sirius::ast::node

// cudf
#include <cudf/table/table_view.hpp>

namespace sirius::detail {

/**
 * @brief Classifies one conjunct of a filter's top-level AND as cheap-prefilter material for the
 * filter cascade in sirius::expression_evaluator::select()
 *
 * A conjunct is cheap iff every node in its subtree is an elementwise, fixed-width-carried
 * operation: a reference whose column index is in bounds for @p input and whose *runtime carrier*
 * satisfies cudf::is_fixed_width, a non-VARCHAR constant, or a comparison / BETWEEN / IN-list /
 * nested conjunction (AND or OR) / NOT / IS NULL / IS NOT NULL over cheap operands. Everything
 * else — string-carried references, string literals, CAST, CASE, COALESCE, TRY, function calls,
 * aggregates — is expensive: real per-row cost and/or an AST-breaking lowering.
 *
 * The runtime carrier decides because cost lives in the materialized column, not the declared
 * type: under compressed materialization a reference's declared type and physical carrier differ,
 * and decode-time predicate substitution turns a string filter into a BOOL8 mask reference —
 * fixed-width, hence cheap. cudf::is_fixed_width is false for exactly the expensive carriers
 * (STRING, LIST, STRUCT, DICTIONARY32, EMPTY), so the single cuDF trait is the whole type policy.
 *
 * Misclassification cannot change results: a conjunction selects the same rows under any
 * partition of its children (Kleene AND is associative/commutative and cudf::apply_boolean_mask
 * keeps only valid-and-TRUE rows), so classification only moves cost between the prefilter and
 * the residual. The safe default direction is *expensive* — a conjunct wrongly classified
 * expensive merely fails to prefilter, while one wrongly classified cheap drags real per-row cost
 * into the stage that runs on all rows. The visitor's default arm therefore returns false, and —
 * unlike ast::clone / ast::visit_references, which static_assert exhaustiveness because omission
 * there is a correctness bug — this classifier deliberately does not force handling of new
 * ast::node alternatives: they land in the residual automatically. Conservative arms (e.g. a
 * fixed-width cast is elementwise-cheap in principle) are only widened with measured evidence.
 *
 * @param n     Root of the conjunct's AST subtree
 * @param input Table the filter runs over; supplies each ast::reference's bounds check and
 *              runtime carrier type
 * @return true when the conjunct belongs in the cheap prefilter group, false when it belongs in
 * the expensive residual
 */
[[nodiscard]] bool is_cheap_prefilter_conjunct(sirius::ast::node const& n,
                                               cudf::table_view const& input);

}  // namespace sirius::detail
