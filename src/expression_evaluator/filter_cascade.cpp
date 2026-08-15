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

// Cheap-conjunct filter cascade policy for sirius::expression_evaluator::select().
//
// This translation unit holds the policy of the cascade — the conjunct classifier and the
// decision procedure in try_select_cascade — following the house pattern of defining one class's
// concern-specific members across dedicated files (see specializations/*.cpp). All mechanism
// (compute_mask, cudf::apply_boolean_mask, cudf::bools_to_mask, NULL_LOGICAL_AND) is the
// existing evaluator/cuDF machinery, reused unmodified; expression_evaluator.cpp keeps the
// baseline select path.
//
// Correctness rests on Kleene partition invariance: for a top-level AND with children C and any
// partition C = A + B, AND(C) is TRUE for a row iff AND(A) and AND(B) are both TRUE, because
// Kleene AND is associative/commutative and a Kleene conjunction is TRUE iff all operands are
// TRUE (a NULL or FALSE anywhere makes it non-TRUE). cudf::apply_boolean_mask keeps exactly the
// valid-and-TRUE rows and NULL_LOGICAL_AND is Kleene AND (matching the lowering in
// specializations/conjunction.cpp), so every route below — cascaded, combined_masks,
// short_circuited, and the caller's monolithic fallback — produces the identical row set. In
// particular, a row where the cheap group evaluates to NULL is correctly dropped before the
// residual ever runs (TRUE AND NULL != TRUE), and a row where the cheap group is TRUE but the
// residual is NULL is dropped by the residual stage; the cascade therefore needs no
// null-handling code of its own.

// sirius
#include <config.hpp>
#include <expression/ast/node.hpp>   // sirius::ast::node alternatives
#include <expression/ast/utils.hpp>  // sirius::ast::clone, sirius::ast::visit_references
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/filter_cascade_internal.hpp>
#include <log/logging.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>  // cudf::empty_like
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>  // cudf::bools_to_mask
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>  // cudf::is_fixed_width

// standard library
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::detail {

// Contract and the deliberate non-exhaustiveness of the visitor's default arm are documented on
// the declaration in filter_cascade_internal.hpp.
bool is_cheap_prefilter_conjunct(sirius::ast::node const& n, cudf::table_view const& input)
{
  namespace ast = sirius::ast;
  return std::visit(
    [&](auto const& alt) -> bool {
      using T = std::decay_t<decltype(alt)>;
      if constexpr (std::is_same_v<T, ast::reference>) {
        if (alt.column_index >= static_cast<uint32_t>(input.num_columns())) { return false; }
        return cudf::is_fixed_width(
          input.column(static_cast<cudf::size_type>(alt.column_index)).type());
      } else if constexpr (std::is_same_v<T, ast::constant>) {
        // Literals are free; only string payloads (which can only pair with string-carried
        // operands) mark a conjunct expensive.
        return !alt.return_type().is_varchar();
      } else if constexpr (std::is_same_v<T, ast::comparison>) {
        return alt.left && alt.right && is_cheap_prefilter_conjunct(*alt.left, input) &&
               is_cheap_prefilter_conjunct(*alt.right, input);
      } else if constexpr (std::is_same_v<T, ast::between>) {
        return alt.input && alt.lower && alt.upper &&
               is_cheap_prefilter_conjunct(*alt.input, input) &&
               is_cheap_prefilter_conjunct(*alt.lower, input) &&
               is_cheap_prefilter_conjunct(*alt.upper, input);
      } else if constexpr (std::is_same_v<T, ast::conjunction>) {
        // A nested AND or OR of fixed-width comparisons is still elementwise.
        return !alt.children.empty() &&
               std::all_of(alt.children.begin(), alt.children.end(), [&](auto const& child) {
                 return child && is_cheap_prefilter_conjunct(*child, input);
               });
      } else if constexpr (std::is_same_v<T, ast::in_list>) {
        return alt.probe && is_cheap_prefilter_conjunct(*alt.probe, input) &&
               std::all_of(alt.values.begin(), alt.values.end(), [&](auto const& value) {
                 return value && is_cheap_prefilter_conjunct(*value, input);
               });
      } else if constexpr (std::is_same_v<T, ast::unary_op>) {
        switch (alt.op) {
          case ast::unary_op::kind::op_not:
          case ast::unary_op::kind::op_is_null:
          case ast::unary_op::kind::op_is_not_null:
            return alt.child && is_cheap_prefilter_conjunct(*alt.child, input);
          default: return false;
        }
      } else {
        // cast, case_expr, coalesce, function_call, aggregate, and any future alternative.
        return false;
      }
    },
    n.v);
}

}  // namespace sirius::detail

namespace sirius {

// Sub-evaluators produce, per row, the same value a monolithic evaluation would: every
// specialization's lowering decisions (restored-reference casts, narrow-domain comparisons)
// depend only on the conjunct's own operands and carrier types, never on sibling conjuncts.
// Future specializations must preserve this context-free property; the byte-equivalence tests in
// test_filter_cascade.cpp are its regression net.
std::optional<std::unique_ptr<cudf::table>> expression_evaluator::try_select_cascade(
  cudf::table_view input, std::span<cudf::size_type const> output_indices)
{
  namespace ast = sirius::ast;
  // One coherent knob snapshot per call: the SET handlers write these globals from other
  // connections, and the branch taken must match what any log line reports.
  if (!duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS) { return std::nullopt; }
  auto const min_rows      = duckdb::Config::FILTER_CASCADE_MIN_ROWS;
  auto const max_pass_rate = duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE;
  auto const num_rows      = input.num_rows();
  if (num_rows <= 0 || static_cast<std::uint64_t>(num_rows) < min_rows) { return std::nullopt; }
  if (_ast_expressions.size() != 1 || _ast_expressions[0] == nullptr) { return std::nullopt; }
  auto const& root = *_ast_expressions[0];
  if (!root.holds<ast::conjunction>()) { return std::nullopt; }
  auto const& conj = root.get<ast::conjunction>();
  if (conj.op != ast::conjunction::kind::op_and) { return std::nullopt; }

  std::vector<ast::node const*> cheap;
  std::vector<ast::node const*> expensive;
  for (auto const& child : conj.children) {
    if (!child) { return std::nullopt; }  // malformed AST: let the ordinary path report it
    (detail::is_cheap_prefilter_conjunct(*child, input) ? cheap : expensive).push_back(child.get());
  }
  if (cheap.empty() || expensive.empty()) { return std::nullopt; }

  // Gather-scope guard: the cascaded branch gathers `input` wholesale, which is acceptable only
  // when that is shape-equivalent to what the monolithic path materializes anyway. Under a
  // projection that drops a column the expensive residual never reads, the gather would
  // materialize up to pass_rate x (its bytes) of pure peak-memory waste on memory-tight scans —
  // and a cascade that may never gather has negative expected value versus the single monolithic
  // kernel (extra launches plus a host sync for no possible win), so refuse outright rather than
  // salvage. The needed set is output_indices union referenced(expensive residual); it
  // deliberately excludes cheap-only references — a column read only by the cheap group is dead
  // after the prefilter, so gathering it is waste too.
  if (!output_indices.empty()) {
    std::vector<bool> needed(static_cast<std::size_t>(input.num_columns()), false);
    for (auto const output_index : output_indices) {
      if (output_index >= 0 && output_index < input.num_columns()) {
        needed[static_cast<std::size_t>(output_index)] = true;
      }
    }
    for (auto const* conjunct : expensive) {
      ast::visit_references(*conjunct, [&](ast::reference const& ref) {
        if (ref.column_index < static_cast<uint32_t>(input.num_columns())) {
          needed[ref.column_index] = true;
        }
      });
    }
    if (!std::all_of(needed.begin(), needed.end(), [](bool used) { return used; })) {
      return std::nullopt;
    }
  }

  // From here the cascade is committed: once the cheap mask and its survivor count exist, every
  // outcome below is cheaper than restarting on the monolithic path.

  // A single conjunct is borrowed in place (owned by the operator's expression, alive across
  // this call); a multi-conjunct group needs an owning AND wrapper, so its members are
  // deep-cloned (predicate trees are tiny — a few host allocations against a multi-million-row
  // kernel).
  auto make_group = [](std::vector<ast::node const*> const& parts)
    -> std::pair<ast::node const*, std::unique_ptr<ast::node>> {
    if (parts.size() == 1) { return {parts.front(), nullptr}; }
    ast::conjunction group;
    group.op = ast::conjunction::kind::op_and;
    group.children.reserve(parts.size());
    for (auto const* part : parts) {
      group.children.push_back(ast::clone(*part));
    }
    auto owned      = std::make_unique<ast::node>(std::move(group));
    auto const* ptr = owned.get();
    return {ptr, std::move(owned)};
  };

  auto const [cheap_root, cheap_owned] = make_group(cheap);
  expression_evaluator cheap_evaluator(cheap_root, _mr, _stream, _strategy, _min_ast_size);
  auto cheap_mask = cheap_evaluator.compute_mask(input);

  // Survivor count: bools_to_mask reports how many entries are false-or-null, which is exactly
  // the set apply_boolean_mask would drop, so passed = num_rows - dropped. The transient bitmask
  // (num_rows/8 bytes) is discarded immediately. The count is the cascade's one host-blocking
  // 4-byte device-to-host sync — the price of the adaptive decision.
  cudf::size_type dropped = 0;
  {
    auto const mask_and_count = cudf::bools_to_mask(cheap_mask->view(), _stream, _mr);
    dropped                   = mask_and_count.second;
  }
  auto const passed    = num_rows - dropped;
  auto const pass_rate = static_cast<double>(passed) / static_cast<double>(num_rows);

  // The engaged path is the hottest filter route in the engine, so per-batch detail stays at
  // DEBUG; one INFO line per process on first engagement keeps activation evidence in every
  // run's log without per-batch volume.
  static std::atomic<bool> activation_logged{false};
  if (!activation_logged.exchange(true, std::memory_order_relaxed)) {
    SIRIUS_LOG_INFO(
      "[expression_evaluator] filter cascade engaged (first activation in this process): "
      "{} of {} rows (rate {:.3f}) pass the cheap prefilter ({} cheap / {} expensive "
      "conjuncts); per-batch decisions log at debug level",
      passed,
      num_rows,
      pass_rate,
      cheap.size(),
      expensive.size());
  }

  auto project = [&](cudf::table_view t) {
    return output_indices.empty() ? t : t.select(output_indices.begin(), output_indices.end());
  };

  if (passed == 0) {
    // Nothing survives the cheap prefilter, so skipping the residual is legal (a non-TRUE cheap
    // group already makes the whole AND non-TRUE). cudf::empty_like matches the zero-row
    // structure apply_boolean_mask itself returns for empty inputs.
    _last_filter_cascade_decision = filter_cascade_decision::short_circuited;
    SIRIUS_LOG_DEBUG(
      "[expression_evaluator] filter cascade: cheap prefilter dropped all {} rows "
      "({} cheap / {} expensive conjuncts)",
      num_rows,
      cheap.size(),
      expensive.size());
    return cudf::empty_like(project(input));
  }

  auto const [residual_root, residual_owned] = make_group(expensive);
  expression_evaluator residual_evaluator(residual_root, _mr, _stream, _strategy, _min_ast_size);

  if (pass_rate <= max_pass_rate) {
    // Selective prefilter: compact once, then run the expensive residual only on survivors.
    // apply_boolean_mask is a stable compaction and two stacked stable compactions compose to a
    // stable compaction, so output order equals input order equals the monolithic path's order.
    _last_filter_cascade_decision = filter_cascade_decision::cascaded;
    SIRIUS_LOG_DEBUG(
      "[expression_evaluator] filter cascade: {} of {} rows (rate {:.3f}) pass the cheap "
      "prefilter ({} cheap / {} expensive conjuncts); evaluating the residual on survivors",
      passed,
      num_rows,
      pass_rate,
      cheap.size(),
      expensive.size());
    auto const survivors = cudf::apply_boolean_mask(input, cheap_mask->view(), _stream, _mr);
    cheap_mask.reset();
    auto const residual_mask = residual_evaluator.compute_mask(survivors->view());
    return cudf::apply_boolean_mask(
      project(survivors->view()), residual_mask->view(), _stream, _mr);
  }

  // The prefilter is too unselective for the gather to pay for itself. The cheap mask is a sunk
  // cost at this point, so ANDing it with the residual mask (Kleene, matching the conjunction
  // lowering) is strictly cheaper than recomputing the monolithic predicate. Same row set either
  // way.
  _last_filter_cascade_decision = filter_cascade_decision::combined_masks;
  SIRIUS_LOG_DEBUG(
    "[expression_evaluator] filter cascade: {} of {} rows (rate {:.3f}) pass the cheap "
    "prefilter ({} cheap / {} expensive conjuncts); pass rate above "
    "filter_cascade_max_pass_rate, combining masks without a gather",
    passed,
    num_rows,
    pass_rate,
    cheap.size(),
    expensive.size());
  auto const residual_mask = residual_evaluator.compute_mask(input);
  auto const combined      = cudf::binary_operation(cheap_mask->view(),
                                               residual_mask->view(),
                                               cudf::binary_operator::NULL_LOGICAL_AND,
                                               cudf::data_type{cudf::type_id::BOOL8},
                                               _stream,
                                               _mr);
  return cudf::apply_boolean_mask(project(input), combined->view(), _stream, _mr);
}

}  // namespace sirius
