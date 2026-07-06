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

// Single translation unit for sirius::ast::node member definitions. Keep every
// node:: method here so they do not spread across per-feature .cpp files. The
// per-alternative method bodies (e.g. reference::cudf_ast_op_count) live with
// their own concerns; only node:: members belong in this file.

#include "expression/ast/node.hpp"

#include "sirius/exception.hpp"

#include <cstddef>
#include <string_view>
#include <variant>

namespace sirius::ast {

namespace {

// Boundary helper for the require_* accessors below, which run during GPU operator
// construction (under create_plan). A null or wrong-type node means the GPU path cannot
// handle this expression shape — e.g. wrap() returned null for an unsupported aggregate, or
// a group key that is not a bound reference. not_implemented_exception is deliberate: it is
// caught by OnFinalizePrepare's `catch (std::exception&)` and triggers graceful CPU fallback
// rather than signalling an internal Sirius invariant violation.
[[noreturn]] void throw_wrong_node_type(std::string_view context, std::string_view expected)
{
  throw not_implemented_exception("{}: expected {}", context, expected);
}

}  // namespace

bool node::is_reference() const noexcept { return holds<reference>(); }

bool node::is_aggregate() const noexcept { return holds<aggregate>(); }

bool node::is_function_call() const noexcept { return holds<function_call>(); }

reference const& node::as_reference() const
{
  if (!is_reference()) { throw std::bad_variant_access(); }
  return get<reference>();
}

aggregate const& node::as_aggregate() const
{
  if (!is_aggregate()) { throw std::bad_variant_access(); }
  return get<aggregate>();
}

function_call const& node::as_function_call() const
{
  if (!is_function_call()) { throw std::bad_variant_access(); }
  return get<function_call>();
}

std::size_t node::cudf_ast_op_count() const
{
  return std::visit([](auto const& alt) { return alt.cudf_ast_op_count(); }, v);
}

logical_type node::return_type() const
{
  return std::visit([](auto const& alt) -> logical_type { return alt.return_type(); }, v);
}

reference const& require_reference(node const* n, std::string_view context)
{
  if (n == nullptr) { throw_wrong_node_type(context, "non-null reference node"); }
  try {
    return n->as_reference();
  } catch (std::bad_variant_access const&) {
    throw_wrong_node_type(context, "reference node");
  }
}

aggregate const& require_aggregate(node const* n, std::string_view context)
{
  if (n == nullptr) { throw_wrong_node_type(context, "non-null aggregate node"); }
  try {
    return n->as_aggregate();
  } catch (std::bad_variant_access const&) {
    throw_wrong_node_type(context, "aggregate node");
  }
}

}  // namespace sirius::ast
