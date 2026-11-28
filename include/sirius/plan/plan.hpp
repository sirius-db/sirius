// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Context-free construction and inspection of immutable physical plans.
 */
#pragma once
#include <sirius-c/plan/plan.h>
#include <sirius/schema.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace sirius {
/// Predicate operator for Expression::compare(). Values may be copied between threads.
/// @code{.cpp}
/// auto predicate = id->compare(sirius::Comparison::greater, *ten);
/// @endcode
enum class Comparison : sirius_comparison {
  equal         = SIRIUS_COMPARISON_EQUAL,
  not_equal     = SIRIUS_COMPARISON_NOT_EQUAL,
  less          = SIRIUS_COMPARISON_LESS,
  less_equal    = SIRIUS_COMPARISON_LESS_EQUAL,
  greater       = SIRIUS_COMPARISON_GREATER,
  greater_equal = SIRIUS_COMPARISON_GREATER_EQUAL,
};
/// Operator returned by PlanNode::kind(). Values may be copied between threads.
/// @code{.cpp}
/// bool is_scan = scan->kind() == sirius::PlanNodeKind::parquet_scan;
/// @endcode
enum class PlanNodeKind : sirius_plan_node_kind {
  parquet_scan = SIRIUS_NODE_PARQUET_SCAN,
  filter       = SIRIUS_NODE_FILTER
};

/** @brief A typed expression retaining its input scope and children.
 * @par Errors
 * Construction returns ErrorCode::allocation_failure if allocation fails and
 * ErrorCode::internal_error for unexpected native failures.
 * @see @ref cpp_plan_construction "Construct a physical plan" for a complete example.
 * @par Thread safety
 * Copies share immutable storage. Concurrent const operations and copying are safe
 * while the source is alive. Assignment and destruction require exclusive access
 * to that wrapper. Copies can be transferred and released on any thread.
 */
class Expression {
 public:
  /// Share ownership of the same immutable object.
  /// @code{.cpp}
  /// sirius::Expression copy = *id;
  /// @endcode
  Expression(const Expression& other) noexcept : handle_(other.handle_)
  {
    sirius_expression_retain(handle_);
  }
  /// Replace this reference with another owning reference.
  /// @code{.cpp}
  /// copy = *id;
  /// @endcode
  Expression& operator=(const Expression& other) noexcept
  {
    sirius_expression_retain(other.handle_);
    sirius_expression_release(handle_);
    handle_ = other.handle_;
    return *this;
  }
  /// Release this owning reference.
  /// @code{.cpp}
  /// { auto copy = *id; }
  /// @endcode
  ~Expression() noexcept { sirius_expression_release(handle_); }

  /// Create a non-null signed 64-bit literal usable with any input node.
  /// @code{.cpp}
  /// auto ten = sirius::Expression::literal_int64(10);
  /// if (!ten) return std::unexpected(ten.error());
  /// @endcode
  [[nodiscard]] static std::expected<Expression, Error> literal_int64(std::int64_t value) noexcept
  {
    sirius_expression* result = nullptr;
    sirius_error* diagnostic  = nullptr;
    const auto status         = sirius_expression_literal_int64(value, &result, &diagnostic);
    if (status != SIRIUS_SUCCESS) return std::unexpected(Error::from_c(status, diagnostic));
    return Expression(result);
  }

  /// Create a non-null boolean literal usable with any input node.
  /// @code{.cpp}
  /// auto enabled = sirius::Expression::literal_boolean(true);
  /// @endcode
  [[nodiscard]] static std::expected<Expression, Error> literal_boolean(bool value) noexcept
  {
    sirius_expression* result = nullptr;
    sirius_error* diagnostic  = nullptr;
    const auto status         = sirius_expression_literal_boolean(value, &result, &diagnostic);
    if (status != SIRIUS_SUCCESS) return std::unexpected(Error::from_c(status, diagnostic));
    return Expression(result);
  }

  /// Compare two expressions of the same type. Int64 supports all comparisons.
  /// Booleans support equality and inequality. A NULL operand gives a NULL result.
  /// Column references on both sides must come from the same input node. Literals
  /// can be used on either side. An incompatible type, operator, or input node, or
  /// an expression tree deeper than 256 levels, returns ErrorCode::invalid_argument.
  /// @code{.cpp}
  /// auto predicate = id->compare(sirius::Comparison::greater, *ten);
  /// if (!predicate) return std::unexpected(predicate.error());
  /// @endcode
  [[nodiscard]] std::expected<Expression, Error> compare(Comparison op,
                                                         const Expression& right) const noexcept
  {
    sirius_expression* result = nullptr;
    sirius_error* diagnostic  = nullptr;
    const auto status         = sirius_expression_compare(
      handle_, static_cast<sirius_comparison>(op), right.handle_, &result, &diagnostic);
    if (status != SIRIUS_SUCCESS) return std::unexpected(Error::from_c(status, diagnostic));
    return Expression(result);
  }

  /// Return the result type.
  /// @code{.cpp}
  /// auto type = id->type();
  /// @endcode
  [[nodiscard]] DataType type() const noexcept
  {
    return static_cast<DataType>(sirius_expression_type(handle_));
  }
  /// Return whether the result admits nulls.
  /// @code{.cpp}
  /// auto nullable = id->nullable();
  /// @endcode
  [[nodiscard]] bool nullable() const noexcept { return sirius_expression_nullable(handle_) != 0; }

 private:
  explicit Expression(sirius_expression* owned) noexcept : handle_(owned) {}
  sirius_expression* handle_;
  friend class PlanNode;
};
/** @brief An immutable physical operator retaining its schema, input, and predicate.
 * @par Errors
 * Construction returns ErrorCode::allocation_failure if allocation fails and
 * ErrorCode::internal_error for unexpected native failures.
 * @see @ref cpp_plan_construction "Construct a physical plan" for a complete example.
 * @par Thread safety
 * Copies share immutable storage. Concurrent const operations and copying are safe
 * while the source is alive. Assignment and destruction require exclusive access
 * to that wrapper. Copies can be transferred and released on any thread.
 */
class PlanNode {
 public:
  /// Share ownership of the same immutable object.
  /// @code{.cpp}
  /// sirius::PlanNode copy = *scan;
  /// @endcode
  PlanNode(const PlanNode& other) noexcept : handle_(other.handle_)
  {
    sirius_plan_node_retain(handle_);
  }
  /// Replace this reference with another owning reference.
  /// @code{.cpp}
  /// copy = *scan;
  /// @endcode
  PlanNode& operator=(const PlanNode& other) noexcept
  {
    sirius_plan_node_retain(other.handle_);
    sirius_plan_node_release(handle_);
    handle_ = other.handle_;
    return *this;
  }
  /// Release this owning reference.
  /// @code{.cpp}
  /// { auto copy = *scan; }
  /// @endcode
  ~PlanNode() noexcept { sirius_plan_node_release(handle_); }

  /// Describe a local Parquet file using its native path bytes and an explicit schema.
  /// The caller supplies the expected columns. Construction copies the path and retains
  /// the schema. File access and schema matching are deferred until preparation.
  /// Relative paths use the working directory at preparation time. Glob characters
  /// are literal. Empty paths, embedded NUL bytes, and URI schemes return
  /// ErrorCode::invalid_argument. Use "./data:1.parquet" for a local name with a
  /// scheme-shaped prefix.
  /// @code{.cpp}
  /// auto scan = sirius::PlanNode::scan_parquet("data.parquet", *schema);
  /// if (!scan) return std::unexpected(scan.error());
  /// @endcode
  [[nodiscard]] static std::expected<PlanNode, Error> scan_parquet(std::string_view path,
                                                                   const Schema& schema) noexcept
  {
    sirius_plan_node* result = nullptr;
    sirius_error* diagnostic = nullptr;
    const auto status =
      sirius_plan_scan_parquet(path.data(), path.size(), schema.handle_, &result, &diagnostic);
    if (status != SIRIUS_SUCCESS) return std::unexpected(Error::from_c(status, diagnostic));
    return PlanNode(result);
  }

  /// Reference the column at a zero-based position in this node's output. The
  /// expression retains this node and inherits the field's type and nullability.
  /// An index at or beyond schema().size() returns ErrorCode::invalid_argument.
  /// @code{.cpp}
  /// auto id = scan->column(0);
  /// if (!id) return std::unexpected(id.error());
  /// @endcode
  [[nodiscard]] std::expected<Expression, Error> column(std::size_t index) const noexcept
  {
    sirius_expression* result = nullptr;
    sirius_error* diagnostic  = nullptr;
    const auto status         = sirius_expression_column(handle_, index, &result, &diagnostic);
    if (status != SIRIUS_SUCCESS) return std::unexpected(Error::from_c(status, diagnostic));
    return Expression(result);
  }

  /// Keep rows for which the predicate is true. False and NULL both discard a row.
  /// The returned node retains this input and its predicate and preserves the schema.
  /// Column references in the predicate must belong to this exact input node.
  /// A non-boolean predicate, a different input, or a node chain deeper than 256
  /// levels returns ErrorCode::invalid_argument.
  /// @code{.cpp}
  /// auto filtered = scan->filter(*predicate);
  /// if (!filtered) return std::unexpected(filtered.error());
  /// @endcode
  [[nodiscard]] std::expected<PlanNode, Error> filter(const Expression& predicate) const noexcept
  {
    sirius_plan_node* result = nullptr;
    sirius_error* diagnostic = nullptr;
    const auto status        = sirius_plan_filter(handle_, predicate.handle_, &result, &diagnostic);
    if (status != SIRIUS_SUCCESS) return std::unexpected(Error::from_c(status, diagnostic));
    return PlanNode(result);
  }

  /// Return the operator kind.
  /// @code{.cpp}
  /// auto kind = scan->kind();
  /// @endcode
  [[nodiscard]] PlanNodeKind kind() const noexcept
  {
    return static_cast<PlanNodeKind>(sirius_plan_node_get_kind(handle_));
  }
  /// Share ownership of the output schema.
  /// @code{.cpp}
  /// auto output_schema = filtered->schema();
  /// @endcode
  [[nodiscard]] Schema schema() const noexcept
  {
    auto* borrowed = sirius_plan_node_schema(handle_);
    sirius_schema_retain(borrowed);
    return Schema(borrowed);
  }
  /// Share ownership of the input node, or return an empty optional for a scan.
  /// @code{.cpp}
  /// auto input = filtered->input();
  /// @endcode
  [[nodiscard]] std::optional<PlanNode> input() const noexcept
  {
    auto* borrowed = sirius_plan_node_input(handle_);
    if (!borrowed) return std::nullopt;
    sirius_plan_node_retain(borrowed);
    return PlanNode(borrowed);
  }

 private:
  explicit PlanNode(sirius_plan_node* owned) noexcept : handle_(owned) {}
  sirius_plan_node* handle_;
  friend class Plan;
};
/** @brief An immutable physical plan with a single root.
 * @par Errors
 * Construction returns ErrorCode::allocation_failure if allocation fails and
 * ErrorCode::internal_error for unexpected native failures.
 * @see @ref cpp_plan_construction "Construct a physical plan" for a complete example.
 * @par Thread safety
 * Copies share immutable storage. Concurrent const operations and copying are safe
 * while the source is alive. Assignment and destruction require exclusive access
 * to that wrapper. Copies can be transferred and released on any thread.
 */
class Plan {
 public:
  /// Share ownership of the same immutable object.
  /// @code{.cpp}
  /// sirius::Plan copy = *plan;
  /// @endcode
  Plan(const Plan& other) noexcept : handle_(other.handle_) { sirius_plan_retain(handle_); }
  /// Replace this reference with another owning reference.
  /// @code{.cpp}
  /// copy = *plan;
  /// @endcode
  Plan& operator=(const Plan& other) noexcept
  {
    sirius_plan_retain(other.handle_);
    sirius_plan_release(handle_);
    handle_ = other.handle_;
    return *this;
  }
  /// Release this owning reference.
  /// @code{.cpp}
  /// { auto copy = *plan; }
  /// @endcode
  ~Plan() noexcept { sirius_plan_release(handle_); }

  /// Finish a plan and retain its root and all reachable nodes. The caller may
  /// release intermediate handles once construction succeeds.
  /// @code{.cpp}
  /// auto plan = sirius::Plan::from_root(*filtered);
  /// if (!plan) return std::unexpected(plan.error());
  /// @endcode
  [[nodiscard]] static std::expected<Plan, Error> from_root(const PlanNode& root) noexcept
  {
    sirius_plan* result      = nullptr;
    sirius_error* diagnostic = nullptr;
    const auto status        = sirius_plan_from_root(root.handle_, &result, &diagnostic);
    if (status != SIRIUS_SUCCESS) return std::unexpected(Error::from_c(status, diagnostic));
    return Plan(result);
  }

  /// Share ownership of the root operator.
  /// @code{.cpp}
  /// auto root = plan->root();
  /// @endcode
  [[nodiscard]] PlanNode root() const noexcept
  {
    auto* borrowed = sirius_plan_root(handle_);
    sirius_plan_node_retain(borrowed);
    return PlanNode(borrowed);
  }
  /// Return an owning copy of the human-readable plan description. Its format is intended
  /// for diagnostics. The returned text can be used after the plan is released.
  /// Returns ErrorCode::allocation_failure if the description cannot fit in the string.
  /// @code{.cpp}
  /// auto description = plan->describe();
  /// if (description) { std::cout << *description; }
  /// @endcode
  [[nodiscard]] std::expected<std::string, Error> describe() const noexcept
  {
    try {
      return std::string(sirius_plan_describe(handle_));
    } catch (const std::bad_alloc&) {
      return std::unexpected(Error{ErrorCode::allocation_failure, {}});
    } catch (const std::length_error&) {
      return std::unexpected(Error{ErrorCode::allocation_failure, {}});
    }
  }

 private:
  explicit Plan(sirius_plan* owned) noexcept : handle_(owned) {}
  sirius_plan* handle_;
};
}  // namespace sirius
