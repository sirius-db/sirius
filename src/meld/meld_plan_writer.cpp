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

#include "meld/meld_plan_writer.hpp"

#include <llvm/Support/raw_ostream.h>
#include <meld/Dialect/Meld/IR/MeldOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>

#include <cstdint>
#include <format>
#include <map>
#include <type_traits>
#include <variant>

namespace sirius::meld {

namespace {

/// The Substrait version a plan is written against.
constexpr std::uint32_t substrait_major_version = 0;
constexpr std::uint32_t substrait_minor_version = 99;
constexpr std::uint32_t substrait_patch_version = 0;

constexpr char const* comparison_urn =
  "https://github.com/substrait-io/substrait/blob/main/extensions/functions_comparison.yaml";
constexpr char const* boolean_urn =
  "https://github.com/substrait-io/substrait/blob/main/extensions/functions_boolean.yaml";
constexpr char const* arithmetic_urn =
  "https://github.com/substrait-io/substrait/blob/main/extensions/functions_arithmetic.yaml";
constexpr char const* aggregate_urn =
  "https://github.com/substrait-io/substrait/blob/main/extensions/functions_aggregate_generic.yaml";

struct refusal {
  std::string why;
};

template <typename... Args>
[[noreturn]] void refuse(std::format_string<Args...> format, Args&&... args)
{
  throw refusal{std::format(format, std::forward<Args>(args)...)};
}

/// A type as a Substrait function key spells it.
std::string key_type_of(sirius::logical_type const& type)
{
  switch (type.id()) {
    case type_id::BOOLEAN: return "bool";
    case type_id::TINYINT: return "i8";
    case type_id::SMALLINT: return "i16";
    case type_id::INTEGER: return "i32";
    case type_id::BIGINT: return "i64";
    case type_id::FLOAT: return "fp32";
    case type_id::DOUBLE: return "fp64";
    case type_id::DATE: return "date";
    default: refuse("Meld does not support {}", type.to_string());
  }
}

bool is_float(sirius::logical_type const& type)
{
  return type.id() == type_id::FLOAT || type.id() == type_id::DOUBLE;
}

/// A value in a region, and the Sirius type of it.
struct value {
  mlir::Value ir;
  sirius::logical_type type;
  /// Whether it is known not to be a NaN, as a literal of a number is.
  bool never_nan = false;

  [[nodiscard]] bool nullable() const { return mlir::isa<mlir::meld::NullableType>(ir.getType()); }
};

/// A relation, and the Sirius type of each of its fields.
struct relation {
  mlir::Value ir;
  std::vector<sirius::logical_type> types;
};

std::vector<mlir::Value> mlir_values_of(std::vector<value> const& values)
{
  std::vector<mlir::Value> mlir_values;
  for (auto const& v : values) {
    mlir_values.push_back(v.ir);
  }
  return mlir_values;
}

std::vector<sirius::logical_type> types_of(std::vector<value> const& values)
{
  std::vector<sirius::logical_type> types;
  for (auto const& v : values) {
    types.push_back(v.type);
  }
  return types;
}

class plan_builder {
 public:
  plan_builder() { _context.loadDialect<mlir::meld::MeldDialect>(); }

  meld_plan build(meld_chain const& chain)
  {
    if (chain.aggregates.empty()) { refuse("Meld does not support an aggregate of no measures"); }
    meld_plan result;
    mlir::OwningOpRef<mlir::meld::PlanOp> plan = mlir::meld::PlanOp::create(
      _builder, _loc, substrait_major_version, substrait_minor_version, substrait_patch_version);
    _builder.createBlock(&plan->getBody());
    _root = mlir::meld::RootOp::create(_builder, _loc, _builder.getArrayAttr({}));
    _builder.createBlock(&_root.getBody());

    std::vector<mlir::Type> field_types;
    for (std::size_t i = 0; i < chain.input_types.size(); ++i) {
      if (chain.input_types[i].id() == type_id::BOOLEAN) {
        refuse("Meld does not support reading a BOOLEAN column of a batch");
      }
      field_types.push_back(meld_type(chain.input_types[i], true));
      result.column_names.push_back(std::format("c{}", i));
    }
    relation table{
      mlir::meld::NamedTableOp::create(_builder,
                                       _loc,
                                       mlir::meld::RelationType::get(&_context, field_types),
                                       strings({"sirius", "batch"}),
                                       strings(result.column_names)),
      chain.input_types};

    for (auto const& step : chain.steps) {
      if (auto const* filter = std::get_if<meld_chain::filter>(&step)) {
        table = build_filter(*filter->condition, table);
      } else {
        table = build_project(std::get<meld_chain::projection>(step).expressions, table);
      }
    }
    table = build_aggregate(chain.aggregates, table);
    mlir::meld::YieldOp::create(_builder, _loc, table.ir);

    std::vector<std::string> root_names;
    for (std::size_t i = 0; i < table.types.size(); ++i) {
      root_names.push_back(std::format("o{}", i));
    }
    _root.setNamesAttr(strings(root_names));

    std::string diagnostics;
    mlir::ScopedDiagnosticHandler handler(&_context, [&](mlir::Diagnostic& diagnostic) {
      diagnostics += (diagnostics.empty() ? "" : "\n") + diagnostic.str();
      return mlir::success();
    });
    if (mlir::failed(mlir::verify(*plan))) {
      refuse("Meld does not support the plan built for it, which is not valid: {}", diagnostics);
    }
    llvm::raw_string_ostream stream(result.text);
    plan->print(stream, mlir::OpPrintingFlags().assumeVerified());
    return result;
  }

 private:
  mlir::ArrayAttr strings(std::vector<std::string> const& parts)
  {
    std::vector<llvm::StringRef> refs(parts.begin(), parts.end());
    return _builder.getStrArrayAttr(refs);
  }

  mlir::Type meld_type(sirius::logical_type const& type, bool nullable)
  {
    mlir::Type scalar;
    switch (type.id()) {
      case type_id::BOOLEAN: scalar = _builder.getI1Type(); break;
      case type_id::TINYINT: scalar = _builder.getI8Type(); break;
      case type_id::SMALLINT: scalar = _builder.getI16Type(); break;
      case type_id::INTEGER: scalar = _builder.getI32Type(); break;
      case type_id::BIGINT: scalar = _builder.getI64Type(); break;
      case type_id::FLOAT: scalar = _builder.getF32Type(); break;
      case type_id::DOUBLE: scalar = _builder.getF64Type(); break;
      case type_id::DATE: scalar = mlir::meld::DateType::get(&_context); break;
      default: refuse("Meld does not support {}", type.to_string());
    }
    return nullable ? mlir::meld::NullableType::get(&_context, scalar) : scalar;
  }

  mlir::meld::RelationType relation_type(std::vector<value> const& fields)
  {
    std::vector<mlir::Type> types;
    for (auto const& field : fields) {
      types.push_back(field.ir.getType());
    }
    return mlir::meld::RelationType::get(&_context, types);
  }

  /// Opens a block of @p region with a row of @p input for arguments, which references then read.
  void open_region(mlir::Region& region, relation const& input)
  {
    auto fields = mlir::cast<mlir::meld::RelationType>(input.ir.getType()).getFieldTypes();
    std::vector<mlir::Location> locations(fields.size(), _loc);
    auto* block = _builder.createBlock(&region, region.end(), fields, locations);
    _scope.clear();
    for (std::size_t i = 0; i < fields.size(); ++i) {
      _scope.push_back({block->getArgument(i), input.types[i]});
    }
  }

  mlir::FlatSymbolRefAttr declare(char const* urn, std::string const& key)
  {
    auto [function, added] = _functions.emplace(key, std::format("f{}", _functions.size()));
    if (added) {
      mlir::OpBuilder::InsertionGuard guard(_builder);
      _builder.setInsertionPoint(_root);
      auto [entry, new_urn] = _urns.emplace(urn, std::format("u{}", _urns.size()));
      if (new_urn) { mlir::meld::ExtensionUrnOp::create(_builder, _loc, entry->second, urn); }
      mlir::meld::FunctionOp::create(_builder, _loc, function->second, entry->second, key);
    }
    return mlir::FlatSymbolRefAttr::get(&_context, function->second);
  }

  static std::string key_of(std::string const& name, std::vector<value> const& args)
  {
    std::string key = name + ":";
    for (std::size_t i = 0; i < args.size(); ++i) {
      key += (i ? "_" : "") + key_type_of(args[i].type);
    }
    return key;
  }

  value call(char const* urn,
             std::string const& key,
             std::vector<value> const& args,
             sirius::logical_type result_type)
  {
    auto callee   = declare(urn, key);
    bool nullable = false;
    for (auto const& arg : args) {
      nullable |= arg.nullable();
    }
    auto result = mlir::meld::CallOp::create(_builder,
                                             _loc,
                                             meld_type(result_type, nullable),
                                             callee,
                                             mlir_values_of(args),
                                             nullptr,
                                             nullptr,
                                             nullptr);
    return {result, std::move(result_type)};
  }

  value logical(char const* name, std::vector<value> const& args)
  {
    // Meld's AND and OR take two arguments; Sirius's take any number, which is the same nested.
    value running = args.front();
    for (std::size_t i = 1; i < args.size(); ++i) {
      running = call(boolean_urn,
                     std::string(name) + ":bool",
                     {running, args[i]},
                     logical_type::make(type_id::BOOLEAN));
    }
    return running;
  }

  value comparison(sirius::comparison_type op, value left, value right)
  {
    if (is_float(left.type) || is_float(right.type)) {
      return float_comparison(op, std::move(left), std::move(right));
    }
    return ieee_comparison(op, std::move(left), std::move(right));
  }

  value ieee_comparison(sirius::comparison_type op, value left, value right)
  {
    static std::map<sirius::comparison_type, char const*> const names = {
      {sirius::comparison_type::equal, "equal"},
      {sirius::comparison_type::not_equal, "not_equal"},
      {sirius::comparison_type::lt, "lt"},
      {sirius::comparison_type::le, "lte"},
      {sirius::comparison_type::gt, "gt"},
      {sirius::comparison_type::ge, "gte"},
    };
    auto found = names.find(op);
    if (found == names.end()) { refuse("Meld does not support IS [NOT] DISTINCT FROM"); }
    std::vector<value> args{std::move(left), std::move(right)};
    return call(
      comparison_urn, key_of(found->second, args), args, logical_type::make(type_id::BOOLEAN));
  }

  /// A comparison of floats as DuckDB orders them, where a NaN equals a NaN and is greater than
  /// every other value; Meld compares them as IEEE 754 does, where a NaN is unordered.
  value float_comparison(sirius::comparison_type op, value left, value right)
  {
    using sirius::comparison_type;
    switch (op) {
      case comparison_type::equal: return float_equal(left, right);
      case comparison_type::not_equal: return negate(float_equal(left, right));
      case comparison_type::lt: return float_less(left, right, false);
      case comparison_type::le: return float_less(left, right, true);
      case comparison_type::gt: return float_less(right, left, false);
      case comparison_type::ge: return float_less(right, left, true);
      default: refuse("Meld does not support IS [NOT] DISTINCT FROM");
    }
  }

  value float_equal(value const& left, value const& right)
  {
    auto equal = ieee_comparison(sirius::comparison_type::equal, left, right);
    if (left.never_nan || right.never_nan) { return equal; }
    return logical("or", {equal, logical("and", {is_nan(left), is_nan(right)})});
  }

  /// `left < right`, or `left <= right` where @p or_equal, with a NaN above every other value.
  value float_less(value const& left, value const& right, bool or_equal)
  {
    auto ordered = ieee_comparison(
      or_equal ? sirius::comparison_type::le : sirius::comparison_type::lt, left, right);
    if (right.never_nan) { return ordered; }
    // Below a NaN is every value but a NaN, and at most a NaN is every value but NULL.
    auto below_nan = or_equal ? is_not_null(left) : negate(is_nan(left));
    return logical("or", {ordered, logical("and", {is_nan(right), below_nan})});
  }

  /// Whether @p v is a NaN, and NULL where it is: a NaN alone is not equal to itself.
  value is_nan(value const& v)
  {
    if (v.never_nan) { return literal_bool(false); }
    return negate(ieee_comparison(sirius::comparison_type::equal, v, v));
  }

  value is_not_null(value const& v)
  {
    if (!v.nullable()) { return literal_bool(true); }
    std::vector<value> args{v};
    auto callee = declare(comparison_urn, key_of("is_not_null", args));
    auto result = mlir::meld::CallOp::create(_builder,
                                             _loc,
                                             _builder.getI1Type(),
                                             callee,
                                             mlir_values_of(args),
                                             nullptr,
                                             nullptr,
                                             nullptr);
    return {result, logical_type::make(type_id::BOOLEAN)};
  }

  value negate(value const& v)
  {
    return call(boolean_urn, "not:bool", {v}, logical_type::make(type_id::BOOLEAN));
  }

  value literal_bool(bool b)
  {
    return {mlir::meld::LiteralOp::create(_builder, _loc, _builder.getBoolAttr(b)),
            logical_type::make(type_id::BOOLEAN)};
  }

  value emit(sirius::ast::node const& node)
  {
    return std::visit([&](auto const& alt) { return emit(alt); }, node.v);
  }

  value emit(sirius::ast::reference const& ref)
  {
    if (ref.column_index >= _scope.size()) {
      refuse("Meld does not support reading column {}, which is not there", ref.column_index);
    }
    return _scope[ref.column_index];
  }

  value emit(sirius::ast::constant const& constant)
  {
    auto type    = constant.return_type();
    auto literal = std::visit(
      [&](auto const& payload) -> mlir::TypedAttr {
        using T = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<T, sirius::null_value>) {
          return {};
        } else if constexpr (std::is_same_v<T, bool>) {
          return _builder.getBoolAttr(payload);
        } else if constexpr (std::is_same_v<T, float>) {
          return _builder.getF32FloatAttr(payload);
        } else if constexpr (std::is_same_v<T, double>) {
          return _builder.getF64FloatAttr(payload);
        } else if constexpr (std::is_same_v<T, sirius::date_value>) {
          return mlir::meld::DateAttr::get(&_context, payload.days);
        } else if constexpr (std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::int16_t> ||
                             std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>) {
          return _builder.getIntegerAttr(meld_type(type, false), payload);
        } else {
          refuse("Meld does not support a literal of type {}", type.to_string());
        }
      },
      constant.payload);
    if (!literal) {
      return {mlir::meld::NullOp::create(_builder, _loc, meld_type(type, true)), type};
    }
    auto number = mlir::dyn_cast<mlir::FloatAttr>(literal);
    return {mlir::meld::LiteralOp::create(_builder, _loc, literal),
            type,
            !number || !number.getValue().isNaN()};
  }

  value emit(sirius::ast::comparison const& cmp)
  {
    auto left  = emit(*cmp.left);
    auto right = emit(*cmp.right);
    return comparison(cmp.op, std::move(left), std::move(right));
  }

  value emit(sirius::ast::conjunction const& conj)
  {
    if (conj.op == sirius::ast::conjunction::kind::invalid || conj.children.empty()) {
      refuse("Meld does not support a conjunction with no operator or no operands");
    }
    std::vector<value> args;
    for (auto const& child : conj.children) {
      args.push_back(emit(*child));
    }
    return logical(conj.op == sirius::ast::conjunction::kind::op_and ? "and" : "or", args);
  }

  value emit(sirius::ast::between const& range)
  {
    auto input = emit(*range.input);
    auto lower = emit(*range.lower);
    auto upper = emit(*range.upper);
    // Between bounds that are not NaN, a NaN is above the upper one, which is where IEEE 754's
    // unordered NaN falls too.
    bool const ieee_order = !is_float(input.type) || (lower.never_nan && upper.never_nan);
    if (range.lower_inclusive && range.upper_inclusive && ieee_order) {
      std::vector<value> args{input, lower, upper};
      return call(
        comparison_urn, key_of("between", args), args, logical_type::make(type_id::BOOLEAN));
    }
    auto above =
      comparison(range.lower_inclusive ? sirius::comparison_type::ge : sirius::comparison_type::gt,
                 input,
                 lower);
    auto below =
      comparison(range.upper_inclusive ? sirius::comparison_type::le : sirius::comparison_type::lt,
                 input,
                 upper);
    return logical("and", {above, below});
  }

  value emit(sirius::ast::function_call const& fn)
  {
    // Floating point only: integer arithmetic overflows differently in Meld than in DuckDB, and
    // DuckDB divides by zero to NULL.
    static std::map<sirius::function_id, char const*> const names = {
      {sirius::function_id::add, "add"},
      {sirius::function_id::sub, "subtract"},
      {sirius::function_id::mul, "multiply"},
    };
    auto found = names.find(fn.function());
    if (found == names.end() || fn.arguments().size() != 2 || !is_float(fn.return_type())) {
      refuse("Meld does not support the function {}",
             sirius::to_duckdb_function_name(fn.function()));
    }
    std::vector<value> args;
    for (auto const& argument : fn.arguments()) {
      args.push_back(emit(*argument));
    }
    return call(arithmetic_urn, key_of(found->second, args), args, fn.return_type());
  }

  value emit(sirius::ast::cast const& cast)
  {
    if (cast.kind != sirius::ast::cast_kind::semantic) {
      refuse("Meld does not support restoring a physical carrier");
    }
    auto input = emit(*cast.child);
    if (meld_type(input.type, false) == meld_type(cast.target_type, false)) {
      return {input.ir, cast.target_type};
    }
    auto result =
      mlir::meld::CastOp::create(_builder,
                                 _loc,
                                 meld_type(cast.target_type, input.nullable() || cast.try_cast),
                                 input.ir,
                                 cast.try_cast ? mlir::meld::CastFailureBehavior::ReturnNull
                                               : mlir::meld::CastFailureBehavior::Error);
    return {result, cast.target_type};
  }

  template <typename Other>
  value emit(Other const&)
  {
    refuse("Meld does not support this kind of expression");
  }

  relation build_filter(sirius::ast::node const& condition, relation const& input)
  {
    auto filter = mlir::meld::FilterOp::create(_builder, _loc, input.ir);
    mlir::OpBuilder::InsertionGuard guard(_builder);
    open_region(filter.getCondition(), input);
    auto kept = emit(condition);
    if (kept.type.id() != type_id::BOOLEAN) {
      refuse("Meld does not support a filter on {}", kept.type.to_string());
    }
    mlir::meld::YieldOp::create(_builder, _loc, kept.ir);
    return {filter, input.types};
  }

  relation build_project(std::vector<sirius::ast::node const*> const& projections,
                         relation const& input)
  {
    auto project = mlir::meld::ProjectOp::create(_builder, _loc, input.ir.getType(), input.ir);
    mlir::OpBuilder::InsertionGuard guard(_builder);
    open_region(project.getExpressions(), input);
    std::vector<value> outputs;
    for (auto const* projection : projections) {
      outputs.push_back(emit(*projection));
    }
    mlir::meld::YieldOp::create(_builder, _loc, mlir_values_of(outputs));
    project.getResult().setType(relation_type(outputs));
    return {project, types_of(outputs)};
  }

  relation build_aggregate(std::vector<sirius::ast::aggregate const*> const& aggregates,
                           relation const& input)
  {
    auto aggregate = mlir::meld::AggregateOp::create(_builder, _loc, input.ir.getType(), input.ir);
    mlir::OpBuilder::InsertionGuard guard(_builder);
    open_region(aggregate.getKeys(), input);
    mlir::meld::YieldOp::create(_builder, _loc, mlir::ValueRange{});
    open_region(aggregate.getMeasures(), input);
    std::vector<value> measures;
    for (auto const* each : aggregates) {
      emit_measures(*each, measures);
    }
    mlir::meld::YieldOp::create(_builder, _loc, mlir_values_of(measures));
    aggregate.getResult().setType(relation_type(measures));
    return {aggregate, types_of(measures)};
  }

  void measure(char const* urn,
               char const* name,
               std::vector<value> const& args,
               sirius::logical_type result_type,
               bool nullable,
               std::vector<value>& measures)
  {
    auto callee = declare(urn, key_of(name, args));
    auto result = mlir::meld::MeasureOp::create(
      _builder,
      _loc,
      meld_type(result_type, nullable),
      mlir::meld::AggregateInvocationAttr::get(&_context, mlir::meld::AggregateInvocation::All),
      callee,
      mlir_values_of(args),
      nullptr);
    measures.push_back({result, std::move(result_type)});
  }

  void emit_measures(sirius::ast::aggregate const& aggregate, std::vector<value>& measures)
  {
    if (aggregate.distinct()) { refuse("Meld does not support DISTINCT aggregates"); }
    std::vector<value> args;
    for (auto const& argument : aggregate.arguments()) {
      if (!argument->holds<sirius::ast::reference>()) {
        refuse("Meld does not support an aggregate over an expression");
      }
      args.push_back(emit(*argument));
    }
    auto const count_type = logical_type::make(type_id::BIGINT);
    switch (aggregate.function()) {
      case sirius::aggregate_id::count_star:
        measure(aggregate_urn, "count", {}, count_type, false, measures);
        return;
      case sirius::aggregate_id::count:
        measure(aggregate_urn, "count", args, count_type, false, measures);
        return;
      case sirius::aggregate_id::sum:
      case sirius::aggregate_id::sum_no_overflow:
      case sirius::aggregate_id::avg:
        // An integer sum is accumulated wider than its type in Sirius, to keep it from
        // overflowing.
        if (args.size() != 1 || !is_float(args[0].type)) {
          refuse("Meld does not support SUM or AVG other than of floating point");
        }
        measure(arithmetic_urn, "sum", args, args[0].type, true, measures);
        if (aggregate.function() == sirius::aggregate_id::avg) {
          measure(aggregate_urn, "count", args, count_type, false, measures);
        }
        return;
      case sirius::aggregate_id::min:
      case sirius::aggregate_id::max:
        if (args.size() != 1) {
          refuse("Meld does not support MIN or MAX over {} arguments", args.size());
        }
        measure(arithmetic_urn,
                aggregate.function() == sirius::aggregate_id::min ? "min" : "max",
                args,
                args[0].type,
                true,
                measures);
        return;
      default:
        refuse("Meld does not support the aggregate {}",
               sirius::to_duckdb_aggregate_name(aggregate.function()));
    }
  }

  mlir::MLIRContext _context{mlir::MLIRContext::Threading::DISABLED};
  mlir::OpBuilder _builder{&_context};
  mlir::Location _loc = _builder.getUnknownLoc();
  mlir::meld::RootOp _root;
  std::map<std::string, std::string> _urns;
  std::map<std::string, std::string> _functions;
  std::vector<value> _scope;
};

}  // namespace

std::optional<meld_plan> write_meld_plan(meld_chain const& chain, std::string& why)
{
  try {
    return plan_builder().build(chain);
  } catch (refusal const& refused) {
    why = refused.why;
    return std::nullopt;
  }
}

}  // namespace sirius::meld
