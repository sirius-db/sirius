#include "statement_router.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/statement/call_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/statement/set_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

#include <array>

namespace sirius_gateway {

using namespace duckdb;

namespace {

/// Set on each rank by the gateway; a client changing one would desynchronize the ranks.
constexpr std::array<char const*, 6> rank_settings{"sirius_distributed_query_id",
                                                   "sirius_exchange_peers",
                                                   "sirius_exchange_rank",
                                                   "sirius_exchange_host_staging",
                                                   "enable_duckdb_fallback",
                                                   "gpu_execution"};

bool is_gateway_name(std::string const& name)
{
  return StringUtil::StartsWith(name, "sirius_gateway_") ||
         StringUtil::StartsWith(name, "quack_") || StringUtil::StartsWith(name, "sirius_rank_");
}

/// What a statement calls and reads, at any depth: subqueries, CTEs and join sides included.
struct references {
  std::vector<std::string> functions;  ///< every function called, table functions included
  /// The statement reads a table, a file, or a table function that is not a gateway function.
  bool other_source = false;
};

void collect(QueryNode& node, references& refs);

void collect(ParsedExpression& e, references& refs)
{
  if (e.GetExpressionClass() == ExpressionClass::FUNCTION) {
    refs.functions.push_back(
      StringUtil::Lower(e.Cast<FunctionExpression>().FunctionName().GetIdentifierName()));
  }
  // The iterator stops at a subquery expression: its query is not one of its children.
  if (e.GetExpressionClass() == ExpressionClass::SUBQUERY) {
    collect(*e.Cast<SubqueryExpression>().SubqueryMutable()->node, refs);
  }
  ParsedExpressionIterator::EnumerateChildren(
    e, [&](ParsedExpression& child) { collect(child, refs); });
}

void collect(QueryNode& node, references& refs)
{
  ParsedExpressionIterator::EnumerateQueryNodeChildren(
    node,
    [&](unique_ptr<ParsedExpression>& e) {
      if (e) { collect(*e, refs); }
    },
    [&](TableRef& ref) {
      switch (ref.type) {
        case TableReferenceType::JOIN:
        case TableReferenceType::SUBQUERY:
        case TableReferenceType::EMPTY_FROM:
        case TableReferenceType::EXPRESSION_LIST: break;  // their parts are visited on their own
        case TableReferenceType::TABLE_FUNCTION: {
          auto& f = *ref.Cast<TableFunctionRef>().function;
          refs.other_source |= f.GetExpressionClass() != ExpressionClass::FUNCTION ||
                               !is_gateway_name(StringUtil::Lower(
                                 f.Cast<FunctionExpression>().FunctionName().GetIdentifierName()));
          break;
        }
        default: refs.other_source = true;
      }
    });
}

references references_of(SQLStatement& stmt)
{
  references refs;
  if (stmt.type == StatementType::CALL_STATEMENT) {
    collect(*stmt.Cast<CallStatement>().function, refs);
  }
  if (stmt.type == StatementType::SELECT_STATEMENT) {
    collect(*stmt.Cast<SelectStatement>().node, refs);
  }
  return refs;
}

/// Whether a client may call @p name, a gateway function, which then runs on the gateway.
bool client_may_call(std::string const& name, bool allow_join)
{
  if (name == "sirius_gateway_status" || name == "sirius_gateway_log") { return true; }
  return allow_join && (name == "sirius_gateway_add_rank" || name == "sirius_gateway_drop_rank");
}

/// Whether @p stmt runs on the gateway rather than on every rank.
bool runs_on_gateway(SQLStatement& stmt, bool allow_join)
{
  // Every client shares one session per rank, so a client's transaction there would hold every
  // other client's statements. It stays in the client's own gateway session instead; each
  // statement still commits on the ranks as it runs.
  if (stmt.type == StatementType::TRANSACTION_STATEMENT) { return true; }
  // A prepared plan keeps the distributed query number it was planned with, which the gateway
  // does not track across statements.
  if (stmt.type == StatementType::PREPARE_STATEMENT ||
      stmt.type == StatementType::EXECUTE_STATEMENT) {
    throw InvalidInputException("PREPARE and EXECUTE are not supported through the gateway");
  }
  if (stmt.type == StatementType::SET_STATEMENT) {
    auto const name = StringUtil::Lower(stmt.Cast<SetStatement>().name.GetIdentifierName());
    for (auto const* owned : rank_settings) {
      if (name == owned) {
        throw InvalidInputException("%s is set on every rank by the gateway", name);
      }
    }
    if (is_gateway_name(name)) {
      throw InvalidInputException("%s can only be set on the gateway's console", name);
    }
  }
  references refs;
  try {
    refs = references_of(stmt);
  } catch (NotImplementedException&) {
    // A tree the iterator cannot walk goes to the ranks, which have no gateway functions.
    return false;
  }
  bool local = false;
  for (auto const& name : refs.functions) {
    if (!is_gateway_name(name)) { continue; }
    if (!client_may_call(name, allow_join)) {
      throw InvalidInputException("%s can only be called on the gateway's console", name);
    }
    local = true;
  }
  // The statement runs as written in the client's gateway session. Anything else it read there
  // would be the gateway host's files, or SQL run by a function such as query().
  if (local && refs.other_source) {
    throw InvalidInputException(
      "a statement that calls a gateway function cannot read tables, files or other table "
      "functions");
  }
  return local;
}

}  // namespace

std::string route(std::string const& sql, bool allow_join)
{
  Parser parser{ParserOptions::Builtin()};
  try {
    parser.ParseQuery(sql);
  } catch (std::exception&) {
    return sql;  // DuckDB reports the parse error when it runs the text
  }
  std::string routed;
  for (auto& stmt : parser.statements) {
    // The parser leaves each statement's own text in query (and its location at 0).
    auto text = stmt->query;
    StringUtil::RTrim(text, ";\n\t ");
    routed += runs_on_gateway(*stmt, allow_join)
                ? text
                : "FROM sirius_gateway_run(" + KeywordHelper::WriteQuoted(text) + ")";
    routed += ";\n";
  }
  return routed.empty() ? sql : routed;
}

}  // namespace sirius_gateway
