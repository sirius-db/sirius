#include "statement_router.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/call_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/statement/set_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

#include <array>

namespace sirius_gateway {

using namespace duckdb;

namespace {

/// Set on each rank by the gateway; a client changing one would desynchronize the ranks.
constexpr std::array<char const*, 11> rank_settings{"sirius_distributed_query_id",
                                                    "sirius_exchange_peers",
                                                    "sirius_exchange_rank",
                                                    "sirius_exchange_host_staging",
                                                    "sirius_exchange_mnnvl",
                                                    "sirius_exchange_frame_bytes",
                                                    "sirius_exchange_send_frames",
                                                    "sirius_exchange_recv_frames_per_peer",
                                                    "sirius_exchange_tcp_lanes",
                                                    "enable_duckdb_fallback",
                                                    "gpu_execution"};

bool is_gateway_name(std::string const& name)
{
  return StringUtil::StartsWith(name, "sirius_gateway_") ||
         StringUtil::StartsWith(name, "quack_") || StringUtil::StartsWith(name, "sirius_rank_");
}

/// Functions a statement calls at its top level: `CALL f()`, `FROM f()`, `SELECT f()`.
std::vector<std::string> called_functions(SQLStatement& stmt)
{
  std::vector<std::string> names;
  auto add = [&](ParsedExpression& e) {
    if (e.GetExpressionClass() == ExpressionClass::FUNCTION) {
      names.push_back(
        StringUtil::Lower(e.Cast<FunctionExpression>().FunctionName().GetIdentifierName()));
    }
  };
  if (stmt.type == StatementType::CALL_STATEMENT) { add(*stmt.Cast<CallStatement>().function); }
  if (stmt.type == StatementType::SELECT_STATEMENT) {
    auto& node = *stmt.Cast<SelectStatement>().node;
    if (node.type == QueryNodeType::SELECT_NODE) {
      auto& select = node.Cast<SelectNode>();
      if (select.from_table && select.from_table->type == TableReferenceType::TABLE_FUNCTION) {
        add(*select.from_table->Cast<TableFunctionRef>().function);
      }
      for (auto& e : select.select_list) {
        add(*e);
      }
    }
  }
  return names;
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
  bool local = false;
  for (auto const& name : called_functions(stmt)) {
    if (!is_gateway_name(name)) { continue; }
    if (!client_may_call(name, allow_join)) {
      throw InvalidInputException("%s can only be called on the gateway's console", name);
    }
    local = true;
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
