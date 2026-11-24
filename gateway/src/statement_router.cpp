#include "statement_router.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/set_statement.hpp"
#include "statement_facts.hpp"

#include <algorithm>
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

/// The gateway function that @p sql calls by name, or empty: an identifier with such a name,
/// quoted or not, that an opening parenthesis follows. A check on the tokens, so it also finds
/// the name of a table or a column there; for the statements that facts_of() does not walk.
std::string gateway_call_in(std::string const& sql)
{
  auto const tokens = Parser::Tokenize(sql);
  for (idx_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i].type != SimplifiedTokenType::SIMPLIFIED_TOKEN_IDENTIFIER) { continue; }
    auto next = i + 1;
    while (next < tokens.size() &&
           tokens[next].type == SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT) {
      ++next;
    }
    if (next == tokens.size() || sql[tokens[next].start] != '(') { continue; }
    auto name = sql.substr(tokens[i].start, tokens[i + 1].start - tokens[i].start);
    StringUtil::Trim(name);
    if (name.size() >= 2 && name.front() == '"' && name.back() == '"') {
      name = name.substr(1, name.size() - 2);
    }
    name = StringUtil::Lower(name);
    if (is_gateway_name(name)) { return name; }
  }
  return {};
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
  // The walk covers CALL and SELECT. Any other statement, and a tree the iterator cannot walk,
  // goes to the ranks, which have the quack_* and sirius_rank_* functions: a CREATE TABLE AS, an
  // INSERT, a COPY, a view or a macro must not carry a call there.
  bool walked =
    stmt.type == StatementType::CALL_STATEMENT || stmt.type == StatementType::SELECT_STATEMENT;
  statement_facts facts;
  try {
    facts = facts_of(stmt);
  } catch (NotImplementedException&) {
    walked = false;
  }
  if (!walked) {
    if (auto const name = gateway_call_in(stmt.query); !name.empty()) {
      throw InvalidInputException(client_may_call(name, allow_join)
                                    ? "%s can only be called in a SELECT or a CALL"
                                    : "%s can only be called on the gateway's console",
                                  name);
    }
    return false;
  }
  bool local = false;
  for (auto const& name : facts.functions) {
    if (!is_gateway_name(name)) { continue; }
    if (!client_may_call(name, allow_join)) {
      throw InvalidInputException("%s can only be called on the gateway's console", name);
    }
    local = true;
  }
  // The statement runs as written in the client's gateway session. Anything else it read or
  // wrote there would be the gateway host's files, or SQL run by a function such as query().
  bool const other_source =
    facts.reads_table || std::any_of(facts.table_functions.begin(),
                                     facts.table_functions.end(),
                                     [](auto const& name) { return !is_gateway_name(name); });
  if (local && facts.writes) {
    throw InvalidInputException(
      "a statement that calls a gateway function cannot write tables or files");
  }
  if (local && other_source) {
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
  // Text that does not parse is refused whole. Run as written, the client's gateway session
  // would execute, unrouted, each statement before the one that does not parse.
  parser.ParseQuery(sql);
  std::string routed;
  for (auto& stmt : parser.statements) {
    // The parser leaves each statement's own text in query (and its location at 0).
    auto text = stmt->query;
    // DuckDB 2.0.1 starts a statement's text right after the terminator of the one before.
    StringUtil::LTrim(text);
    StringUtil::RTrim(text, ";\n\t ");
    // The newline ends a trailing line comment, which would otherwise hold the terminator.
    routed += runs_on_gateway(*stmt, allow_join)
                ? text + "\n"
                : "FROM sirius_gateway_run(" + KeywordHelper::WriteQuoted(text) + ")";
    routed += ";\n";
  }
  return routed.empty() ? sql : routed;
}

}  // namespace sirius_gateway
