#include "statement_facts.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/query_node/copy_query_node.hpp"
#include "duckdb/parser/query_node/insert_query_node.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/statement/call_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

#include <algorithm>
#include <set>

namespace sirius_gateway {

using namespace duckdb;

namespace {

struct walk {
  statement_facts facts;
  std::set<std::string> ctes;
  std::vector<std::string> tables;  ///< unqualified names, matched against ctes at the end

  void expression(ParsedExpression& e)
  {
    if (e.GetExpressionClass() == ExpressionClass::FUNCTION) {
      facts.functions.push_back(
        StringUtil::Lower(e.Cast<FunctionExpression>().FunctionName().GetIdentifierName()));
    }
    // The iterator stops at a subquery expression: its query is not one of its children.
    if (e.GetExpressionClass() == ExpressionClass::SUBQUERY) {
      query(*e.Cast<SubqueryExpression>().SubqueryMutable()->node);
    }
    ParsedExpressionIterator::EnumerateChildren(e,
                                                [&](ParsedExpression& child) { expression(child); });
  }

  /// The iterator calls back for each expression and table reference below @p node, but not for
  /// the query nodes on the way: nodes() covers those.
  void query(QueryNode& node)
  {
    nodes(node);
    ParsedExpressionIterator::EnumerateQueryNodeChildren(
      node,
      [&](unique_ptr<ParsedExpression>& e) {
        if (e) { expression(*e); }
      },
      [&](TableRef& ref) { table(ref); });
  }

  /// @p node and the query nodes it holds directly: CTE bodies, the sides of a set operation, the
  /// query of an INSERT or a COPY. A node under a table reference or an expression is entered
  /// from table() or expression().
  void nodes(QueryNode& node)
  {
    for (auto& kv : node.cte_map.map) {
      ctes.insert(StringUtil::Lower(kv.first.GetIdentifierName()));
      if (kv.second->query_node) { nodes(*kv.second->query_node); }
    }
    switch (node.type) {
      case QueryNodeType::SET_OPERATION_NODE:
        for (auto& child : node.Cast<SetOperationNode>().children) {
          nodes(*child);
        }
        break;
      case QueryNodeType::RECURSIVE_CTE_NODE:
        nodes(*node.Cast<RecursiveCTENode>().left);
        nodes(*node.Cast<RecursiveCTENode>().right);
        break;
      case QueryNodeType::INSERT_QUERY_NODE:
        facts.writes = true;
        if (auto& select = node.Cast<InsertQueryNode>().select_statement) { nodes(*select->node); }
        break;
      case QueryNodeType::COPY_QUERY_NODE:
        facts.writes = true;
        if (auto& select = node.Cast<CopyQueryNode>().info->select_statement) { nodes(*select); }
        break;
      case QueryNodeType::UPDATE_QUERY_NODE:
      case QueryNodeType::DELETE_QUERY_NODE:
      case QueryNodeType::MERGE_QUERY_NODE: facts.writes = true; break;
      default: break;
    }
  }

  void table(TableRef& ref)
  {
    switch (ref.type) {
      case TableReferenceType::JOIN:
      case TableReferenceType::EMPTY_FROM:
      case TableReferenceType::EXPRESSION_LIST: break;  // their parts are visited on their own
      case TableReferenceType::SUBQUERY: nodes(*ref.Cast<SubqueryRef>().subquery->node); break;
      case TableReferenceType::BASE_TABLE: {
        auto const& name = ref.Cast<BaseTableRef>().GetQualifiedName();
        if (name.Catalog().empty() && name.Schema().empty()) {
          tables.push_back(StringUtil::Lower(name.Name().GetIdentifierName()));
        } else {
          facts.reads_table = true;
        }
        break;
      }
      case TableReferenceType::TABLE_FUNCTION: {
        auto& f = *ref.Cast<TableFunctionRef>().function;
        facts.table_functions.push_back(
          f.GetExpressionClass() == ExpressionClass::FUNCTION
            ? StringUtil::Lower(f.Cast<FunctionExpression>().FunctionName().GetIdentifierName())
            : std::string{});
        break;
      }
      default: facts.reads_table = true;
    }
  }
};

}  // namespace

statement_facts facts_of(SQLStatement& stmt)
{
  walk w;
  if (stmt.type == StatementType::CALL_STATEMENT) {
    w.expression(*stmt.Cast<CallStatement>().function);
  }
  if (stmt.type == StatementType::SELECT_STATEMENT) { w.query(*stmt.Cast<SelectStatement>().node); }
  w.facts.reads_table =
    w.facts.reads_table || std::any_of(w.tables.begin(), w.tables.end(), [&](auto const& name) {
      return w.ctes.count(name) == 0;
    });
  return std::move(w.facts);
}

}  // namespace sirius_gateway
