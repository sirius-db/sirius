#pragma once

#include <string>
#include <vector>

namespace duckdb {
class SQLStatement;
}  // namespace duckdb

namespace sirius_gateway {

/// What a statement calls, reads and writes, at any depth: CTEs, subqueries and join sides
/// included. Names are in lower case. Only a SELECT and a CALL are walked: any other statement
/// has no facts.
struct statement_facts {
  std::vector<std::string> functions;  ///< every function called, table functions included
  /// The table functions read; "" for one that is not a plain call.
  std::vector<std::string> table_functions;
  /// Reads a table that is not one of the statement's CTEs, or a source that is neither a table
  /// nor a table function. A CTE is matched by name over the whole statement, not by scope.
  bool reads_table = false;
  bool writes      = false;  ///< holds DML or a COPY, which a CTE's body can be
};

/// @throws duckdb::NotImplementedException for a tree that the parser's iterator cannot walk.
statement_facts facts_of(duckdb::SQLStatement& stmt);

}  // namespace sirius_gateway
