//
// Created by andy on 11/23/25.
//
#include "logical_graph_operator.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

LogicalGraphOperator::LogicalGraphOperator(const ParsedGraphQuery& parsed)
    : LogicalOperator(LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR),
      edge_table(parsed.edge_table),
      source_column(parsed.source_column),
      dest_column(parsed.dest_column),
      source_vertex(parsed.source_vertex),
      algorithm_type(parsed.algorithm_type),
      max_hops(parsed.max_hops),
      is_left_directed(parsed.is_left_directed),
      is_right_directed(parsed.is_right_directed),
      is_any_directed(parsed.is_any_directed),
      is_left_right_directed(parsed.is_left_right_directed),
      is_path_query(parsed.is_path_query),
      path_pattern(parsed.path_pattern),
      weight_column(parsed.weight_column) {

  // Set up output schema: (vertex_id BIGINT, distance BIGINT)
  types.push_back(LogicalType::BIGINT);
  types.push_back(LogicalType::BIGINT);
  if (is_path_query) {
    types.push_back(LogicalType::BIGINT);  // path_length
  }
}

string LogicalGraphOperator::GetName() const {
  return "GRAPH_TRAVERSAL";
}

string LogicalGraphOperator::ToString() const {
  return StringUtil::Format(
    "GRAPH_TRAVERSAL[table=%s, source=%lld, algo=%s, path=%s]",
    edge_table.c_str(),
    (long long) source_vertex,
    algorithm_type.c_str(),
    is_path_query ? "true" : "false"
  );
}

} // namespace duckdb
