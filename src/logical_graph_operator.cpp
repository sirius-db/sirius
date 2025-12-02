#include "logical_graph_operator.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

LogicalGraphOperator::LogicalGraphOperator(const ParsedGraphQuery& parsed)
    : LogicalOperator(LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR),
      edge_table(parsed.edge_table),
      source_column(parsed.source_column),
      dest_column(parsed.dest_column),
      weight_column(parsed.weight_column),
      source_vertex(parsed.source_vertex),
      dest_vertex(parsed.dest_vertex),
      source_vertices(parsed.source_vertices),
      dest_vertices(parsed.dest_vertices),
      algorithm_type(parsed.algorithm_type),
      path_pattern(parsed.path_pattern),
      is_path_query(parsed.is_path_query),
      max_hops(parsed.max_hops),
      is_left_directed(parsed.is_left_directed),
      is_right_directed(parsed.is_right_directed),
      is_any_directed(parsed.is_any_directed),
      is_left_right_directed(parsed.is_left_right_directed),
      output_columns(parsed.output_columns) {

  // Set up output schema based on output_columns or defaults
  if (!output_columns.empty()) {
    for (const auto& col : output_columns) {
      types.push_back(LogicalType::BIGINT);
    }
  } else {
    types.push_back(LogicalType::BIGINT);    // vertex_id
    types.push_back(LogicalType::BIGINT);    // distance
    if (is_path_query) {
      types.push_back(LogicalType::BIGINT);  // predecessor
    }
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
