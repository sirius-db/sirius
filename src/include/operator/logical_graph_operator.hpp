#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/common/types.hpp"
#include <string>

namespace duckdb {

struct ParsedGraphQuery {
  // Edge table info
  string edge_table;
  string source_column = "src";
  string dest_column = "dst";
  string weight_column = "";

  // Vertex constraints
  int64_t source_vertex = -1;
  int64_t dest_vertex = -1;
  std::vector<int64_t> source_vertices;
  std::vector<int64_t> dest_vertices;

  // Query type
  string algorithm_type;
  string path_pattern = "";
  int max_hops = -1;

  // Edge direction
  bool is_left_directed = false;
  bool is_right_directed = false;
  bool is_any_directed = false;
  bool is_left_right_directed = false;

  // Output result
  std::vector<string> output_columns;
  bool parse_success = false;
};

class LogicalGraphOperator : public LogicalOperator {
public:
  string edge_table;
  string source_column;
  string dest_column;
  string weight_column;
  int64_t source_vertex;
  int64_t dest_vertex;
  std::vector<int64_t> source_vertices;
  std::vector<int64_t> dest_vertices;
  string algorithm_type;
  string path_pattern;
  int max_hops;
  bool is_left_directed;
  bool is_right_directed;
  bool is_any_directed;
  bool is_left_right_directed;
  std::vector<string> output_columns;

  explicit LogicalGraphOperator(const ParsedGraphQuery& parsed);

  string GetName() const override;
  string ToString() const;

  void Serialize(Serializer &serializer) const override {
    throw NotImplementedException("LogicalGraphOperator::Serialize not implemented");
  }

  static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer) {
    throw NotImplementedException("LogicalGraphOperator::Deserialize not implemented");
  }

  vector<ColumnBinding> GetColumnBindings() override {
    vector<ColumnBinding> bindings;

    if (!output_columns.empty()) {
      for (size_t i = 0; i < output_columns.size(); i++) {
        bindings.push_back(ColumnBinding(0, i));
      }
    } else {
      // Default bindings
      bindings.push_back(ColumnBinding(0, 0));    // vertex_id
      bindings.push_back(ColumnBinding(0, 1));    // distance
    }

    return bindings;
  }

protected:
  void ResolveTypes() override {
    types.clear();

    if (!output_columns.empty()) {
      for (const auto& col : output_columns) {
        // Assume all are BIGINT
        types.push_back(LogicalType::BIGINT);
      }
    } else {
      // Default columns
      types.push_back(LogicalType::BIGINT);    // vertex id
      types.push_back(LogicalType::BIGINT);    // distance
    }
  }
};

} // namespace duckdb