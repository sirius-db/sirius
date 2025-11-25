//
// Created by andy on 11/23/25.
//

#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/common/types.hpp"
#include <string>

namespace duckdb {

struct ParsedGraphQuery {
  string edge_table;
  string source_column = "src";
  string dest_column = "dst";
  int64_t source_vertex = -1;
  string algorithm_type;
  int max_hops = -1;

  bool is_left_directed = false;
  bool is_right_directed = false;
  bool is_any_directed = false;
  bool is_left_right_directed = false;

  // Path query flags
  bool is_path_query = false;
  string path_pattern = "";
  string weight_column = "";

  bool parse_success = false;
};

class LogicalGraphOperator : public LogicalOperator {
public:
  string edge_table;
  string source_column;
  string dest_column;
  int64_t source_vertex;
  string algorithm_type;
  int max_hops;
  bool is_left_directed;
  bool is_right_directed;
  bool is_any_directed;
  bool is_left_right_directed;
  bool is_path_query;
  string path_pattern;
  string weight_column;

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
    // Graph operator produces vertex_id and distance columns
    // They don't come from any table, so we create synthetic bindings
    if (is_path_query) {
      // may need additional columns (path_length, edges, etc.)
      return {
        ColumnBinding(0, 0),  // vertex_id
        ColumnBinding(0, 1),  // distance/path_length
        ColumnBinding(0, 2)   // (optional) predecessor, for path reconsutrtrcuuction
      };
    }
    return {
      ColumnBinding(0, 0),  // vertex_id
      ColumnBinding(0, 1)   // distance
    };
  }

protected:
  void ResolveTypes() override {
    types.clear();
    types.push_back(LogicalType::BIGINT);
    types.push_back(LogicalType::BIGINT);
    if (is_path_query) {
      types.push_back(LogicalType::BIGINT);  // path_length or path info
    }
  }
};

} // namespace duckdb