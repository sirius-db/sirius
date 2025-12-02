#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace duckdb {

// Helper class to encapsulate parser utilities
class GraphQueryParser {
public:
  // Extract a list of integers from "IN (id1, id2, id3)"
  static std::vector<int64_t> ExtractIDList(const std::string& clause) {
    std::vector<int64_t> ids;

    size_t open = clause.find("(");
    size_t close = clause.find(")", open);

    if (open == std::string::npos || close == std::string::npos) {
      SIRIUS_LOG_WARN("Could not find parentheses in IN clause: {}", clause);
      return ids;
    }

    std::string ids_str = clause.substr(open + 1, close - open - 1);
    SIRIUS_LOG_DEBUG("Extracting IDs from: '{}'", ids_str);

    std::stringstream ss(ids_str);
    std::string id_str;

    while (std::getline(ss, id_str, ',')) {
      // Trim whitespace
      id_str.erase(0, id_str.find_first_not_of(" \t\n\r"));
      id_str.erase(id_str.find_last_not_of(" \t\n\r") + 1);

      SIRIUS_LOG_DEBUG("Processing ID token: '{}'", id_str);

      if (!id_str.empty()) {
        // Check if it's a valid number
        bool is_number = true;
        for (size_t i = 0; i < id_str.length(); i++) {
          if (i == 0 && id_str[i] == '-') continue; // Allow negative sign
          if (!std::isdigit(id_str[i])) {
            is_number = false;
            break;
          }
        }

        if (is_number) {
          int64_t id = std::stoll(id_str);
          ids.push_back(id);
          SIRIUS_LOG_DEBUG("Extracted ID: {}", id);
        } else {
          SIRIUS_LOG_WARN("Skipping non-numeric token: '{}'", id_str);
        }
      }
    }

    return ids;
  }

  // Extract a single integer from "= value"
  static int64_t ExtractSingleID(const std::string& clause, size_t start_pos) {
    size_t eq = clause.find("=", start_pos);
    if (eq == std::string::npos) {
      return -1;
    }

    size_t num_start = eq + 1;
    while (num_start < clause.length() && std::isspace(clause[num_start])) {
      num_start++;
    }

    size_t num_end = num_start;
    while (num_end < clause.length() &&
           (std::isdigit(clause[num_end]) || clause[num_end] == '-')) {
      num_end++;
    }

    if (num_end > num_start) {
      std::string num_str = clause.substr(num_start, num_end - num_start);
      return std::stoll(num_str);
    }

    return -1;
  }

  // Extract variable name from vertex pattern like "p:Person"
  static std::string ExtractVariableName(const std::string& vertex_pattern) {
    size_t colon = vertex_pattern.find(":");
    if (colon == std::string::npos) {
      return "";
    }

    std::string var_name = vertex_pattern.substr(0, colon);
    var_name.erase(0, var_name.find_first_not_of(" \t\n"));
    var_name.erase(var_name.find_last_not_of(" \t\n") + 1);

    return var_name;
  }

  // Extract columns from "COLUMNS (col1, col2, col3)"
  static std::vector<std::string> ExtractColumns(const std::string& query) {
    std::vector<std::string> columns;

    std::string query_upper = query;
    std::transform(query_upper.begin(), query_upper.end(),
                   query_upper.begin(), ::toupper);

    size_t columns_pos = query_upper.find("COLUMNS");
    if (columns_pos == std::string::npos) {
      return columns;
    }

    size_t open = query.find("(", columns_pos);
    size_t close = query.find(")", open);

    if (open == std::string::npos || close == std::string::npos) {
      return columns;
    }

    std::string cols_str = query.substr(open + 1, close - open - 1);
    std::stringstream ss(cols_str);
    std::string col;

    while (std::getline(ss, col, ',')) {
      col.erase(0, col.find_first_not_of(" \t\n"));
      col.erase(col.find_last_not_of(" \t\n") + 1);
      if (!col.empty()) {
        columns.push_back(col);
      }
    }

    return columns;
  }

  // Extract all vertex patterns from query
  static std::vector<std::string> ExtractVertexPatterns(const std::string& query) {
    std::vector<std::string> patterns;

    size_t pos = 0;
    while ((pos = query.find("(", pos)) != std::string::npos) {
      // Count nested parentheses to find the matching close paren
      int paren_count = 1;
      size_t search_pos = pos + 1;
      size_t close = std::string::npos;

      while (search_pos < query.length() && paren_count > 0) {
        if (query[search_pos] == '(') {
          paren_count++;
        } else if (query[search_pos] == ')') {
          paren_count--;
          if (paren_count == 0) {
            close = search_pos;
            break;
          }
        }
        search_pos++;
      }

      if (close == std::string::npos) {
        pos++;
        continue;
      }

      std::string pattern = query.substr(pos + 1, close - pos - 1);

      // Check if this looks like a vertex pattern (has : and optionally WHERE)
      if (pattern.find(":") != std::string::npos) {
        patterns.push_back(pattern);
      }

      pos = close + 1;
    }

    return patterns;
  }

  // Parse WHERE clause for vertex IDs
  struct WhereClauseResult {
    bool has_where = false;
    bool is_in_clause = false;
    std::vector<int64_t> vertex_ids;
  };

  static WhereClauseResult ParseWhereClause(const std::string& vertex_pattern) {
    WhereClauseResult result;

    std::string pattern_upper = vertex_pattern;
    std::transform(pattern_upper.begin(), pattern_upper.end(),
                   pattern_upper.begin(), ::toupper);

    size_t where_pos = pattern_upper.find("WHERE");
    if (where_pos == std::string::npos) {
      return result;
    }

    result.has_where = true;

    // Check for IN clause
    size_t in_pos = pattern_upper.find("IN", where_pos);

    if (in_pos != std::string::npos) {
      // Make sure it's actually the keyword "IN" (not part of another word)
      bool is_in_keyword = true;

      // Check character before "IN"
      if (in_pos > 0) {
        char before = pattern_upper[in_pos - 1];
        if (!std::isspace(before) && before != '.') {
          is_in_keyword = false;
        }
      }

      // Check character after "IN"
      if (is_in_keyword && in_pos + 2 < pattern_upper.length()) {
        char after = pattern_upper[in_pos + 2];
        if (!std::isspace(after) && after != '(') {
          is_in_keyword = false;
        }
      }

      if (is_in_keyword) {
        // IN clause: WHERE p.id IN (14, 25, 37)
        result.is_in_clause = true;

        // Find the opening paren after IN
        size_t paren_start = vertex_pattern.find("(", in_pos);
        if (paren_start != std::string::npos) {
          result.vertex_ids = ExtractIDList(vertex_pattern.substr(in_pos));

          SIRIUS_LOG_DEBUG("Parsed IN clause, found {} IDs", result.vertex_ids.size());
          for (auto id : result.vertex_ids) {
            SIRIUS_LOG_DEBUG("  - ID: {}", id);
          }
        }
      }
    }

    // If not IN clause, try single value
    if (!result.is_in_clause) {
      // Single value: WHERE p.id = 14
      int64_t id = ExtractSingleID(vertex_pattern, where_pos);
      if (id >= 0) {
        result.vertex_ids.push_back(id);
      }
    }

    return result;
  }

  // Detect algorithm type and path pattern
  struct AlgorithmInfo {
    std::string algorithm_type;
    std::string path_pattern;
  };

  static AlgorithmInfo DetectAlgorithm(const std::string& query) {
    AlgorithmInfo info;

    std::string query_upper = query;
    std::transform(query_upper.begin(), query_upper.end(),
                   query_upper.begin(), ::toupper);

    // Check for ANY SHORTEST
    if (query_upper.find("ANY SHORTEST") != std::string::npos) {
      info.algorithm_type = "UNWEIGHTED_SHORTEST_PATH";

      // Check for hop patterns
      if (query.find("->+") != std::string::npos) {
        info.path_pattern = "ONE_OR_MORE";
      } else if (query.find("->*") != std::string::npos) {
        info.path_pattern = "ZERO_OR_MORE";
      } else {
        info.path_pattern = "DIRECT";
      }
    }
    // Check for SHORTEST (weighted)
    else if (query_upper.find("SHORTEST DISTANCE") != std::string::npos ||
             query_upper.find("CHEAPEST PATH") != std::string::npos) {
      info.algorithm_type = "WEIGHTED_SHORTEST_PATH";
             }
    // Check for SHORTEST with MATCH (also weighted)
    else if (query_upper.find("SHORTEST") != std::string::npos) {
      info.algorithm_type = "WEIGHTED_SHORTEST_PATH";
    }
    // Check for BFS patterns (->* or ->+)
    else if (query.find("->*") != std::string::npos ||
             query.find("->+") != std::string::npos) {
      info.algorithm_type = "BFS";

      // NEW: Detect path pattern for BFS too!
      if (query.find("->+") != std::string::npos) {
        info.path_pattern = "ONE_OR_MORE";
      } else if (query.find("->*") != std::string::npos) {
        info.path_pattern = "ZERO_OR_MORE";
      }
             }
    // Default to edge traversal
    else {
      info.algorithm_type = "EDGE_TRAVERSAL";
      info.path_pattern = "DIRECT";
    }

    return info;
  }

  // Detect edge direction
  struct DirectionInfo {
    bool is_left_directed = false;
    bool is_right_directed = false;
    bool is_any_directed = false;
    bool is_left_right_directed = false;
  };

  static DirectionInfo DetectDirection(const std::string& query) {
    DirectionInfo info;

    bool has_left_arrow = query.find("<-") != std::string::npos;
    bool has_right_arrow = query.find("->") != std::string::npos;

    if (has_left_arrow && has_right_arrow) {
      // Check if it's <-[]-> (bidirectional)
      size_t left_pos = query.find("<-");
      size_t right_pos = query.find("->");

      if (right_pos > left_pos && right_pos - left_pos < 10) {
        info.is_left_right_directed = true;
      } else {
        // Separate <- and -> in query
        info.is_left_directed = true;
        info.is_right_directed = true;
      }
    } else if (has_left_arrow) {
      info.is_left_directed = true;
    } else if (has_right_arrow) {
      info.is_right_directed = true;
    } else {
      info.is_any_directed = true;
    }

    return info;
  }

  // Extract edge table name from pattern -[:label]->
  static std::string ExtractEdgeTable(const std::string& query) {
    size_t edge_start = query.find("-[");
    if (edge_start == std::string::npos) {
      return "";
    }

    edge_start += 2;  // Move past "-["

    // Skip optional colon at the start (e.g., [:knows] not [e:knows])
    if (edge_start < query.length() && query[edge_start] == ':') {
      edge_start++;
    }

    // Now check if there's a variable name before another colon
    // Pattern: [e:knows] - skip 'e:'
    // Pattern: [:knows] - already skipped ':'
    size_t colon_pos = query.find(":", edge_start);
    size_t bracket_end = query.find("]", edge_start);

    if (colon_pos != std::string::npos &&
        bracket_end != std::string::npos &&
        colon_pos < bracket_end) {
      // There's another colon, so skip variable name
      edge_start = colon_pos + 1;
        }

    if (bracket_end == std::string::npos) {
      return "";
    }

    std::string edge_table = query.substr(edge_start, bracket_end - edge_start);

    // Trim whitespace
    edge_table.erase(0, edge_table.find_first_not_of(" \t"));
    edge_table.erase(edge_table.find_last_not_of(" \t") + 1);

    return edge_table;
  }

  // Extract graph name from GRAPH_TABLE(graph_name ...)
  static std::string ExtractGraphName(const std::string& query) {
    std::string query_upper = query;
    std::transform(query_upper.begin(), query_upper.end(),
                   query_upper.begin(), ::toupper);

    size_t graph_table_pos = query_upper.find("GRAPH_TABLE");
    if (graph_table_pos == std::string::npos) {
      return "";
    }

    size_t open_paren = query_upper.find("(", graph_table_pos);
    size_t space_or_match = query_upper.find_first_of(" \t\n", open_paren + 1);

    if (open_paren != std::string::npos && space_or_match != std::string::npos) {
      std::string graph_name = query.substr(open_paren + 1, space_or_match - open_paren - 1);
      graph_name.erase(0, graph_name.find_first_not_of(" \t\n"));
      graph_name.erase(graph_name.find_last_not_of(" \t\n") + 1);
      return graph_name;
    }

    return "";
  }

  // Extract weight column variable from edge pattern [w:label]
  static std::string ExtractWeightColumn(const std::string& query) {
    size_t bracket_start = query.find("[");
    size_t bracket_end = query.find("]");

    if (bracket_start == std::string::npos || bracket_end == std::string::npos) {
      return "";
    }

    std::string bracket_content = query.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    size_t colon = bracket_content.find(":");

    if (colon != std::string::npos) {
      std::string weight_var = bracket_content.substr(0, colon);
      weight_var.erase(0, weight_var.find_first_not_of(" \t"));
      weight_var.erase(weight_var.find_last_not_of(" \t") + 1);
      return weight_var;
    }

    return "";
  }

  // Check if this is a path query
  static bool IsPathQuery(const std::string& query) {
    std::string query_upper = query;
    std::transform(query_upper.begin(), query_upper.end(),
                   query_upper.begin(), ::toupper);

    return false;
  }
};

} // namespace duckdb