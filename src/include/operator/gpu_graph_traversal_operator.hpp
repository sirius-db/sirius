#pragma once

#include "gpu_physical_operator.hpp"
#include "gpu_csr_construction_operator.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <cugraph_c/resource_handle.h>
#include <cugraph_c/graph.h>

namespace duckdb {

enum class GraphAlgorithmType {
  EDGE_TRAVERSAL,           // Simple edge scanning
  BFS,                      // Breadth-first search (unweighted)
  UNWEIGHTED_SHORTEST_PATH, // BFS but return full paths
  WEIGHTED_SHORTEST_PATH,   // SSSP (Bellman-Ford or Dijkstra)
  SHORTEST_DISTANCE         // BFS but only return distances
};

class GPUGraphTraversalOperator : public GPUPhysicalOperator {
public:
  unique_ptr<GPUPhysicalOperator> child;  // CSR construction operator
  int64_t source_vertex;
  std::vector<int64_t> source_vertices;
  int64_t dest_vertex = -1;
  std::vector<int64_t> dest_vertices;
  string weight_column;
  GraphAlgorithmType algorithm_type;
  string path_pattern;
  bool is_path_query;
  int max_hops;
  std::vector<string> output_columns;

  // Result data
  mutable vector<int64_t> result_vertices;
  mutable vector<int64_t> result_distances;
  mutable vector<int64_t> result_predecessors;  // For path reconstruction

  // cuGraph handles
  mutable void* cugraph_handle = nullptr;
  mutable bool handle_initialized = false;

  GPUGraphTraversalOperator(
    unique_ptr<GPUPhysicalOperator> child_op,
    int64_t source_vertex,
    std::vector<int64_t> source_vertices,
    int64_t dest_vertex,
    std::vector<int64_t> dest_vertices,
    const string& weight_col,
    const string& algo_str,
    string path_pattern,
    bool is_path,
    int max_hops,
    std::vector<string> output_columns,
    ClientContext& context,
    GPUContext& gpu_context
  );

  ~GPUGraphTraversalOperator() override;

  OperatorResultType Execute(
    GPUIntermediateRelation &input_relation,
    GPUIntermediateRelation &output_relation
  ) const override;

  SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

private:
  void InitializeCuGraph() const;
  void RunEdgeTraversal(int64_t* d_offsets, int64_t* d_indices, int64_t num_vertices, int64_t num_edges) const;
  void RunBFS(int64_t* d_offsets, int64_t* d_indices, int64_t num_vertices, int64_t num_edges) const;
  void RunMultiSourceBFS(int64_t* d_offsets, int64_t* d_indices, int64_t num_vertices, int64_t num_edges) const;
  void RunSSSP(int64_t* d_offsets, int64_t* d_indices, double* d_weights, int64_t num_vertices, int64_t num_edges) const;

  // Helpers
  void BuildOutputRelation(GPUIntermediateRelation& output_relation, size_t num_results) const;
  static GraphAlgorithmType StringToAlgorithmType(const string& algo_str);
};

} // namespace duckdb