//
// Created by andy on 11/23/25.
//
#include "gpu_graph_traversal_operator.hpp"
#include "gpu_context.hpp"
#include "log/logging.hpp"
#include <cugraph_c/error.h>
#include <cugraph/graph.hpp>
#include <cugraph/algorithms.hpp>
#include <cugraph/graph_view.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/device_span.hpp>

namespace duckdb {

GPUGraphTraversalOperator::GPUGraphTraversalOperator(
  unique_ptr<GPUPhysicalOperator> child_op,
  int64_t source,
  const string& algo_str,
  bool is_path,
  int max_hops,
  const string& weight_col,
  ClientContext& context,
  GPUContext& gpu_context
) : GPUPhysicalOperator(
    PhysicalOperatorType::EXTENSION,
    vector<LogicalType>{LogicalType::BIGINT, LogicalType::BIGINT},  // vertex_id, distance
    0  // estimated_cardinality
  ),
    child(std::move(child_op)),
    source_vertex(source),
    algorithm_type(StringToAlgorithmType(algo_str)),
    max_hops(max_hops),
    is_path_query(is_path),
    weight_column(weight_col) {
}

GPUGraphTraversalOperator::~GPUGraphTraversalOperator() {
  if (handle_initialized && cugraph_handle) {
    auto* handle = static_cast<raft::handle_t*>(cugraph_handle);
    delete handle;
    cugraph_handle = nullptr;
  }
}

OperatorResultType
GPUGraphTraversalOperator::Execute(
    GPUIntermediateRelation &input_relation,
    GPUIntermediateRelation &output_relation) const {

  SIRIUS_LOG_INFO("Graph Traversal: Starting execution");

  // Execute CSR construction operator
  GPUIntermediateRelation csr_data(2);  // offsets, indices
  auto result = child->Execute(input_relation, csr_data);
  if (result != OperatorResultType::FINISHED) {
    return result;
  }

  // Get CSR pointers from child output
  auto csr_op = dynamic_cast<GPUCSRConstructionOperator*>(child.get());
  if (!csr_op) {
    throw InternalException("Child operator is not GPUCSRConstructionOperator");
  }
  int64_t* d_offsets = csr_op->d_offsets;
  int64_t* d_indices = csr_op->d_indices;
  int64_t num_vertices = csr_op->num_vertices;
  int64_t num_edges = csr_op->indices.size();

  SIRIUS_LOG_DEBUG("CSR: {} vertices, {} edges", num_vertices, num_edges);

  if (!handle_initialized) {
    InitializeCuGraph();
  }

  // Execute the appropriate algorithm
  switch (algorithm_type) {
    case GraphAlgorithmType::EDGE_TRAVERSAL:
      RunEdgeTraversal(d_offsets, d_indices, num_vertices, num_edges);
      break;
    case GraphAlgorithmType::BFS:
    case GraphAlgorithmType::UNWEIGHTED_SHORTEST_PATH:
    case GraphAlgorithmType::SHORTEST_DISTANCE:
      RunBFS(d_offsets, d_indices, num_vertices, num_edges);
      break;
    case GraphAlgorithmType::WEIGHTED_SHORTEST_PATH:
      // Get weights from CSR operator
      if (!csr_op->has_weights) {
        throw InvalidInputException("WEIGHTED_SHORTEST_PATH requires edge weights");
      }
      RunSSSP(d_offsets, d_indices, csr_op->d_weights, num_vertices, num_edges);
      break;
    default:
      throw InternalException("Unknown algorithm type");
  }

  // FiFilter out unreachable nodes (on CPU)
  vector<int64_t> filtered_vertices;
  vector<int64_t> filtered_distances;
  vector<int64_t> filtered_predecessors;

  for (int64_t i = 0; i < num_vertices; i++) {
    // Only include reachable vertices (distance != infinity)
    if (result_distances[i] != std::numeric_limits<int64_t>::max()) {
      filtered_vertices.push_back(i);
      filtered_distances.push_back(result_distances[i]);
      if (is_path_query) {
        filtered_predecessors.push_back(result_predecessors[i]);
      }
    }
  }

  result_vertices = std::move(filtered_vertices);
  result_distances = std::move(filtered_distances);
  if (is_path_query) {
    result_predecessors = std::move(filtered_predecessors);
  }

  size_t num_results = filtered_vertices.size();
  SIRIUS_LOG_DEBUG("Graph traversal: {} reachable vertices out of {}",
                   num_results, num_vertices);

  // Transfer filtered results to GPU
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

  // Allocate GPU memory for filtered results
  int64_t* d_result_vertices = gpuBufferManager->customCudaMalloc<int64_t>(num_results, 0, 0);
  int64_t* d_result_distances = gpuBufferManager->customCudaMalloc<int64_t>(num_results, 0, 0);
  int64_t* d_result_predecessors = nullptr;

  // Copy to GPU
  cudaMemcpy(d_result_vertices, filtered_vertices.data(),
             num_results * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_result_distances, filtered_distances.data(),
             num_results * sizeof(int64_t), cudaMemcpyHostToDevice);
  if (is_path_query && !filtered_predecessors.empty()) {
    d_result_predecessors = gpuBufferManager->customCudaMalloc<int64_t>(num_results, 0, 0);
    cudaMemcpy(d_result_predecessors, filtered_predecessors.data(),
               num_results * sizeof(int64_t), cudaMemcpyHostToDevice);
  }

  // Create output relation with results
  output_relation.columns.clear();

  // Column 0: vertex_id
  auto vertex_col = make_shared_ptr<GPUColumn>(
    num_results,
    GPUColumnType(GPUColumnTypeId::INT64),
    reinterpret_cast<uint8_t*>(d_result_vertices),  // GPU pointer
    nullptr  // validity_mask (all valid)
  );
  output_relation.columns.push_back(vertex_col);

  // Column 1: distance
  auto distance_col = make_shared_ptr<GPUColumn>(
    num_results,
    GPUColumnType(GPUColumnTypeId::INT64),
    reinterpret_cast<uint8_t*>(d_result_distances),  // GPU pointer
    nullptr  // validity_mask (all valid)
  );
  output_relation.columns.push_back(distance_col);

  // Column 2: predecessor (if path query)
  if (is_path_query && d_result_predecessors != nullptr) {
    auto pred_col = make_shared_ptr<GPUColumn>(
      num_results,
      GPUColumnType(GPUColumnTypeId::INT64),
      reinterpret_cast<uint8_t*>(d_result_predecessors),  // GPU pointer
      nullptr  // validity_mask (all valid)
    );
    output_relation.columns.push_back(pred_col);
  }

  SIRIUS_LOG_DEBUG("Graph traversal returned {} results", result_vertices.size());

  return OperatorResultType::FINISHED;
}

void
GPUGraphTraversalOperator::InitializeCuGraph() const {
  if (handle_initialized) {
    return;
  }

  SIRIUS_LOG_INFO("Initializing cuGraph handle");

  // Create RAFT handle (cuGraph's resource manager)
  auto* handle = new raft::handle_t();
  cugraph_handle = static_cast<void*>(handle);

  handle_initialized = true;

  SIRIUS_LOG_INFO("cuGraph handle initialized");
}

void
GPUGraphTraversalOperator::RunEdgeTraversal(
  int64_t* d_offsets,
  int64_t* d_indices,
  int64_t num_vertices,
  int64_t num_edges
) const {
  SIRIUS_LOG_INFO("Running edge traversal from source vertex {}", source_vertex);

  // Validate source vertex
  if (source_vertex < 0 || source_vertex >= num_vertices) {
    throw InvalidInputException(
      StringUtil::Format("Source vertex %lld out of range [0, %lld)",
                        source_vertex, num_vertices)
    );
  }

  // Get offset range for this vertex in CSR
  vector<int64_t> h_offsets(2);
  cudaMemcpy(h_offsets.data(), &d_offsets[source_vertex],
             2 * sizeof(int64_t), cudaMemcpyDeviceToHost);

  int64_t edge_start = h_offsets[0];
  int64_t edge_end = h_offsets[1];
  int64_t num_edges_from_source = edge_end - edge_start;

  SIRIUS_LOG_DEBUG("Source vertex {} has {} outgoing edges",
                   source_vertex, num_edges_from_source);

  if (num_edges_from_source == 0) {
    // No edges from this vertex
    result_vertices.clear();
    result_distances.clear();
    SIRIUS_LOG_INFO("No edges from source vertex {}", source_vertex);
    return;
  }

  // Copy the destination vertices for these edges
  result_vertices.resize(num_edges_from_source);
  cudaMemcpy(result_vertices.data(), &d_indices[edge_start],
             num_edges_from_source * sizeof(int64_t), cudaMemcpyDeviceToHost);

  // All edges have distance 1 (one hop)
  result_distances.resize(num_edges_from_source);
  std::fill(result_distances.begin(), result_distances.end(), 1);

  // If path query, set predecessors (all point back to source)
  if (is_path_query) {
    result_predecessors.resize(num_edges_from_source);
    std::fill(result_predecessors.begin(), result_predecessors.end(), source_vertex);
  }

  SIRIUS_LOG_INFO("Edge traversal complete: {} edges from vertex {}",
                  num_edges_from_source, source_vertex);
}

void
GPUGraphTraversalOperator::RunBFS(
    int64_t* d_offsets,
    int64_t* d_indices,
    int64_t num_vertices,
    int64_t num_edges) const {

  SIRIUS_LOG_INFO("Running BFS from source vertex {}", source_vertex);

  // Get RAFT handle
  auto* handle = static_cast<raft::handle_t*>(cugraph_handle);

  // Create device spans
  auto offsets_span = raft::device_span<int64_t const>(d_offsets, num_vertices + 1);
  auto indices_span = raft::device_span<int64_t const>(d_indices, num_edges);

  // Create graph metadata
  auto graph_meta = cugraph::graph_view_meta_t<int64_t, int64_t, false, false>{
    num_vertices,
    num_edges,
    cugraph::graph_properties_t{false, false}
  };

  // Create graph view
  auto graph_view = cugraph::graph_view_t<int64_t, int64_t, false, false>(
    offsets_span,
    indices_span,
    graph_meta
  );

  // Allocate output arrays on GPU
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  int64_t* d_distances = gpuBufferManager->customCudaMalloc<int64_t>(num_vertices, 0, 0);
  int64_t* d_predecessors = gpuBufferManager->customCudaMalloc<int64_t>(num_vertices, 0, 0);

  // Run BFS
  cugraph::bfs(
    *handle,
    graph_view,
    d_distances,
    d_predecessors,
    &source_vertex,
    1,
    false,
    std::numeric_limits<int64_t>::max()
  );

  SIRIUS_LOG_INFO("BFS complete");

  // Copy results back to CPU for now (TODO: keep on GPU)
  result_vertices.resize(num_vertices);
  result_distances.resize(num_vertices);

  // Vertex IDs are just 0, 1, 2, ..., num_vertices-1
  for (int64_t i = 0; i < num_vertices; i++) {
    result_vertices[i] = i;
  }

  // Copy distances from GPU to CPU
  cudaMemcpy(result_distances.data(), d_distances,
             num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost);

  // Copy predecessors if path query
  if (is_path_query) {
    result_predecessors.resize(num_vertices);
    cudaMemcpy(result_predecessors.data(), d_predecessors,
               num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost);
    SIRIUS_LOG_DEBUG("Copied {} predecessors for path reconstruction", num_vertices);
  }

  SIRIUS_LOG_INFO("BFS results: {} vertices reached",
                  std::count_if(result_distances.begin(), result_distances.end(),
                                [](int64_t d) { return d != std::numeric_limits<int64_t>::max(); }));
}

void
GPUGraphTraversalOperator::RunSSSP(
    int64_t* d_offsets,
    int64_t* d_indices,
    double* d_weights,
    int64_t num_vertices,
    int64_t num_edges) const {

  SIRIUS_LOG_INFO("Running SSSP from source vertex {}", source_vertex);

  auto* handle = static_cast<raft::handle_t*>(cugraph_handle);

  // Create device spans
  auto offsets_span = raft::device_span<int64_t const>(d_offsets, num_vertices + 1);
  auto indices_span = raft::device_span<int64_t const>(d_indices, num_edges);

  // Create graph metadata
  auto graph_meta = cugraph::graph_view_meta_t<int64_t, int64_t, false, false>{
    num_vertices,
    num_edges,
    cugraph::graph_properties_t{false, false} // not symmetric, not multigraph
  };

  // Create graph view
  auto graph_view = cugraph::graph_view_t<int64_t, int64_t, false, false>(
    offsets_span,
    indices_span,
    graph_meta
  );

  // Allocate output
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  float* d_distances = gpuBufferManager->customCudaMalloc<float>(num_vertices, 0, 0);
  int64_t* d_predecessors = gpuBufferManager->customCudaMalloc<int64_t>(num_vertices, 0, 0);

  // Allocate and fill weights array with 1.0
  float* d_weights_float = nullptr;
  if (d_weights) {
    d_weights_float = gpuBufferManager->customCudaMalloc<float>(num_edges, 0, 0);

    // TODO: Convert double to float (simple kernel or use thrust)
    // For now, create on CPU and copy
    vector<double> weights_host(num_edges);
    cudaMemcpy(weights_host.data(), d_weights, num_edges * sizeof(double),
               cudaMemcpyDeviceToHost);

    vector<float> weights_float(num_edges);
    for (size_t i = 0; i < num_edges; i++) {
      weights_float[i] = static_cast<float>(weights_host[i]);
    }

    cudaMemcpy(d_weights_float, weights_float.data(),
               num_edges * sizeof(float), cudaMemcpyHostToDevice);
  }

  // Wrap in edge_property_view with vectors
  std::vector<float const*> weight_ptrs = {d_weights_float};
  std::vector<int64_t> edge_counts = {num_edges};

  auto edge_weights = cugraph::edge_property_view_t<int64_t, float const*>(
      weight_ptrs,
      edge_counts
  );

  // Run SSSP
  cugraph::sssp(
    *handle,
    graph_view,
    edge_weights,      // edge weights (nullptr for unweighted = all 1.0)
    d_distances,        // output: distances from source
    d_predecessors,     // output: predecessor in shortest path tree
    source_vertex,      // source vertex
    std::numeric_limits<float>::max(),  // cutoff distance
    false               // do_expensive_check
  );

  SIRIUS_LOG_INFO("SSSP complete");

  // Copy results back (TODO: optimize)
  result_vertices.resize(num_vertices);
  result_distances.resize(num_vertices);

  // Temporary buffer for float distances
  vector<float> distances_float(num_vertices);
  cudaMemcpy(distances_float.data(), d_distances,
             num_vertices * sizeof(float), cudaMemcpyDeviceToHost);

  // Convert to int64 and populate results
  for (int64_t i = 0; i < num_vertices; i++) {
    result_vertices[i] = i;
    // Convert float distance to int64
    result_distances[i] = static_cast<int64_t>(distances_float[i]);
  }

  SIRIUS_LOG_INFO("SSSP results computed for {} vertices", num_vertices);
}

GraphAlgorithmType
GPUGraphTraversalOperator::StringToAlgorithmType(const string& algo_str) {
  if (algo_str == "EDGE_TRAVERSAL") return GraphAlgorithmType::EDGE_TRAVERSAL;
  if (algo_str == "BFS") return GraphAlgorithmType::BFS;
  if (algo_str == "UNWEIGHTED_SHORTEST_PATH") return GraphAlgorithmType::UNWEIGHTED_SHORTEST_PATH;
  if (algo_str == "WEIGHTED_SHORTEST_PATH") return GraphAlgorithmType::WEIGHTED_SHORTEST_PATH;
  if (algo_str == "SHORTEST_DISTANCE") return GraphAlgorithmType::SHORTEST_DISTANCE;

  throw NotImplementedException("Unknown algorithm type: " + algo_str);
}

} // namespace duckdb