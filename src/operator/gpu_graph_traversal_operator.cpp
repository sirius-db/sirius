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
  int64_t source_vertex,
  std::vector<int64_t> source_vertices,
  int64_t dest_vertex,
  std::vector<int64_t> dest_vertices,
  const string& weight_col,
  const string& algo_str,
  string path_pattern,
  int max_hops,
  std::vector<string> output_columns,
  ClientContext& context,
  GPUContext& gpu_context
) : GPUPhysicalOperator(
    PhysicalOperatorType::EXTENSION,
    vector<LogicalType>{LogicalType::BIGINT, LogicalType::BIGINT},  // default: vertex_id, distance
    0  // estimated_cardinality
  ),
    child(std::move(child_op)),
    source_vertex(source_vertex),
    source_vertices(source_vertices),
    dest_vertex(dest_vertex),
    dest_vertices(dest_vertices),
    output_columns(output_columns),
    algorithm_type(StringToAlgorithmType(algo_str)),
    path_pattern(path_pattern),
    max_hops(max_hops),
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
  double* d_weights = csr_op->d_weights;
  int64_t num_vertices = csr_op->offsets.size() - 1;
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
      if (source_vertices.size() > 1) {
        RunMultiSourceBFS(d_offsets, d_indices, num_vertices, num_edges);
      } else {
        RunBFS(d_offsets, d_indices, num_vertices, num_edges);
      }
      break;
    case GraphAlgorithmType::WEIGHTED_SHORTEST_PATH:
      // Get weights from CSR operator
      if (!d_weights) {
        throw InvalidInputException("WEIGHTED_SHORTEST_PATH requires edge weights");
      }
      RunSSSP(d_offsets, d_indices, d_weights, num_vertices, num_edges);
      break;
    default:
      throw InternalException("Unknown algorithm type");
  }

  // Filter out unreachable nodes (on CPU)
  vector<int64_t> filtered_vertices;
  vector<double> filtered_distances;
  vector<int64_t> filtered_predecessors;

  for (int64_t i = 0; i < num_vertices; i++) {
    if (result_distances[i] == std::numeric_limits<double>::max()) {
      continue; // Skip unreachable vertices
    }

    // Apply filtering based on path pattern
    if (algorithm_type == GraphAlgorithmType::BFS) {
      // For BFS, check the path pattern from the parser
      if (path_pattern == "ONE_OR_MORE" && result_distances[i] == 0.0) {
        continue; // Skip source for ->+
      }
      // For ZERO_OR_MORE (->*), include everything (including source)
    } else if (algorithm_type == GraphAlgorithmType::EDGE_TRAVERSAL) {
      // Direct edges only, no source
      if (result_distances[i] != 1.0) {
        continue;
      }
    } else {
      // For other algorithms (SSSP, etc.), skip source
      if (result_distances[i] == 0.0) {
        continue;
      }
    }

    filtered_vertices.push_back(result_vertices[i]);
    filtered_distances.push_back(result_distances[i]);
    if (!result_predecessors.empty() && i < result_predecessors.size()) {
      filtered_predecessors.push_back(result_predecessors[i]);
    }
  }

  // Filter by destination if specified
  if (dest_vertex >= 0) {
    vector<int64_t> dest_filtered_vertices;
    vector<double> dest_filtered_distances;
    vector<int64_t> dest_filtered_predecessors;

    for (size_t i = 0; i < filtered_vertices.size(); i++) {
      if (filtered_vertices[i] == dest_vertex) {
        dest_filtered_vertices.push_back(filtered_vertices[i]);
        dest_filtered_distances.push_back(filtered_distances[i]);
        if (!result_predecessors.empty() && i < filtered_predecessors.size()) {
          dest_filtered_predecessors.push_back(filtered_predecessors[i]);
        }
      }
    }

    filtered_vertices = std::move(dest_filtered_vertices);
    filtered_distances = std::move(dest_filtered_distances);
    filtered_predecessors = std::move(dest_filtered_predecessors);

    SIRIUS_LOG_DEBUG("Filtered to destination vertex {}: {} results",
                     dest_vertex, filtered_vertices.size());
  }

  result_vertices = std::move(filtered_vertices);
  result_distances = std::move(filtered_distances);
  result_predecessors = std::move(filtered_predecessors);

  size_t num_results = filtered_vertices.size();
  SIRIUS_LOG_DEBUG("Graph traversal: {} reachable vertices out of {}", num_results, num_vertices);

  BuildOutputRelation(output_relation, num_results);
  SIRIUS_LOG_DEBUG("Graph traversal returned {} results", result_vertices.size());

  return OperatorResultType::FINISHED;
}

SourceResultType
GPUGraphTraversalOperator::GetData(GPUIntermediateRelation& output_relation) const {

  SIRIUS_LOG_INFO("Graph Traversal: GetData called");

  // Execute child to get CSR data
  GPUIntermediateRelation csr_data(2);
  auto child_result = child->GetData(csr_data);

  if (child_result != SourceResultType::FINISHED) {
    return child_result;
  }

  // Get CSR pointers from child output
  auto csr_op = dynamic_cast<GPUCSRConstructionOperator*>(child.get());
  if (!csr_op) {
    throw InternalException("Child operator is not GPUCSRConstructionOperator");
  }

  int64_t* d_offsets = csr_op->d_offsets;
  int64_t* d_indices = csr_op->d_indices;
  double* d_weights = csr_op->d_weights;
  int64_t num_vertices = csr_op->offsets.size() - 1;
  int64_t num_edges = csr_op->indices.size();

  SIRIUS_LOG_DEBUG("CSR: {} vertices, {} edges", num_vertices, num_edges);

  // Initialize cuGraph if needed
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
      if (source_vertices.size() > 1) {
        RunMultiSourceBFS(d_offsets, d_indices, num_vertices, num_edges);
      } else {
        RunBFS(d_offsets, d_indices, num_vertices, num_edges);
      }
      break;
    case GraphAlgorithmType::WEIGHTED_SHORTEST_PATH:
      if (!d_weights) {
        throw InvalidInputException("WEIGHTED_SHORTEST_PATH requires edge weights");
      }
      RunSSSP(d_offsets, d_indices, d_weights, num_vertices, num_edges);
      break;
    default:
      throw InternalException("Unknown algorithm type");
  }

  // Filter unreachable vertices (on CPU)
  vector<int64_t> filtered_vertices;
  vector<double> filtered_distances;
  vector<int64_t> filtered_predecessors;
  if (result_vertices.size() != result_distances.size()) {
    throw InternalException(
      StringUtil::Format("Vertex/distance size mismatch: %zu vs %zu",
                        result_vertices.size(), result_distances.size())
    );
  }

  for (size_t i = 0; i < result_vertices.size(); i++) {
    if (result_distances[i] == std::numeric_limits<double>::max()) {
      continue; // Skip unreachable vertices
    }

    // Apply filtering based on path pattern
    if (algorithm_type == GraphAlgorithmType::BFS) {
      // For BFS, check the path pattern from the parser
      if (path_pattern == "ONE_OR_MORE" && result_distances[i] == 0.0) {
        continue; // Skip source for ->+
      }
      // For ZERO_OR_MORE (->*), include everything (including source)
    } else if (algorithm_type == GraphAlgorithmType::EDGE_TRAVERSAL) {
      // Direct edges only, no source
      if (result_distances[i] != 1.0) {
        continue;
      }
    } else {
      // For other algorithms (SSSP, etc.), skip source
      if (result_distances[i] == 0.0) {
        continue;
      }
    }

    filtered_vertices.push_back(result_vertices[i]);
    filtered_distances.push_back(result_distances[i]);
    if (!result_predecessors.empty() && i < result_predecessors.size()) {
      filtered_predecessors.push_back(result_predecessors[i]);
    }
  }

  // Filter by destination if specified
  if (dest_vertex >= 0) {
    vector<int64_t> dest_filtered_vertices;
    vector<double> dest_filtered_distances;
    vector<int64_t> dest_filtered_predecessors;

    for (size_t i = 0; i < filtered_vertices.size(); i++) {
      if (filtered_vertices[i] == dest_vertex) {
        dest_filtered_vertices.push_back(filtered_vertices[i]);
        dest_filtered_distances.push_back(filtered_distances[i]);
        if (!result_predecessors.empty() && i < filtered_predecessors.size()) {
          dest_filtered_predecessors.push_back(filtered_predecessors[i]);
        }
      }
    }

    filtered_vertices = std::move(dest_filtered_vertices);
    filtered_distances = std::move(dest_filtered_distances);
    filtered_predecessors = std::move(dest_filtered_predecessors);

    SIRIUS_LOG_DEBUG("Filtered to destination vertex {}: {} results",
                     dest_vertex, filtered_vertices.size());
  }

  size_t num_results = filtered_vertices.size();
  SIRIUS_LOG_DEBUG("Graph traversal: {} reachable vertices", num_results);

  result_vertices = std::move(filtered_vertices);
  result_distances = std::move(filtered_distances);
  result_predecessors = std::move(filtered_predecessors);
  BuildOutputRelation(output_relation, num_results);
  SIRIUS_LOG_INFO("Graph traversal complete: {} results", num_results);

  return SourceResultType::FINISHED;
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

  auto csr_op = dynamic_cast<GPUCSRConstructionOperator*>(child.get());
  if (!csr_op) {
    throw InternalException("Child operator is not GPUCSRConstructionOperator");
  }

  // Validate source vertex
  int64_t source_index = -1;
  for (size_t i = 0; i < csr_op->vertex_id_map.size(); i++) {
    if (csr_op->vertex_id_map[i] == source_vertex) {
      source_index = i;
      break;
    }
  }
  if (source_index == -1) {
    throw InvalidInputException(
      StringUtil::Format("Source vertex %lld not found in graph", source_vertex)
    );
  }
  SIRIUS_LOG_DEBUG("Mapped source vertex {} to index {}", source_vertex, source_index);

  // Get offset range for this vertex in CSR
  vector<int64_t> h_offsets(2);
  cudaMemcpy(h_offsets.data(), &d_offsets[source_index],
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
  vector<int64_t> dest_indices(num_edges_from_source);
  cudaMemcpy(dest_indices.data(), &d_indices[edge_start],
             num_edges_from_source * sizeof(int64_t), cudaMemcpyDeviceToHost);

  // Map destination indices back to original IDs
  result_vertices.resize(num_edges_from_source);
  for (int64_t i = 0; i < num_edges_from_source; i++) {
    result_vertices[i] = csr_op->vertex_id_map[dest_indices[i]];
  }

  // All edges have distance 1 (one hop)
  result_distances.resize(num_edges_from_source);
  std::fill(result_distances.begin(), result_distances.end(), 1.0);

  bool need_predecessors = std::find(output_columns.begin(), output_columns.end(), "predecessor") != output_columns.end();

  if (need_predecessors) {
    result_predecessors.resize(num_edges_from_source);
    std::fill(result_predecessors.begin(), result_predecessors.end(), source_vertex);
    SIRIUS_LOG_DEBUG("Set {} predecessors (all pointing to source vertex {})",
                     num_edges_from_source, source_vertex);
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

  auto* handle = static_cast<raft::handle_t*>(cugraph_handle);
  auto csr_op = dynamic_cast<GPUCSRConstructionOperator*>(child.get());
  if (!csr_op) {
    throw InternalException("Child operator is not GPUCSRConstructionOperator");
  }

  // Convert source vertex ID to array index
  int64_t source_index = -1;
  for (size_t i = 0; i < csr_op->vertex_id_map.size(); i++) {
    if (csr_op->vertex_id_map[i] == source_vertex) {
      source_index = i;
      break;
    }
  }
  if (source_index == -1) {
    throw InvalidInputException(
      StringUtil::Format("Source vertex %lld not found in graph", source_vertex)
    );
  }

  SIRIUS_LOG_DEBUG("Mapped source vertex {} to index {}", source_vertex, source_index);

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
  int64_t* d_sources = gpuBufferManager->customCudaMalloc<int64_t>(1, 0, 0);
  cudaMemcpy(d_sources, &source_index, sizeof(int64_t), cudaMemcpyHostToDevice);

  // Run BFS (TODO: BFS supports multiple sources, i.e., MATCH (p:Person WHERE p.id IN [14, 25])-[:knows]->*(p2:Person))
  cugraph::bfs(
    *handle,
    graph_view,
    d_distances,
    d_predecessors,
    d_sources,
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
    result_vertices[i] = csr_op->vertex_id_map[i];
  }

  // Copy distances from GPU to CPU
  vector<int64_t> distances_int(num_vertices);
  cudaMemcpy(distances_int.data(), d_distances,
             num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost);
  for (int64_t i = 0; i < num_vertices; i++) {
    if (distances_int[i] == std::numeric_limits<int64_t>::max()) {
      result_distances[i] = std::numeric_limits<double>::max();
    } else {
      result_distances[i] = static_cast<double>(distances_int[i]);
    }
  }

  bool need_predecessors = std::find(output_columns.begin(), output_columns.end(), "predecessor") != output_columns.end();

  if (need_predecessors) {
    result_predecessors.resize(num_vertices);
    vector<int64_t> predecessors_indices(num_vertices);

    // Copy predecessor indices from GPU
    cudaMemcpy(predecessors_indices.data(), d_predecessors,
               num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Map predecessor indices back to original vertex IDs
    for (int64_t i = 0; i < num_vertices; i++) {
      int64_t pred_index = predecessors_indices[i];
      if (pred_index >= 0 && pred_index < num_vertices) {
        result_predecessors[i] = csr_op->vertex_id_map[pred_index];
      } else {
        result_predecessors[i] = -1;  // No predecessor (unreachable or source)
      }
    }
    SIRIUS_LOG_DEBUG("Copied {} predecessors for path reconstruction", num_vertices);
  }

  SIRIUS_LOG_INFO("BFS results: {} vertices reached",
                  std::count_if(result_distances.begin(), result_distances.end(),
                                [](int64_t d) { return d != std::numeric_limits<int64_t>::max(); }));
}

void
GPUGraphTraversalOperator::RunMultiSourceBFS(
    int64_t* d_offsets,
    int64_t* d_indices,
    int64_t num_vertices,
    int64_t num_edges) const {

  SIRIUS_LOG_INFO("Running multi-source BFS from {} sources", source_vertices.size());

  auto* handle = static_cast<raft::handle_t*>(cugraph_handle);
  auto csr_op = dynamic_cast<GPUCSRConstructionOperator*>(child.get());
  if (!csr_op) {
    throw InternalException("Child operator is not GPUCSRConstructionOperator");
  }

  // Convert all source vertex IDs to array indices
  std::vector<int64_t> source_indices;
  for (int64_t src_id : source_vertices) {
    int64_t src_index = -1;
    for (size_t i = 0; i < csr_op->vertex_id_map.size(); i++) {
      if (csr_op->vertex_id_map[i] == src_id) {
        src_index = i;
        break;
      }
    }
    if (src_index == -1) {
      throw InvalidInputException(
        StringUtil::Format("Source vertex %lld not found in graph", src_id)
      );
    }
    source_indices.push_back(src_index);
  }

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

  // Copy sources to GPU
  int64_t* d_sources = gpuBufferManager->customCudaMalloc<int64_t>(source_vertices.size(), 0, 0);
  cudaMemcpy(d_sources, source_indices.data(),
             source_indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

  // Run multi-source BFS
  cugraph::bfs(
    *handle,
    graph_view,
    d_distances,
    d_predecessors,
    d_sources,
    source_vertices.size(),  // Number of sources
    false,
    std::numeric_limits<int64_t>::max()
  );

  SIRIUS_LOG_INFO("Multi-source BFS complete");

  // Copy results back to CPU
  result_vertices.resize(num_vertices);
  result_distances.resize(num_vertices);

  for (int64_t i = 0; i < num_vertices; i++) {
    result_vertices[i] = csr_op->vertex_id_map[i];
  }

  vector<int64_t> distances_int(num_vertices);
  cudaMemcpy(distances_int.data(), d_distances,
             num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost);
  for (int64_t i = 0; i < num_vertices; i++) {
    if (distances_int[i] == std::numeric_limits<int64_t>::max()) {
      result_distances[i] = std::numeric_limits<double>::max();
    } else {
      result_distances[i] = static_cast<double>(distances_int[i]);
    }
  }

  bool need_predecessors = std::find(output_columns.begin(), output_columns.end(), "predecessor") != output_columns.end();

  if (need_predecessors) {
    result_predecessors.resize(num_vertices);
    vector<int64_t> predecessors_indices(num_vertices);

    // Copy predecessor indices from GPU
    cudaMemcpy(predecessors_indices.data(), d_predecessors,
               num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Map predecessor indices back to original vertex IDs
    for (int64_t i = 0; i < num_vertices; i++) {
      int64_t pred_index = predecessors_indices[i];
      if (pred_index >= 0 && pred_index < num_vertices) {
        result_predecessors[i] = csr_op->vertex_id_map[pred_index];
      } else {
        result_predecessors[i] = -1;  // No predecessor (unreachable or source)
      }
    }
    SIRIUS_LOG_DEBUG("Copied {} predecessors for path reconstruction", num_vertices);

  }

  SIRIUS_LOG_INFO("Multi-source BFS results: {} vertices reached",
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
  auto csr_op = dynamic_cast<GPUCSRConstructionOperator*>(child.get());
  if (!csr_op) {
    throw InternalException("Child operator is not GPUCSRConstructionOperator");
  }

  // Convert source vertex ID to array index
  int64_t source_index = -1;
  for (size_t i = 0; i < csr_op->vertex_id_map.size(); i++) {
    if (csr_op->vertex_id_map[i] == source_vertex) {
      source_index = i;
      break;
    }
  }
  if (source_index == -1) {
    throw InvalidInputException(
      StringUtil::Format("Source vertex %lld not found in graph", source_vertex)
    );
  }

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

  // Allocate and fill weights array
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
    edge_weights,      // edge weights
    d_distances,        // output: distances from source
    d_predecessors,     // output: predecessor in shortest path tree
    source_index,       // source vertex mapped index
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

  // Convert to int64 and populate results with mapped vertex IDs
  for (int64_t i = 0; i < num_vertices; i++) {
    result_vertices[i] = csr_op->vertex_id_map[i];
    if (distances_float[i] == std::numeric_limits<float>::max()) {
      result_distances[i] = std::numeric_limits<double>::max();
    } else {
      result_distances[i] = static_cast<double>(distances_float[i]);
    }
  }

  bool need_predecessors = std::find(output_columns.begin(), output_columns.end(), "predecessor") != output_columns.end();

  if (need_predecessors) {
    result_predecessors.resize(num_vertices);
    vector<int64_t> predecessors_indices(num_vertices);

    // Copy predecessor indices from GPU
    cudaMemcpy(predecessors_indices.data(), d_predecessors,
               num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Map predecessor indices back to original vertex IDs
    for (int64_t i = 0; i < num_vertices; i++) {
      int64_t pred_index = predecessors_indices[i];
      if (pred_index >= 0 && pred_index < num_vertices) {
        result_predecessors[i] = csr_op->vertex_id_map[pred_index];
      } else {
        result_predecessors[i] = -1;  // No predecessor (unreachable or source)
      }
    }

    SIRIUS_LOG_DEBUG("Copied {} predecessors for path reconstruction", num_vertices);
  }

  SIRIUS_LOG_INFO("SSSP results computed for {} vertices", num_vertices);
}

void
GPUGraphTraversalOperator::BuildOutputRelation(
    GPUIntermediateRelation& output_relation,
    size_t num_results) const {

  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  output_relation.columns.clear();

  std::vector<string> cols_to_output = output_columns.empty()
    ? vector<string>{"vertex_id", "distance"}
    : output_columns;

  for (const auto& col_spec : cols_to_output) {
    // Extract field name from dotted notation
    string col_name = col_spec;
    size_t dot_pos = col_name.find(".");
    if (dot_pos != string::npos) {
      col_name = col_name.substr(dot_pos + 1);
    }

    // Allocate and copy vertices
    if (col_name == "id" || col_name == "vertex_id") {
      int64_t* d_result_vertices = gpuBufferManager->customCudaMalloc<int64_t>(num_results, 0, 0);
      cudaMemcpy(d_result_vertices, result_vertices.data(),
                 num_results * sizeof(int64_t), cudaMemcpyHostToDevice);

      auto col = make_shared_ptr<GPUColumn>(
        num_results,
        GPUColumnType(GPUColumnTypeId::INT64),
        reinterpret_cast<uint8_t*>(d_result_vertices),
        nullptr
      );
      output_relation.columns.push_back(col);
    }

    // Allocate and copy distances
    else if (col_name == "distance" || col_name == "weight" || col_name == "cost") {
      bool is_weighted = algorithm_type == GraphAlgorithmType::WEIGHTED_SHORTEST_PATH;

      if (is_weighted) {
        // Weighted: return as DOUBLE
        double* d_result_distances = gpuBufferManager->customCudaMalloc<double>(num_results, 0, 0);
        cudaMemcpy(d_result_distances, result_distances.data(),
                   num_results * sizeof(double), cudaMemcpyHostToDevice);

        auto col = make_shared_ptr<GPUColumn>(
          num_results,
          GPUColumnType(GPUColumnTypeId::FLOAT64),
          reinterpret_cast<uint8_t*>(d_result_distances),
          nullptr
        );
        output_relation.columns.push_back(col);
      } else {
        // Unweighted: return as INT64 (hop count)
        vector<int64_t> distances_int(num_results);
        for (size_t i = 0; i < num_results; i++) {
          distances_int[i] = static_cast<int64_t>(result_distances[i]);
        }

        int64_t* d_result_distances = gpuBufferManager->customCudaMalloc<int64_t>(num_results, 0, 0);
        cudaMemcpy(d_result_distances, distances_int.data(),
                   num_results * sizeof(int64_t), cudaMemcpyHostToDevice);

        auto col = make_shared_ptr<GPUColumn>(
          num_results,
          GPUColumnType(GPUColumnTypeId::INT64),
          reinterpret_cast<uint8_t*>(d_result_distances),
          nullptr
        );
        output_relation.columns.push_back(col);
      }
    }

    // Only add if predecessor is available
    else if (col_name == "predecessor") {
      if (result_predecessors.empty() || result_predecessors.size() != num_results) {
        // Create empty predecessor column
        SIRIUS_LOG_DEBUG("Creating empty predecessor column for {} results", num_results);

        int64_t* d_result_predecessors = gpuBufferManager->customCudaMalloc<int64_t>(
          num_results > 0 ? num_results : 1, 0, 0  // Allocate at least 1 element
        );

        auto col = make_shared_ptr<GPUColumn>(
          num_results,
          GPUColumnType(GPUColumnTypeId::INT64),
          reinterpret_cast<uint8_t*>(d_result_predecessors),
          nullptr
        );
        output_relation.columns.push_back(col);
      } else {
        // Predecessors are available
        int64_t* d_result_predecessors = gpuBufferManager->customCudaMalloc<int64_t>(num_results, 0, 0);
        cudaMemcpy(d_result_predecessors, result_predecessors.data(),
                   num_results * sizeof(int64_t), cudaMemcpyHostToDevice);

        auto col = make_shared_ptr<GPUColumn>(
          num_results,
          GPUColumnType(GPUColumnTypeId::INT64),
          reinterpret_cast<uint8_t*>(d_result_predecessors),
          nullptr
        );
        output_relation.columns.push_back(col);
      }
    }
  }

  SIRIUS_LOG_DEBUG("Built {} output columns", output_relation.columns.size());
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