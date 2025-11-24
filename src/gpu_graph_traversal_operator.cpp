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
  const string& algo,
  int hops,
  ClientContext& context,
  GPUContext& gpu_context)
: GPUPhysicalOperator(
    PhysicalOperatorType::EXTENSION,
    vector<LogicalType>{LogicalType::BIGINT, LogicalType::BIGINT},  // vertex_id, distance
    0  // estimated_cardinality
  ),
    child(std::move(child_op)),
    source_vertex(source),
    algorithm_type(algo),
    max_hops(hops) {
}

GPUGraphTraversalOperator::~GPUGraphTraversalOperator() {
  if (handle_initialized && cugraph_handle) {
    auto* handle = static_cast<raft::handle_t*>(cugraph_handle);
    delete handle;
    cugraph_handle = nullptr;
  }
}

OperatorResultType GPUGraphTraversalOperator::Execute(
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
  auto offsets_col = csr_data.columns[0];
  auto indices_col = csr_data.columns[1];

  int64_t* d_offsets = reinterpret_cast<int64_t*>(offsets_col->GetData());
  int64_t* d_indices = reinterpret_cast<int64_t*>(indices_col->GetData());
  int64_t num_vertices = offsets_col->column_length - 1;
  int64_t num_edges = indices_col->column_length;

  SIRIUS_LOG_INFO("Graph Traversal: Running {} on graph with {} vertices, {} edges",
                  algorithm_type, num_vertices, num_edges);

  // Call cuGraph BFS/SSSP
  // TODO: Implement cuGraph call here
  // For now, just create empty output

  SIRIUS_LOG_WARN("Graph Traversal: cuGraph integration not yet implemented");

  // Set up output with results
  output_relation.column_count = 2;
  output_relation.columns.resize(2);

  // Placeholder: return empty result
  // Later: return actual BFS/SSSP results
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
    cugraph::graph_properties_t{false, false} // not symmetric, not multigraph
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
    d_distances,        // output: distances from source
    d_predecessors,     // output: predecessor in BFS tree
    &source_vertex,     // source vertices (can be multiple)
    1,                  // number of sources
    false,              // direction_optimizing
    std::numeric_limits<int64_t>::max()  // depth_limit
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

  SIRIUS_LOG_INFO("BFS results: {} vertices reached",
                  std::count_if(result_distances.begin(), result_distances.end(),
                                [](int64_t d) { return d != std::numeric_limits<int64_t>::max(); }));
}

void
GPUGraphTraversalOperator::RunSSSP(
    int64_t* d_offsets,
    int64_t* d_indices,
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

  // TODO: handle and add a third column for weight in the edge table
  // Allocate and fill weights array with 1.0
  float* d_weights = gpuBufferManager->customCudaMalloc<float>(num_edges, 0, false);

  // Create a host array of 1.0f values and copy
  std::vector<float> weights_host(num_edges, 1.0f);
  cudaMemcpy(d_weights, weights_host.data(), num_edges * sizeof(float), cudaMemcpyHostToDevice);

  // Wrap in edge_property_view with vectors
  std::vector<float const*> weight_ptrs = {d_weights};
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
    // Convert float distance to int64 (round or cast as appropriate)
    result_distances[i] = static_cast<int64_t>(distances_float[i]);
  }

  SIRIUS_LOG_INFO("SSSP results computed for {} vertices", num_vertices);
}

} // namespace duckdb