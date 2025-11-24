//
// Created by andy on 11/23/25.
//

#pragma once

#include "gpu_physical_operator.hpp"
#include "gpu_csr_construction_operator.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <cugraph_c/resource_handle.h>
#include <cugraph_c/graph.h>

namespace duckdb {

class GPUGraphTraversalOperator : public GPUPhysicalOperator {
public:
  unique_ptr<GPUPhysicalOperator> child;  // CSR construction operator
  int64_t source_vertex;
  string algorithm_type;
  int max_hops;

  // Result data
  mutable vector<int64_t> result_vertices;
  mutable vector<int64_t> result_distances;

  // cuGraph handles
  mutable void* cugraph_handle = nullptr;
  mutable bool handle_initialized = false;

  GPUGraphTraversalOperator(
    unique_ptr<GPUPhysicalOperator> child,
    int64_t source,
    const string& algo,
    int max_hops,
    ClientContext& context,
    GPUContext& gpu_context
  );

  ~GPUGraphTraversalOperator() override;

  OperatorResultType Execute(
    GPUIntermediateRelation &input_relation,
    GPUIntermediateRelation &output_relation
  ) const override;

private:
  void InitializeCuGraph() const;
  void RunBFS(int64_t* d_offsets, int64_t* d_indices, int64_t num_vertices, int64_t num_edges) const;
  void RunSSSP(int64_t* d_offsets, int64_t* d_indices, int64_t num_vertices, int64_t num_edges) const;
};

} // namespace duckdb