//
// Created by andy on 11/23/25.
//
#include "gpu_csr_construction_operator.hpp"
#include "log/logging.hpp"
#include <algorithm>
#include <cuda_runtime.h>

namespace duckdb {

GPUCSRConstructionOperator::GPUCSRConstructionOperator(
    unique_ptr<GPUPhysicalOperator> child_op,
    const string& src_col,
    const string& dst_col,
    ClientContext& context,
    GPUContext& gpu_context)
: GPUPhysicalOperator(
      PhysicalOperatorType::EXTENSION,
      vector<LogicalType>{LogicalType::BIGINT, LogicalType::BIGINT},  // offsets, indices
      0  // estimated_cardinality
  ),
      child(std::move(child_op)),
      source_column(src_col),
      dest_column(dst_col),
      num_vertices(0) {
}

GPUCSRConstructionOperator::~GPUCSRConstructionOperator() {
  d_offsets = nullptr;
  d_indices = nullptr;
}


OperatorResultType GPUCSRConstructionOperator::Execute(
  GPUIntermediateRelation &input_relation,
  GPUIntermediateRelation &output_relation) const {

  SIRIUS_LOG_INFO("CSR Construction: Starting execution");

    // Execute child operator (table scan) to get edge data
    if (!csr_built) {
        SIRIUS_LOG_INFO("CSR Construction: Executing child operator (table scan)");

        GPUIntermediateRelation edge_data(2);  // 2 columns: src, dst
        auto result = child->Execute(input_relation, edge_data);

        if (result != OperatorResultType::FINISHED) {
            return result;  // Propagate if not finished
        }

        // Extract src and dst columns from edge_data
        SIRIUS_LOG_INFO("CSR Construction: Extracting edge columns");

        if (edge_data.columns.size() < 2) {
            throw InternalException("Edge table must have at least 2 columns");
        }

        // Get the data from GPU columns
        auto src_column = edge_data.columns[0];
        auto dst_column = edge_data.columns[1];

        size_t num_edges = src_column->column_length;
        SIRIUS_LOG_INFO("CSR Construction: Processing {} edges", num_edges);

        // TODO: Let cuGraph handle CSR construction internally
        // Copy data from GPU to CPU for CSR construction, assuming the data is int64_t
        vector<int64_t> src_cpu(num_edges);
        vector<int64_t> dst_cpu(num_edges);
        cudaMemcpy(src_cpu.data(), src_column->GetData(),
                   num_edges * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(dst_cpu.data(), dst_column->GetData(),
                   num_edges * sizeof(int64_t), cudaMemcpyDeviceToHost);

        // Build CSR on CPU
        BuildCSR(src_cpu, dst_cpu);

        // Transfer CSR to GPU
        TransferCSRToGPU();

        csr_built = true;
        SIRIUS_LOG_INFO("CSR Construction: Complete");
    }

    // Set up output relation with CSR data
    // The output should contain pointers to the CSR structure
    output_relation.column_count = 2;
    output_relation.columns.resize(2);

    // Create columns that hold the CSR pointers
    // Column 0: offsets array
    output_relation.columns[0] = make_shared_ptr<GPUColumn>(
        num_vertices + 1,
        GPUColumnType(GPUColumnTypeId::INT64),
        reinterpret_cast<uint8_t*>(d_offsets),
        nullptr  // no row_ids
    );

    // Column 1: indices array
    output_relation.columns[1] = make_shared_ptr<GPUColumn>(
        indices.size(),
        GPUColumnType(GPUColumnTypeId::INT64),
        reinterpret_cast<uint8_t*>(d_indices),
        nullptr
    );

    return OperatorResultType::FINISHED;
}

void GPUCSRConstructionOperator::BuildCSR(
  const vector<int64_t>& src,
  const vector<int64_t>& dst) const {

  if (src.empty()) {
    SIRIUS_LOG_WARN("CSR Construction: Empty edge list");
    return;
  }

  SIRIUS_LOG_DEBUG("CSR Construction: Building CSR from {} edges", src.size());

  // Find max vertex ID to determine number of vertices
  num_vertices = 0;
  for (auto v : src) {
    num_vertices = std::max(num_vertices, v + 1);
  }
  for (auto v : dst) {
    num_vertices = std::max(num_vertices, v + 1);
  }

  SIRIUS_LOG_DEBUG("CSR Construction: Number of vertices = {}", num_vertices);

  // Initialize offsets array (size = num_vertices + 1)
  offsets.resize(num_vertices + 1, 0);

  // Count out-degree for each vertex
  for (auto v : src) {
    offsets[v + 1]++;
  }

  // Compute prefix sum to get offsets
  for (int64_t i = 1; i <= num_vertices; i++) {
    offsets[i] += offsets[i - 1];
  }

  // Fill in the indices array
  indices.resize(src.size());
  vector<int64_t> temp_offsets = offsets;  // Temporary copy for filling

  for (size_t i = 0; i < src.size(); i++) {
    int64_t v = src[i];
    int64_t pos = temp_offsets[v];
    indices[pos] = dst[i];
    temp_offsets[v]++;
  }

  SIRIUS_LOG_DEBUG("CSR Construction: Offsets size = {}, Indices size = {}", offsets.size(), indices.size());
}

void GPUCSRConstructionOperator::TransferCSRToGPU() const {
  SIRIUS_LOG_INFO("CSR Construction: Transferring CSR to GPU");
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

  // Allocate GPU memory for offsets
  d_offsets = gpuBufferManager->customCudaMalloc<int64_t>(offsets.size(), 0, 0);
  cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

  // Allocate GPU memory for indices
  d_indices = gpuBufferManager->customCudaMalloc<int64_t>(indices.size(), 0, 0);
  cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);


  SIRIUS_LOG_INFO("CSR Construction: Transfer complete");
}

} // namespace duckdb
