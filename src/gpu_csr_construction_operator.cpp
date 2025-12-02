#include "gpu_csr_construction_operator.hpp"
#include "log/logging.hpp"
#include <algorithm>

namespace duckdb {

GPUCSRConstructionOperator::GPUCSRConstructionOperator(
    unique_ptr<GPUPhysicalOperator> child_op,
    const string& src_col,
    const string& dst_col,
    const string& weight_col,
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
      weight_column(weight_col),
      num_vertices(0),
      has_weights(!weight_col.empty()) {
  SIRIUS_LOG_DEBUG("CSR Construction: weight column = '{}', has_weights = {}",
    weight_column, has_weights);
}

GPUCSRConstructionOperator::~GPUCSRConstructionOperator() {
  // GPUBufferManager manages its lifecycle so no need to free here
  d_offsets = nullptr;
  d_indices = nullptr;
  d_weights = nullptr;
}


SourceResultType
GPUCSRConstructionOperator::GetData(GPUIntermediateRelation& output_relation) const {

  SIRIUS_LOG_INFO("CSR Construction: Starting execution");

    // Execute child operator (table scan) to get edge data
    if (!csr_built) {
      SIRIUS_LOG_INFO("CSR Construction: Executing child operator (table scan)");

      size_t expected_cols = has_weights ? 3 : 2; // 2 or 3 columns: src, dst, (weights)
      GPUIntermediateRelation edge_data(expected_cols);
      auto result = child->GetData(edge_data);
      if (result != SourceResultType::FINISHED) {
        return result;  // Propagate if not finished
      }

      // Extract src and dst columns from edge_data
      SIRIUS_LOG_INFO("CSR Construction: Extracting edge columns");

      if (edge_data.columns.size() < expected_cols) {
        throw InternalException(
        StringUtil::Format("Edge table must have at least %zu columns (src, dst%s)",
                        expected_cols, has_weights ? ", weight" : "")
        );
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

      if (!has_weights) {
        // Build CSR without weights on CPU
        BuildCSR(src_cpu, dst_cpu);
      }
      else {
        auto weight_column = edge_data.columns[2];
        vector<double> weights_cpu(num_edges);

        // Handle different weight types (int32, int64, float, double)
        auto weight_type = weight_column->data_wrapper.type.id();

        if (weight_type == GPUColumnTypeId::FLOAT64) {
          cudaMemcpy(weights_cpu.data(), weight_column->GetData(),
                     num_edges * sizeof(double), cudaMemcpyDeviceToHost);
        } else if (weight_type == GPUColumnTypeId::FLOAT32) {
          vector<float> temp_weights(num_edges);
          cudaMemcpy(temp_weights.data(), weight_column->GetData(),
                     num_edges * sizeof(float), cudaMemcpyDeviceToHost);
          // Convert float to double
          for (size_t i = 0; i < num_edges; i++) {
            weights_cpu[i] = static_cast<double>(temp_weights[i]);
          }
        } else if (weight_type == GPUColumnTypeId::INT64) {
          vector<int64_t> temp_weights(num_edges);
          cudaMemcpy(temp_weights.data(), weight_column->GetData(),
                     num_edges * sizeof(int64_t), cudaMemcpyDeviceToHost);
          // Convert int64 to double
          for (size_t i = 0; i < num_edges; i++) {
            weights_cpu[i] = static_cast<double>(temp_weights[i]);
          }
        } else if (weight_type == GPUColumnTypeId::INT32) {
          vector<int32_t> temp_weights(num_edges);
          cudaMemcpy(temp_weights.data(), weight_column->GetData(),
                     num_edges * sizeof(int32_t), cudaMemcpyDeviceToHost);
          // Convert int32 to double
          for (size_t i = 0; i < num_edges; i++) {
            weights_cpu[i] = static_cast<double>(temp_weights[i]);
          }
        } else {
          throw NotImplementedException(
            StringUtil::Format("Unsupported weight column type: %d",
                              static_cast<int>(weight_type))
          );
        }

        SIRIUS_LOG_DEBUG("CSR Construction: Extracted {} weights", weights_cpu.size());
        BuildCSRWithWeights(src_cpu, dst_cpu, weights_cpu);
      }

      // Transfer CSR to GPU
      TransferCSRToGPU();

      csr_built = true;
      SIRIUS_LOG_INFO("CSR Construction: Complete");
    }

  // Set up output relation with CSR data
  // The output contains pointers to the CSR structure
  output_relation.column_count = has_weights ? 3 : 2;
  output_relation.columns.resize(output_relation.column_count);

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

  // Column 2: weights array (if present)
  if (has_weights) {
    output_relation.columns[2] = make_shared_ptr<GPUColumn>(
        weights.size(),
        GPUColumnType(GPUColumnTypeId::FLOAT64),
        reinterpret_cast<uint8_t*>(d_weights),
        nullptr
    );
  }

  return SourceResultType::FINISHED;
}

// TODO: build csr internally with cuGraph
void
GPUCSRConstructionOperator::BuildCSR(
  const vector<int64_t>& src,
  const vector<int64_t>& dst) const {

  if (src.empty()) {
    SIRIUS_LOG_WARN("CSR Construction: Empty edge list");
    return;
  }

  SIRIUS_LOG_DEBUG("CSR Construction: Building CSR from {} edges", src.size());

  // Collect all unique vertices and create mapping
  std::set<int64_t> unique_vertices;
  for (auto v : src) {
    unique_vertices.insert(v);
  }
  for (auto v : dst) {
    unique_vertices.insert(v);
  }

  // Create the vertex ID mapping (sorted order)
  vertex_id_map.assign(unique_vertices.begin(), unique_vertices.end());
  num_vertices = vertex_id_map.size();

  // Create reverse mapping: original ID -> array index
  std::unordered_map<int64_t, int64_t> id_to_index;
  for (size_t i = 0; i < vertex_id_map.size(); i++) {
    id_to_index[vertex_id_map[i]] = i;
  }

  SIRIUS_LOG_DEBUG("CSR Construction: Number of vertices = {}", num_vertices);

  // Initialize offsets array (size = num_vertices + 1)
  offsets.resize(num_vertices + 1, 0);

  // Count out-degree for each vertex
  for (auto v : src) {
    int64_t index = id_to_index[v];
    offsets[index + 1]++;
  }

  // Compute prefix sum to get offsets
  for (int64_t i = 1; i <= num_vertices; i++) {
    offsets[i] += offsets[i - 1];
  }

  // Fill in the indices array
  indices.resize(src.size());
  vector<int64_t> temp_offsets = offsets;

  for (size_t i = 0; i < src.size(); i++) {
    int64_t src_index = id_to_index[src[i]];
    int64_t dst_index = id_to_index[dst[i]];
    int64_t pos = temp_offsets[src_index];
    indices[pos] = dst_index;  // Store destination as array index
    temp_offsets[src_index]++;
  }

  SIRIUS_LOG_DEBUG("CSR Construction: Offsets size = {}, Indices size = {}",
    offsets.size(), indices.size());
}

void
GPUCSRConstructionOperator::BuildCSRWithWeights(
  const vector<int64_t>& src,
  const vector<int64_t>& dst,
  const vector<double>& weights_in) const {

  if (src.empty()) {
    SIRIUS_LOG_WARN("CSR Construction: Empty edge list");
    return;
  }

  SIRIUS_LOG_DEBUG("CSR Construction: Building CSR (weighted) from {} edges", src.size());

  // Collect all unique vertices and create mapping
  std::set<int64_t> unique_vertices;
  for (auto v : src) {
    unique_vertices.insert(v);
  }
  for (auto v : dst) {
    unique_vertices.insert(v);
  }

  // Create the vertex ID mapping (sorted order)
  vertex_id_map.assign(unique_vertices.begin(), unique_vertices.end());
  num_vertices = vertex_id_map.size();

  // Create reverse mapping: original ID -> array index
  std::unordered_map<int64_t, int64_t> id_to_index;
  for (size_t i = 0; i < vertex_id_map.size(); i++) {
    id_to_index[vertex_id_map[i]] = i;
  }

  SIRIUS_LOG_DEBUG("CSR Construction: Number of vertices = {}", num_vertices);

  // Initialize offsets
  offsets.resize(num_vertices + 1, 0);

  // Count out-degree
  for (auto v : src) {
    int64_t index = id_to_index[v];
    offsets[index + 1]++;
  }

  // Prefix sum
  for (int64_t i = 1; i <= num_vertices; i++) {
    offsets[i] += offsets[i - 1];
  }

  // Fill indices and weights
  indices.resize(src.size());
  weights.resize(src.size());
  vector<int64_t> temp_offsets = offsets;

  for (size_t i = 0; i < src.size(); i++) {
    int64_t src_index = id_to_index[src[i]];
    int64_t dst_index = id_to_index[dst[i]];
    int64_t pos = temp_offsets[src_index];
    indices[pos] = dst_index;  // Store as array index
    weights[pos] = weights_in[i];
    temp_offsets[src_index]++;
  }

  SIRIUS_LOG_DEBUG("CSR Construction: Offsets size = {}, Indices size = {}, Weights size = {}",
                   offsets.size(), indices.size(), weights.size());
}

void
GPUCSRConstructionOperator::TransferCSRToGPU() const {
  SIRIUS_LOG_INFO("CSR Construction: Transferring CSR to GPU");
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

  // Allocate GPU memory for offsets
  d_offsets = gpuBufferManager->customCudaMalloc<int64_t>(offsets.size(), 0, 0);
  cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int64_t),
    cudaMemcpyHostToDevice);

  // Allocate GPU memory for indices
  d_indices = gpuBufferManager->customCudaMalloc<int64_t>(indices.size(), 0, 0);
  cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int64_t),
    cudaMemcpyHostToDevice);


  SIRIUS_LOG_INFO("CSR Construction: Transfer complete");
}

} // namespace duckdb
