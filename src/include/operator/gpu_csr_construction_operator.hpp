//
// Created by andy on 11/23/25.
//

#pragma once

#include "gpu_physical_operator.hpp"
#include "duckdb/common/types.hpp"
#include "gpu_context.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>

namespace duckdb {

class GPUCSRConstructionOperator : public GPUPhysicalOperator {
public:
  unique_ptr<GPUPhysicalOperator> child;
  string source_column;
  string dest_column;

  // CSR data structures (CPU side for now)
  mutable vector<int64_t> offsets;
  mutable vector<int64_t> indices;
  mutable int64_t num_vertices;

  // GPU pointers (will be allocated later)
  mutable int64_t* d_offsets = nullptr;
  mutable int64_t* d_indices = nullptr;

  GPUCSRConstructionOperator(
    unique_ptr<GPUPhysicalOperator> child,
    const string& src_col,
    const string& dst_col,
    ClientContext& context,
    GPUContext& gpu_context
  );

  ~GPUCSRConstructionOperator() override;

  OperatorResultType Execute(
    GPUIntermediateRelation &input_relation,
    GPUIntermediateRelation &output_relation
  ) const override;

private:
  void BuildCSR(const vector<int64_t>& src, const vector<int64_t>& dst) const;
  void TransferCSRToGPU() const;
  mutable bool csr_built = false;
};

} // namespace duckdb