#include "gpu_physical_dummy_scan.hpp"

namespace duckdb {

// SourceResultType GPUPhysicalDummyScan::GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation,
//                                             OperatorSourceInput &input) const {
SourceResultType 
GPUPhysicalDummyScan::GetData(GPUIntermediateRelation &output_relation) const {
	// return a single row on the first call to the dummy scan
	// chunk.SetCardinality(1);

    printf("Reading data from dummy scan\n");
    for (int col_idx = 0; col_idx < output_relation.columns.size(); col_idx++) {
        output_relation.columns[col_idx] = nullptr;
    }

	return SourceResultType::FINISHED;
}

} // namespace duckdb
