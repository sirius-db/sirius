#include "gpu_physical_dummy_scan.hpp"

namespace duckdb {

SourceResultType 
GPUPhysicalDummyScan::GetData(GPUIntermediateRelation &output_relation) const {

    printf("Reading data from dummy scan\n");
    for (int col_idx = 0; col_idx < output_relation.columns.size(); col_idx++) {
        output_relation.columns[col_idx] = nullptr;
    }

	return SourceResultType::FINISHED;
}

} // namespace duckdb
