#include "operator/gpu_physical_empty_result.hpp"

namespace duckdb {

SourceResultType 
GPUPhysicalEmptyResult::GetData(GPUIntermediateRelation &output_relation) const {
    printf("Reading data from empty result\n");
    for (int col = 0; col < types.size(); col++) {
        output_relation.columns[col] = make_shared_ptr<GPUColumn>(0, ColumnType::INT64, nullptr);
    }
	return SourceResultType::FINISHED;
}

} // namespace duckdb
