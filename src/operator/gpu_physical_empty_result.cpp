#include "operator/gpu_physical_empty_result.hpp"

namespace duckdb {

SourceResultType 
GPUPhysicalEmptyResult::GetData(GPUIntermediateRelation &output_relation) const {
    printf("Reading data from empty result\n");
	return SourceResultType::FINISHED;
}

} // namespace duckdb
