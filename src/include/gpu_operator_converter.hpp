#include "gpu_physical_operator.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

class GPUOperatorConverter {

public:
	GPUOperatorConverter();
    GPUPhysicalOperator& ConvertOperator(PhysicalOperator& op);

    PhysicalOperator& original_op;

};

} // namespace duckdb