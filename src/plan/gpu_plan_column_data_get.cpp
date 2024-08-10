#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_column_data_get.hpp"

#include "gpu_physical_plan_generator.hpp"
#include "gpu_physical_column_data_scan.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalColumnDataGet &op) {
	D_ASSERT(op.children.size() == 0);
	D_ASSERT(op.collection);

	return make_uniq<GPUPhysicalColumnDataScan>(op.types, PhysicalOperatorType::COLUMN_DATA_SCAN, op.estimated_cardinality,
	                                         std::move(op.collection));
}

} // namespace duckdb
