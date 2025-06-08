#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_expression_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_expression_get.hpp"
// #include "gpu_physical_expression_scan.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "gpu_physical_column_data_scan.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "log/logging.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalExpressionGet &op) {
	D_ASSERT(op.children.size() == 1);
	auto plan = CreatePlan(*op.children[0]);

	// auto expr_scan = make_uniq<PhysicalExpressionScan>(op.types, std::move(op.expressions), op.estimated_cardinality);
	// expr_scan->children.push_back(std::move(plan));
	// if (!expr_scan->IsFoldable()) {
	// 	return std::move(expr_scan);
	// }
	// auto &allocator = Allocator::Get(context);
	// simple expression scan (i.e. no subqueries to evaluate and no prepared statement parameters)
	// we can evaluate all the expressions right now and turn this into a chunk collection scan
	auto chunk_scan = make_uniq<GPUPhysicalColumnDataScan>(op.types, PhysicalOperatorType::COLUMN_DATA_SCAN,
	                                                    op.expressions.size(),
	                                                    make_uniq<ColumnDataCollection>(context, op.types));

	// DataChunk chunk;
	// chunk.Initialize(allocator, op.types);

	// ColumnDataAppendState append_state;
	// chunk_scan->collection->InitializeAppend(append_state);
	// for (idx_t expression_idx = 0; expression_idx < expr_scan->expressions.size(); expression_idx++) {
	// 	chunk.Reset();
	// 	expr_scan->EvaluateExpression(context, expression_idx, nullptr, chunk);
	// 	chunk_scan->collection->Append(append_state, chunk);
	// }
	// return std::move(chunk_scan);
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);
    for (idx_t expression_idx = 0; expression_idx < op.expressions.size(); expression_idx++) {
        SIRIUS_LOG_DEBUG("Expression idx: {}", expression_idx);
        if (op.expressions[expression_idx].size() > 1) {
            throw NotImplementedException("Expression get not supported");
        }
        uint64_t* h_data = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        uint64_t* d_data = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
        if (op.expressions[expression_idx][0]->type == ExpressionType::VALUE_CONSTANT) {
            auto &constant_expr = op.expressions[expression_idx][0]->Cast<BoundConstantExpression>();
            h_data[0] = constant_expr.value.GetValue<uint64_t>();
            // callCudaMemcpy(d_data, h_data, 1 * sizeof(uint64_t), cudaMemcpyHostToDevice);
            callCudaMemcpyHostToDevice<uint64_t>(d_data, h_data, 1, 0);
        } else {
            throw NotImplementedException("Expression get not supported");
        }
        chunk_scan->intermediate_relation = make_shared_ptr<GPUIntermediateRelation>(1);
        chunk_scan->intermediate_relation->columns[0] = make_shared_ptr<GPUColumn>(1, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(d_data));
    }
    return std::move(chunk_scan);
}

} // namespace duckdb
