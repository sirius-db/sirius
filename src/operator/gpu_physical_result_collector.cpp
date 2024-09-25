// #include "duckdb/execution/operator/helper/physical_result_collector.hpp"

// #include "duckdb/execution/operator/helper/physical_batch_collector.hpp"
// #include "duckdb/execution/operator/helper/physical_materialized_collector.hpp"
// #include "duckdb/execution/operator/helper/physical_buffered_collector.hpp"
// #include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/config.hpp"
// #include "duckdb/main/prepared_statement_data.hpp"
// #include "duckdb/parallel/meta_pipeline.hpp"
// #include "duckdb/parallel/pipeline.hpp"

#include "operator/gpu_physical_result_collector.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "duckdb/main/prepared_statement_data.hpp"

namespace duckdb {

GPUPhysicalResultCollector::GPUPhysicalResultCollector(GPUPreparedStatementData &data)
    : GPUPhysicalOperator(PhysicalOperatorType::RESULT_COLLECTOR, {LogicalType::BOOLEAN}, 0),
      statement_type(data.prepared->statement_type), properties(data.prepared->properties), plan(*data.gpu_physical_plan), names(data.prepared->names) {
	this->types = data.prepared->types;
}

// unique_ptr<GPUPhysicalResultCollector> GPUPhysicalResultCollector::GetResultCollector(ClientContext &context,
//                                                                                 PreparedStatementData &data) {
// 	if (!PhysicalPlanGenerator::PreserveInsertionOrder(context, *data.plan)) {
// 		// the plan is not order preserving, so we just use the parallel materialized collector
// 		if (data.is_streaming) {
// 			return make_uniq_base<GPUPhysicalResultCollector, PhysicalBufferedCollector>(data, true);
// 		}
// 		return make_uniq_base<PhysicalResultCollector, PhysicalMaterializedCollector>(data, true);
// 	} else if (!PhysicalPlanGenerator::UseBatchIndex(context, *data.plan)) {
// 		// the plan is order preserving, but we cannot use the batch index: use a single-threaded result collector
// 		if (data.is_streaming) {
// 			return make_uniq_base<GPUPhysicalResultCollector, PhysicalBufferedCollector>(data, false);
// 		}
// 		return make_uniq_base<PhysicalResultCollector, PhysicalMaterializedCollector>(data, false);
// 	} else {
// 		// we care about maintaining insertion order and the sources all support batch indexes
// 		// use a batch collector
// 		if (data.is_streaming) {
// 			return make_uniq_base<GPUPhysicalResultCollector, PhysicalBufferedCollector>(data, false);
// 		}
// 		return make_uniq_base<GPUPhysicalResultCollector, PhysicalBatchCollector>(data);
// 	}
// }

vector<const_reference<GPUPhysicalOperator>> 
GPUPhysicalResultCollector::GetChildren() const {
	return {plan};
}

void 
GPUPhysicalResultCollector::BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) {
	// operator is a sink, build a pipeline
	sink_state.reset();

	D_ASSERT(children.empty());

	// single operator: the operator becomes the data source of the current pipeline
	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	// we create a new pipeline starting from the child
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	child_meta_pipeline.Build(plan);
}

GPUPhysicalMaterializedCollector::GPUPhysicalMaterializedCollector(GPUPreparedStatementData &data)
    : GPUPhysicalResultCollector(data) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class GPUMaterializedCollectorGlobalState : public GlobalSinkState {
public:
	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;
};

class GPUMaterializedCollectorLocalState : public LocalSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

SinkResultType GPUPhysicalMaterializedCollector::Sink(GPUIntermediateRelation &input_relation) const {
	// auto &lstate = input.local_state.Cast<MaterializedCollectorLocalState>();
	// lstate.collection->Append(lstate.append_state, chunk);
	return SinkResultType::FINISHED;
}

// SinkCombineResultType PhysicalMaterializedCollector::Combine(ExecutionContext &context,
//                                                              OperatorSinkCombineInput &input) const {
// 	auto &gstate = input.global_state.Cast<MaterializedCollectorGlobalState>();
// 	auto &lstate = input.local_state.Cast<MaterializedCollectorLocalState>();
// 	if (lstate.collection->Count() == 0) {
// 		return SinkCombineResultType::FINISHED;
// 	}

// 	lock_guard<mutex> l(gstate.glock);
// 	if (!gstate.collection) {
// 		gstate.collection = std::move(lstate.collection);
// 	} else {
// 		gstate.collection->Combine(*lstate.collection);
// 	}

// 	return SinkCombineResultType::FINISHED;
// }

unique_ptr<GlobalSinkState> GPUPhysicalMaterializedCollector::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<GPUMaterializedCollectorGlobalState>();
	state->context = context.shared_from_this();
	return std::move(state);
}

unique_ptr<LocalSinkState> GPUPhysicalMaterializedCollector::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<GPUMaterializedCollectorLocalState>();
	state->collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

unique_ptr<QueryResult> GPUPhysicalMaterializedCollector::GetResult(GlobalSinkState &state) {
	auto &gstate = state.Cast<GPUMaterializedCollectorGlobalState>();
	// Currently the result will be empty
	if (!gstate.collection) {
		gstate.collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	}
	// printf("I am here\n");
	if (!gstate.context) throw InvalidInputException("No context set in GPUMaterializedCollectorState");
	if (!gstate.collection) throw InvalidInputException("No context set in GPUMaterializedCollectorState");
	auto prop = gstate.context->GetClientProperties();
	// printf("I am here\n");
	auto result = make_uniq<MaterializedQueryResult>(statement_type, properties, names, std::move(gstate.collection),
	                                                 prop);
	// printf("I am here\n");
	return std::move(result);
}

// bool PhysicalMaterializedCollector::ParallelSink() const {
// 	return parallel;
// }

// bool PhysicalMaterializedCollector::SinkOrderDependent() const {
// 	return true;
// }

} // namespace duckdb