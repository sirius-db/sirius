#include "gpu_physical_operator.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_executor.hpp"

namespace duckdb {

string GPUPhysicalOperator::GetName() const {
	return PhysicalOperatorToString(type);
}

vector<const_reference<GPUPhysicalOperator>> GPUPhysicalOperator::GetChildren() const {
	vector<const_reference<GPUPhysicalOperator>> result;
	for (auto &child : children) {
		result.push_back(*child);
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
// LCOV_EXCL_START
unique_ptr<OperatorState> GPUPhysicalOperator::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<OperatorState>();
}

unique_ptr<GlobalOperatorState> GPUPhysicalOperator::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<GlobalOperatorState>();
}

OperatorResultType GPUPhysicalOperator::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                             GlobalOperatorState &gstate, OperatorState &state) const {
	throw InternalException("Calling Execute on a node that is not an operator!");
}

OperatorFinalizeResultType GPUPhysicalOperator::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                          GlobalOperatorState &gstate, OperatorState &state) const {
	throw InternalException("Calling FinalExecute on a node that is not an operator!");
}
// LCOV_EXCL_STOP

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
unique_ptr<LocalSourceState> GPUPhysicalOperator::GetLocalSourceState(ExecutionContext &context,
                                                                   GlobalSourceState &gstate) const {
	return make_uniq<LocalSourceState>();
}

unique_ptr<GlobalSourceState> GPUPhysicalOperator::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<GlobalSourceState>();
}


//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
unique_ptr<LocalSinkState> GPUPhysicalOperator::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<LocalSinkState>();
}

unique_ptr<GlobalSinkState> GPUPhysicalOperator::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<GlobalSinkState>();
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void GPUPhysicalOperator::BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) {
	op_state.reset();

	auto &state = meta_pipeline.GetState();
	if (IsSink()) {
		// operator is a sink, build a pipeline
		sink_state.reset();
		D_ASSERT(children.size() == 1);

		// single operator: the operator becomes the data source of the current pipeline
		state.SetPipelineSource(current, *this);

		// we create a new pipeline starting from the child
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
		child_meta_pipeline.Build(*children[0]);
	} else {
		// operator is not a sink! recurse in children
		if (children.empty()) {
			// source
			state.SetPipelineSource(current, *this);
		} else {
			if (children.size() != 1) {
				throw InternalException("Operator not supported in BuildPipelines");
			}
			state.AddPipelineOperator(current, *this);
			children[0]->BuildPipelines(current, meta_pipeline);
		}
	}
}

vector<const_reference<GPUPhysicalOperator>> GPUPhysicalOperator::GetSources() const {
	vector<const_reference<GPUPhysicalOperator>> result;
	if (IsSink()) {
		D_ASSERT(children.size() == 1);
		result.push_back(*this);
		return result;
	} else {
		if (children.empty()) {
			// source
			result.push_back(*this);
			return result;
		} else {
			if (children.size() != 1) {
				throw InternalException("Operator not supported in GetSource");
			}
			return children[0]->GetSources();
		}
	}
}

void GPUPhysicalOperator::Verify() {
#ifdef DEBUG
	auto sources = GetSources();
	D_ASSERT(!sources.empty());
	for (auto &child : children) {
		child->Verify();
	}
#endif
}

} // namespace duckdb
