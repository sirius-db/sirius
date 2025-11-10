/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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


OperatorResultType GPUPhysicalOperator::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
	throw InternalException("Calling Execute on a node that is not an operator!");
}

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

SourceResultType GPUPhysicalOperator::GetData(GPUIntermediateRelation& output_relation) const {
	throw InternalException("Calling GetData on a node that is not a source!");
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

SinkResultType GPUPhysicalOperator::Sink(GPUIntermediateRelation& input_relation) const {
	throw InternalException("Calling Sink on a node that is not a sink!");
}

SinkFinalizeType GPUPhysicalOperator::CombineFinalize(vector<shared_ptr<GPUIntermediateRelation>> &input,
																				 					  	GPUIntermediateRelation& output) const {
	throw InternalException("Calling CombineFinalize on a node that is not a sink!");
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
