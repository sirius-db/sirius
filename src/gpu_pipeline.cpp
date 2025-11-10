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

#include "gpu_pipeline.hpp"
#include "gpu_executor.hpp"
#include "gpu_meta_pipeline.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/tree_renderer/text_tree_renderer.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parallel/pipeline_event.hpp"
#include "duckdb/parallel/pipeline_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "log/logging.hpp"

namespace duckdb {

GPUPipeline::GPUPipeline(GPUExecutor &executor_p)
    : executor(executor_p), ready(false), initialized(false), source(nullptr), sink(nullptr) {
}

ClientContext &GPUPipeline::GetClientContext() {
	return executor.context;
}

bool GPUPipeline::IsOrderDependent() const {
	auto &config = DBConfig::GetConfig(executor.context);
	if (source) {
		auto source_order = source->SourceOrder();
		if (source_order == OrderPreservationType::FIXED_ORDER) {
			return true;
		}
		if (source_order == OrderPreservationType::NO_ORDER) {
			return false;
		}
	}
	for (auto &op_ref : operators) {
		auto &op = op_ref.get();
		if (op.OperatorOrder() == OrderPreservationType::NO_ORDER) {
			return false;
		}
		if (op.OperatorOrder() == OrderPreservationType::FIXED_ORDER) {
			return true;
		}
	}
	if (!config.options.preserve_insertion_order) {
		return false;
	}
	if (sink && sink->SinkOrderDependent()) {
		return true;
	}
	return false;
}

void GPUPipeline::ResetSink() {
	if (sink) {
		if (!sink->IsSink()) {
			throw InternalException("Sink of pipeline does not have IsSink set");
		}
		lock_guard<mutex> guard(sink->lock);
		if (!sink->sink_state) {
			sink->sink_state = sink->GetGlobalSinkState(GetClientContext());
		}
	}
}

void GPUPipeline::Reset() {
	ResetSink();
	for (auto &op_ref : operators) {
		auto &op = op_ref.get();
		lock_guard<mutex> guard(op.lock);
		if (!op.op_state) {
			op.op_state = op.GetGlobalOperatorState(GetClientContext());
		}
	}
	ResetSource(false);
	// we no longer reset source here because this function is no longer guaranteed to be called by the main thread
	// source reset needs to be called by the main thread because resetting a source may call into clients like R
	initialized = true;
}

void GPUPipeline::ResetSource(bool force) {
	if (source && !source->IsSource()) {
		throw InternalException("Source of pipeline does not have IsSource set");
	}
	if (force || !source_state) {
		source_state = source->GetGlobalSourceState(GetClientContext());
	}
}

void GPUPipeline::Ready() {
	if (ready) {
		return;
	}
	ready = true;
	std::reverse(operators.begin(), operators.end());
}

void GPUPipeline::AddDependency(shared_ptr<GPUPipeline> &pipeline) {
	D_ASSERT(pipeline);
	// dependencies.push_back(weak_ptr<GPUPipeline>(pipeline));
	dependencies.push_back(pipeline);
	pipeline->parents.push_back(weak_ptr<GPUPipeline>(shared_from_this()));
}

// string GPUPipeline::ToString() const {
// 	TreeRenderer renderer;
// 	return renderer.ToString(*this);
// }

// void GPUPipeline::Print() const {
// 	Printer::Print(ToString());
// }

// void GPUPipeline::PrintDependencies() const {
// 	for (auto &dep : dependencies) {
// 		shared_ptr<GPUPipeline>(dep)->Print();
// 	}
// }

vector<reference<GPUPhysicalOperator>> GPUPipeline::GetOperators() {
	vector<reference<GPUPhysicalOperator>> result;
	D_ASSERT(source);
	result.push_back(*source);
	for (auto &op : operators) {
		result.push_back(op.get());
	}
	if (sink) {
		result.push_back(*sink);
	}
	return result;
}

vector<const_reference<GPUPhysicalOperator>> GPUPipeline::GetOperators() const {
	vector<const_reference<GPUPhysicalOperator>> result;
	D_ASSERT(source);
	result.push_back(*source);
	for (auto &op : operators) {
		result.push_back(op.get());
	}
	if (sink) {
		result.push_back(*sink);
	}
	return result;
}

void GPUPipeline::ClearSource() {
	source_state.reset();
	batch_indexes.clear();
}

idx_t GPUPipeline::RegisterNewBatchIndex() {
	lock_guard<mutex> l(batch_lock);
	idx_t minimum = batch_indexes.empty() ? base_batch_index : *batch_indexes.begin();
	batch_indexes.insert(minimum);
	return minimum;
}

idx_t GPUPipeline::UpdateBatchIndex(idx_t old_index, idx_t new_index) {
	lock_guard<mutex> l(batch_lock);
	if (new_index < *batch_indexes.begin()) {
		throw InternalException("Processing batch index %llu, but previous min batch index was %llu", new_index,
		                        *batch_indexes.begin());
	}
	auto entry = batch_indexes.find(old_index);
	if (entry == batch_indexes.end()) {
		throw InternalException("Batch index %llu was not found in set of active batch indexes", old_index);
	}
	batch_indexes.erase(entry);
	batch_indexes.insert(new_index);
	return *batch_indexes.begin();
}

//===--------------------------------------------------------------------===//
// GPU Pipeline Build State
//===--------------------------------------------------------------------===//
void GPUPipelineBuildState::SetPipelineSource(GPUPipeline &pipeline, GPUPhysicalOperator &op) {
	SIRIUS_LOG_DEBUG("Setting pipeline source {}", PhysicalOperatorToString(op.type));
	pipeline.source = &op;
}

void GPUPipelineBuildState::SetPipelineSink(GPUPipeline &pipeline, optional_ptr<GPUPhysicalOperator> op,
                                         idx_t sink_pipeline_count) {
	pipeline.sink = op;
	if (pipeline.sink) SIRIUS_LOG_DEBUG("Setting pipeline sink {}", PhysicalOperatorToString((*pipeline.sink).type));
	// set the base batch index of this pipeline based on how many other pipelines have this node as their sink
	pipeline.base_batch_index = BATCH_INCREMENT * sink_pipeline_count;
}

void GPUPipelineBuildState::AddPipelineOperator(GPUPipeline &pipeline, GPUPhysicalOperator &op) {
	SIRIUS_LOG_DEBUG("Adding operator to pipeline {}", PhysicalOperatorToString(op.type));
	pipeline.operators.push_back(op);
}

optional_ptr<GPUPhysicalOperator> GPUPipelineBuildState::GetPipelineSource(GPUPipeline &pipeline) {
	return pipeline.source;
}

optional_ptr<GPUPhysicalOperator> GPUPipelineBuildState::GetPipelineSink(GPUPipeline &pipeline) {
	return pipeline.sink;
}

void GPUPipelineBuildState::SetPipelineOperators(GPUPipeline &pipeline, vector<reference<GPUPhysicalOperator>> operators) {
	pipeline.operators = std::move(operators);
}

shared_ptr<GPUPipeline> GPUPipelineBuildState::CreateChildPipeline(GPUExecutor &executor, GPUPipeline &pipeline,
                                                             GPUPhysicalOperator &op) {
	return executor.CreateChildPipeline(pipeline, op);
}

vector<reference<GPUPhysicalOperator>> GPUPipelineBuildState::GetPipelineOperators(GPUPipeline &pipeline) {
	return pipeline.operators;
}

} // namespace duckdb