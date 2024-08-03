#include "gpu_meta_pipeline.hpp"
#include "gpu_executor.hpp"

namespace duckdb {

GPUMetaPipeline::GPUMetaPipeline(GPUExecutor &executor_p, GPUPipelineBuildState &state_p, optional_ptr<GPUPhysicalOperator> sink_p)
    : executor(executor_p), state(state_p), sink(sink_p), recursive_cte(false), next_batch_index(0) {
	CreatePipeline();
}

GPUExecutor &GPUMetaPipeline::GetExecutor() const {
	return executor;
}

GPUPipelineBuildState &GPUMetaPipeline::GetState() const {
	return state;
}

optional_ptr<GPUPhysicalOperator> GPUMetaPipeline::GetSink() const {
	return sink;
}

shared_ptr<GPUPipeline> &GPUMetaPipeline::GetBasePipeline() {
	return pipelines[0];
}

void GPUMetaPipeline::GetPipelines(vector<shared_ptr<GPUPipeline>> &result, bool recursive) {
	result.insert(result.end(), pipelines.begin(), pipelines.end());
	if (recursive) {
		for (auto &child : children) {
			child->GetPipelines(result, true);
		}
	}
}

void GPUMetaPipeline::GetMetaPipelines(vector<shared_ptr<GPUMetaPipeline>> &result, bool recursive, bool skip) {
	if (!skip) {
		result.push_back(enable_shared_from_this<GPUMetaPipeline>::shared_from_this());
	}
	if (recursive) {
		for (auto &child : children) {
			child->GetMetaPipelines(result, true, false);
		}
	}
}

optional_ptr<const vector<reference<GPUPipeline>>> GPUMetaPipeline::GetDependencies(GPUPipeline &dependant) const {
	auto it = dependencies.find(dependant);
	if (it == dependencies.end()) {
		return nullptr;
	} else {
		return &it->second;
	}
}

bool GPUMetaPipeline::HasRecursiveCTE() const {
	return recursive_cte;
}

void GPUMetaPipeline::SetRecursiveCTE() {
	recursive_cte = true;
}

void GPUMetaPipeline::AssignNextBatchIndex(GPUPipeline &pipeline) {
	pipeline.base_batch_index = next_batch_index++ * GPUPipelineBuildState::BATCH_INCREMENT;
}

void GPUMetaPipeline::BuildGPUPipelines(GPUPhysicalOperator &node, GPUPipeline &current) {
	node.op_state.reset();

	auto &state = GetState();
	if (node.IsSink()) {
		// operator is a sink, build a pipeline
		node.sink_state.reset();
		D_ASSERT(node.children.size() == 1);

		// single operator: the operator becomes the data source of the current pipeline
		state.SetPipelineSource(current, node);

		// we create a new pipeline starting from the child
		auto &child_meta_pipeline = CreateChildMetaPipeline(current, node);
		child_meta_pipeline.Build(*node.children[0]);
	} else {
		// operator is not a sink! recurse in children
		if (node.children.empty()) {
			// source
			state.SetPipelineSource(current, node);
		} else {
			if (node.children.size() != 1) {
				throw InternalException("Operator not supported in BuildPipelines");
			}
			state.AddPipelineOperator(current, node);
			// node.children[0]->BuildPipelines(current, meta_pipeline);
			BuildGPUPipelines(*node.children[0], current);
		}
	}
}

void GPUMetaPipeline::Build(GPUPhysicalOperator &op) {
	D_ASSERT(pipelines.size() == 1);
	D_ASSERT(children.empty());
	BuildGPUPipelines(op, *pipelines.back());
}

void GPUMetaPipeline::Ready() {
	for (auto &pipeline : pipelines) {
		pipeline->Ready();
	}
	for (auto &child : children) {
		child->Ready();
	}
}

GPUMetaPipeline &GPUMetaPipeline::CreateChildMetaPipeline(GPUPipeline &current, GPUPhysicalOperator &op) {
	children.push_back(make_shared_ptr<GPUMetaPipeline>(executor, state, &op));
	auto child_meta_pipeline = children.back().get();
	// child GPUMetaPipeline must finish completely before this GPUMetaPipeline can start
	current.AddDependency(child_meta_pipeline->GetBasePipeline());
	// child meta pipeline is part of the recursive CTE too
	child_meta_pipeline->recursive_cte = recursive_cte;
	return *child_meta_pipeline;
}

GPUPipeline &GPUMetaPipeline::CreatePipeline() {
	pipelines.emplace_back(make_shared_ptr<GPUPipeline>(executor));
	state.SetPipelineSink(*pipelines.back(), sink, next_batch_index++);
	return *pipelines.back();
}

void GPUMetaPipeline::AddDependenciesFrom(GPUPipeline &dependant, GPUPipeline &start, bool including) {
	// find 'start'
	auto it = pipelines.begin();
	for (; !RefersToSameObject(**it, start); it++) {
	}

	if (!including) {
		it++;
	}

	// collect pipelines that were created from then
	vector<reference<GPUPipeline>> created_pipelines;
	for (; it != pipelines.end(); it++) {
		if (RefersToSameObject(**it, dependant)) {
			// cannot depend on itself
			continue;
		}
		created_pipelines.push_back(**it);
	}

	// add them to the dependencies
	auto &deps = dependencies[dependant];
	deps.insert(deps.begin(), created_pipelines.begin(), created_pipelines.end());
}

void GPUMetaPipeline::AddFinishEvent(GPUPipeline &pipeline) {
	D_ASSERT(finish_pipelines.find(pipeline) == finish_pipelines.end());
	finish_pipelines.insert(pipeline);

	// add all pipelines that were added since 'pipeline' was added (including 'pipeline') to the finish group
	auto it = pipelines.begin();
	for (; !RefersToSameObject(**it, pipeline); it++) {
	}
	it++;
	for (; it != pipelines.end(); it++) {
		finish_map.emplace(**it, pipeline);
	}
}

bool GPUMetaPipeline::HasFinishEvent(GPUPipeline &pipeline) const {
	return finish_pipelines.find(pipeline) != finish_pipelines.end();
}

optional_ptr<GPUPipeline> GPUMetaPipeline::GetFinishGroup(GPUPipeline &pipeline) const {
	auto it = finish_map.find(pipeline);
	return it == finish_map.end() ? nullptr : &it->second;
}

GPUPipeline &GPUMetaPipeline::CreateUnionPipeline(GPUPipeline &current, bool order_matters) {
	// create the union pipeline (batch index 0, should be set correctly afterwards)
	auto &union_pipeline = CreatePipeline();
	state.SetPipelineOperators(union_pipeline, state.GetPipelineOperators(current));
	state.SetPipelineSink(union_pipeline, sink, 0);

	// 'union_pipeline' inherits ALL dependencies of 'current' (within this GPUMetaPipeline, and across MetaPipelines)
	union_pipeline.dependencies = current.dependencies;
	auto current_deps = GetDependencies(current);
	if (current_deps) {
		dependencies[union_pipeline] = *current_deps;
	}

	if (order_matters) {
		// if we need to preserve order, or if the sink is not parallel, we set a dependency
		dependencies[union_pipeline].push_back(current);
	}

	return union_pipeline;
}

void GPUMetaPipeline::CreateChildPipeline(GPUPipeline &current, GPUPhysicalOperator &op, GPUPipeline &last_pipeline) {
	// rule 2: 'current' must be fully built (down to the source) before creating the child pipeline
	D_ASSERT(current.source);

	// create the child pipeline (same batch index)
	pipelines.emplace_back(state.CreateChildPipeline(executor, current, op));
	auto &child_pipeline = *pipelines.back();
	child_pipeline.base_batch_index = current.base_batch_index;

	// child pipeline has a dependency (within this GPUMetaPipeline on all pipelines that were scheduled
	// between 'current' and now (including 'current') - set them up
	dependencies[child_pipeline].push_back(current);
	AddDependenciesFrom(child_pipeline, last_pipeline, false);
	D_ASSERT(!GetDependencies(child_pipeline)->empty());
}

} // namespace duckdb
