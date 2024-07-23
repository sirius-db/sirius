#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/execution/task_error_manager.hpp"
#include "gpu_pipeline.hpp"

namespace duckdb {

class GPUExecutor {
	friend class GPUPipeline;
	friend class GPUPipelineBuildState;

public:
	explicit GPUExecutor(ClientContext &context);
	~GPUExecutor();

	ClientContext &context;

	shared_ptr<GPUPipeline> CreateChildPipeline(GPUPipeline &current, PhysicalOperator &op) {
		D_ASSERT(!current.operators.empty());
		D_ASSERT(op.IsSource());
		// found another operator that is a source, schedule a child pipeline
		// 'op' is the source, and the sink is the same
		auto child_pipeline = make_shared_ptr<GPUPipeline>(*this);
		child_pipeline->sink = current.sink;
		child_pipeline->source = &op;

		// the child pipeline has the same operators up until 'op'
		for (auto current_op : current.operators) {
			if (&current_op.get() == &op) {
				break;
			}
			child_pipeline->operators.push_back(current_op);
		}

		return child_pipeline;
	}

};

} // namespace duckdb
