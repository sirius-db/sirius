#include "gpu_context.hpp"
#include "gpu_operator_converter.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"

namespace duckdb {

void 
GPUExecutor::Reset() {
	// lock_guard<mutex> elock(executor_lock);
	physical_plan = nullptr;
	// cancelled = false;
	owned_plan.reset();
	// root_executor.reset();
	root_pipelines.clear();
	root_pipeline_idx = 0;
	completed_pipelines = 0;
	total_pipelines = 0;
	// error_manager.Reset();
	pipelines.clear();
	// events.clear();
	// to_be_rescheduled_tasks.clear();
	// execution_result = PendingExecutionResult::RESULT_NOT_READY;
}

void ConvertDuckDBPlantoGPU(PhysicalOperator& physical_plan) {
	//converting from PhysicalOperator to GPUPhysicalOperator recursively
		// unique_ptr<GPUPhysicalOperator> gpu_operator = make_unique<GPUPhysicalOperator>(*physical_plan);

		// Recursively convert child operators
		// for (auto &child : physical_plan->children) {
		// 	ConvertDuckDBPlantoGPU(std::move(child), gpu_operator);
		// }

		// Add the converted operator to the GPU plan
		// gpu_physical_plan = std::move(gpu_operator);

		// physical_plan->Print();
		printf("Node: %s\n", physical_plan.GetName().c_str());
		// ConvertOperator(physical_plan);
		
		// Recursively convert child operators
		for (auto &child : physical_plan.children) {
			printf("Going to child\n");
			ConvertDuckDBPlantoGPU(*child);
		}
}

void GPUExecutor::Initialize(unique_ptr<PhysicalResultCollector> physical_result_collector) {
	Reset();

	// unique_ptr<GPUPhysicalOperator> gpu_physical_plan = nullptr;
	//convert cpu plan to gpu plan 
	ConvertDuckDBPlantoGPU(physical_result_collector->plan);
	// physical_plan->Print();

	throw NotImplementedException("GPUExecutor::Initialize");

	InitializeInternal(*physical_result_collector);
}

void GPUExecutor::InitializeInternal(PhysicalResultCollector &physical_result_collector) {

	// auto &scheduler = TaskScheduler::GetScheduler(context);
	{
		// lock_guard<mutex> elock(executor_lock);
		physical_plan = &physical_result_collector;


		// this->profiler = ClientData::Get(context).profiler;
		// profiler->Initialize(plan);
		// this->producer = scheduler.CreateProducer();

		// build and ready the pipelines
		GPUPipelineBuildState state;
		auto root_pipeline = make_shared_ptr<GPUMetaPipeline>(*this, state, nullptr);
		root_pipeline->Build(*gpu_physical_plan);
		root_pipeline->Ready();

		// ready recursive cte pipelines too
		// TODO: SUPPORT RECURSIVE CTE FOR GPU
		// for (auto &rec_cte_ref : recursive_ctes) {
		// 	auto &rec_cte = rec_cte_ref.get().Cast<PhysicalRecursiveCTE>();
		// 	// rec_cte.recursive_meta_pipeline->Ready();
		// }

		// set root pipelines, i.e., all pipelines that end in the final sink
		root_pipeline->GetPipelines(root_pipelines, false);
		root_pipeline_idx = 0;

		// collect all meta-pipelines from the root pipeline
		vector<shared_ptr<GPUMetaPipeline>> to_schedule;
		root_pipeline->GetMetaPipelines(to_schedule, true, true);

		// number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to set it here
		total_pipelines = to_schedule.size();

		// collect all pipelines from the root pipelines (recursively) for the progress bar and verify them
		root_pipeline->GetPipelines(pipelines, true);

		// finally, verify and schedule
		// VerifyPipelines();
		// ScheduleEvents(to_schedule);
	}
}

void 
GPUExecutor::CancelTasks() {
	// task.reset();

	// {
	// 	lock_guard<mutex> elock(executor_lock);
	// 	// mark the query as cancelled so tasks will early-out
	// 	cancelled = true;
		// destroy all pipelines, events and states
		// TODO: SUPPORT RECURSIVE CTE FOR GPU
		// for (auto &rec_cte_ref : recursive_ctes) {
		// 	auto &rec_cte = rec_cte_ref.get().Cast<PhysicalRecursiveCTE>();
		// 	rec_cte.recursive_meta_pipeline.reset();
		// }
		pipelines.clear();
		root_pipelines.clear();
	// 	to_be_rescheduled_tasks.clear();
	// 	events.clear();
	// }
	// // Take all pending tasks and execute them until they cancel
	// while (executor_tasks > 0) {
	// 	WorkOnTasks();
	// }
}

shared_ptr<GPUPipeline> 
GPUExecutor::CreateChildPipeline(GPUPipeline &current, GPUPhysicalOperator &op) {
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

bool 
GPUExecutor::HasResultCollector() {
	return physical_plan->type == PhysicalOperatorType::RESULT_COLLECTOR;
}

unique_ptr<QueryResult> 
GPUExecutor::GetResult() {
	D_ASSERT(HasResultCollector());
	auto &result_collector = physical_plan->Cast<PhysicalResultCollector>();
	D_ASSERT(result_collector.sink_state);
	return result_collector.GetResult(*result_collector.sink_state);
}

}; // namespace duckdb