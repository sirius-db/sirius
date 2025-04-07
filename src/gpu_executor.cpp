#include "gpu_context.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "gpu_physical_operator.hpp"
#include "operator/gpu_physical_result_collector.hpp"
#include "operator/gpu_physical_hash_join.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include <iostream>
#include <stdio.h>

namespace duckdb {

void 
GPUExecutor::Reset() {
	// lock_guard<mutex> elock(executor_lock);
	gpu_physical_plan = nullptr;
	// cancelled = false;
	gpu_owned_plan.reset();
	// root_executor.reset();
	root_pipelines.clear();
	root_pipeline_idx = 0;
	completed_pipelines = 0;
	total_pipelines = 0;
	// error_manager.Reset();
	pipelines.clear();
	gpuBufferManager->ResetBuffer();
	// events.clear();
	// to_be_rescheduled_tasks.clear();
	// execution_result = PendingExecutionResult::RESULT_NOT_READY;
}

void GPUExecutor::Initialize(unique_ptr<GPUPhysicalOperator> plan) {
	Reset();
	gpu_owned_plan = std::move(plan);
	InitializeInternal(*gpu_owned_plan);
}

void GPUExecutor::Execute() {

	int initial_idx = 0;

	printf("Total meta pipelines %d\n", scheduled.size());

	for (int i = 0; i < scheduled.size(); i++) {
		printf("\n\n\n");

		auto pipeline = scheduled[i];

		// TODO: This is temporary solution
		// if (pipeline->source->type == PhysicalOperatorType::HASH_JOIN || pipeline->source->type == PhysicalOperatorType::RESULT_COLLECTOR) {
		// 	continue;
		// }

		vector<GPUIntermediateRelation*> intermediate_relations;
		GPUIntermediateRelation* final_relation;
		// vector<unique_ptr<OperatorState>> intermediate_states;
		intermediate_relations.reserve(pipeline->operators.size());
		// intermediate_states.reserve(pipeline->operators.size());

		// printf("Executing pipeline op size %d\n", pipeline->operators.size());
		for (idx_t i = 0; i < pipeline->operators.size(); i++) {
			auto &prev_operator = i == 0 ? *(pipeline->source) : pipeline->operators[i - 1].get();
			auto &current_operator = pipeline->operators[i].get();

			// auto chunk = make_uniq<DataChunk>();
			// chunk->Initialize(Allocator::Get(context.client), prev_operator.GetTypes());
			GPUIntermediateRelation* inter_rel = new GPUIntermediateRelation(prev_operator.GetTypes().size());
			intermediate_relations.push_back(std::move(inter_rel));

			// auto op_state = current_operator.GetOperatorState(context);
			// intermediate_states.push_back(std::move(op_state));

			// if (current_operator.IsSink() && current_operator.sink_state->state == SinkFinalizeType::NO_OUTPUT_POSSIBLE) {
			// 	// one of the operators has already figured out no output is possible
			// 	// we can skip executing the pipeline
			// 	FinishProcessing();
			// }
		}
		// InitializeChunk(final_chunk);
		auto &last_op = pipeline->operators.empty() ? *pipeline->source : pipeline->operators.back().get();
		final_relation = new GPUIntermediateRelation(last_op.GetTypes().size());

		// auto thread_context = ThreadContext(context);
		// auto exec_context = GPUExecutionContext(context, thread_context, pipeline.get());

		// pipeline->Reset();
		// auto prop = pipeline->executor.context.GetClientProperties();
		// std::cout << "Properties: " << prop.time_zone << std::endl;
		auto is_empty = pipeline->operators.empty();
		auto &source_relation = is_empty ? final_relation : intermediate_relations[0];
		// auto source_result = FetchFromSource(source_chunk);

		// StartOperator(*pipeline.source);
		// auto interrupt_state = InterruptState();
		// auto local_source_state = pipeline.source->GetLocalSourceState(exec_context, *pipeline.source_state);
		// OperatorSourceInput source_input = {*pipeline.source_state, *local_source_state, interrupt_state};
		// pipeline->source->GetData(exec_context, source_relation, source_input);
		auto source_type = pipeline->source.get()->type;
		std::cout << "pipeline source type " << PhysicalOperatorToString(source_type) << std::endl;
		if (source_type == PhysicalOperatorType::TABLE_SCAN) {
			// initialize pipeline
			Pipeline duckdb_pipeline(*executor);
			ThreadContext thread_context(context);
			ExecutionContext exec_context(context, thread_context, &duckdb_pipeline);
			auto &table_scan = pipeline->source->Cast<GPUPhysicalTableScan>();
			table_scan.GetDataDuckDB(exec_context);
		}
		pipeline->source->GetData(*source_relation);
		// printf("source relation size %d\n", source_relation->columns.size());
		// for (auto col : source_relation->columns) {
		// 	printf("source relation column size %d column name %s\n", col->column_length, col->name.c_str());
		// }
		// printf("\n");
		// EndOperator(*pipeline.source, &result);

		//call source
		// std::cout << pipeline->source.get()->GetName() << std::endl;
		for (int current_idx = 1; current_idx <= pipeline->operators.size(); current_idx++) {
			auto op = pipeline->operators[current_idx-1];
			auto op_type = op.get().type;
			std::cout << "pipeline operator type " << PhysicalOperatorToString(op_type) << std::endl;
			// std::cout << op.get().GetName() << std::endl;
			//call operator

			auto current_intermediate = current_idx;
			auto &current_relation =
				current_intermediate >= intermediate_relations.size() ? final_relation : intermediate_relations[current_intermediate];
			// current_chunk.Reset();

			auto &prev_relation =
			    current_intermediate == initial_idx + 1 ? source_relation : intermediate_relations[current_intermediate - 1];
			auto operator_idx = current_idx - 1;
			auto &current_operator = pipeline->operators[operator_idx];

			// auto op_state = current_operator.GetOperatorState(context);
			// intermediate_states.push_back(std::move(op_state));
			
			// StartOperator(current_operator);
			// auto result = current_operator.get().Execute(exec_context, prev_relation, current_relation, *current_operator.op_state,
			//                                        *intermediate_states[current_intermediate - 1]);

			auto result = current_operator.get().Execute(*prev_relation, *current_relation);
			// EndOperator(current_operator, &current_chunk);
		}
		if (pipeline->sink) {
			auto sink_type = pipeline->sink.get()->type;
			std::cout << "pipeline sink type " << PhysicalOperatorToString(sink_type) << std::endl;
			// std::cout << pipeline->sink.get()->GetName() << std::endl;
			//call sink
			auto &sink_relation = final_relation;
			// printf("sink relation size %d\n", final_relation->columns.size());
			// int i = 0;
			// for (auto col : final_relation->columns) {
			// 	if (col == nullptr) printf("%d\n", i);
			// 	i++;
			// 	// printf("sink relation column size %d\n", col->column_length);
			// }
			// auto interrupt_state = InterruptState();
			// auto local_sink_state = pipeline->sink->GetLocalSinkState(exec_context);
			// OperatorSinkInput sink_input {*pipeline->sink->sink_state, *local_sink_state, interrupt_state};
			// pipeline->sink->Sink(exec_context, *sink_relation, sink_input);
			pipeline->sink->Sink(*sink_relation);
			printf("\n");
		}
	}
}

void GPUExecutor::InitializeInternal(GPUPhysicalOperator &plan) {

	// auto &scheduler = TaskScheduler::GetScheduler(context);
	{
		// lock_guard<mutex> elock(executor_lock);
		gpu_physical_plan = &plan;

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
		// for (auto &pipeline : root_pipelines) {
		// 	auto type = pipeline->source.get()->type;
		// 	printf("root pipeline operators size = %d\n", pipeline->operators.size());
		// 	std::cout << "root pipeline source type " << PhysicalOperatorToString(type) << std::endl;
		// }

		// collect all meta-pipelines from the root pipeline
		vector<shared_ptr<GPUMetaPipeline>> to_schedule;
		root_pipeline->GetMetaPipelines(to_schedule, true, true);

		// number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to set it here
		total_pipelines = to_schedule.size();
		
		printf("Total meta pipelines %d\n", to_schedule.size());
		int schedule_count = 0;
		int meta = 0;
		while (schedule_count < to_schedule.size()) {
			vector<shared_ptr<GPUMetaPipeline>> children;
			to_schedule[to_schedule.size() - 1 - meta]->GetMetaPipelines(children, false, true);
			auto base_pipeline = to_schedule[to_schedule.size() - 1 - meta]->GetBasePipeline();
			bool should_schedule = true;

			//already scheduled
			if (find(scheduled.begin(), scheduled.end(), base_pipeline) != scheduled.end()) {
				should_schedule = false;
			} else {
				//check if all children are scheduled
				for (auto &child : children) {
					if (find(scheduled.begin(), scheduled.end(), child->GetBasePipeline()) == scheduled.end()) {
						should_schedule = false;
						break;
					}
				}
				//check if all dependencies are scheduled
				for (int dep = 0; dep < base_pipeline->dependencies.size(); dep++) {
					if (find(scheduled.begin(), scheduled.end(), base_pipeline->dependencies[dep]) == scheduled.end()) {
						should_schedule = false;
						break;
					}
				}
			}
			if (should_schedule) {
				vector<shared_ptr<GPUPipeline>> pipeline_inside;
				to_schedule[to_schedule.size() - 1 - meta]->GetPipelines(pipeline_inside, false);
				for (int pipeline_idx = 0; pipeline_idx < pipeline_inside.size(); pipeline_idx++) {
					auto &pipeline = pipeline_inside[pipeline_idx];
					if (pipeline_inside[pipeline_idx]->source->type == PhysicalOperatorType::HASH_JOIN) {
						auto& temp = pipeline_inside[pipeline_idx]->source.get()->Cast<GPUPhysicalHashJoin>();
						if (temp.join_type == JoinType::RIGHT || temp.join_type == JoinType::RIGHT_SEMI || temp.join_type == JoinType::RIGHT_ANTI) {
							scheduled.push_back(pipeline);
						}
					} else {
						scheduled.push_back(pipeline);
					}
				}
				// scheduled.push_back(base_pipeline);
				schedule_count++;
			}
			meta = (meta + 1) % to_schedule.size();
		}

		// collect all pipelines from the root pipelines (recursively) for the progress bar and verify them
		root_pipeline->GetPipelines(pipelines, true);
		printf("total_pipelines = %d\n", pipelines.size());

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
	return gpu_physical_plan->type == PhysicalOperatorType::RESULT_COLLECTOR;
}

unique_ptr<QueryResult> 
GPUExecutor::GetResult() {
	D_ASSERT(HasResultCollector());
	if (!gpu_physical_plan) throw InvalidInputException("gpu_physical_plan is NULL");
	if (gpu_physical_plan.get() == NULL) throw InvalidInputException("gpu_physical_plan is NULL");
	// auto &result_collector = gpu_physical_plan.get()->Cast<GPUPhysicalResultCollector>();
	auto &result_collector = gpu_physical_plan.get()->Cast<GPUPhysicalMaterializedCollector>();
	D_ASSERT(result_collector.sink_state);
	// printf("we are getting result\n");
	result_collector.sink_state = result_collector.GetGlobalSinkState(context);
	unique_ptr<QueryResult> res = result_collector.GetResult(*(result_collector.sink_state));
	// printf("we can get result\n");
	return res;
}

}; // namespace duckdb