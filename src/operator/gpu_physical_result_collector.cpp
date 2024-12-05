// #include "duckdb/execution/operator/helper/physical_result_collector.hpp"

// #include "duckdb/execution/operator/helper/physical_batch_collector.hpp"
// #include "duckdb/execution/operator/helper/physical_materialized_collector.hpp"
// #include "duckdb/execution/operator/helper/physical_buffered_collector.hpp"
// #include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/config.hpp"
// #include "duckdb/main/prepared_statement_data.hpp"
// #include "duckdb/parallel/meta_pipeline.hpp"
// #include "duckdb/parallel/pipeline.hpp"
#include "duckdb/common/types/string_type.hpp"

#include "operator/gpu_physical_result_collector.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

GPUPhysicalResultCollector::GPUPhysicalResultCollector(GPUPreparedStatementData &data)
    : GPUPhysicalOperator(PhysicalOperatorType::RESULT_COLLECTOR, {LogicalType::BOOLEAN}, 0),
      statement_type(data.prepared->statement_type), properties(data.prepared->properties), plan(*data.gpu_physical_plan), names(data.prepared->names) {
	this->types = data.prepared->types;
	gpuBufferManager = &(GPUBufferManager::GetInstance());
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
	: GPUPhysicalResultCollector(data), collection(make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types)) {
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

void GPUPhysicalMaterializedCollector::FinalizeMaterializeString(GPUIntermediateRelation input_relation, GPUIntermediateRelation& output_relation, size_t col) const {
	bool need_to_late_materalize = input_relation.checkLateMaterialization(col);
	output_relation.columns[col] = new GPUColumn(*input_relation.columns[col]);

	if (need_to_late_materalize) {
		// Late materalize the input relationship
		size_t num_rows = output_relation.columns[col]->row_id_count;
		uint64_t* row_ids = output_relation.columns[col]->row_ids;
		DataWrapper input_data_wrapper = output_relation.columns[col]->data_wrapper;

		// First create the new offsets
		int* materalized_offsets = gpuBufferManager->customCudaMalloc<int>(num_rows + 1, 0, 0);
		int new_chars_len = strMateralizeOffsets(materalized_offsets, input_data_wrapper.offsets, row_ids, num_rows);

		// Copy over the chars
		uint8_t* materalized_chars = gpuBufferManager->customCudaMalloc<uint8_t>(new_chars_len, 0, 0);
		strMateralizeChars(materalized_chars, input_data_wrapper.data, materalized_offsets, input_data_wrapper.offsets, row_ids, num_rows);
		
		// Update the data wrapper
		input_data_wrapper.offsets = materalized_offsets;
		input_data_wrapper.num_strings = static_cast<int>(num_rows);
		input_data_wrapper.data = materalized_chars;
		input_data_wrapper.size = static_cast<size_t>(new_chars_len);
		
		// Mark that record has been late materalized
		output_relation.columns[col]->data_wrapper = input_data_wrapper;
		output_relation.columns[col]->column_length = num_rows;
		output_relation.columns[col]->row_id_count = 0;
		output_relation.columns[col]->row_ids = nullptr;

		// Also reset the input marking it as materalized for future queries
		input_relation.columns[col]->row_id_count = 0;
		input_relation.columns[col]->row_ids = nullptr;
	} 
}

template <typename T>
void 
GPUPhysicalMaterializedCollector::FinalMaterializeInternal(GPUIntermediateRelation input_relation, GPUIntermediateRelation &output_relation, size_t col) const {
	if (input_relation.checkLateMaterialization(col)) {
		T* data = reinterpret_cast<T*> (input_relation.columns[col]->data_wrapper.data);
		uint64_t* row_ids = reinterpret_cast<uint64_t*> (input_relation.columns[col]->row_ids);
		T* materialized = gpuBufferManager->customCudaMalloc<T>(input_relation.columns[col]->row_id_count, 0, 0);
		printf("input_relation.columns[col]->row_id_count %zu\n", input_relation.columns[col]->row_id_count);
		materializeExpression<T>(data, materialized, row_ids, input_relation.columns[col]->row_id_count);
		output_relation.columns[col] = new GPUColumn(input_relation.columns[col]->row_id_count, input_relation.columns[col]->data_wrapper.type, reinterpret_cast<uint8_t*>(materialized));
		output_relation.columns[col]->row_id_count = 0;
		output_relation.columns[col]->row_ids = nullptr;
	} else {
		output_relation.columns[col] = input_relation.columns[col];
	}
}

size_t
GPUPhysicalMaterializedCollector::FinalMaterialize(GPUIntermediateRelation input_relation, GPUIntermediateRelation &output_relation, size_t col) const {
	size_t size_bytes;
	bool is_str_col = input_relation.columns[col]->data_wrapper.is_string_data;
	
	switch (input_relation.columns[col]->data_wrapper.type) {
	case ColumnType::INT64:
		FinalMaterializeInternal<uint64_t>(input_relation, output_relation, col);
		size_bytes = output_relation.columns[col]->column_length * sizeof(uint64_t);
		break;
	case ColumnType::INT32:
		FinalMaterializeInternal<int>(input_relation, output_relation, col);
		size_bytes = output_relation.columns[col]->column_length * sizeof(int);
		break;
	case ColumnType::FLOAT64:
		FinalMaterializeInternal<double>(input_relation, output_relation, col);
		size_bytes = output_relation.columns[col]->column_length * sizeof(uint64_t);
		break;
	case ColumnType::FLOAT32:
		FinalMaterializeInternal<float>(input_relation, output_relation, col);
		size_bytes = output_relation.columns[col]->column_length * sizeof(uint64_t);
		break;
	case ColumnType::VARCHAR:
		FinalizeMaterializeString(input_relation, output_relation, col);
		size_bytes = output_relation.columns[col]->data_wrapper.size * sizeof(uint8_t);
		break;
	default:
		throw NotImplementedException("FinalMaterialize Unsupported column type");
	}

	printf("Final materialize size %d and output col length %d\n", size_bytes, output_relation.columns[col]->column_length);
	return size_bytes;
}

LogicalType ColumnTypeToLogicalType(ColumnType type) {
	switch (type) {
		case ColumnType::INT32:
			return LogicalType::INTEGER;
		case ColumnType::INT64:
			return LogicalType::BIGINT;
		case ColumnType::FLOAT32:
			return LogicalType::FLOAT;
		case ColumnType::FLOAT64:
			return LogicalType::DOUBLE;
		case ColumnType::VARCHAR:
			return LogicalType::VARCHAR;
		default:
			throw NotImplementedException("ColumnTypeToLogicalType Unsupported column type");
	}
}

Vector rawDataToVector(uint8_t* host_data, size_t vector_offset, ColumnType type) {
	size_t sizeof_type;
	switch (type) {
		case ColumnType::INT32:
			sizeof_type = sizeof(int); break;
		case ColumnType::INT64:
			sizeof_type = sizeof(uint64_t); break;
		case ColumnType::FLOAT32:
			sizeof_type = sizeof(float); break;
		case ColumnType::FLOAT64:
			sizeof_type = sizeof(double); break;
		case ColumnType::VARCHAR:
			sizeof_type = sizeof(uint8_t); break;
		default:
			throw NotImplementedException("rawDataToVector Unsupported column type");
	}

	int initial_offset = vector_offset * STANDARD_VECTOR_SIZE * sizeof_type;
	uint8_t* data = host_data + initial_offset;
	printf("rawDataToVector got offset of %d\n", initial_offset);
	return Vector(ColumnTypeToLogicalType(type), data);
}

SinkResultType GPUPhysicalMaterializedCollector::Sink(GPUIntermediateRelation &input_relation) const {
	//TODO: Don't forget to check the if input relation is already materialized or not, if not then materialize it
	if (types.size() != input_relation.columns.size()) {
		throw InvalidInputException("Column count mismatch");
	}
	// auto &gstate = GetGlobalSinkState(input_relation.context);

	for(int i = 0; i < types.size(); i++) {
		std::cout << "GPUPhysicalMaterializedCollector Sink Col " << i << " has types " << types[i].ToString() << std::endl;
	}

	size_t size_bytes = 0;
	int num_columns = input_relation.columns.size();
	Allocator& allocator = Allocator::DefaultAllocator();
	uint8_t** host_data = new uint8_t*[num_columns];
	GPUIntermediateRelation materialized_relation(num_columns);
	std::cout << "Initialized the materialized_relation with " << num_columns << " columns" << std::endl;

	for (int col = 0; col < num_columns; col++) {
		// Log the current column
		GPUColumn* curr_col = input_relation.columns[col];
		if(curr_col == nullptr) {
			throw InvalidInputException("Got null column in the input relationship");
		}

		// Final materialization for all non string columns
		std::cout << "Calling finalized materalized for column " << col << std::endl;
		size_bytes = FinalMaterialize(input_relation, materialized_relation, col);

		if(input_relation.columns[col]->data_wrapper.type != ColumnType::VARCHAR) {
			host_data[col] = allocator.AllocateData(size_bytes);
			callCudaMemcpyDeviceToHost<uint8_t>(host_data[col], materialized_relation.columns[col]->data_wrapper.data, size_bytes, 0);
			std::cout << "Allocated " << size_bytes << " bytes on CPU for col " << col << std::endl;
			std::cout << "Got data count of " << reinterpret_cast<uint64_t*>(host_data[col])[0] << std::endl;
		} else {
			DataWrapper materialized_col_data = materialized_relation.columns[col]->data_wrapper;
			std::cout << "Got materalized col data with " << materialized_col_data.num_strings << " and " << materialized_col_data.size << " chars" << std::endl;
			
			// Copy over the pointers from the gpu to the cpu
			size_t offset_bytes = (materialized_col_data.num_strings + 1) * sizeof(int);
			int* cpu_offsets = reinterpret_cast<int*>(allocator.AllocateData(offset_bytes));
			callCudaMemcpyDeviceToHost<int>(cpu_offsets, materialized_col_data.offsets, materialized_col_data.num_strings + 1, 0);
			std::cout << "Got cpu offsets of " << cpu_offsets[0] << "," << cpu_offsets[1] << std::endl;
			materialized_col_data.offsets = cpu_offsets;
			
			// Do the same for the chars
			size_t data_bytes = materialized_col_data.size * sizeof(uint8_t);
			uint8_t* cpu_chars = reinterpret_cast<uint8_t*>(allocator.AllocateData(data_bytes));
			callCudaMemcpyDeviceToHost<uint8_t>(cpu_chars, materialized_col_data.data, data_bytes, 0);
			materialized_col_data.data = cpu_chars;

			// Copy over the data wrapper
			materialized_relation.columns[col]->data_wrapper = materialized_col_data;
			std::cout << "Copied over strings to the CPU with offset " << materialized_relation.columns[col]->data_wrapper.offsets[0];
			std::cout << " and chars " << materialized_relation.columns[col]->data_wrapper.data[0] << std::endl;
		}
	}
	
	ColumnDataAppendState append_state;
	collection->InitializeAppend(append_state);
	size_t total_chunks = (materialized_relation.columns[0]->column_length + STANDARD_VECTOR_SIZE - 1) / STANDARD_VECTOR_SIZE;
	size_t remaining = materialized_relation.columns[0]->column_length;
	printf("Total vectors of %zu with standard vector size of %zu and remaining of %zu\n", total_chunks, STANDARD_VECTOR_SIZE, remaining);
	for (int vec = 0; vec < total_chunks; vec++) {
		DataChunk chunk;
		chunk.InitializeEmpty(types);
		chunk.SetCapacity(STANDARD_VECTOR_SIZE);
		for (int col = 0; col < materialized_relation.columns.size(); col++) {
			if(materialized_relation.columns[col]->data_wrapper.type != ColumnType::VARCHAR) {
				std::cout << "Converting non string col " << col << " into vector " << std::endl;
				Vector vector = rawDataToVector(host_data[col], vec, materialized_relation.columns[col]->data_wrapper.type);
				chunk.data[col].Reference(vector);
			} else {
				// Create the array of duckdb strings
				DataWrapper strings_data_wrapper = materialized_relation.columns[col]->data_wrapper;
				char* str_all_chars = reinterpret_cast<char*>(strings_data_wrapper.data);
				int num_strings = strings_data_wrapper.num_strings;
				string_t* duckdb_strings = new string_t[num_strings];
				for(int i = 0; i < num_strings; i++) {
					int str_start_offset = strings_data_wrapper.offsets[i];
					int curr_str_length = strings_data_wrapper.offsets[i + 1] - str_start_offset;
					duckdb_strings[i] = string_t(str_all_chars + str_start_offset, curr_str_length);
				}

				// Create the strings vector
				Vector str_vector(LogicalType::VARCHAR, reinterpret_cast<data_ptr_t>(duckdb_strings));
				chunk.data[col].Reference(str_vector);
			}
		}

		// Set the cardinality of the chunk
		if (remaining < STANDARD_VECTOR_SIZE) {
			chunk.SetCardinality(remaining);
			remaining = 0;
		} else {
			chunk.SetCardinality(STANDARD_VECTOR_SIZE);
			remaining -= STANDARD_VECTOR_SIZE;
		}
		
		collection->Append(append_state, chunk);
	}

	printf("Returning finished\n");
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
	if (!gstate.context) throw InvalidInputException("No context set in GPUMaterializedCollectorState");
	if (!gstate.collection) throw InvalidInputException("No context set in GPUMaterializedCollectorState");
	auto prop = gstate.context->GetClientProperties();
	// auto result = make_uniq<MaterializedQueryResult>(statement_type, properties, names, std::move(gstate.collection),
	//                                                  prop);
	auto result = make_uniq<MaterializedQueryResult>(statement_type, properties, names, std::move(collection),
	                                                 prop);
	return std::move(result);
}

// bool PhysicalMaterializedCollector::ParallelSink() const {
// 	return parallel;
// }

// bool PhysicalMaterializedCollector::SinkOrderDependent() const {
// 	return true;
// }

} // namespace duckdb