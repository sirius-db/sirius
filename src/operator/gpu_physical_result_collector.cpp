#include "duckdb/main/config.hpp"

#include "operator/gpu_physical_result_collector.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"

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
	: GPUPhysicalResultCollector(data), result_collection(make_uniq<GPUResultCollection>()) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class GPUMaterializedCollectorGlobalState : public GlobalSinkState {
public:
	mutex glock;
	shared_ptr<ClientContext> context;
};

class GPUMaterializedCollectorLocalState : public LocalSinkState {
public:
	ColumnDataAppendState append_state;
};

template <typename T>
void 
GPUPhysicalMaterializedCollector::FinalMaterializeInternal(GPUIntermediateRelation input_relation, GPUIntermediateRelation &output_relation, size_t col) const {
	if (input_relation.checkLateMaterialization(col)) {
		T* data = reinterpret_cast<T*> (input_relation.columns[col]->data_wrapper.data);
		uint64_t* row_ids = reinterpret_cast<uint64_t*> (input_relation.columns[col]->row_ids);
		T* materialized;
		// printf("input_relation.columns[col]->row_id_count %d\n", input_relation.columns[col]->row_id_count);
		materializeExpression<T>(data, materialized, row_ids, input_relation.columns[col]->row_id_count, input_relation.columns[col]->column_length);
		output_relation.columns[col] = make_shared_ptr<GPUColumn>(input_relation.columns[col]->row_id_count, input_relation.columns[col]->data_wrapper.type, reinterpret_cast<uint8_t*>(materialized));
		output_relation.columns[col]->row_id_count = 0;
		output_relation.columns[col]->row_ids = nullptr;
		output_relation.columns[col]->is_unique = input_relation.columns[col]->is_unique;
	} else {
		// output_relation.columns[col] = input_relation.columns[col];
		output_relation.columns[col] = make_shared_ptr<GPUColumn>(input_relation.columns[col]->column_length, input_relation.columns[col]->data_wrapper.type, input_relation.columns[col]->data_wrapper.data);
		output_relation.columns[col]->is_unique = input_relation.columns[col]->is_unique;
	}
}

void 
GPUPhysicalMaterializedCollector::FinalMaterializeString(GPUIntermediateRelation input_relation, GPUIntermediateRelation& output_relation, size_t col) const {
	// bool need_to_late_materalize = input_relation.checkLateMaterialization(col);
	if (input_relation.checkLateMaterialization(col)) {
		// Late materalize the input relationship
		uint8_t* data = input_relation.columns[col]->data_wrapper.data;
		uint64_t* offset = input_relation.columns[col]->data_wrapper.offset;
		uint64_t* row_ids = input_relation.columns[col]->row_ids;
		size_t num_rows = input_relation.columns[col]->row_id_count;
		uint8_t* result; uint64_t* result_offset; uint64_t* new_num_bytes;

		std::cout << "Running string late materalization with " << num_rows << " rows" << std::endl;

		materializeString(data, offset, result, result_offset, row_ids, new_num_bytes, num_rows, input_relation.columns[col]->column_length, input_relation.columns[col]->data_wrapper.num_bytes);

		output_relation.columns[col] = make_shared_ptr<GPUColumn>(num_rows, ColumnType::VARCHAR, reinterpret_cast<uint8_t*>(result), result_offset, new_num_bytes[0], true);
		output_relation.columns[col]->row_id_count = 0;
		output_relation.columns[col]->row_ids = nullptr;
		output_relation.columns[col]->is_unique = input_relation.columns[col]->is_unique;
	} else {
		output_relation.columns[col] = make_shared_ptr<GPUColumn>(*input_relation.columns[col]);
		output_relation.columns[col]->is_unique = input_relation.columns[col]->is_unique;
	}
}

size_t
GPUPhysicalMaterializedCollector::FinalMaterialize(GPUIntermediateRelation input_relation, GPUIntermediateRelation &output_relation, size_t col) const {
	size_t size_bytes;
	
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
		size_bytes = output_relation.columns[col]->column_length * sizeof(double);
		break;
	case ColumnType::FLOAT32:
		FinalMaterializeInternal<float>(input_relation, output_relation, col);
		size_bytes = output_relation.columns[col]->column_length * sizeof(float);
		break;
	case ColumnType::BOOLEAN:
		FinalMaterializeInternal<uint8_t>(input_relation, output_relation, col);
		size_bytes = output_relation.columns[col]->column_length * sizeof(uint8_t);
		break;
	case ColumnType::VARCHAR:
		FinalMaterializeString(input_relation, output_relation, col);
		break;
	default:
		throw NotImplementedException("Unsupported column type");
	}
	// output_relation.length = output_relation.columns[col]->column_length;
	// printf("Final materialize size %d bytes\n", size_bytes);
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
		case ColumnType::BOOLEAN:
			return LogicalType::BOOLEAN;
		case ColumnType::VARCHAR:
			return LogicalType::VARCHAR;
		default:
			throw NotImplementedException("Unsupported column type");
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
		case ColumnType::BOOLEAN:
			sizeof_type = sizeof(uint8_t); break;
		default:
			throw NotImplementedException("Unsupported column type");
	}
	uint8_t* data = host_data + vector_offset * STANDARD_VECTOR_SIZE * sizeof_type;
	return Vector(ColumnTypeToLogicalType(type), data);
}

SinkResultType GPUPhysicalMaterializedCollector::Sink(GPUIntermediateRelation &input_relation) const {
	//TODO: Don't forget to check the if input relation is already materialized or not, if not then materialize it
	if (types.size() != input_relation.columns.size()) {
		throw InvalidInputException("Column count mismatch");
	}

	//measure time
	auto start = std::chrono::high_resolution_clock::now();
	// auto &gstate = GetGlobalSinkState(input_relation.context);

	auto materialize_start_time = std::chrono::high_resolution_clock::now();

	// First figure out the total number of strings and chars in all of the columns
	size_t all_columns_num_strings = 0;
	size_t all_columns_total_chars = 0;
	for (int col = 0; col < input_relation.columns.size(); col++) {
		DataWrapper column_data_wrapper = input_relation.columns[col]->data_wrapper;
		if(column_data_wrapper.type == ColumnType::VARCHAR) {
			all_columns_num_strings += column_data_wrapper.size;
			all_columns_total_chars += column_data_wrapper.num_bytes;
		}
	}

	size_t all_columns_strings_buffer_size = all_columns_num_strings * sizeof(string_t);
	size_t all_columns_chars_buffer_size = all_columns_total_chars * sizeof(char);
	std::cout << "GPU RESULT COLLECTOR: Creating string buffer " << all_columns_num_strings << " strings with " << all_columns_total_chars << " chars" << std::endl;

	// Now allocate the buffers for the columns
	size_t total_buffer_size = all_columns_strings_buffer_size + all_columns_chars_buffer_size;
	uint8_t* combined_buffer = gpuBufferManager->customCudaHostAlloc<uint8_t>(total_buffer_size);
	string_t* all_columns_string = reinterpret_cast<string_t*>(combined_buffer);
	char* all_columns_chars = reinterpret_cast<char*>(combined_buffer + all_columns_strings_buffer_size);
	std::cout << "GPU RESULT COLLECTOR: Created string buffer of " << all_columns_strings_buffer_size << " bytes and chars buffer of " << all_columns_chars_buffer_size << " bytes" << std::endl;

	size_t size_bytes = 0;
	Allocator& allocator = Allocator::DefaultAllocator();
	uint8_t** host_data = new uint8_t*[input_relation.columns.size()];
	GPUIntermediateRelation materialized_relation(input_relation.columns.size());
	string_t** duckdb_strings = new string_t*[input_relation.columns.size()];
	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

	string_t* curr_column_string_buffer = all_columns_string;
	char* curr_column_chars_buffer = all_columns_chars;
	for (int col = 0; col < input_relation.columns.size(); col++) {
		// TODO: Need to fix this for the future, but for now, we will just return when there is null column
		if (input_relation.columns[col]->data_wrapper.data == nullptr) return SinkResultType::FINISHED;
		// Final materialization
		auto col_materialize_start_time = std::chrono::high_resolution_clock::now();
		size_bytes = FinalMaterialize(input_relation, materialized_relation, col);

		bool is_string = false;
		if(input_relation.columns[col]->data_wrapper.type != ColumnType::VARCHAR) {
			// host_data[col] = allocator.AllocateData(size_bytes);
			host_data[col] = gpuBufferManager->customCudaHostAlloc<uint8_t>(size_bytes);
			callCudaMemcpyDeviceToHost<uint8_t>(host_data[col], materialized_relation.columns[col]->data_wrapper.data, size_bytes, 0);
		} else {
			// Use the helper method to materialize the string on the GPU
			shared_ptr<GPUColumn> str_column = materialized_relation.columns[col];
			materializeStringColumnToDuckdbFormat(str_column, curr_column_chars_buffer, curr_column_string_buffer);
			duckdb_strings[col] = curr_column_string_buffer;
			materialized_relation.columns[col] = str_column;
			is_string = true;

			// Advance the buffer pointers based on this column's details
			DataWrapper str_column_data = str_column->data_wrapper;
			curr_column_chars_buffer += str_column_data.num_bytes * sizeof(char);
			curr_column_string_buffer += str_column_data.size * sizeof(string_t);
		}

		auto col_materialize_end_time = std::chrono::high_resolution_clock::now();
		auto col_materialize_duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(col_materialize_end_time - col_materialize_start_time).count()/1000.0;
		std::cout << "GPU RESULT COLLECTOR: Materializing column with is_string of " << is_string << " took " << col_materialize_duration_ms << " ms" << std::endl;
	}
	auto materialize_end_time = std::chrono::high_resolution_clock::now();
	auto materialize_duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(materialize_end_time - materialize_start_time).count()/1000.0;
	std::cout << "GPU RESULT COLLECTOR: Materialize time of " << materialize_duration_ms << " ms" << std::endl; 

	// // free all input relation columns
	// for (int col = 0; col < input_relation.columns.size(); col++) {
	// 	gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[col]->data_wrapper.data), 0);
    //     if (input_relation.columns[col]->data_wrapper.type == ColumnType::VARCHAR) {
    //         gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[col]->data_wrapper.offset), 0);
    //     }
	// }

	auto chunk_start_time = std::chrono::high_resolution_clock::now();
	size_t total_vector = (materialized_relation.columns[0]->column_length + STANDARD_VECTOR_SIZE - 1) / STANDARD_VECTOR_SIZE;
	result_collection->SetCapacity(total_vector);

	uint64_t read_index = 0;
	size_t remaining = materialized_relation.columns[0]->column_length;
	for (uint64_t vec = 0; vec < total_vector; vec++) {
		DataChunk chunk;
		chunk.InitializeEmpty(types);
		for (int col = 0; col < materialized_relation.columns.size(); col++) {
			if(materialized_relation.columns[col]->data_wrapper.type != ColumnType::VARCHAR) {
				Vector vector = rawDataToVector(host_data[col], vec, materialized_relation.columns[col]->data_wrapper.type);
				chunk.data[col].Reference(vector);
			} else {
				// Add the strings to the vector
				Vector str_vector(LogicalType::VARCHAR, reinterpret_cast<data_ptr_t>(duckdb_strings[col] + read_index));
				chunk.data[col].Reference(str_vector);
			}
		}

		size_t cardinality = std::min(remaining, (size_t) STANDARD_VECTOR_SIZE);
		chunk.SetCardinality(cardinality);
		result_collection->AddChunk(chunk);
		
		remaining -= cardinality; read_index += cardinality;
	}
	auto chunk_end_time = std::chrono::high_resolution_clock::now();
	auto chunking_duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(chunk_end_time - chunk_start_time).count()/1000.0;
	std::cout << "GPU RESULT COLLECTOR: Chunking Time of " << chunking_duration_ms << " ms" << std::endl;

	//measure time
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("Result collector time: %.2f ms\n", duration.count()/1000.0);
	return SinkResultType::FINISHED;
}

unique_ptr<GlobalSinkState> GPUPhysicalMaterializedCollector::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<GPUMaterializedCollectorGlobalState>();
	state->context = context.shared_from_this();
	return std::move(state);
}

unique_ptr<LocalSinkState> GPUPhysicalMaterializedCollector::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<GPUMaterializedCollectorLocalState>();
	return std::move(state);
}

unique_ptr<QueryResult> GPUPhysicalMaterializedCollector::GetResult(GlobalSinkState &state) {
	auto &gstate = state.Cast<GPUMaterializedCollectorGlobalState>();
	// Currently the result will be empty
	if (!gstate.context) throw InvalidInputException("No context set in GPUMaterializedCollectorState");
	auto prop = gstate.context->GetClientProperties();
	auto result = make_uniq<GPUQueryResult>(statement_type, properties, names, types, prop, std::move(result_collection));
	std::cout << "Returning GPUQueryResult with value of " << result->ToString() << std::endl;
	return std::move(result);
}

// bool PhysicalMaterializedCollector::ParallelSink() const {
// 	return parallel;
// }

// bool PhysicalMaterializedCollector::SinkOrderDependent() const {
// 	return true;
// }

} // namespace duckdb