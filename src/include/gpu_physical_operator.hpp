#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/enums/operator_result_type.hpp"
#include "duckdb/common/enums/physical_operator_type.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/optimizer/join_order/join_node.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/execution/physical_operator_states.hpp"
#include "duckdb/common/enums/order_preservation_type.hpp"
#include "gpu_columns.hpp"
#include "helper/types.hpp"

namespace duckdb {
class GPUExecutor;
class GPUPhysicalOperator;
class GPUPipeline;
class GPUPipelineBuildState;
class GPUMetaPipeline;

//! GPUPhysicalOperator is the base class of the physical operators present in the
//! execution plan
class GPUPhysicalOperator{
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

public:
	
	GPUPhysicalOperator(PhysicalOperatorType type, vector<LogicalType> types, idx_t estimated_cardinality)
	    : type(type), types(std::move(types)), estimated_cardinality(estimated_cardinality) {
	}
	GPUPhysicalOperator() = default;

	virtual ~GPUPhysicalOperator() {
	}
	// ~GPUPhysicalOperator() = default;

	//! The physical operator type
	PhysicalOperatorType type;
	//! The set of children of the operator
	vector<unique_ptr<GPUPhysicalOperator>> children;
	//! The types returned by this physical operator
	vector<LogicalType> types;
	//! The estimated cardinality of this physical operator
	idx_t estimated_cardinality;

	//! The global sink state of this operator
	unique_ptr<GlobalSinkState> sink_state;
	//! The global state of this operator
	unique_ptr<GlobalOperatorState> op_state;
	//! Lock for (re)setting any of the operator states
	mutex lock;

public:
	virtual string GetName() const;
	// virtual string ParamsToString() const {
	// 	return "";
	// }
	// virtual string ToString() const;
	// void Print() const;
	virtual vector<const_reference<GPUPhysicalOperator>> GetChildren() const;

	//! Return a vector of the types that will be returned by this operator
	const vector<LogicalType> &GetTypes() const {
		return types;
	}

	virtual bool Equals(const GPUPhysicalOperator &other) const {
		return false;
	}

	virtual void Verify();

public:
	// Operator interface
	virtual unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const;
	virtual unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const;
	// virtual OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	//                                    GlobalOperatorState &gstate, OperatorState &state) const;
	// virtual OperatorResultType Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation,
	// 									GlobalOperatorState &gstate, OperatorState &state) const;
	// virtual OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk,
	//                                                 GlobalOperatorState &gstate, OperatorState &state) const;

	virtual OperatorResultType Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const;

	virtual bool ParallelOperator() const {
		return false;
	}

	virtual bool RequiresFinalExecute() const {
		return false;
	}

	//! The influence the operator has on order (insertion order means no influence)
	virtual OrderPreservationType OperatorOrder() const {
		return OrderPreservationType::INSERTION_ORDER;
	}

public:
	//Source Interface
	// virtual SourceResultType GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const;
	virtual SourceResultType GetData(GPUIntermediateRelation &output_relation) const;
	virtual unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                         GlobalSourceState &gstate) const;
	virtual unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const;

	virtual bool IsSource() const {
		return false;
	}

	virtual bool ParallelSource() const {
		return false;
	}

	//! The type of order emitted by the operator (as a source)
	virtual OrderPreservationType SourceOrder() const {
		return OrderPreservationType::INSERTION_ORDER;
	}

public:
	//Sink interface
	// virtual SinkResultType Sink(ExecutionContext &context, GPUIntermediateRelation &input, OperatorSinkInput &input) const;
	// virtual SinkResultType Sink(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input) const;
	virtual SinkResultType Sink(GPUIntermediateRelation &input_relation) const;
	virtual unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const;
	virtual unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const;
	
	virtual bool IsSink() const {
		return false;
	}

	virtual bool ParallelSink() const {
		return false;
	}

	virtual bool RequiresBatchIndex() const {
		return false;
	}

	//! Whether or not the sink operator depends on the order of the input chunks
	//! If this is set to true, we cannot do things like caching intermediate vectors
	virtual bool SinkOrderDependent() const {
		return false;
	}

public:
	// Pipeline construction
	virtual vector<const_reference<GPUPhysicalOperator>> GetSources() const;
	// bool AllSourcesSupportBatchIndex() const;

	virtual void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline);

public:
	template <class TARGET>
	TARGET &Cast() {
		if (TARGET::TYPE != PhysicalOperatorType::INVALID && type != TARGET::TYPE) {
			throw InternalException("Failed to cast physical operator to type - physical operator type mismatch");
		}
		return reinterpret_cast<TARGET &>(*this);
	}

	template <class TARGET>
	const TARGET &Cast() const {
		if (TARGET::TYPE != PhysicalOperatorType::INVALID && type != TARGET::TYPE) {
			throw InternalException("Failed to cast physical operator to type - physical operator type mismatch");
		}
		return reinterpret_cast<const TARGET &>(*this);
	}
};

} // namespace duckdb
