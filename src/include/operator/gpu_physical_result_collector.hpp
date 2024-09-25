#pragma once

#include "gpu_physical_operator.hpp"
#include "duckdb/common/enums/statement_type.hpp"

namespace duckdb {

class GPUPreparedStatementData;

class GPUPhysicalResultCollector : public GPUPhysicalOperator {

public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::RESULT_COLLECTOR;

public:
	explicit GPUPhysicalResultCollector(GPUPreparedStatementData &data);

	StatementType statement_type;
	StatementProperties properties;
	GPUPhysicalOperator &plan;
	vector<string> names;

// public:
// 	static unique_ptr<PhysicalResultCollector> GetResultCollector(ClientContext &context, PreparedStatementData &data);

public:
	// //! The final method used to fetch the query result from this operator
	virtual unique_ptr<QueryResult> GetResult(GlobalSinkState &state) = 0;
	
	bool IsSink() const override {
		return true;
	}

public:
	vector<const_reference<GPUPhysicalOperator>> GetChildren() const override;
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;

	bool IsSource() const override {
		return true;
	}
};


class GPUPhysicalMaterializedCollector : public GPUPhysicalResultCollector {
public:
	GPUPhysicalMaterializedCollector(GPUPreparedStatementData &data);

	// bool parallel;

public:
	unique_ptr<QueryResult> GetResult(GlobalSinkState &state) override;

public:
	// Sink interface
	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;
	// SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	// bool ParallelSink() const override;
	// bool SinkOrderDependent() const override;
};

} // namespace duckdb