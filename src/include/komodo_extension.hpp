#pragma once

#include "duckdb.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_executor.hpp"
#include "communication.hpp"

namespace duckdb {

// Declaration of the CUDA kernel
extern void myKernel();

class KomodoExtension : public Extension {
public:
	void Load(DuckDB &db) override;
	std::string Name() override;
};

} // namespace duckdb