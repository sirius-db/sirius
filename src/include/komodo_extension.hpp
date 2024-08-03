#pragma once

#include "duckdb.hpp"
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