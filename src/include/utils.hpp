#pragma once

#include "duckdb.hpp"

namespace duckdb {

void warmup_gpu();
void convertInt64ToInt128(uint8_t *input, uint8_t *output, size_t count);
void convertInt32ToInt128(uint8_t *input, uint8_t *output, size_t count);
void convertInt32ToInt64(uint8_t *input, uint8_t *output, size_t count);

} // namespace duckdb