/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "duckdb.hpp"

namespace duckdb {

void warmup_gpu();
void convertInt64ToInt128(uint8_t *input, uint8_t *output, size_t count);
void convertInt32ToInt128(uint8_t *input, uint8_t *output, size_t count);
void convertInt32ToInt64(uint8_t *input, uint8_t *output, size_t count);

} // namespace duckdb