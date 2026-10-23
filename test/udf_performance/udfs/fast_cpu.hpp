/*
 * Copyright 2026, Sirius Contributors.
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

#include <duckdb.hpp>
#include <duckdb/common/vector/string_vector.hpp>
#include <duckdb/common/vector_operations/unary_executor.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/extension/extension_loader.hpp>

#include <cstdint>
#include <utility>

// CPU-only benchmark baselines for the frozen, independently synthetic UDFs.
// Typed vector execution propagates validity and handles flat, constant and
// dictionary inputs without constructing a DuckDB Value for each row.
inline void register_fast_cpu(duckdb::ExtensionLoader& loader)
{
  duckdb::ScalarFunction mix(
    "bench_mix_cpu",
    {duckdb::LogicalType::UBIGINT},
    duckdb::LogicalType::UBIGINT,
    [](duckdb::DataChunk& input, duckdb::ExpressionState&, duckdb::Vector& output) {
      duckdb::UnaryExecutor::Execute<std::uint64_t, std::uint64_t>(
        input.data[0],
        output,
        input.size(),
        [](std::uint64_t value) {
          auto word = value ^ (value >> 17);
          // Unsigned C++ arithmetic wraps modulo 2^64, as the original requires.
          word *= UINT64_C(0x9E3779B185EBCA87);
          auto const rotated = (word << 23) | (word >> 41);
          return rotated ^ UINT64_C(0xD1B54A32D192ED03);
        },
        duckdb::FunctionErrors::CANNOT_ERROR);
    });
  mix.SetNullHandling(duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING);
  loader.RegisterFunction(std::move(mix));

  duckdb::ScalarFunction normalize(
    "bench_normalize_cpu",
    {duckdb::LogicalType::VARCHAR},
    duckdb::LogicalType::VARCHAR,
    [](duckdb::DataChunk& input, duckdb::ExpressionState&, duckdb::Vector& output) {
      duckdb::UnaryExecutor::Execute<duckdb::string_t, duckdb::string_t>(
        input.data[0],
        output,
        input.size(),
        [&output](duckdb::string_t value) {
          auto const* source  = value.GetData();
          duckdb::idx_t begin = 0;
          auto end            = value.GetSize();
          while (begin < end && (source[begin] == ' ' || source[begin] == '\t')) {
            ++begin;
          }
          while (end > begin && (source[end - 1] == ' ' || source[end - 1] == '\t')) {
            --end;
          }

          // ASCII contract: trimming sets the final size, and replacement and
          // uppercasing each preserve it. Write once into the result's storage.
          auto result  = duckdb::StringVector::EmptyString(output, end - begin);
          auto* target = result.GetDataWriteable();
          for (duckdb::idx_t index = begin; index < end; ++index) {
            auto character = source[index];
            if (character == '-') {
              character = '_';
            } else if (character >= 'a' && character <= 'z') {
              character = static_cast<char>(character - 'a' + 'A');
            }
            target[index - begin] = character;
          }
          result.Finalize();
          return result;
        },
        duckdb::FunctionErrors::CANNOT_ERROR);
    });
  normalize.SetNullHandling(duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING);
  loader.RegisterFunction(std::move(normalize));
}
