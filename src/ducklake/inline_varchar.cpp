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

#include "ducklake/inline_varchar.hpp"

#include <duckdb/common/types/vector.hpp>
#include <duckdb/common/vector/constant_vector.hpp>
#include <duckdb/common/vector/dictionary_vector.hpp>
#include <duckdb/common/vector/flat_vector.hpp>

#include <stdexcept>
namespace sirius::ducklake {
std::optional<std::string_view> borrow_inline_varchar(duckdb::Vector const& input, size_t row)
{
  auto* current = &input;
  for (;;) {
    if (current->GetType() != duckdb::LogicalType::VARCHAR || row >= current->size())
      throw std::invalid_argument("DuckLake inline VARCHAR has invalid type or row");
    switch (current->GetVectorType()) {
      case duckdb::VectorType::DICTIONARY_VECTOR:
        row     = duckdb::DictionaryVector::SelVector(*current).get_index(row);
        current = &duckdb::DictionaryVector::Child(*current);
        break;
      case duckdb::VectorType::FLAT_VECTOR: {
        if (!duckdb::FlatVector::Validity(*current).RowIsValid(row)) return std::nullopt;
        auto const& value = duckdb::FlatVector::GetData<duckdb::string_t>(*current)[row];
        return std::string_view(value.GetData(), value.GetSize());
      }
      case duckdb::VectorType::CONSTANT_VECTOR: {
        if (duckdb::ConstantVector::IsNull(*current)) return std::nullopt;
        auto const& value = duckdb::ConstantVector::GetData<duckdb::string_t>(*current)[0];
        return std::string_view(value.GetData(), value.GetSize());
      }
      default:
        throw std::invalid_argument(
          "DuckLake inline VARCHAR requires a borrowable vector representation");
    }
  }
}
}  // namespace sirius::ducklake
