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
#include <cstddef>
#include <optional>
#include <string_view>
namespace duckdb {
class Vector;
}
namespace sirius::ducklake {
// Borrow from an owner-lane native chunk, never flatten/decompress/copy payload.
// nullopt is SQL NULL; the view expires with the source vector or its next write.
std::optional<std::string_view> borrow_inline_varchar(duckdb::Vector const&, size_t row);
}  // namespace sirius::ducklake
