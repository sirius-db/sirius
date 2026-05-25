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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace duckdb {

struct PinTableArgs {
  std::string path;
  std::string tier;
  std::string name;
  std::optional<std::vector<std::string>> cols;
  std::optional<int64_t> n_rows;
};

void pin_table_to(const PinTableArgs& args);

void unpin_table_to(const std::string& name);

// Test-only helpers used by unit tests to verify pin_table forwards the
// correct parameters to pin_table_to. Not part of the stable public API.
namespace pin_table_testing {
const std::vector<PinTableArgs>& recorded_calls();
void clear_recorded_calls();

const std::vector<std::string>& recorded_unpin_calls();
void clear_recorded_unpin_calls();
}  // namespace pin_table_testing

}  // namespace duckdb
