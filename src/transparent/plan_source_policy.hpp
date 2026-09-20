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

#include <cstdint>
#include <string>
#include <vector>

namespace duckdb {
class ClientContext;
class LogicalOperator;
class PhysicalOperator;
}  // namespace duckdb
namespace sirius::transparent {
enum class byte_source_class : uint8_t {
  local_file,
  sirius_owned_s3,
  duckdb_native,
  stream,
  unclassified
};
struct scan_source_policy {
  std::string function_name;
  byte_source_class source = byte_source_class::unclassified;
  bool permits_cpu_replay  = true;
  std::string reason;
};
struct plan_source_policy {
  std::vector<scan_source_policy> scans;
  bool discovery_complete = true;
  [[nodiscard]] bool cpu_replay_permitted() const noexcept;
  [[nodiscard]] bool reads_sirius_owned_s3() const noexcept;
  [[nodiscard]] std::string reason() const;
};
plan_source_policy derive_plan_source_policy(duckdb::PhysicalOperator const&,
                                             duckdb::ClientContext&);
plan_source_policy derive_plan_source_policy(duckdb::LogicalOperator const&,
                                             duckdb::ClientContext&);
// The existing S3 veto precedes the fallback setting; other source vetoes do not.
void require_s3_cpu_replay(plan_source_policy const&,
                           std::string const& sql,
                           std::string const& gpu_error);
// Check remaining source vetoes after require_s3_cpu_replay has succeeded.
void require_non_s3_cpu_replay(plan_source_policy const&, std::string const& gpu_error);
void require_cpu_replay(plan_source_policy const&,
                        std::string const& sql,
                        std::string const& gpu_error);
}  // namespace sirius::transparent
