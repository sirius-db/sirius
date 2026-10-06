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

#include "transparent/plan_source_policy.hpp"

#include "planner/connector_registry.hpp"
#include "sirius_sql_rewrite.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/multi_file/multi_file_list.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/planner/operator/logical_get.hpp>

#include <algorithm>
#include <stdexcept>

namespace sirius::transparent {
bool plan_source_policy::cpu_replay_permitted() const noexcept
{
  return discovery_complete && std::all_of(scans.begin(), scans.end(), [](auto const& scan) {
           return scan.permits_cpu_replay;
         });
}
bool plan_source_policy::reads_sirius_owned_s3() const noexcept
{
  return std::any_of(scans.begin(), scans.end(), [](auto const& scan) {
    return scan.source == byte_source_class::sirius_owned_s3;
  });
}
std::string plan_source_policy::reason() const
{
  for (auto const& scan : scans) {
    if (!scan.permits_cpu_replay) return scan.function_name + ": " + scan.reason;
  }
  return discovery_complete ? "" : "source discovery incomplete";
}

namespace {
scan_source_policy classify(duckdb::TableFunction const& function,
                            duckdb::FunctionData const* bind,
                            duckdb::ClientContext& context)
{
  scan_source_policy result{function.name, byte_source_class::unclassified, true, ""};
  auto const* entry = planner::lookup_connector(function, bind, context);
  if (entry) {
    result.source             = entry->byte_source;
    result.permits_cpu_replay = entry->permits_cpu_replay;
    if (result.source == byte_source_class::stream) result.reason = "stream source has no CPU body";
    if (result.source == byte_source_class::sirius_owned_s3)
      result.reason = "S3 CPU fallback is not supported";
  }
  if (auto const* files = dynamic_cast<duckdb::MultiFileBindData const*>(bind)) {
    if (!files->file_list) throw std::runtime_error("Missing bound file list");
    result.source             = byte_source_class::local_file;
    result.permits_cpu_replay = true;
    for (auto const& file : files->file_list->Files()) {
      auto const& path = file.path;
      if (path.size() > 5 && (path[0] == 's' || path[0] == 'S') && path[1] == '3' &&
          path[2] == ':' && path[3] == '/' && path[4] == '/') {
        result.source             = byte_source_class::sirius_owned_s3;
        result.permits_cpu_replay = false;
        result.reason             = "S3 CPU fallback is not supported";
        break;
      }
    }
  }
  return result;
}
// Keep discovering siblings so an incomplete node cannot hide another source's S3 veto.
template <class F>
void discover(plan_source_policy& policy, F&& operation)
{
  try {
    operation();
  } catch (duckdb::InterruptException&) {
    throw;
  } catch (...) {
    policy.discovery_complete = false;
  }
}

void walk(duckdb::LogicalOperator const& node,
          duckdb::ClientContext& context,
          plan_source_policy& policy)
{
  discover(policy, [&] {
    if (node.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
      auto const& get = node.Cast<duckdb::LogicalGet>();
      policy.scans.push_back(classify(get.function, get.bind_data.get(), context));
    }
  });
  for (auto const& child : node.children) {
    walk(*child, context, policy);
  }
}

void walk(duckdb::PhysicalOperator const& node,
          duckdb::ClientContext& context,
          plan_source_policy& policy)
{
  discover(policy, [&] {
    if (node.type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
      auto const& get = node.Cast<duckdb::PhysicalTableScan>();
      policy.scans.push_back(classify(get.function, get.bind_data.get(), context));
    }
  });
  discover(policy, [&] {
    for (auto const& child : node.GetChildren()) {
      walk(child.get(), context, policy);
    }
  });
}

template <class Plan>
plan_source_policy derive(Plan const& plan, duckdb::ClientContext& context)
{
  plan_source_policy result;
  try {
    walk(plan, context, result);
  } catch (duckdb::InterruptException&) {
    throw;
  } catch (...) {
    result.discovery_complete = false;
  }
  return result;
}
}  // namespace

plan_source_policy derive_plan_source_policy(duckdb::PhysicalOperator const& plan,
                                             duckdb::ClientContext& context)
{
  return derive(plan, context);
}
plan_source_policy derive_plan_source_policy(duckdb::LogicalOperator const& plan,
                                             duckdb::ClientContext& context)
{
  return derive(plan, context);
}

void require_s3_cpu_replay(plan_source_policy const& policy,
                           std::string const& sql,
                           std::string const& gpu_error)
{
  if (policy.reads_sirius_owned_s3() || sirius::references_sirius_owned_s3_parquet(sql)) {
    throw std::runtime_error(
      "S3 CPU fallback is not supported: this query reads s3:// data, GPU execution failed, and "
      "Sirius has no CPU fallback for S3 data sources. Underlying GPU error: " +
      gpu_error);
  }
}

void require_cpu_replay(plan_source_policy const& policy,
                        std::string const& sql,
                        std::string const& gpu_error)
{
  require_s3_cpu_replay(policy, sql, gpu_error);
  require_non_s3_cpu_replay(policy, gpu_error);
}

void require_non_s3_cpu_replay(plan_source_policy const& policy, std::string const& gpu_error)
{
  if (!policy.cpu_replay_permitted()) {
    throw std::runtime_error("CPU fallback is not supported: " + policy.reason() +
                             ". Underlying GPU error: " + gpu_error);
  }
}
}  // namespace sirius::transparent
