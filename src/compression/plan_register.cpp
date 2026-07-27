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

#include "plan_register.hpp"

#include <api/simpatico_codegen.hpp>

#include <mutex>
#include <shared_mutex>
#include <utility>

namespace sirius::compression {

plan_register& plan_register::global()
{
  static plan_register instance;
  return instance;
}

void plan_register::set_table_plan(const std::string& table_name, std::string full_plan_dsl)
{
  std::unique_lock lock(_mutex);
  _table_plans[table_name] = std::move(full_plan_dsl);
}

void plan_register::clear_table_plan(const std::string& table_name)
{
  std::unique_lock lock(_mutex);
  _table_plans.erase(table_name);
}

std::optional<std::string> plan_register::resolve_table_plan(const std::string& table_name) const
{
  std::shared_lock lock(_mutex);
  auto it = _table_plans.find(table_name);
  if (it != _table_plans.end() && !it->second.empty()) { return it->second; }
  return std::nullopt;
}

void plan_register::set_plan(const std::string& table_name,
                             const std::string& column_name,
                             std::string plan_dsl)
{
  std::unique_lock lock(_mutex);
  _col_plans[table_name + "::" + column_name] = std::move(plan_dsl);
}

void plan_register::clear_plan(const std::string& table_name, const std::string& column_name)
{
  std::unique_lock lock(_mutex);
  _col_plans.erase(table_name + "::" + column_name);
}

void plan_register::set_spill_plan(const cucascade::shared_data_repository* repo,
                                   std::string plan_dsl)
{
  std::unique_lock lock(_mutex);
  _spill_plans[repo] = std::move(plan_dsl);
}

void plan_register::clear_spill_plan(const cucascade::shared_data_repository* repo)
{
  std::unique_lock lock(_mutex);
  _spill_plans.erase(repo);
}

std::optional<std::string> plan_register::resolve_spill_plan(
  const cucascade::shared_data_repository* repo) const
{
  std::shared_lock lock(_mutex);
  auto it = _spill_plans.find(repo);
  if (it != _spill_plans.end() && !it->second.empty()) { return it->second; }
  return std::nullopt;
}

void plan_register::clear_all()
{
  std::unique_lock lock(_mutex);
  _table_plans.clear();
  _col_plans.clear();
  _spill_plans.clear();
}

std::optional<std::string> select_plan_blocks(const std::string& full_plan_dsl,
                                              const std::vector<std::size_t>& column_indices)
{
  auto blocks = simpatico::split_plan_dsl(full_plan_dsl);
  std::string out;
  for (std::size_t k = 0; k < column_indices.size(); ++k) {
    std::size_t const idx = column_indices[k];
    if (idx >= blocks.size()) { return std::nullopt; }
    if (k != 0) { out += "\n---\n"; }
    out += blocks[idx];
  }
  return out;
}

}  // namespace sirius::compression
