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

void plan_register::clear_all()
{
  std::unique_lock lock(_mutex);
  _table_plans.clear();
  _col_plans.clear();
}

}  // namespace sirius::compression
