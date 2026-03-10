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

#include <string>
#include <string_view>
#include <unordered_map>

namespace sirius::op::scan {

/// Controls where and how scan results are cached between query executions.
enum class cache_level {
  NONE,        ///< No caching
  TABLE_GPU,   ///< Cache decoded table in GPU memory
  TABLE_HOST,  ///< Cache decoded table in host memory
  PARQUET,     ///< Cache raw parquet data in host memory
};

inline bool string_to_enum(std::string_view sv, cache_level& level)
{
  static const std::unordered_map<std::string_view, cache_level> map = {
    {"none", cache_level::NONE},
    {"off", cache_level::NONE},
    {"table_gpu", cache_level::TABLE_GPU},
    {"table_host", cache_level::TABLE_HOST},
    {"parquet", cache_level::PARQUET},
  };
  auto it = map.find(sv);
  if (it != map.end()) {
    level = it->second;
    return true;
  }
  return false;
}

inline bool enum_to_string(cache_level level, std::string& s)
{
  switch (level) {
    case cache_level::NONE: s = "none"; return true;
    case cache_level::TABLE_GPU: s = "table_gpu"; return true;
    case cache_level::TABLE_HOST: s = "table_host"; return true;
    case cache_level::PARQUET: s = "parquet"; return true;
  }
  return false;
}

}  // namespace sirius::op::scan
