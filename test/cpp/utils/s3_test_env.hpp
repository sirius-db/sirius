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

#include "catch.hpp"
#include "memory/topology_index.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::test::s3 {

// require_env, require_rest_ioctx and skip_or_fail_unless assert via Catch2; test thread only.

inline std::string env_or(std::string_view name, std::string fallback = {})
{
  auto const* value = std::getenv(std::string{name}.c_str());
  return value ? std::string{value} : std::move(fallback);
}

inline std::string require_env(std::string_view name)
{
  auto value = env_or(name);
  REQUIRE_FALSE(value.empty());
  return value;
}

inline std::string sql_quote(std::string_view value)
{
  std::string out{"'"};
  for (char c : value) {
    if (c == '\'') { out.push_back('\''); }
    out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

inline cucascade::memory::system_topology_info single_gpu_topology(int numa_node)
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = numa_node;
  topology.gpus.push_back(std::move(gpu));
  return topology;
}

inline std::shared_ptr<const sirius::memory::topology_index> single_gpu_index(int numa_node)
{
  return std::make_shared<sirius::memory::topology_index>(single_gpu_topology(numa_node),
                                                          std::vector<int>{0});
}

inline cucascade::io::rest::rest_ioctx* require_rest_ioctx(
  std::shared_ptr<cucascade::io::datasource> const& ds)
{
  REQUIRE(ds != nullptr);
  REQUIRE(ds->io_ctx() != nullptr);
  CHECK(ds->io_ctx()->type() == cucascade::io::io_context_type::restful);
  auto* rest_ctx = dynamic_cast<cucascade::io::rest::rest_ioctx*>(ds->io_ctx().get());
  REQUIRE(rest_ctx != nullptr);
  return rest_ctx;
}

inline bool truthy_env(std::string_view name)
{
  auto const* raw  = std::getenv(std::string{name}.c_str());
  auto const value = raw ? std::string_view{raw} : std::string_view{};
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

inline bool strict_mode() { return truthy_env("SIRIUS_TEST_S3_STRICT"); }

[[nodiscard]] inline bool skip_or_fail_unless(bool ready, std::string_view reason)
{
  if (ready) { return false; }
  if (strict_mode()) { FAIL(reason); }
  SUCCEED(reason);
  return true;
}

}  // namespace sirius::test::s3
