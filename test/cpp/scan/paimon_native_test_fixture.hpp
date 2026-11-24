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
#include "op/scan/paimon/paimon_native_read.hpp"

#include <dlfcn.h>
#include <paimon/memory/memory_pool.h>

#include <fstream>
#include <sstream>
namespace sirius::test::paimon {
struct native_fixture {
  std::shared_ptr<::paimon::MemoryPool> pool{::paimon::GetMemoryPool()};

  native_fixture()
  {
    static bool const loaded = [] {
      for (auto path :
           {SIRIUS_PAIMON_LOCAL_MODULE, SIRIUS_PAIMON_AVRO_MODULE, SIRIUS_PAIMON_PARQUET_MODULE}) {
        auto module = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!module) throw std::runtime_error(dlerror());
        // SDK's process-global registry owns factories from this DSO until exit.
        // Keep its mapping resident; per-reader dlclose would leave dangling factories.
      }
      return true;
    }();
    (void)loaded;
  }
};
inline sirius::op::scan::paimon_read_metadata fixture_metadata()
{
  auto path = std::string(SIRIUS_PROJECT_ROOT) +
              "/test/cpp/integration/data/paimon_conformance/warehouse/reference.db/append_basic";
  std::ifstream file(path + "/schema/schema-0");
  REQUIRE(file.good());
  std::ostringstream schema;
  schema << file.rdbuf();
  sirius::op::scan::paimon_read_metadata metadata{path, "main", schema.str(), 0, 1, {}};
  return metadata;
}
}  // namespace sirius::test::paimon
