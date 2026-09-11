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

#include "catch.hpp"
#include "sirius_config.hpp"

#include <sched.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct scoped_yaml {
  scoped_yaml(std::string const& name, std::string const& text)
    : path(std::filesystem::temp_directory_path() / name)
  {
    std::ofstream out(path);
    out << text;
    REQUIRE(out);
  }

  ~scoped_yaml()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  scoped_yaml(scoped_yaml const&)            = delete;
  scoped_yaml& operator=(scoped_yaml const&) = delete;

  std::filesystem::path path;
};

int first_allowed_cpu()
{
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  REQUIRE(sched_getaffinity(0, sizeof(cpu_set_t), &allowed) == 0);
  for (int id = 0; id < CPU_SETSIZE; ++id) {
    if (CPU_ISSET(id, &allowed)) { return id; }
  }
  FAIL("current process has no allowed CPUs");
  return -1;
}

int first_disallowed_cpu()
{
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  REQUIRE(sched_getaffinity(0, sizeof(cpu_set_t), &allowed) == 0);
  for (int id = 0; id < CPU_SETSIZE; ++id) {
    if (!CPU_ISSET(id, &allowed)) { return id; }
  }
  return -1;
}

std::string affinity_yaml(std::string const& block, std::string const& value)
{
  return "sirius:\n"
         "  executor:\n"
         "    " +
         block +
         ":\n"
         "      cpu_affinity: " +
         value + "\n";
}

}  // namespace

TEST_CASE("sirius_config accepts shared affinity masks for the three user-configurable pools",
          "[config][cpu_affinity]")
{
  int const cpu = first_allowed_cpu();
  scoped_yaml yaml("sirius_valid_cpu_affinity.yaml",
                   "sirius:\n"
                   "  executor:\n"
                   "    task_creator:\n"
                   "      cpu_affinity: [" +
                     std::to_string(cpu) +
                     "]\n"
                     "    scan_manager:\n"
                     "      cpu_affinity: [" +
                     std::to_string(cpu) +
                     "]\n"
                     "    downgrade:\n"
                     "      cpu_affinity: [" +
                     std::to_string(cpu) + "]\n");

  sirius::sirius_config config;
  REQUIRE_NOTHROW(config.load_from_file(yaml.path));
  CHECK(config.get_task_creator_config().thread_pool.cpu_affinity_list == std::vector<int>{cpu});
  CHECK(config.get_scan_manager_config().thread_pool.cpu_affinity_list == std::vector<int>{cpu});
  CHECK(config.get_downgrade_executor_config().thread_pool.cpu_affinity_list ==
        std::vector<int>{cpu});
}

TEST_CASE("sirius_config preserves inherited affinity when lists are empty",
          "[config][cpu_affinity]")
{
  scoped_yaml yaml("sirius_empty_cpu_affinity.yaml",
                   "sirius:\n"
                   "  executor:\n"
                   "    task_creator: {cpu_affinity: []}\n"
                   "    scan_manager: {cpu_affinity: []}\n"
                   "    downgrade: {cpu_affinity: []}\n");

  sirius::sirius_config config;
  REQUIRE_NOTHROW(config.load_from_file(yaml.path));
  CHECK(config.get_task_creator_config().thread_pool.cpu_affinity_list.empty());
  CHECK(config.get_scan_manager_config().thread_pool.cpu_affinity_list.empty());
  CHECK(config.get_downgrade_executor_config().thread_pool.cpu_affinity_list.empty());
}

TEST_CASE("sirius_config rejects invalid affinity on every user-configurable pool",
          "[config][cpu_affinity]")
{
  auto const block =
    GENERATE(std::string{"task_creator"}, std::string{"scan_manager"}, std::string{"downgrade"});
  CAPTURE(block);

  SECTION("negative CPU ID")
  {
    scoped_yaml yaml("sirius_negative_" + block + "_cpu_affinity.yaml",
                     affinity_yaml(block, "[-1]"));
    sirius::sirius_config config;
    CHECK_THROWS_WITH(config.load_from_file(yaml.path), Catch::Contains("must be non-negative"));
  }

  SECTION("CPU ID at CPU_SETSIZE")
  {
    scoped_yaml yaml("sirius_large_" + block + "_cpu_affinity.yaml",
                     affinity_yaml(block, "[" + std::to_string(CPU_SETSIZE) + "]"));
    sirius::sirius_config config;
    CHECK_THROWS_WITH(config.load_from_file(yaml.path), Catch::Contains("CPU_SETSIZE"));
  }

  SECTION("scalar instead of list")
  {
    scoped_yaml yaml("sirius_scalar_" + block + "_cpu_affinity.yaml",
                     affinity_yaml(block, std::to_string(first_allowed_cpu())));
    sirius::sirius_config config;
    CHECK_THROWS_WITH(config.load_from_file(yaml.path), Catch::Contains("expected a sequence"));
  }

  SECTION("CPU ID outside current process allowed mask")
  {
    int const disallowed = first_disallowed_cpu();
    if (disallowed < 0) {
      WARN("current process is allowed on every CPU_SETSIZE slot; disallowed-ID case skipped");
      return;
    }
    scoped_yaml yaml("sirius_disallowed_" + block + "_cpu_affinity.yaml",
                     affinity_yaml(block, "[" + std::to_string(disallowed) + "]"));
    sirius::sirius_config config;
    CHECK_THROWS_WITH(config.load_from_file(yaml.path),
                      Catch::Contains("current process allowed CPU mask"));
  }
}
