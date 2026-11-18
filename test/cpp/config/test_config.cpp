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
#include "config/user_config.hpp"
#include "sirius_config.hpp"

#include <filesystem>
#include <fstream>
#include <limits>

using namespace sirius;

TEST_CASE("operator parameters start with hardware-independent defaults", "[config_opt][defaults]")
{
  const operator_params params;
  CHECK(params.scan_task_batch_size == config::DEFAULT_BATCH_SIZE);
  CHECK(params.hash_partition_bytes == config::DEFAULT_BATCH_SIZE);
  CHECK(params.concat_batch_bytes == config::DEFAULT_BATCH_SIZE);
  CHECK(params.sort_sample_bytes == config::DEFAULT_BATCH_SIZE);
  CHECK(params.max_build_hash_table_bytes == 8 * config::DEFAULT_BATCH_SIZE);
  CHECK(params.max_build_probe_resident_bytes == 8 * config::DEFAULT_BATCH_SIZE);
  CHECK(params.max_dynamic_filter_bloom_bytes_per_gpu() == 4 * params.scan_task_batch_size);
}

TEST_CASE("the domain-coverage threshold is validated where it enters the engine",
          "[config_opt][conditional][dynamic_filter]")
{
  // YAML and SQL configuration share this ingress validator.
  config::valid_domain_coverage_threshold const accepts;

  REQUIRE(accepts(0.9));
  REQUIRE(accepts(1.5));  // values above 1.0 disable the gate

  REQUIRE_FALSE(accepts(0.0));  // would suppress every filter
  REQUIRE_FALSE(accepts(-0.5));
  REQUIRE_FALSE(accepts(std::numeric_limits<double>::quiet_NaN()));
  REQUIRE_FALSE(accepts(std::numeric_limits<double>::infinity()));

  // The YAML surface rejects the same values, naming the key.
  REQUIRE_THROWS_WITH(config::read_user_config("sirius:\n  operator_params:\n"
                                               "    dynamic_filter_domain_coverage_threshold: 0\n"),
                      Catch::Matchers::StartsWith(
                        "'sirius.operator_params.dynamic_filter_domain_coverage_threshold'"));
  REQUIRE(config::read_user_config(
            "sirius:\n  operator_params:\n    dynamic_filter_domain_coverage_threshold: 1.5\n")
            .operator_params.dynamic_filter_domain_coverage_threshold.get() == 1.5);
}

TEST_CASE("the dynamic-filter switch is consumed from the operator_params YAML section",
          "[config_opt][dynamic_filter]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_dynamic_filter.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  operator_params:\n"
           "    enable_dynamic_filter: false\n";
  }

  // Parsing false is the non-vacuous direction because the default is true.
  CHECK(operator_params{}.enable_dynamic_filter);

  sirius_config cfg;
  cfg.load_from_file(path);
  CHECK_FALSE(cfg.get_operator_params().enable_dynamic_filter);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("multi-partition dynamic-filter switch preserves the derived budget",
          "[config_opt][dynamic_filter][multi_partition]")
{
  auto const path = std::filesystem::temp_directory_path() / "sirius_dynamic_filter_multi.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  operator_params:\n"
           "    enable_dynamic_filter_multi_partition: false\n";
  }

  sirius_config cfg;
  CHECK(cfg.get_operator_params().enable_dynamic_filter_multi_partition);
  CHECK(cfg.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu() ==
        4 * cfg.get_operator_params().scan_task_batch_size);
  cfg.load_from_file(path);
  CHECK_FALSE(cfg.get_operator_params().enable_dynamic_filter_multi_partition);
  CHECK(cfg.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu() ==
        4 * cfg.get_operator_params().scan_task_batch_size);
  CHECK(cfg.get_operator_params().enable_dynamic_filter);
  CHECK(cfg.get_operator_params().dynamic_filter_domain_coverage_threshold == Approx(0.9));
  cfg.apply_defaults();
  CHECK(cfg.get_operator_params().enable_dynamic_filter_multi_partition);
  CHECK(cfg.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu() ==
        4 * cfg.get_operator_params().scan_task_batch_size);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("multi-partition dynamic-filter budget rejects YAML overrides",
          "[config_opt][dynamic_filter][multi_partition]")
{
  auto const budget = GENERATE(std::string{"-1"}, std::string{"0"}, std::string{"32MiB"});
  auto const path = std::filesystem::temp_directory_path() / "sirius_dynamic_filter_multi_bad.yaml";
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  operator_params:\n"
           "    max_dynamic_filter_bloom_bytes_per_gpu: "
        << budget << '\n';
  }
  sirius_config cfg;
  CHECK_THROWS_WITH(cfg.load_from_file(path),
                    Catch::Matchers::ContainsSubstring("always 4x scan_task_batch_size"));
  std::error_code ec;
  std::filesystem::remove(path, ec);
}
