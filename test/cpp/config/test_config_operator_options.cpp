// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/options/operator.hpp"
#include "config/options/topology.hpp"
#include "config/yaml.hpp"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>

using namespace sirius::config;
using namespace sirius::config::size_literals;

namespace {

struct operator_root : block_base<operator_root> {
  topology_options topology;
  operator_options operator_params;
  auto fields()
  {
    return field_list(sub("topology", topology), sub("operator_params", operator_params));
  }
};

operator_root read(std::string_view text)
{
  operator_root root;
  read_yaml(root, text, "sirius", environment::prod);
  return root;
}

std::string read_error(std::string_view text)
{
  operator_root root;
  try {
    read_yaml(root, text, "sirius", environment::prod);
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

std::string fixture(std::string_view name)
{
  auto const dir =
    std::filesystem::path(std::source_location::current().file_name()).parent_path() / "data";
  std::ifstream in(dir / name);
  REQUIRE(in);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

}  // namespace

TEST_CASE("operator options declare their keys once", "[config][config_operator]")
{
  operator_root{}.check_declaration();
}

TEST_CASE("absent operator options take operator_params' defaults", "[config][config_operator]")
{
  auto const root      = read("sirius: {}\n");
  auto const& op       = root.operator_params;
  auto const& defaults = operator_options::defaults();
  REQUIRE(op.enable_late_materialization.get() == defaults.enable_late_materialization);
  REQUIRE(op.late_materialization_min_boundaries.get() == 4);
  REQUIRE(op.fused_scan_max_selectivity.get() == 0.35);
  REQUIRE(op.max_sort_partition_bytes.get() == 0_B);
  REQUIRE(op.max_broadcast_join_size.get() == 256_MiB);
  REQUIRE(op.mark_join_build_switch_ratio.get() == 8.0);
  REQUIRE(op.dynamic_filter_domain_coverage_threshold.get() == 0.9);
  REQUIRE(op.avg_variable_column_bytes.get() == 32_B);
  REQUIRE_FALSE(op.use_hw_decompression.get());

  // Batch sizes derive from GPU memory when the configuration is resolved.
  for (auto const* option : {&op.scan_task_batch_size,
                             &op.hash_partition_bytes,
                             &op.concat_batch_bytes,
                             &op.cross_join_task_bytes,
                             &op.sort_sample_bytes,
                             &op.max_build_hash_table_bytes,
                             &op.max_build_probe_resident_bytes}) {
    INFO(option->key());
    REQUIRE(option->is_unset());
  }
}

TEST_CASE("explicit batch sizes are user-defined", "[config][config_operator]")
{
  auto const root = read(
    "sirius:\n  operator_params:\n    scan_task_batch_size: 1MiB\n"
    "    max_build_hash_table_bytes: 0\n");
  REQUIRE(root.operator_params.scan_task_batch_size.user_defined());
  REQUIRE(root.operator_params.scan_task_batch_size.get() == 1_MiB);
  REQUIRE(root.operator_params.max_build_hash_table_bytes.get() == 0_B);  // BUILD_PROBE off
  REQUIRE(root.operator_params.hash_partition_bytes.is_unset());
}

TEST_CASE("operator scopes follow the DuckDB settings", "[config][config_operator]")
{
  operator_options op;
  REQUIRE(op.hash_partition_bytes.scope() == option_scope::query);
  REQUIRE(op.enable_compressed_materialization.scope() == option_scope::query);
  REQUIRE(op.pin_table_prove_unique_keys.scope() == option_scope::query);
  // Overridable only under SIRIUS_ENABLE_TEST_OPTIONS: session for users.
  REQUIRE(op.scan_task_batch_size.scope() == option_scope::session);
  REQUIRE(op.enable_dynamic_filter.scope() == option_scope::session);
  REQUIRE(op.use_hw_decompression.scope() == option_scope::session);
}

TEST_CASE("operator fixtures read through the new block", "[config][config_operator]")
{
  struct expectation {
    char const* file;
    char const* key;
  };
  for (auto const& [file, key] : std::initializer_list<expectation>{
         {"invalid_dynamic_filter_inlist_max_l2_fraction_above_one.yaml",
          "dynamic_filter_inlist_max_l2_fraction"},
         {"invalid_dynamic_filter_inlist_max_l2_fraction_negative.yaml",
          "dynamic_filter_inlist_max_l2_fraction"},
         {"invalid_dynamic_filter_inlist_max_l2_fraction_nan.yaml",
          "dynamic_filter_inlist_max_l2_fraction"},
         {"invalid_dense_count_join_engine_policy.yaml", "dense_count_join_max_bytes"},
         {"invalid_dense_count_join_enable_type.yaml", "enable_dense_count_join"},
         {"invalid_mark_join_switch_negative.yaml", "mark_join_build_switch_ratio"},
         {"invalid_mark_join_switch_nan.yaml", "mark_join_build_switch_ratio"},
         {"invalid_dynamic_filter_keep_threshold_above_one.yaml", "dynamic_filter_keep_threshold"},
         {"invalid_dynamic_filter_keep_threshold_negative.yaml", "dynamic_filter_keep_threshold"},
         {"invalid_dynamic_filter_keep_threshold_nan.yaml", "dynamic_filter_keep_threshold"},
         {"invalid_hash_partition_zero.yaml", "hash_partition_bytes"},
         {"invalid_scan_task_batch_zero.yaml", "scan_task_batch_size"},
         {"invalid_dynamic_filter_domain_coverage_threshold.yaml",
          "dynamic_filter_domain_coverage_threshold"},
         {"invalid_dynamic_filter_domain_coverage_threshold_nan.yaml",
          "dynamic_filter_domain_coverage_threshold"},
         {"invalid_runtime_distinct_build_probe.yaml", "enable_runtime_distinct_build_probe"}}) {
    INFO(file);
    REQUIRE_THAT(read_error(fixture(file)),
                 Catch::Matchers::StartsWith(std::string("'sirius.operator_params.") + key + "'"));
  }
  REQUIRE(read(fixture("valid_dynamic_filter_threshold_boundaries.yaml"))
            .operator_params.dynamic_filter_domain_coverage_threshold.get() == 1.5);
  REQUIRE_FALSE(read(fixture("valid_dense_count_join_disable.yaml"))
                  .operator_params.enable_dense_count_join.get());
  REQUIRE(read(fixture("valid_mark_join_switch_zero.yaml"))
            .operator_params.mark_join_build_switch_ratio.get() == 0.0);
}

TEST_CASE("operator option messages", "[config][config_operator]")
{
  REQUIRE(
    read_error("sirius:\n  operator_params:\n    hash_partition_bytes: 0\n") ==
    "'sirius.operator_params.hash_partition_bytes': value out of range: '0 B' (must be greater "
    "than 0)");
  REQUIRE(
    read_error("sirius:\n  operator_params:\n    fused_scan_max_selectivity: .inf\n") ==
    "'sirius.operator_params.fused_scan_max_selectivity': value out of range: '.inf' (must be "
    "finite and greater than 0)");
  REQUIRE(
    read_error("sirius:\n  operator_params:\n    late_materialization_min_boundaries: -1\n") ==
    "'sirius.operator_params.late_materialization_min_boundaries': value out of range: '-1' "
    "(must not be negative)");
  REQUIRE(read_error("sirius:\n  operator_params:\n    fused_scan_max_membership_sources: "
                     "9223372036854775808\n") ==
          "'sirius.operator_params.fused_scan_max_membership_sources': value out of range: "
          "'9223372036854775808' (must be at most 9223372036854775807)");
  REQUIRE(
    read_error("sirius:\n  operator_params:\n    max_dynamic_filter_bloom_bytes_per_gpu: ~\n") ==
    "'sirius.operator_params.max_dynamic_filter_bloom_bytes_per_gpu': removed; the Bloom budget "
    "is always 4x scan_task_batch_size; remove this key. Set "
    "enable_dynamic_filter_multi_partition to false to disable Bloom accumulation");
  REQUIRE(read_error("sirius:\n  operator_params:\n    enable_dynamic_filter_pushdown: true\n") ==
          "unknown config key: 'enable_dynamic_filter_pushdown' in sirius.operator_params");
  // An infinite switch ratio is accepted, as before.
  REQUIRE(read("sirius:\n  operator_params:\n    mark_join_build_switch_ratio: .inf\n")
            .operator_params.mark_join_build_switch_ratio.get() ==
          std::numeric_limits<double>::infinity());
}

TEST_CASE("operator options write what they read", "[config][config_operator]")
{
  constexpr std::string_view text = R"(sirius:
  operator_params:
    enable_fused_scan_filter: false
    hash_partition_bytes: 3 MiB
    dynamic_filter_keep_threshold: 0.5
    pin_table_unique_exact_max_bytes: 1 GiB
)";
  auto const root                 = read(text);
  REQUIRE(parse_yaml(write_yaml(root, "sirius")) == parse_yaml(text));
  auto const all = read(write_yaml(root, "sirius", true));
  REQUIRE(all.operator_params.max_broadcast_join_size.user_defined());
  REQUIRE(all.operator_params.scan_task_batch_size.is_unset());  // derived, never written
}
