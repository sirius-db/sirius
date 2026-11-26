// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/options/operator.hpp"
#include "config/options/topology.hpp"
#include "config/yaml.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

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
  read_yaml(root, text, "sirius");
  return root;
}

std::string read_error(std::string_view text)
{
  operator_root root;
  try {
    read_yaml(root, text, "sirius");
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
    "than zero); target partition size for hash joins and group-bys");
  REQUIRE(
    read_error("sirius:\n  operator_params:\n    fused_scan_max_selectivity: .inf\n") ==
    "'sirius.operator_params.fused_scan_max_selectivity': value out of range: '.inf' (must be "
    "finite and greater than 0)"
    "; surviving-row share limit for fused-scan compaction");
  REQUIRE(
    read_error("sirius:\n  operator_params:\n    late_materialization_min_boundaries: -1\n") ==
    "'sirius.operator_params.late_materialization_min_boundaries': value out of range: '-1' "
    "(must not be negative)"
    "; minimum port crossings late materialization must save");
  REQUIRE(read_error("sirius:\n  operator_params:\n    fused_scan_max_membership_sources: "
                     "9223372036854775808\n") ==
          "'sirius.operator_params.fused_scan_max_membership_sources': value out of range: "
          "'9223372036854775808' (must be at most 9223372036854775807)"
          "; join-filter probes carried by one fused decode");
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

TEST_CASE("every operator option's fallback and audience", "[config][config_operator]")
{
  operator_options const op;
  auto const production = std::vector<std::string>{"max_broadcast_join_size",
                                                   "max_sort_partition_bytes",
                                                   "hash_partition_bytes",
                                                   "concat_batch_bytes",
                                                   "cross_join_task_bytes",
                                                   "sort_sample_bytes",
                                                   "max_build_hash_table_bytes",
                                                   "scan_task_batch_size",
                                                   "max_build_probe_resident_bytes"};
  auto check            = [&](const auto& option, bool has_fallback) {
    INFO(option.key());
    REQUIRE(option.fallback().has_value() == has_fallback);
    REQUIRE(option.is_developer() ==
            (std::find(production.begin(), production.end(), option.key()) == production.end()));
  };
  check(op.enable_late_materialization, true);
  check(op.enable_fused_scan_filter, true);
  check(op.enable_low_cardinality_groupby, true);
  check(op.late_materialization_min_boundaries, true);
  check(op.late_materialization_min_value_x_boundaries, true);
  check(op.late_materialization_count_defer, true);
  check(op.late_materialization_min_value_compressed, true);
  check(op.fused_scan_diagnostics, true);
  check(op.fused_scan_max_selectivity, true);
  check(op.fused_scan_full_route_max_selectivity, true);
  check(op.fused_scan_index_walk_max_selectivity, true);
  check(op.fused_scan_max_membership_sources, true);
  check(op.scan_task_batch_size, false);
  check(op.max_sort_partition_bytes, true);
  check(op.max_sort_partition_memory_fraction, true);
  check(op.hash_partition_bytes, false);
  check(op.concat_batch_bytes, false);
  check(op.cross_join_task_bytes, false);
  check(op.sort_sample_bytes, false);
  check(op.max_build_hash_table_bytes, false);
  check(op.max_build_probe_resident_bytes, false);
  check(op.max_broadcast_join_size, true);
  check(op.mark_join_build_switch_ratio, true);
  check(op.enable_dynamic_filter, true);
  check(op.enable_dynamic_filter_multi_partition, true);
  check(op.enable_dynamic_zone_map_filter, true);
  check(op.dynamic_filter_domain_coverage_threshold, true);
  check(op.dynamic_filter_inlist_max_l2_fraction, true);
  check(op.dynamic_filter_keep_threshold, true);
  check(op.enable_pinned_zone_map_pruning, true);
  check(op.enable_decimal_sum_stats_bound, true);
  check(op.enable_compressed_materialization, true);
  check(op.enable_dictionary_carrier, true);
  check(op.enable_dense_count_join, true);
  check(op.enable_delim_aggregate_reuse, true);
  check(op.pin_table_prove_unique_keys, true);
  check(op.pin_table_unique_exact_max_bytes, true);
  check(op.enable_exists_minmax_rewrite, true);
  check(op.enable_partial_aggregate_below_join, true);
  check(op.enable_subquery_key_carry, true);
  check(op.enable_exists_from_aggregate, true);
  check(op.parquet_footer_statistics_max_remote_reads, true);
  check(op.parquet_footer_statistics_max_local_reads, true);
  check(op.admission_bytes_per_gpu, true);
  check(op.avg_variable_column_bytes, true);
  check(op.enable_runtime_size_estimation, true);
  check(op.use_hw_decompression, true);
}

TEST_CASE("operator validator boundaries", "[config][config_operator]")
{
  for (auto const* entry : {"late_materialization_min_value_x_boundaries: -1",
                            "late_materialization_min_value_compressed: -1",
                            "fused_scan_full_route_max_selectivity: 0",
                            "fused_scan_index_walk_max_selectivity: 0",
                            "avg_variable_column_bytes: 0",
                            "max_sort_partition_memory_fraction: 1.5",
                            "mark_join_build_switch_ratio: -.inf",
                            "parquet_footer_statistics_max_remote_reads: -2",
                            "parquet_footer_statistics_max_local_reads: -2"}) {
    std::string_view const text = entry;
    INFO(text);
    REQUIRE_THAT(read_error(std::string("sirius:\n  operator_params:\n    ") + entry + "\n"),
                 Catch::Matchers::StartsWith("'sirius.operator_params." +
                                             std::string(text.substr(0, text.find(':'))) +
                                             "': value out of range"));
  }
  REQUIRE(read_error(
            "sirius:\n  operator_params:\n    parquet_footer_statistics_max_local_reads: -2\n") ==
          "'sirius.operator_params.parquet_footer_statistics_max_local_reads': value out of range: "
          "'-2' (must be at least -1); local Parquet footers one scan may read for statistics");
  auto const null_batch = read("sirius:\n  operator_params:\n    scan_task_batch_size: ~\n");
  REQUIRE(null_batch.operator_params.scan_task_batch_size.is_unset());
  // Sizes reach UINT64_MAX, as the engine's fields and the UBIGINT SET options do.
  REQUIRE(read("sirius:\n  operator_params:\n    scan_task_batch_size: 18446744073709551615\n")
            .operator_params.scan_task_batch_size.get()
            .count() == std::numeric_limits<std::uint64_t>::max());
  REQUIRE(read_error("sirius:\n  operator_params: []\n") ==
          "'sirius.operator_params': expected a mapping, got a sequence");
  REQUIRE(
    read("sirius:\n  operator_params: ~\n").operator_params.max_broadcast_join_size.is_fallback());
}

TEST_CASE("removed operator keys keep their messages", "[config][config_operator]")
{
  REQUIRE(
    read_error("sirius:\n  operator_params:\n    enable_runtime_distinct_build_probe: true\n") ==
    "'sirius.operator_params.enable_runtime_distinct_build_probe': removed; runtime distinct "
    "build probing is an internal join policy (temporarily disabled pending issue #1600); "
    "remove this key");
  REQUIRE(
    read_error("sirius:\n  operator_params:\n    dense_count_join_max_bytes: ~\n") ==
    "'sirius.operator_params.dense_count_join_max_bytes': removed; dense count-join histogram "
    "sizing is an internal engine policy; remove this key");
}
