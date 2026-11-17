// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/option.hpp"
#include "config/size.hpp"
#include "config/validators.hpp"
#include "sirius_config.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace sirius::config {

namespace detail {

inline const operator_params& operator_defaults()
{
  static const operator_params value{};
  return value;
}

}  // namespace detail

/// `sirius.operator_params`: operator sizing and planning policy. Options a DuckDB SET overrides
/// per connection are scope query; the file value is where a connection starts. The planner
/// switches SQL exposes only under SIRIUS_ENABLE_TEST_OPTIONS are developer-only; the batch sizes
/// it guards the same way are not, as every derived batch size must be written for an effective
/// configuration to resolve alike.
struct operator_options : block_base<operator_options> {
  static const operator_params& defaults() { return detail::operator_defaults(); }

  // -- late materialization and fused scans
  config_option<bool> enable_late_materialization =
    option<bool>("enable_late_materialization")
      .fallback(defaults().enable_late_materialization)
      .scope(option_scope::query)
      .build();
  config_option<bool> enable_fused_scan_filter = option<bool>("enable_fused_scan_filter")
                                                   .fallback(defaults().enable_fused_scan_filter)
                                                   .scope(option_scope::query)
                                                   .build();
  config_option<bool> enable_low_cardinality_groupby =
    option<bool>("enable_low_cardinality_groupby")
      .fallback(defaults().enable_low_cardinality_groupby)
      .scope(option_scope::query)
      .build();
  config_option<int> late_materialization_min_boundaries =
    option<int>("late_materialization_min_boundaries")
      .validate(non_negative<int>{})
      .fallback(defaults().late_materialization_min_boundaries)
      .scope(option_scope::query)
      .build();
  config_option<std::int64_t> late_materialization_min_value_x_boundaries =
    option<std::int64_t>("late_materialization_min_value_x_boundaries")
      .validate(non_negative<std::int64_t>{})
      .fallback(defaults().late_materialization_min_value_x_boundaries)
      .scope(option_scope::query)
      .build();
  config_option<bool> late_materialization_count_defer =
    option<bool>("late_materialization_count_defer")
      .fallback(defaults().late_materialization_count_defer)
      .scope(option_scope::query)
      .build();
  config_option<std::int64_t> late_materialization_min_value_compressed =
    option<std::int64_t>("late_materialization_min_value_compressed")
      .validate(non_negative<std::int64_t>{})
      .fallback(defaults().late_materialization_min_value_compressed)
      .scope(option_scope::query)
      .build();
  config_option<bool> fused_scan_diagnostics = option<bool>("fused_scan_diagnostics")
                                                 .fallback(defaults().fused_scan.diagnostics)
                                                 .scope(option_scope::query)
                                                 .build();
  config_option<double> fused_scan_max_selectivity =
    option<double>("fused_scan_max_selectivity")
      .validate(positive_finite{})
      .fallback(defaults().fused_scan.max_selectivity)
      .scope(option_scope::query)
      .build();
  config_option<double> fused_scan_full_route_max_selectivity =
    option<double>("fused_scan_full_route_max_selectivity")
      .validate(positive_finite{})
      .fallback(defaults().fused_scan.full_route_max_selectivity)
      .scope(option_scope::query)
      .build();
  config_option<double> fused_scan_index_walk_max_selectivity =
    option<double>("fused_scan_index_walk_max_selectivity")
      .validate(positive_finite{})
      .fallback(defaults().fused_scan.index_walk_max_selectivity)
      .scope(option_scope::query)
      .build();
  config_option<std::size_t> fused_scan_max_membership_sources =
    option<std::size_t>("fused_scan_max_membership_sources")
      .validate(
        at_most<std::size_t>{static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())})
      .fallback(defaults().fused_scan.max_membership_sources)
      .scope(option_scope::query)
      .build();

  // -- batch and partition sizes. Unset sizes derive from the GPU memory when the configuration
  // is resolved (max_build_hash_table_bytes from hash_partition_bytes).
  config_option<ubytes> scan_task_batch_size =
    option<ubytes>("scan_task_batch_size").validate(positive<ubytes>{}).build();
  /// 0 = max_sort_partition_memory_fraction of the GPU memory.
  config_option<ubytes> max_sort_partition_bytes =
    option<ubytes>("max_sort_partition_bytes")
      .fallback(ubytes(defaults().max_sort_partition_bytes))
      .scope(option_scope::query)
      .build();
  config_option<double> max_sort_partition_memory_fraction =
    option<double>("max_sort_partition_memory_fraction")
      .validate(fraction{})
      .fallback(defaults().max_sort_partition_memory_fraction)
      .scope(option_scope::query)
      .build();
  config_option<ubytes> hash_partition_bytes = option<ubytes>("hash_partition_bytes")
                                                 .validate(positive<ubytes>{})
                                                 .scope(option_scope::query)
                                                 .build();
  config_option<ubytes> concat_batch_bytes    = option<ubytes>("concat_batch_bytes").build();
  config_option<ubytes> cross_join_task_bytes = option<ubytes>("cross_join_task_bytes").build();
  config_option<ubytes> sort_sample_bytes =
    option<ubytes>("sort_sample_bytes").scope(option_scope::query).build();
  /// 0 turns BUILD_PROBE off, MARK joins included.
  config_option<ubytes> max_build_hash_table_bytes =
    option<ubytes>("max_build_hash_table_bytes").scope(option_scope::query).build();
  /// 0 turns BUILD_PROBE off, MARK joins included.
  config_option<ubytes> max_build_probe_resident_bytes =
    option<ubytes>("max_build_probe_resident_bytes").scope(option_scope::query).build();
  config_option<ubytes> max_broadcast_join_size =
    option<ubytes>("max_broadcast_join_size")
      .fallback(ubytes(defaults().max_broadcast_join_size))
      .scope(option_scope::query)
      .build();
  /// 0 always uses filtered_join.
  config_option<double> mark_join_build_switch_ratio =
    option<double>("mark_join_build_switch_ratio")
      .validate(non_negative<double>{})
      .fallback(defaults().mark_join_build_switch_ratio)
      .scope(option_scope::query)
      .build();

  // -- dynamic filters
  config_option<bool> enable_dynamic_filter = option<bool>("enable_dynamic_filter")
                                                .for_developers()
                                                .fallback(defaults().enable_dynamic_filter)
                                                .build();
  config_option<bool> enable_dynamic_filter_multi_partition =
    option<bool>("enable_dynamic_filter_multi_partition")
      .for_developers()
      .fallback(defaults().enable_dynamic_filter_multi_partition)
      .build();
  config_option<bool> enable_dynamic_zone_map_filter =
    option<bool>("enable_dynamic_zone_map_filter")
      .for_developers()
      .fallback(defaults().enable_dynamic_zone_map_filter)
      .build();
  /// Above 1 turns the coverage gate off.
  config_option<double> dynamic_filter_domain_coverage_threshold =
    option<double>("dynamic_filter_domain_coverage_threshold")
      .validate(valid_domain_coverage_threshold{})
      .fallback(defaults().dynamic_filter_domain_coverage_threshold)
      .scope(option_scope::query)
      .build();
  config_option<double> dynamic_filter_inlist_max_l2_fraction =
    option<double>("dynamic_filter_inlist_max_l2_fraction")
      .validate(fraction{})
      .fallback(defaults().dynamic_filter_inlist_max_l2_fraction)
      .scope(option_scope::query)
      .build();
  config_option<double> dynamic_filter_keep_threshold =
    option<double>("dynamic_filter_keep_threshold")
      .validate(fraction{})
      .fallback(defaults().dynamic_filter_keep_threshold)
      .scope(option_scope::query)
      .build();

  // -- pinning, compression and planning rewrites
  config_option<bool> enable_pinned_zone_map_pruning =
    option<bool>("enable_pinned_zone_map_pruning")
      .for_developers()
      .fallback(defaults().enable_pinned_zone_map_pruning)
      .build();
  config_option<bool> enable_decimal_sum_stats_bound =
    option<bool>("enable_decimal_sum_stats_bound")
      .for_developers()
      .fallback(defaults().enable_decimal_sum_stats_bound)
      .build();
  config_option<bool> enable_compressed_materialization =
    option<bool>("enable_compressed_materialization")
      .fallback(defaults().enable_compressed_materialization)
      .scope(option_scope::query)
      .build();
  config_option<bool> enable_dense_count_join = option<bool>("enable_dense_count_join")
                                                  .for_developers()
                                                  .fallback(defaults().enable_dense_count_join)
                                                  .build();
  config_option<bool> enable_delim_aggregate_reuse =
    option<bool>("enable_delim_aggregate_reuse")
      .for_developers()
      .fallback(defaults().enable_delim_aggregate_reuse)
      .build();
  config_option<bool> pin_table_prove_unique_keys = option<bool>("pin_table_prove_unique_keys")
                                                      .fallback(defaults().prove_unique_keys)
                                                      .scope(option_scope::query)
                                                      .build();
  /// 0 = a share of the pinning GPU memory space's capacity.
  config_option<ubytes> pin_table_unique_exact_max_bytes =
    option<ubytes>("pin_table_unique_exact_max_bytes")
      .fallback(ubytes(defaults().unique_exact_max_bytes))
      .scope(option_scope::query)
      .build();
  config_option<bool> enable_exists_minmax_rewrite =
    option<bool>("enable_exists_minmax_rewrite")
      .for_developers()
      .fallback(defaults().enable_exists_minmax_rewrite)
      .build();
  config_option<bool> enable_partial_aggregate_below_join =
    option<bool>("enable_partial_aggregate_below_join")
      .for_developers()
      .fallback(defaults().enable_partial_aggregate_below_join)
      .build();
  config_option<bool> enable_subquery_key_carry = option<bool>("enable_subquery_key_carry")
                                                    .for_developers()
                                                    .fallback(defaults().enable_subquery_key_carry)
                                                    .build();
  config_option<bool> enable_exists_from_aggregate =
    option<bool>("enable_exists_from_aggregate")
      .for_developers()
      .fallback(defaults().enable_exists_from_aggregate)
      .build();

  // -- admission
  /// 0 turns the estimate off: topology.gpus_per_query decides.
  config_option<ubytes> admission_bytes_per_gpu =
    option<ubytes>("admission_bytes_per_gpu")
      .fallback(ubytes(defaults().admission_bytes_per_gpu))
      .scope(option_scope::query)
      .build();
  config_option<ubytes> avg_variable_column_bytes =
    option<ubytes>("avg_variable_column_bytes")
      .validate(positive<ubytes>{})
      .fallback(ubytes(defaults().avg_variable_column_bytes))
      .scope(option_scope::query)
      .build();
  config_option<bool> enable_runtime_size_estimation =
    option<bool>("enable_runtime_size_estimation")
      .fallback(defaults().enable_runtime_size_estimation)
      .scope(option_scope::query)
      .build();
  /// Read once at startup.
  config_option<bool> use_hw_decompression =
    option<bool>("use_hw_decompression").fallback(defaults().use_hw_decompression).build();

  auto fields()
  {
    return field_list(
      removed("enable_runtime_distinct_build_probe",
              "removed; runtime distinct build probing is an internal join policy (temporarily "
              "disabled pending issue #1600); remove this key"),
      removed("max_dynamic_filter_bloom_bytes_per_gpu",
              "removed; the Bloom budget is always 4x scan_task_batch_size; remove this key. Set "
              "enable_dynamic_filter_multi_partition to false to disable Bloom accumulation"),
      removed("dense_count_join_max_bytes",
              "removed; dense count-join histogram sizing is an internal engine policy; remove "
              "this key"),
      enable_late_materialization,
      enable_fused_scan_filter,
      enable_low_cardinality_groupby,
      late_materialization_min_boundaries,
      late_materialization_min_value_x_boundaries,
      late_materialization_count_defer,
      late_materialization_min_value_compressed,
      fused_scan_diagnostics,
      fused_scan_max_selectivity,
      fused_scan_full_route_max_selectivity,
      fused_scan_index_walk_max_selectivity,
      fused_scan_max_membership_sources,
      scan_task_batch_size,
      max_sort_partition_bytes,
      max_sort_partition_memory_fraction,
      hash_partition_bytes,
      concat_batch_bytes,
      cross_join_task_bytes,
      sort_sample_bytes,
      max_build_hash_table_bytes,
      max_build_probe_resident_bytes,
      max_broadcast_join_size,
      mark_join_build_switch_ratio,
      enable_dynamic_filter,
      enable_dynamic_filter_multi_partition,
      enable_dynamic_zone_map_filter,
      dynamic_filter_domain_coverage_threshold,
      dynamic_filter_inlist_max_l2_fraction,
      dynamic_filter_keep_threshold,
      enable_pinned_zone_map_pruning,
      enable_decimal_sum_stats_bound,
      enable_compressed_materialization,
      enable_dense_count_join,
      enable_delim_aggregate_reuse,
      pin_table_prove_unique_keys,
      pin_table_unique_exact_max_bytes,
      enable_exists_minmax_rewrite,
      enable_partial_aggregate_below_join,
      enable_subquery_key_carry,
      enable_exists_from_aggregate,
      admission_bytes_per_gpu,
      avg_variable_column_bytes,
      enable_runtime_size_estimation,
      use_hw_decompression);
  }
};

}  // namespace sirius::config
