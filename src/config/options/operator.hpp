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

/// `sirius.operator_params`: operator sizing and planning policy. Every batch size resolution
/// derives is a production setting: an effective configuration without one would
/// derive it again from the GPU usage limit the file names, smaller than the first time.
struct operator_options : block_base<operator_options> {
  static const operator_params& defaults() { return detail::operator_defaults(); }

  // -- late materialization and fused scans
  config_option<bool> enable_late_materialization =
    option<bool>("enable_late_materialization")
      .fallback(defaults().enable_late_materialization)
      .describe("carry rowids instead of values for pinned-table scans")
      .build();
  config_option<bool> enable_fused_scan_filter =
    option<bool>("enable_fused_scan_filter")
      .fallback(defaults().enable_fused_scan_filter)
      .describe("fuse row filters into compressed scan decoding")
      .build();
  config_option<bool> enable_low_cardinality_groupby =
    option<bool>("enable_low_cardinality_groupby")
      .fallback(defaults().enable_low_cardinality_groupby)
      .describe("dedicated kernel for small string-key group-bys")
      .build();
  config_option<int> late_materialization_min_boundaries =
    option<int>("late_materialization_min_boundaries")
      .validate(non_negative<int>{})
      .fallback(defaults().late_materialization_min_boundaries)
      .describe("minimum port crossings late materialization must save")
      .build();
  config_option<std::int64_t> late_materialization_min_value_x_boundaries =
    option<std::int64_t>("late_materialization_min_value_x_boundaries")
      .validate(non_negative<std::int64_t>{})
      .fallback(defaults().late_materialization_min_value_x_boundaries)
      .describe("minimum bytes per row times crossings a deferral must save")
      .build();
  config_option<bool> late_materialization_count_defer =
    option<bool>("late_materialization_count_defer")
      .fallback(defaults().late_materialization_count_defer)
      .describe("let columns read only by COUNT stay deferred")
      .build();
  config_option<std::int64_t> late_materialization_min_value_compressed =
    option<std::int64_t>("late_materialization_min_value_compressed")
      .validate(non_negative<std::int64_t>{})
      .fallback(defaults().late_materialization_min_value_compressed)
      .describe("minimum deferred bytes per row for compressed origins")
      .build();
  config_option<bool> fused_scan_diagnostics =
    option<bool>("fused_scan_diagnostics")
      .fallback(defaults().fused_scan.diagnostics)
      .describe("verbose fused-scan decision traces and decoder diagnostics")
      .build();
  config_option<double> fused_scan_max_selectivity =
    option<double>("fused_scan_max_selectivity")
      .validate(positive_finite{})
      .fallback(defaults().fused_scan.max_selectivity)
      .describe("surviving-row share limit for fused-scan compaction")
      .build();
  config_option<double> fused_scan_full_route_max_selectivity =
    option<double>("fused_scan_full_route_max_selectivity")
      .validate(positive_finite{})
      .fallback(defaults().fused_scan.full_route_max_selectivity)
      .describe("surviving-row share limit for full-width fused decoding")
      .build();
  config_option<double> fused_scan_index_walk_max_selectivity =
    option<double>("fused_scan_index_walk_max_selectivity")
      .validate(positive_finite{})
      .fallback(defaults().fused_scan.index_walk_max_selectivity)
      .describe("surviving-row share limit for the survivor-index walk")
      .build();
  config_option<std::size_t> fused_scan_max_membership_sources =
    option<std::size_t>("fused_scan_max_membership_sources")
      .validate(
        at_most<std::size_t>{static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())})
      .fallback(defaults().fused_scan.max_membership_sources)
      .describe("join-filter probes carried by one fused decode")
      .build();

  // -- batch and partition sizes. Unset sizes derive from the GPU memory when the configuration
  // is resolved (max_build_hash_table_bytes from hash_partition_bytes).
  config_option<ubytes> scan_task_batch_size =
    option<ubytes>("scan_task_batch_size")
      .validate(positive<ubytes>{})
      .mark_production()
      .describe("target batch size for DuckDB scan tasks")
      .build();
  /// 0 = max_sort_partition_memory_fraction of the GPU memory.
  config_option<ubytes> max_sort_partition_bytes =
    option<ubytes>("max_sort_partition_bytes")
      .fallback(ubytes(defaults().max_sort_partition_bytes))
      .mark_production()
      .describe("maximum bytes per sort partition")
      .build();
  config_option<double> max_sort_partition_memory_fraction =
    option<double>("max_sort_partition_memory_fraction")
      .validate(fraction{})
      .fallback(defaults().max_sort_partition_memory_fraction)
      .describe("share of GPU memory per sort partition when auto-sized")
      .build();
  config_option<ubytes> hash_partition_bytes =
    option<ubytes>("hash_partition_bytes")
      .validate(positive<ubytes>{})
      .mark_production()
      .describe("target partition size for hash joins and group-bys")
      .build();
  config_option<ubytes> concat_batch_bytes =
    option<ubytes>("concat_batch_bytes")
      .mark_production()
      .describe("target output batch size for CONCAT and REPLICATE")
      .build();
  config_option<ubytes> cross_join_task_bytes =
    option<ubytes>("cross_join_task_bytes")
      .mark_production()
      .describe("target output size of each cross product task")
      .build();
  /// 0 = all the sorted input.
  config_option<ubytes> sort_sample_bytes =
    option<ubytes>("sort_sample_bytes")
      .fallback(ubytes(defaults().sort_sample_bytes))
      .mark_production()
      .describe("bytes of sorted input seen before choosing sort range boundaries")
      .build();
  /// 0 turns BUILD_PROBE off, MARK joins included.
  config_option<ubytes> max_build_hash_table_bytes =
    option<ubytes>("max_build_hash_table_bytes")
      .mark_production()
      .describe("per-GPU hash-table size limit for BUILD_PROBE joins")
      .build();
  /// 0 turns BUILD_PROBE off, MARK joins included.
  config_option<ubytes> max_build_probe_resident_bytes =
    option<ubytes>("max_build_probe_resident_bytes")
      .mark_production()
      .describe("per-GPU GPU-resident build payload limit for BUILD_PROBE joins")
      .build();
  config_option<ubytes> max_broadcast_join_size =
    option<ubytes>("max_broadcast_join_size")
      .fallback(ubytes(defaults().max_broadcast_join_size))
      .mark_production()
      .describe("largest build side eligible for a broadcast join")
      .build();
  /// 0 always uses filtered_join.
  config_option<double> mark_join_build_switch_ratio =
    option<double>("mark_join_build_switch_ratio")
      .validate(non_negative<double>{})
      .fallback(defaults().mark_join_build_switch_ratio)
      .describe("row ratio at which MARK joins build on the left side")
      .build();

  // -- dynamic filters
  config_option<bool> enable_dynamic_filter = option<bool>("enable_dynamic_filter")
                                                .fallback(defaults().enable_dynamic_filter)
                                                .describe("runtime filters for eligible hash joins")
                                                .build();
  config_option<bool> enable_dynamic_filter_multi_partition =
    option<bool>("enable_dynamic_filter_multi_partition")
      .fallback(defaults().enable_dynamic_filter_multi_partition)
      .describe("Bloom filters for multi-partition hash join builds")
      .build();
  config_option<bool> enable_dynamic_zone_map_filter =
    option<bool>("enable_dynamic_zone_map_filter")
      .fallback(defaults().enable_dynamic_zone_map_filter)
      .describe("build-key min/max runtime filters")
      .build();
  /// Above 1 turns the coverage gate off.
  config_option<double> dynamic_filter_domain_coverage_threshold =
    option<double>("dynamic_filter_domain_coverage_threshold")
      .validate(valid_domain_coverage_threshold{})
      .fallback(defaults().dynamic_filter_domain_coverage_threshold)
      .describe("build key-domain coverage above which a key is not filtered")
      .build();
  config_option<double> dynamic_filter_inlist_max_l2_fraction =
    option<double>("dynamic_filter_inlist_max_l2_fraction")
      .validate(fraction{})
      .fallback(defaults().dynamic_filter_inlist_max_l2_fraction)
      .describe("largest exact IN-list runtime filter, as a share of GPU L2")
      .build();
  config_option<double> dynamic_filter_keep_threshold =
    option<double>("dynamic_filter_keep_threshold")
      .validate(fraction{})
      .fallback(defaults().dynamic_filter_keep_threshold)
      .describe("kept-row share above which post-decode filtering stops")
      .build();

  // -- pinning, compression and planning rewrites
  config_option<bool> enable_pinned_zone_map_pruning =
    option<bool>("enable_pinned_zone_map_pruning")
      .fallback(defaults().enable_pinned_zone_map_pruning)
      .describe("prune pinned chunks with per-chunk min/max statistics")
      .build();
  config_option<bool> enable_decimal_sum_stats_bound =
    option<bool>("enable_decimal_sum_stats_bound")
      .fallback(defaults().enable_decimal_sum_stats_bound)
      .describe("plan-time SUM/AVG widening decisions from table statistics")
      .build();
  config_option<bool> enable_compressed_materialization =
    option<bool>("enable_compressed_materialization")
      .fallback(defaults().enable_compressed_materialization)
      .describe("narrower carriers for eligible integer and DECIMAL values")
      .build();
  config_option<bool> enable_dictionary_carrier =
    option<bool>("enable_dictionary_carrier")
      .fallback(defaults().enable_dictionary_carrier)
      .describe("dictionary carriers for VARCHAR columns of a dictionary-encoded pin")
      .build();
  config_option<bool> enable_dense_count_join = option<bool>("enable_dense_count_join")
                                                  .fallback(defaults().enable_dense_count_join)
                                                  .describe("fused count-over-outer-join operator")
                                                  .build();
  config_option<bool> enable_delim_aggregate_reuse =
    option<bool>("enable_delim_aggregate_reuse")
      .fallback(defaults().enable_delim_aggregate_reuse)
      .describe("compute decorrelated subquery aggregates from delim CTE rows")
      .build();
  config_option<bool> pin_table_prove_unique_keys =
    option<bool>("pin_table_prove_unique_keys")
      .fallback(defaults().prove_unique_keys)
      .describe("prove pinned columns distinct for join planning")
      .build();
  /// 0 = a share of the pinning GPU memory space's capacity.
  config_option<ubytes> pin_table_unique_exact_max_bytes =
    option<ubytes>("pin_table_unique_exact_max_bytes")
      .fallback(ubytes(defaults().unique_exact_max_bytes))
      .describe("value cap for the pin-time exact uniqueness check")
      .build();
  config_option<ubytes> iceberg_equality_key_bounds_max_bytes =
    option<ubytes>("iceberg_equality_key_bounds_max_bytes")
      .fallback(ubytes(defaults().iceberg_equality_key_bounds_max_bytes))
      .describe("maximum retained Iceberg equality-delete key bounds per scan; 0 disables pruning")
      .build();
  config_option<std::uint64_t> iceberg_equality_key_bounds_max_comparisons =
    option<std::uint64_t>("iceberg_equality_key_bounds_max_comparisons")
      .fallback(defaults().iceberg_equality_key_bounds_max_comparisons)
      .describe("maximum Iceberg equality-delete key comparisons per scan; 0 disables pruning")
      .build();
  config_option<std::uint64_t> iceberg_equality_key_bound_max_length =
    option<std::uint64_t>("iceberg_equality_key_bound_max_length")
      .fallback(defaults().iceberg_equality_key_bound_max_length)
      .describe("maximum Iceberg equality-delete key bound length in bytes; 0 disables pruning")
      .build();
  config_option<bool> enable_exists_minmax_rewrite =
    option<bool>("enable_exists_minmax_rewrite")
      .fallback(defaults().enable_exists_minmax_rewrite)
      .describe("rewrite sibling correlated EXISTS into a MIN/MAX aggregate")
      .build();
  config_option<bool> enable_partial_aggregate_below_join =
    option<bool>("enable_partial_aggregate_below_join")
      .fallback(defaults().enable_partial_aggregate_below_join)
      .describe("aggregate below an inner join that only adds payload")
      .build();
  config_option<bool> enable_subquery_key_carry =
    option<bool>("enable_subquery_key_carry")
      .fallback(defaults().enable_subquery_key_carry)
      .describe("restrict correlated subquery tables to the outer rows' keys")
      .build();
  config_option<bool> enable_exists_from_aggregate =
    option<bool>("enable_exists_from_aggregate")
      .fallback(defaults().enable_exists_from_aggregate)
      .describe("drop a correlated EXISTS covered by an aggregate subquery")
      .build();
  /// Uncached footers the footer-statistics pass reads for one scan: -1 no limit, 0 cached only.
  config_option<std::int64_t> parquet_footer_statistics_max_remote_reads =
    option<std::int64_t>("parquet_footer_statistics_max_remote_reads")
      .validate(at_least<std::int64_t>{-1})
      .fallback(defaults().parquet_footer_statistics_max_remote_reads)
      .describe("remote Parquet footers one scan may read for statistics")
      .build();
  config_option<std::int64_t> parquet_footer_statistics_max_local_reads =
    option<std::int64_t>("parquet_footer_statistics_max_local_reads")
      .validate(at_least<std::int64_t>{-1})
      .fallback(defaults().parquet_footer_statistics_max_local_reads)
      .describe("local Parquet footers one scan may read for statistics")
      .build();

  // -- admission
  /// 0 turns the estimate off: topology.gpus_per_query decides.
  config_option<ubytes> admission_bytes_per_gpu =
    option<ubytes>("admission_bytes_per_gpu")
      .fallback(ubytes(defaults().admission_bytes_per_gpu))
      .describe("target projected scan-output bytes per GPU at admission")
      .build();
  config_option<ubytes> avg_variable_column_bytes =
    option<ubytes>("avg_variable_column_bytes")
      .validate(positive<ubytes>{})
      .fallback(ubytes(defaults().avg_variable_column_bytes))
      .describe("assumed per-row width of variable-width columns")
      .build();
  config_option<bool> enable_runtime_size_estimation =
    option<bool>("enable_runtime_size_estimation")
      .fallback(defaults().enable_runtime_size_estimation)
      .describe("size group-by partitions from projected input")
      .build();
  /// Read once at startup.
  config_option<bool> use_hw_decompression =
    option<bool>("use_hw_decompression")
      .fallback(defaults().use_hw_decompression)
      .describe("hardware decompression for compressed Parquet scans")
      .build();

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
      enable_dictionary_carrier,
      enable_dense_count_join,
      enable_delim_aggregate_reuse,
      pin_table_prove_unique_keys,
      pin_table_unique_exact_max_bytes,
      iceberg_equality_key_bounds_max_bytes,
      iceberg_equality_key_bounds_max_comparisons,
      iceberg_equality_key_bound_max_length,
      enable_exists_minmax_rewrite,
      enable_partial_aggregate_below_join,
      enable_subquery_key_carry,
      enable_exists_from_aggregate,
      parquet_footer_statistics_max_remote_reads,
      parquet_footer_statistics_max_local_reads,
      admission_bytes_per_gpu,
      avg_variable_column_bytes,
      enable_runtime_size_estimation,
      use_hw_decompression);
  }
};

}  // namespace sirius::config
