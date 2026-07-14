// SPDX-License-Identifier: Apache-2.0
//
// Beam-search compression explorer for simpatico_codegen.
//
// The BFS works at the level of individual operators so the caller gets a full
// cascade DSL rather than only being able to evaluate complete plans.

#pragma once

#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <string>
#include <vector>

namespace simpatico {

// ---------------------------------------------------------------------------
// Scoring / ranking configuration
// ---------------------------------------------------------------------------

enum class score_mode {
  Weighted,  ///< score = ratio^wr * comp_gbps^wc * decomp_gbps^wd
  Pareto,    ///< Pareto frontier of (ratio, comp_gbps, decomp_gbps)
};

struct exploration_config {
  size_t beam_width = 100;
  size_t max_depth  = 10;
  bool verbose      = false;

  score_mode rerank_mode   = score_mode::Weighted;
  double rerank_weights[3] = {1.0, 1.0, 1.0};  ///< (ratio, comp, decomp) exponents
  size_t rerank_top        = 8;                ///< finalists to time (selected by ratio)
  size_t simplicity_slots  = 4;  ///< top-N finalists injected per step-count depth level

  /// Ratio-guided BFS sample cap: when >0 and a column has more rows than this,
  /// the beam search runs on a contiguous prefix of this many rows instead of
  /// the full column, then finalists are re-measured on the full column (so the
  /// reported ratio/throughput stay exact — only the throwaway beam ranking
  /// sees the sample). Default 0 = always full column.
  ///
  /// This is an approximate speedup, not on by default: measured to match the
  /// full column for unsorted-numeric columns, but to pick markedly worse
  /// plans for sorted/monotonic columns (e.g. primary keys), whose best
  /// cascade exploits global structure a prefix distorts. A STRING sample is
  /// materialized (not a zero-copy slice) so the byte codecs accept it and
  /// its measured size is exact; cost is proportional to sample_rows.
  size_t sample_rows = 0;

  /// Per-column byte budget for the whole exploration (BFS + rerank): a column
  /// bigger than this is trimmed to a representative row-prefix so no codec
  /// allocates buffers larger than device memory. Default 2 GiB; 0 = unlimited.
  size_t max_explore_bytes = 2ull << 30;
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

struct exploration_result {
  std::string plan_dsl;
  double compression_ratio          = 1.0;
  size_t original_size_bytes        = 0;
  size_t compressed_size_bytes      = 0;
  size_t cascade_depth              = 0;
  double compress_throughput_gbps   = 0.0;
  double decompress_throughput_gbps = 0.0;
  std::string pareto_alternates_summary;
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Memory-bounded BFS over single-column compression cascades.
/// Uses `simpatico::make_compressor` for non-fused operators and thin
/// single-op `compress_column` calls for fused (delta/rle/bitpack/for/zigzag)
/// operators.  The rerank pass calls `compress_column` + `decompress_column`
/// for end-to-end throughput measurement.
exploration_result explore_column_compression(cudf::column_view input,
                                              exploration_config const& config,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

/// Byte size of a column (for compression ratio computation).
size_t column_size_bytes_ex(cudf::column_view const& col, rmm::cuda_stream_view stream);

}  // namespace simpatico
