// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/plan/leaf_desc.hpp"
#include "codegen/plan/plan_dsl.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/plan/representation.hpp"
#include "codegen/selection/selection.hpp"
#include "codegen/util/stream_pool.hpp"

#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simpatico {

/// Builder for assembling a PlanTree from codegen output.
///
/// `leaves` maps each fused op's PlanTree NodeId to its produced rep (which the
/// compress driver moves onto `tree.nodes[nodeid].rep`).
///
/// `raw_passthrough_leaves` holds entries for synthesised Raw passthrough leaves
/// that have no PlanTree op of their own.  Each entry carries:
///   - `parent_id`   : the parent node's PlanTree NodeId
///   - `channel_name`: the output channel name under which the rep is stored
///                     ("values" for RLE parents, "deltas" for FOR parents)
///   - `rep`         : the codegen_fused_representation holding the raw buffers
///
/// The compress driver parks each rep on
/// `tree.nodes[parent_id].channels[channel_output_path]`.
struct RawPassthroughLeaf {
  NodeId parent_id;
  std::string channel_name;
  std::unique_ptr<compressed_representation> rep;
};

struct fused_leaf_builder {
  std::unordered_map<NodeId, std::unique_ptr<compressed_representation>> leaves;
  std::vector<RawPassthroughLeaf> raw_passthrough_leaves;
};

/// True if `op` is a codegen-fusable operator name
/// (delta / rle / bitpack / for / zigzag).
bool is_codegen_compressor(std::string const& op);

/// Compress a single column using the plan DSL.  Returns nullptr on error.
/// The lower-level building block behind the table-level
/// ``simpatico::compress_with_plan``.
///
/// Runs entirely on ``stream`` (single-stream per column).  Cross-column
/// parallelism is the caller's concern: fan one column per worker thread,
/// each on its own stream.  Intermediate device buffers are released
/// eagerly as the tree walk consumes them.
std::unique_ptr<PlanTree> compress_column(cudf::column_view input,
                                          std::string_view plan_dsl,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr,
                                          std::string* error_out);

/// Compress a single column with ONE operator and return the resulting
/// ``compressed_representation``.  A thin wrapper for the BFS explorer:
///
/// * Non-fused (identity, dictionary, alp, ans, snappy, lz4, …): delegates
///   to ``make_compressor(op_name)->compress()``.
/// * Fused (delta / rle / bitpack / for / zigzag): builds a single-node
///   PlanTree and invokes ``encode_fused_subtree``.
///
/// Returns nullptr and sets ``*error_out`` on failure.  The caller can then
/// use ``compressed_representation::named_channels()`` for BFS child outputs
/// and ``compressed_representation::compressed_size_bytes()`` for scoring.
std::unique_ptr<compressed_representation> compress_single_op(std::string const& op_name,
                                                              cudf::column_view input,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr,
                                                              std::string* error_out);

/// Per-column decode-time row selection
/// (`SIRIUS_EXP_FUSED_SCAN_FILTER`). Built by the wave-2 orchestrator in
/// ``decompress_columns_parallel`` AFTER the combine + CNT wave fixed the
/// survivor count; never active on the default path (gate off ⇒ callers pass
/// nullptr and behavior is byte-identical).
struct decode_selection {
  /// Combined AND-of-all-conjuncts scan mask; CNT already ran. Opaque to the
  /// decode driver — only forwarded to the masked JIT launch as kernel
  /// arguments (see codegen/selection/selection.hpp).
  sirius::codegen::selection_mask const* mask = nullptr;
  /// Survivor row count (popcount of @c mask); >= 0 once the CNT wave ran.
  std::int64_t survivor_count = -1;
  /// INT32 survivor row indices, size == @c survivor_count. Built ONCE per
  /// batch by the selection wave's mask→indices kernel and shared by every
  /// column that needs them. Read on the @c full route and by the index walk.
  cudf::column_view survivor_indices{};
  /// How this column produces its compacted output. One value, so the modes
  /// cannot contradict each other and a plan that supports none of them simply
  /// takes @c full. Must match @c probe_column(tree).compact_route — a mismatch
  /// is refused, never silently decoded full width.
  ///
  /// @c str_split has NO generic in-walk fallback (compacted offsets cannot
  /// feed the ordinary str_split reconstruct): if its dedicated route declines,
  /// the call errors and the orchestrator re-runs the batch unfiltered.
  sirius::codegen::decode_route route = sirius::codegen::decode_route::full;
  /// Walk the survivor index list instead of the mask bits — the cheaper
  /// enumeration once few rows survive. Only meaningful for
  /// @c decode_route::bitpack_mask; delta roots IGNORE it (the index walk
  /// rejects them at render) and a dictionary codes region is unaffected. The
  /// orchestrator populates @c survivor_indices whenever it sets this; any
  /// anomaly silently keeps the mask walk (the pick is an optimization).
  bool enumerate_by_index = false;

  [[nodiscard]] bool active() const noexcept { return mask != nullptr && survivor_count >= 0; }
  /// Any compacted route: the result column must be survivor-sized.
  [[nodiscard]] bool compacted() const noexcept
  {
    return sirius::codegen::route_decodes_compacted(route);
  }
};

/// Decompress a plan tree produced by compress_column. DecodeWalk performs a
/// single reverse walk: each codegen-fused subtree root is inverted by one
/// high-level ``decode_fused_subtree`` call and every other step by its rep's
/// own decompress(). Runs entirely on ``stream``.
/// @param pred  Optional set-membership directive. When non-null and active the
///              result is a BOOL8 column of the same row count carrying
///              `value ∈ pred->equals_any` instead of the reconstructed column
///              (see @c decode_predicate). A dictionary-rooted tree resolves it
///              against the key set and never gathers the chars; every other
///              shape decodes normally and compares afterwards, which is correct
///              but no cheaper — gate on @c column_decode_caps::can_answer_equality.
/// @param sel   Optional decode-time row selection (see
///              @c decode_selection). When non-null and active the result holds
///              only the mask's survivor rows: a compacted route decodes
///              compacted in-kernel, @c full decodes full width and is
///              gathered. Composition with @p pred (a survivor-sized BOOL8 at a
///              mask-source slot): allowed for @c dict_codes (the predicate is
///              answered over the compacted codes) and for @c full (full-width
///              BOOL8, then the survivor gather); refused for the write-skipping
///              routes, which never materialize the value to compare. The
///              result is then BOOL8[survivor_count] — enforced by a belt.
///              Non-null columns are required — a null-masked decode fails
///              loudly (never corrupts).
std::unique_ptr<cudf::column> decompress_column(PlanTree const& tree,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr,
                                                std::string* error_out,
                                                decode_predicate const* pred = nullptr,
                                                decode_selection const* sel  = nullptr);

/// What one plan can do for a decode beyond reconstructing its column.
///
/// One probe, so a capability and the route that implements it cannot drift
/// apart: @c compact_route == @c full IS "cannot decode compacted", and only a
/// bitpack root can evaluate a predicate while decoding, which is exactly
/// @c bitpack_mask.
struct column_decode_caps {
  /// The route this plan's shape takes under a selection mask:
  ///   bitpack_mask — a `bitpack` root (mask_consume on the root region),
  ///   delta_mask   — a delta->bitpack root (the same launcher renders it),
  ///   dict_codes   — a non-null `dictionary` root whose `indices` channel is
  ///                  bitpack-compressed: the codes decode under the mask and
  ///                  only the surviving keys are gathered — the biggest win
  ///                  for dictionary string payloads, whose chars gather scales
  ///                  linearly with selectivity,
  ///   str_split    — a non-null `str_split` root whose `chars` channel is
  ///                  TERMINAL (raw/identity — entropy-coded chars cannot be
  ///                  byte-gathered without a full decompress) and whose
  ///                  `offsets` channel is a plain bitpack child. Deeper
  ///                  offsets chains (e.g. delta->rle->bitpack) and
  ///                  widened (>2 GB, non-UINT8) chars stay `full` — widen only
  ///                  in lockstep with the renderer's masked
  ///                  offsets-reconstruction coverage,
  ///   full         — everything else: decode full width, then gather.
  /// Nullable plans (a `null_mask` output channel) take @c full throughout:
  /// selection has no null model yet.
  sirius::codegen::decode_route compact_route = sirius::codegen::decode_route::full;

  /// The plan resolves a set-membership predicate WITHOUT materialising the
  /// column — its root value is produced by a `dictionary` node, so the
  /// predicate is answered against the keys. Pushing a predicate into any other
  /// plan is correct but pointless: it only moves the comparison.
  bool can_answer_equality = false;

  /// The plan can ballot its rows into selection-mask words, which the render
  /// supports for bitpack roots only.
  [[nodiscard]] bool can_produce_mask() const noexcept
  {
    return compact_route == sirius::codegen::decode_route::bitpack_mask;
  }
};

/// Probe @p tree once for everything a decode needs to know about it.
column_decode_caps probe_column(PlanTree const& tree);

/// Wave 1: decode @p tree's root bitpack region fused with the inclusive
/// decoded-domain range @p pred, writing selection-mask words into
/// @p mask_words — NO column output is allocated or written. @p mask_words
/// must hold ``selection_mask::AllocWordsFor(num_rows)`` words; every word of
/// every covered chunk is written (out-of-range lanes ballot to 0, so tail
/// bits are zero by construction). Returns false + @p error_out when the plan
/// is not bitpack-rooted or the launch fails; no device state is corrupted.
bool decompress_column_selection_mask(PlanTree const& tree,
                                      sirius::codegen::range_predicate pred,
                                      std::uint32_t* mask_words,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr,
                                      std::string* error_out);

/// Assemble one filtered decode's ragged output into a uniformly
/// survivor-sized table: compacted-route columns pass through; ALL full-width
/// columns are compacted with ONE ``cudf::gather`` over ``result.row_indices``.
/// When ``result.applied`` is false the columns are assembled unchanged.
/// Null-masked columns are refused (returns nullptr + @p error_out).
/// Synchronizes @p stream before returning, so the caller may free/rebind the
/// inputs immediately.
///
/// Internal to the decode: ``decompress_scan_filter`` calls this before
/// returning, so no caller outside sees the ragged intermediate.
std::unique_ptr<cudf::table> compact_scan_filter_output(
  std::vector<std::unique_ptr<cudf::column>>&& columns,
  sirius::codegen::scan_filter_result const& result,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out);

/// Reconstruct a compressed_representation from named output columns. A thin
/// dispatcher mapping the compressor name (or the ``bitextract_<spec>`` prefix)
/// to the matching rep subclass's ``from_outputs`` factory, which validates
/// names/arity/type and reconstructs the rep. ``meta`` carries per-node decode
/// metadata (e.g. ``leaf_meta::ans`` / ``leaf_meta::bitcomp`` with
/// ``uncompressed_size`` and ``original_type_id``) that cannot be recovered
/// from the channel buffers alone. Used by the decode driver.
std::unique_ptr<compressed_representation> reconstruct_representation(
  std::string const& compressor_name,
  std::vector<std::string> const& output_names,
  std::vector<std::unique_ptr<cudf::column>> outputs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out,
  leaf_meta_v const& meta = leaf_meta::none{});

}  // namespace simpatico
