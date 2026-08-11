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
///
/// Three consumption modes:
///  * ``compact_capable`` (tier_a AND tier_a_delta): the plan's root fused
///    region decodes with the mask via the mask_consume launcher (bitpack or
///    delta->bitpack roots) and writes compacted output directly — the output
///    column is allocated count-first with ``survivor_count`` rows.
///  * ``dict_compact``: for a ``dictionary``-rooted string plan whose
///    ``indices`` channel is bitpack-compressed, the CODE region decodes under
///    the mask, and the key-chars gather runs
///    over the compacted codes only — the strings column comes back
///    survivor-sized without a full-width chars materialization.
///  * neither (Tier B): the column decodes full width exactly as today, then
///    one ``cudf::gather`` over ``survivor_indices`` compacts it.
struct decode_selection {
  /// Combined AND-of-all-conjuncts scan mask; CNT already ran. Opaque to the
  /// decode driver — only forwarded to the masked JIT launch as kernel
  /// arguments (see codegen/selection/selection.hpp).
  sirius::codegen::selection_mask const* mask = nullptr;
  /// Survivor row count (popcount of @c mask); >= 0 once the CNT wave ran.
  std::int64_t survivor_count = -1;
  /// INT32 survivor row indices, size == @c survivor_count. Built ONCE per
  /// batch by the selection wave's mask→indices kernel and shared by every
  /// Tier-B column of the batch. Only read on the Tier-B path.
  cudf::column_view survivor_indices{};
  /// True when the plan's root region consumes the mask in-kernel (Tier A).
  bool compact_capable = false;
  /// True for the dictionary-gather mode above. Mutually exclusive with
  /// @c compact_capable; gate on @c plan_supports_dict_selection_decode.
  bool dict_compact = false;
  /// Prefer walking the survivor index list over walking the mask bits — the
  /// cheaper enumeration once few rows survive. Only meaningful with
  /// @c compact_capable; bitpack LEAF roots route to
  /// ``launch_decode_fused_tree_index_consume`` over @c survivor_indices, delta
  /// roots IGNORE it (the index walk rejects them at render), and a dictionary
  /// codes region is unaffected. The orchestrator populates
  /// @c survivor_indices (int32, survivor_count entries) whenever it sets this;
  /// any anomaly silently keeps the mask walk (the pick is an optimization).
  bool prefer_index_decode = false;
  /// Masked str_split strings: survivor lengths from the masked offsets
  /// decode, an exclusive-sum scan to destination offsets, and a survivor
  /// char gather into a count-first chars buffer — variable-width strings
  /// never materialize full char width. Mutually exclusive with the other
  /// compacted modes; gate on @c plan_supports_str_selection_decode. There is
  /// NO generic in-walk fallback for this mode (compacted offsets cannot feed
  /// the ordinary str_split reconstruct): if the dedicated route declines, the
  /// call errors and the orchestrator re-runs the batch unfiltered.
  bool str_compact = false;

  [[nodiscard]] bool active() const noexcept { return mask != nullptr && survivor_count >= 0; }
  /// Any compacted mode: the result column must be survivor-sized.
  [[nodiscard]] bool compacted() const noexcept
  {
    return compact_capable || dict_compact || str_compact;
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
///              but no cheaper — gate on @c plan_supports_predicate_decode.
/// @param sel   Optional decode-time row selection (see
///              @c decode_selection). When non-null and active the result holds
///              only the mask's survivor rows: a Tier-A plan decodes compacted
///              in-kernel, a Tier-B plan decodes full width and is gathered.
///              Composition with @p pred (dual delivery — a survivor-sized
///              BOOL8 at a mask-source slot): allowed for @c dict_compact (the
///              predicate is answered over the compacted codes) and for the
///              plain Tier-B route (full-width BOOL8, then the survivor
///              gather); refused for @c compact_capable / @c str_compact. The
///              result is then BOOL8[survivor_count] — enforced by a belt.
///              Non-null columns are required — a null-masked decode fails
///              loudly (never corrupts).
std::unique_ptr<cudf::column> decompress_column(PlanTree const& tree,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr,
                                                std::string* error_out,
                                                decode_predicate const* pred = nullptr,
                                                decode_selection const* sel  = nullptr);

/// True iff @p tree decodes a predicate without materialising the column — i.e.
/// its root value is produced by a `dictionary` node, so the predicate resolves
/// against the keys. Callers use this to decide whether pushing a predicate down
/// is worth anything; pushing it into any other plan is correct but pointless.
bool plan_supports_predicate_decode(PlanTree const& tree);

/// True iff @p tree can decode SURVIVOR-COMPACTED under a selection mask by
/// ANY masked route: a `bitpack`-rooted plan (tier_a — mask_consume on the
/// root region), a delta->bitpack plan (tier_a_delta — mask_consume renders
/// Delta roots too), or a dictionary plan with bitpack codes (the codes decode
/// under the mask and only the surviving keys are gathered; see
/// @c plan_supports_dict_selection_decode). This is the umbrella probe callers
/// key compact-capability off; per-route dispatch uses @c plan_selection_tier.
///
/// CAUTION (wave-1 callers): a FILTER column feeding the ballot decode must be
/// bitpack-rooted — gate filter candidates on
/// `plan_selection_tier(tree) == output_tier::tier_a`.
bool plan_supports_selection_decode(PlanTree const& tree);

/// True iff @p tree can decode survivor-compacted through the dictionary
/// gather: its root value is produced by a non-null `dictionary` node whose
/// `indices` channel is bitpack-compressed. The dictionary CODES then decode
/// under the mask and the key-chars gather runs over survivors only — the
/// biggest win for dictionary string payload columns, whose chars gather scales
/// linearly with selectivity. Nullable dictionary plans (a `null_mask` output
/// channel) are refused: selection has no null model yet.
bool plan_supports_dict_selection_decode(PlanTree const& tree);

/// True iff @p tree can decode survivor-compacted through the masked str_split
/// route: its root value is produced by a non-null `str_split` node
/// whose `chars` channel is TERMINAL (raw/identity — entropy-coded chars
/// cannot be byte-gathered without a full decompress) and whose `offsets`
/// channel is routed to a plain bitpack child. Deeper offsets chains
/// (c_phone's delta->rle->bitpack) and widened (>2 GB, non-UINT8) chars stay
/// tier_b — widen only in lockstep with the renderer's masked
/// offsets-reconstruction coverage.
bool plan_supports_str_selection_decode(PlanTree const& tree);

/// Per-tier ground truth for the wave orchestrator's output-shape switch:
/// tier_a for a bitpack-rooted plan, tier_a_delta for a delta->bitpack root,
/// tier_dict_gather for a dictionary plan with bitpack codes, tier_b for
/// everything else (the classifier and the umbrella probe flip together to
/// preserve the invariant).
/// Consistent by construction with @c plan_supports_selection_decode:
/// umbrella true ⇔ classifier != tier_b.
sirius::codegen::output_tier plan_selection_tier(PlanTree const& tree);

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

/// Wave 1: two-column pair-predicate mask (column-vs-column conjuncts). Both
/// plans must be bitpack-rooted over the same row count —
/// the launcher additionally verifies matching chunk geometry. Row r's mask
/// bit is `decoded_a(r) OP decoded_b(r) && decoded_a(r) in pred.range_a &&
/// decoded_b(r) in pred.range_b` (full-domain range = inactive side). Writes
/// mask words only (AllocWordsFor sizing, tail-zero), NO column output.
/// pair_compare_op eq/ne are refused (the render covers lt/le/gt/ge only) —
/// the caller must not emit them. Returns false + @p error_out on any
/// refusal; no device state is corrupted.
bool decompress_column_pair_selection_mask(PlanTree const& tree_a,
                                           PlanTree const& tree_b,
                                           sirius::codegen::pair_predicate pred,
                                           std::uint32_t* mask_words,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr,
                                           std::string* error_out);

/// Wave 2, compacted decode: dispatches on plan shape — a bitpack-rooted plan
/// walks the mask (rows whose mask bit is 0 are never unpacked), a
/// dictionary-rooted plan with bitpack codes decodes the codes under the mask
/// and gathers only the surviving keys. Every output column is
/// allocated count-first with ``mask.survivor_count`` rows. Requires the CNT
/// wave to have run (``mask.survivor_count >= 0`` and ``mask.chunk_offsets !=
/// nullptr``) and a plan passing @c plan_supports_selection_decode or
/// @c plan_supports_dict_selection_decode. Returns nullptr + @p error_out
/// otherwise — never a full-width column.
std::unique_ptr<cudf::column> decompress_column_compacted(
  PlanTree const& tree,
  sirius::codegen::selection_mask const& mask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out);

/// Assemble the filtered decode's output (see
/// ``simpatico::decompress_scan_filter``) into one uniformly survivor-sized
/// table: Tier-A columns pass through (already compacted); ALL Tier-B
/// (full-width) columns are compacted with ONE ``cudf::gather`` over
/// ``result.row_indices``. When ``result.applied`` is false the columns are
/// assembled unchanged (the ordinary full-width decode). Null-masked columns
/// are refused (returns nullptr + @p error_out; the caller must fall back to
/// the ordinary decode — never corrupt). Synchronizes @p stream before
/// returning, so the caller may free/rebind the inputs immediately.
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
