// SPDX-License-Identifier: Apache-2.0
#ifndef SIMPATICO_PLAN_INTERPRETER_HPP
#define SIMPATICO_PLAN_INTERPRETER_HPP

#include "codegen/plan/leaf_desc.hpp"
#include "codegen/plan/plan_dsl.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/plan/representation.hpp"
#include "codegen/util/stream_pool.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simpatico {

struct plan_compound {
  // Canonical plan structure (one node per op, named-edge wiring). Owns every
  // compressed representation: each op's rep lives on its node
  // (`PlanNode::rep` / `PlanNode::channels`). Render back to DSL text on demand
  // via render_plan_tree(tree).
  PlanTree tree;
};

/// Builder for assembling a plan_compound from codegen output.
///
/// `leaves` maps each fused op's PlanTree NodeId to its produced rep (which the
/// compress driver moves onto `tree.nodes[nodeid].rep`).
///
/// `raw_passthrough_leaves` holds entries for synthesised Raw passthrough leaves
/// that have no PlanTree op of their own.  Each entry carries:
///   - `parent_id`   : the parent node's PlanTree NodeId (was "parent_rle")
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

struct plan_compound_builder {
  std::unordered_map<NodeId, std::unique_ptr<compressed_representation>> leaves;
  std::vector<RawPassthroughLeaf> raw_passthrough_leaves;
};

/// True if `op` is a codegen-fusable operator name
/// (delta / rle / bitpack / for / zigzag).
bool is_codegen_compressor(std::string const& op);

/// True if `nid` is a fusable node whose producer is itself a fusable op —
/// decoded as part of its region root's single JIT launch, not on its own.
bool is_fusion_interior(NodeId nid, PlanTree const& tree);

/// Compress a single column using the plan DSL.  Returns nullptr on error.
/// The lower-level building block behind the table-level
/// ``simpatico::compress_with_plan``.
///
/// Runs entirely on ``stream`` (single-stream per column).  Cross-column
/// parallelism is the caller's concern: fan one column per worker thread,
/// each on its own stream.  Intermediate device buffers are released
/// eagerly as the tree walk consumes them.
std::unique_ptr<plan_compound> compress_column(cudf::column_view input,
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
///   PlanTree and invokes ``jit_encode_subtree``.
///
/// Returns nullptr and sets ``*error_out`` on failure.  The caller can then
/// use ``compressed_representation::named_channels()`` for BFS child outputs
/// and ``compressed_representation::compressed_size_bytes()`` for scoring.
std::unique_ptr<compressed_representation> compress_single_op(std::string const& op_name,
                                                              cudf::column_view input,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr,
                                                              std::string* error_out);

/// Decompress a compound produced by compress_column.  A single post-order
/// walk over the plan tree: each codegen-fused subtree root is inverted by one
/// JIT-compiled kernel (``dispatch_codegen_subtree``) and every other step by
/// its rep's own decompress().  Runs entirely on ``stream``.
std::unique_ptr<cudf::column> decompress_column(plan_compound const& compound,
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

#endif  // SIMPATICO_PLAN_INTERPRETER_HPP
