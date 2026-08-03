#pragma once

#include "codegen/bridge/fused_tree_build.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/plan_tree.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace simpatico {

struct CodegenHead {
  std::shared_ptr<codegen::jit::FusedTree> tree;
  // PlanTree node ids covered by the fused region (non-raw-passthrough only),
  // in DFS-preorder. Lets the compress walk mark them done and find the
  // region's boundary outputs (child edges leaving the covered set).
  std::vector<NodeId> covered_nodes;
  // Full DFS-preorder origins (index == jit node_id), including synthesised Raw
  // passthrough leaves. Used by launch_encode_fused_tree to key reps by NodeId.
  std::vector<FusedNodeOrigin> preorder;
};

/// JIT-encode the maximal {delta,rle,bitpack,FOR,zigzag} subtree rooted at
/// PlanTree node ``start_node`` from a contiguous int32/int64 column. Extracts
/// the fusable region, then writes its reps into ``builder`` (NodeId-keyed in
/// builder.leaves; raw passthrough reps in builder.raw_passthrough_leaves);
/// entropy tails on the region's boundary outputs are scheduled by the caller's
/// compress walk. Returns false without setting ``error_out`` when ``start_node``
/// roots no fusable region (caller runs the generic path), true on success. If
/// ``head_out`` is non-null it receives the extracted CodegenHead so the caller
/// can inspect covered_nodes.
///
/// Mirror of ``decode_fused_subtree`` below.
bool encode_fused_subtree(PlanTree const& tree,
                          NodeId start_node,
                          cudf::column_view input_col,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr,
                          fused_leaf_builder& builder,
                          std::string* error_out,
                          CodegenHead* head_out = nullptr);

/// Launch an already-built encode head for ``input_col`` and assemble its
/// compressed leaves in ``builder``.
bool launch_encode_fused_tree(CodegenHead const& head,
                              cudf::column_view const& input_col,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref const& mr,
                              fused_leaf_builder& builder,
                              std::string* error_out);

/// Callback used by the high-level decode bridge to materialize an entropy-tail
/// child while binding a fused subtree.
using decode_materialize_fn = std::function<cudf::column const*(NodeId)>;

/// JIT-decode the maximal fused subtree rooted at ``start_node``. Builds the
/// shared FusedTree once, resolves metadata, binds persisted buffers (using
/// ``materialize`` for entropy tails), allocates the output, and launches the
/// inverse kernel. Returns nullptr and sets ``error_out`` on failure.
///
/// Mirror of ``encode_fused_subtree`` above.
std::unique_ptr<cudf::column> decode_fused_subtree(PlanTree const& tree,
                                                   NodeId start_node,
                                                   decode_materialize_fn const& materialize,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref const& mr,
                                                   std::string* error_out);

/// Fill ``d_bp_offsets`` (int32[num_chunks+1]) with the exclusive-prefix scan
/// of the per-chunk live-word counts derived from chunk_bits x chunk_count,
/// plus the [num_chunks] total sentinel (so bp_offsets[c+1]-bp_offsets[c] ==
/// live_words[c]). ``alloc_scratch(bytes)`` supplies CUB temp storage that
/// must stay live until the enqueued scan completes (stream-ordered lifetime
/// on the same stream is sufficient). Enqueues on ``stream_v`` (a
/// cudaStream_t) only — no sync, no other streams. Returns 0 on success, else
/// a cuda error code.
///
/// Exported for rep-level bp_offsets memoization: compute once per (column,
/// bitpack node), keep the buffer alive rep-side, and pre-bind it as
/// ``buffer_key(node_id, "bp_offsets")`` in the LabeledBuffers handed to the
/// decode launchers — synthesize_decode_transients trusts a pre-bound entry
/// (non-null, length >= num_chunks+1) and skips its per-launch recomputation.
int compute_bp_offsets(const void* d_chunk_count,
                       const void* d_chunk_bits,
                       std::int32_t num_chunks,
                       void* d_bp_offsets,
                       const std::function<void*(std::size_t)>& alloc_scratch,
                       void* stream_v);

/// Launch an already-prepared fused decode tree. ``labeled`` must contain all
/// persisted buffers; this adds decode-only transients (dropped from
/// ``labeled`` again before returning; caller-pre-bound bp_offsets are kept
/// and reused), renders/compiles, and enqueues the kernel into ``out`` on
/// ``stream`` (async — callers join streams before consuming the output).
bool launch_decode_fused_tree(codegen::jit::FusedTree const& tree,
                              codegen::jit::LabeledBuffers& labeled,
                              char const* dtype,
                              std::int64_t num_rows,
                              void* out,
                              rmm::cuda_stream_view stream);

}  // namespace simpatico
