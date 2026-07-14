#pragma once

#include "codegen/bridge/fused_tree_build.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/plan_tree.hpp"

#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cstdint>
#include <memory>
#include <optional>
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
  // passthrough leaves. Used by encode_subtree_impl to key reps by NodeId.
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
                          plan_compound_builder& builder,
                          std::string* error_out,
                          CodegenHead* head_out = nullptr);

/// Decode one codegen-fused subtree (C++-native). The caller passes a
/// fully-built ``FusedTree`` and a ``LabeledBuffers`` of the real device buffers keyed by
/// DFS-preorder ``buffer_key(node_id, field)``. This adds the decode-only
/// transients (Compact ``bp_offsets`` scan, RLE scratch) into ``labeled``,
/// compiles/launches the rendered kernel, and writes ``num_rows`` decoded
/// elements to ``out`` on ``stream``. Returns true on success.
///
/// Mirror of ``encode_fused_subtree`` above.
bool decode_fused_subtree(codegen::jit::FusedTree const& tree,
                          codegen::jit::LabeledBuffers& labeled,
                          char const* dtype,
                          std::int64_t num_rows,
                          void* out,
                          rmm::cuda_stream_view stream);

}  // namespace simpatico
