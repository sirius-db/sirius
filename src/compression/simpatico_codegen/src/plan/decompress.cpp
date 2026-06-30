// SPDX-License-Identifier: Apache-2.0
#include "codegen/bridge/fused_tree_build.hpp"
#include "codegen/codegen_bridge.hpp"
#include "codegen/plan/bitjoin_layout.hpp"
#include "codegen/plan/plan_interpreter.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>

// The C++-native JIT codegen decode entry point (``simpatico::decode_fused_subtree``,
// defined in codegen_runtime.cpp) is declared in codegen_bridge.hpp above. One
// call does the whole pipeline: build tree → compile → bind buffers → launch →
// sync.

namespace simpatico {
// The compress driver (the recursive CompressWalk, compress_column) lives
// in plan/compress.cpp.
// This file owns the decode driver
// (decompress_column). The two halves share the bitjoin_layout / column_copy
// helpers and reconstruct_representation (plan/representation_factory.cpp).

namespace {

// === codegen decode for fused subtrees ===
//
// Codegen compress writes one ``compressed_representation`` per node in a
// fused subtree, keyed by the node's plan path (``"input"`` /
// ``"delta.differences"`` / ...).  Decode recovers the whole subtree's op
// kinds + buffers and reconstructs the root column with a single codegen
// kernel launch.
//
// The per-node op kind comes from the PlanTree node's `op` field; the per-node
// device buffers come from ``compressed_representation::named_channels()``.
// Nothing per-kind lives in the walker — adding a fused op is a decode kernel,
// not new walker code.

const char* codegen_dtype_str_for(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT32: return "int32";
    case cudf::type_id::INT64: return "int64";
    case cudf::type_id::FLOAT32: return "float32";
    case cudf::type_id::FLOAT64: return "float64";
    default: return nullptr;
  }
}

// Map a DSL compressor name to its kind string if it is a codegen (fusable)
// op, or return empty if it isn't. The returned string equals the DSL op name
// so it can be used directly as the key into consumed_slots().
std::string codegen_kind_for_compressor(std::string const& c)
{
  if (c == "bitpack") return "bitpack";
  if (c == "delta") return "delta";
  if (c == "rle") return "rle";
  if (c == "for") return "for";
  if (c == "zigzag") return "zigzag";
  return {};
}

// The per-op buffer slots the decode binder reads from each rep, in order.
// Keys match the DSL op names (lowercase) returned by codegen_kind_for_compressor,
// plus "RawFused" for the synthetic raw-passthrough rep (no DSL counterpart).
// Note bitpack's 5th manifest slot (bp_offsets) is a decode-only transient
// synthesized by ``synthesize_decode_transients``, NOT bound here.  Empty
// list → unknown kind.
std::vector<std::string> consumed_slots(std::string const& kind)
{
  if (kind == "bitpack") return {"chunk_min", "chunk_count", "chunk_bits", "packed"};
  if (kind == "delta") return {"delta_first"};
  if (kind == "rle") return {"rle_runs_offsets"};
  if (kind == "for") return {"references"};
  if (kind == "zigzag") return {"zigzag"};
  if (kind == "RawFused") return {"data", "offsets"};
  return {};
}

// Recover the raw bytes of a channel that was entropy-tail-routed at compress
// time (a fused parent's buffer slot dropped from its producing rep because a
// downstream NON-codegen op further-compressed it, e.g.
// ``delta.differences.values.packed -> snappy``).  *nid* is the tree node of
// that downstream op (the parent's edge target for the slot's channel).
// Returns a fresh column holding the decoded channel bytes (same layout the
// producing rep's channel had), or nullptr on any failure.
//
//   * Single entropy op (the common case: identity / snappy / lz4 / …): its rep
//     self-decodes to the raw channel bytes — ``node.rep->decompress()``.
//   * Multi-step tail chain (e.g. ``…packed -> bitcomp -> output; …output ->
//     ans``): rebuild this op from its outputs (each output is a downstream
//     child to resolve first, or a terminal channel rep on this node), then
//     decompress.  Outputs are walked structurally via the node's edges /
//     channels — no DSL/path lookups.
//
// All work is enqueued on *stream*; the entropy reps' ``decompress`` sync the
// stream internally, so the returned column is safe to read once this returns.
struct TreeDecodeCtx {
  plan_compound const& compound;
  std::unordered_map<std::string, std::unique_ptr<cudf::column>> decompressed;
  std::unordered_set<NodeId> skipped;
  std::vector<std::unique_ptr<compressed_representation>> kept_reprs;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
};

// Returns the rep for node nid, or nullptr if the node has none.
compressed_representation const* node_rep(NodeId nid, PlanTree const& tree)
{
  if (nid < tree.nodes.size()) return tree.nodes[nid].rep.get();
  return nullptr;
}

// Finds the rep for a given output `path` owned by node `owner_nid`.
// Checks `rep` (via `rep_path`) then `channels`.
compressed_representation const* rep_at_path(std::string const& path,
                                             NodeId owner_nid,
                                             PlanTree const& tree)
{
  if (owner_nid < tree.nodes.size()) {
    auto const& owner = tree.nodes[owner_nid];
    if (owner.rep && owner.rep_path == path) return owner.rep.get();
    auto cit = owner.channels.find(path);
    if (cit != owner.channels.end() && cit->second) return cit->second.get();
  }
  return nullptr;
}

// Returns the output port name for `path` on `node`: output_names[i] where
// output_paths[i] == path. Falls back to `path` if not found.
std::string port_for_output_path(PlanNode const& node, std::string const& path)
{
  for (std::size_t i = 0; i < node.output_paths.size(); ++i) {
    if (node.output_paths[i] == path) return node.output_names[i];
  }
  return path;
}

// Per-slot element width for a LabeledBuffer. Fixed-width metadata slots carry
// their own widths; value-typed slots (chunk_min/delta_first/data/
// rle_run_values) carry the column's element width.
std::size_t elem_size_for_slot(std::string const& slot, std::size_t element_size)
{
  if (slot == "chunk_count" || slot == "rle_runs_offsets" || slot == "offsets")
    return sizeof(std::int32_t);
  if (slot == "chunk_bits") return sizeof(std::uint8_t);
  if (slot == "packed") return sizeof(std::uint32_t);
  return element_size;  // chunk_min, delta_first, data, rle_run_values
}

// Forward declaration.
std::unique_ptr<cudf::column> resolve_channel_bytes_node(NodeId nid,
                                                         PlanTree const& tree,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr);

// Bind the ``data``/``offsets`` slots of a synthesized Raw passthrough leaf
// at preorder *node_id*. The Raw leaf has no PlanTree op of its own; its
// bytes live in a RawFused rep parked on the parent node's ``channels``.
//
// The channel name is derived from origin.parent_op:
//   "rle" -> "values" (run values), "for" -> "deltas" (residuals).
bool bind_raw_passthrough_buffers(std::int32_t node_id,
                                  NodeId parent_id,
                                  std::string const& parent_op,
                                  PlanTree const& tree,
                                  std::size_t element_size,
                                  codegen::jit::LabeledBuffers& labeled,
                                  std::string* error_out)
{
  std::string const channel_name       = (parent_op == "for") ? "deltas" : "values";
  compressed_representation const* rep = nullptr;
  if (parent_id < tree.nodes.size()) {
    auto const& parent_node = tree.nodes[parent_id];
    for (auto const& [path, crep] : parent_node.channels) {
      if (!crep) continue;
      std::string port = port_for_output_path(parent_node, path);
      if (port == channel_name) {
        rep = crep.get();
        break;
      }
    }
  }
  if (rep == nullptr) {
    if (error_out)
      *error_out = "codegen decode: RawFused passthrough rep missing at " + parent_op + " node " +
                   std::to_string(parent_id) + " (channel '" + channel_name + "')";
    return false;
  }
  std::vector<compressible_output> nb = rep->named_channels();
  std::unordered_map<std::string, cudf::column_view> by_name;
  for (auto const& o : nb)
    by_name.emplace(o.name, o.view);
  for (auto const& slot : consumed_slots("RawFused")) {
    auto bit = by_name.find(slot);
    if (bit == by_name.end()) {
      if (error_out) *error_out = "codegen decode: RawFused leaf missing slot '" + slot + "'";
      return false;
    }
    labeled[codegen::jit::buffer_key(node_id, slot)] = {
      bit->second.head<void>(),
      static_cast<std::size_t>(bit->second.size()),
      elem_size_for_slot(slot, element_size)};
  }
  return true;
}

// Bind the device buffers for ONE real fused op node (bitpack / delta / rle)
// at preorder *node_id* into *labeled*. Buffers come from the node's
// rep ``named_channels()`` in per-op CONSUMED-slot order (``consumed_slots``);
// every rep is dense, so decode always uses the Compact gather.
//
// Entropy-tail-routed channels — a CONSUMED slot consumed downstream by a
// NON-codegen op (e.g. ``…packed -> snappy``, ``…packed -> bitcomp -> ans``),
// detected as an edge whose child op owns a rep — are RESOLVED here via
// ``resolve_channel_bytes_node`` (the downstream child subtree), and the
// scratch column parked in *tail_scratches_out* so its device buffer outlives
// the (synchronous) decode launch. An identity NO-OP terminal
// (``…chunk_min -> identity``) leaves the bytes inside THIS rep and is bound
// directly.
bool bind_real_node_buffers(std::int32_t node_id,
                            NodeId plan_node,
                            PlanTree const& tree,
                            std::size_t element_size,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr,
                            codegen::jit::LabeledBuffers& labeled,
                            std::vector<std::unique_ptr<cudf::column>>& tail_scratches_out,
                            std::string* error_out)
{
  PlanNode const& node                  = tree.nodes[plan_node];
  compressed_representation const* repr = node_rep(plan_node, tree);

  std::string kind = codegen_kind_for_compressor(node.op);
  if (kind.empty()) {
    if (error_out)
      *error_out = "codegen decode: non-codegen op at node " + std::to_string(plan_node) + " ('" +
                   node.op + "')";
    return false;
  }
  if (repr == nullptr) {
    if (error_out) *error_out = "codegen decode: missing rep at node " + std::to_string(plan_node);
    return false;
  }

  std::vector<compressible_output> nb = repr->named_channels();
  std::unordered_map<std::string, cudf::column_view> by_name;
  by_name.reserve(nb.size());
  for (auto const& o : nb)
    by_name.emplace(o.name, o.view);

  std::unordered_map<std::string, NodeId> edge_by_channel;
  edge_by_channel.reserve(node.children.size());
  for (auto const& e : node.children)
    edge_by_channel.emplace(e.channel, e.child);

  std::vector<std::string> const slots = consumed_slots(kind);
  if (slots.empty()) {
    if (error_out) *error_out = "codegen decode: no consumed-slot list for kind '" + kind + "'";
    return false;
  }
  for (auto const& slot : slots) {
    auto eit                      = edge_by_channel.find(slot);
    const bool has_edge           = eit != edge_by_channel.end();
    const bool downstream_has_rep = has_edge && node_rep(eit->second, tree) != nullptr;

    const void* ptr = nullptr;
    std::size_t len = 0;
    auto bit        = by_name.find(slot);
    if (bit != by_name.end() && !downstream_has_rep) {
      ptr = bit->second.head<void>();
      len = static_cast<std::size_t>(bit->second.size());
    } else if (has_edge) {
      auto col = resolve_channel_bytes_node(eit->second, tree, stream, mr);
      if (!col) {
        if (error_out) {
          *error_out = "codegen decode: failed to resolve tail-routed slot '" + slot +
                       "' at node " + std::to_string(plan_node);
        }
        return false;
      }
      auto v = col->view();
      ptr    = v.head<void>();
      len    = static_cast<std::size_t>(v.size());
      tail_scratches_out.push_back(std::move(col));
    } else {
      if (error_out)
        *error_out = "codegen decode: missing buffer for slot '" + slot + "' at node " +
                     std::to_string(plan_node);
      return false;
    }
    labeled[codegen::jit::buffer_key(node_id, slot)] = {
      ptr, len, elem_size_for_slot(slot, element_size)};
  }
  return true;
}

// Build the fused subtree rooted at *root_nid* via the shared structural
// builder (``build_fused_tree``), then bind every node's device buffers in the
// builder's DFS-preorder (preorder index == rendered kernel node_id). The
// structural shape (op kinds, children, node-id order) is the builder's
// responsibility — shared with the encode bridge — so this binder only sources
// the per-node reps/buffers. Decode uses ``fixed_stride=false`` (Compact).
//
// Returns the built tree, or nullptr + *error_out on failure.
std::shared_ptr<codegen::jit::FusedTree> bind_fused_subtree(
  NodeId root_nid,
  PlanTree const& tree,
  std::size_t element_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  codegen::jit::LabeledBuffers& labeled,
  std::vector<std::unique_ptr<cudf::column>>& tail_scratches_out,
  std::string* error_out)
{
  auto built = build_fused_tree(tree, root_nid, /*fixed_stride=*/false);
  if (!built) {
    if (error_out)
      *error_out =
        "codegen decode: no valid fusable region rooted at node " + std::to_string(root_nid);
    return nullptr;
  }
  for (std::int32_t node_id = 0; node_id < static_cast<std::int32_t>(built->preorder.size());
       ++node_id) {
    auto const& origin = built->preorder[node_id];
    if (origin.is_raw_passthrough) {
      if (!bind_raw_passthrough_buffers(
            node_id, origin.parent_rle, origin.parent_op, tree, element_size, labeled, error_out)) {
        return nullptr;
      }
    } else {
      if (!bind_real_node_buffers(node_id,
                                  origin.plan_node,
                                  tree,
                                  element_size,
                                  stream,
                                  mr,
                                  labeled,
                                  tail_scratches_out,
                                  error_out)) {
        return nullptr;
      }
    }
  }
  return built->tree;
}

// Build the codegen-fused subtree rooted at tree node *root_nid* + bind its
// buffers, then launch the codegen decode kernel into a fresh output column and
// return it.  The launch is synchronous (``decode_fused_subtree`` syncs the
// stream before returning) so there is no cross-call state to cache — every
// call re-binds the reps and re-launches; the JIT compile is cached
// process-wide in KernelCache.
//
// Returns nullptr + *error_out on failure.
std::unique_ptr<cudf::column> dispatch_codegen_subtree(NodeId root_nid,
                                                       PlanTree const& tree,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr,
                                                       std::string* error_out)
{
  compressed_representation const* root_repr = node_rep(root_nid, tree);
  if (root_repr == nullptr) {
    if (error_out)
      *error_out = "codegen decompress: no rep at root node " + std::to_string(root_nid);
    return nullptr;
  }
  cudf::data_type root_type = root_repr->decoded_type();
  cudf::size_type num_rows  = root_repr->num_rows;
  const char* dtype         = codegen_dtype_str_for(root_type);
  if (dtype == nullptr) {
    if (error_out) *error_out = "codegen decompress: unsupported root dtype";
    return nullptr;
  }

  // Build the FusedTree + bind the real device buffers (keyed by DFS-preorder
  // node_id) directly from the node-owned reps.  Scratch columns holding bytes
  // resolved for entropy-tail-routed channels must outlive the (synchronous)
  // decode launch below — they live to end-of-function, so the device buffers
  // are freed only after decode_fused_subtree returns.
  std::vector<std::unique_ptr<cudf::column>> tail_scratches;
  codegen::jit::LabeledBuffers labeled;
  const std::size_t element_size = static_cast<std::size_t>(cudf::size_of(root_type));
  std::string bind_err;
  auto fused = bind_fused_subtree(
    root_nid, tree, element_size, stream, mr, labeled, tail_scratches, &bind_err);
  if (!fused) {
    if (error_out) {
      *error_out = bind_err.empty() ? "codegen decompress: incomplete fused subtree" : bind_err;
    }
    return nullptr;
  }

  auto out_col =
    cudf::make_fixed_width_column(root_type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  if (!out_col) {
    if (error_out) *error_out = "codegen decompress: output column alloc failed";
    return nullptr;
  }
  std::uintptr_t out_ptr = reinterpret_cast<std::uintptr_t>(out_col->mutable_view().head<void>());

  int rc = decode_fused_subtree(*fused,
                                labeled,
                                dtype,
                                static_cast<std::int64_t>(num_rows),
                                out_ptr,
                                reinterpret_cast<std::uintptr_t>(stream.value()));
  if (rc != 1) {
    if (error_out) *error_out = "codegen decompress: decode failed";
    return nullptr;
  }
  return out_col;
}

// Mark every descendant of *root* (inclusive) as handled after a fused-subtree
// JIT dispatch at *root*.
void mark_subtree_nodes_skipped(NodeId root,
                                PlanTree const& tree,
                                std::unordered_set<NodeId>& skipped)
{
  std::vector<NodeId> stack{root};
  while (!stack.empty()) {
    NodeId const n = stack.back();
    stack.pop_back();
    if (!skipped.insert(n).second) continue;
    for (auto const& e : tree.nodes[n].children)
      stack.push_back(e.child);
  }
}

// Recover the output path produced by node `nid` on channel `channel`, reading
// directly from the node's output_names/output_paths metadata. For the input
// root (nid==0) the output is always "input" (the channel is "input" by
// convention). For other nodes, output_names[i]==channel -> output_paths[i].
std::string path_for_node_channel(NodeId nid, std::string const& channel, PlanTree const& tree)
{
  if (nid == 0) return "input";
  if (nid >= tree.nodes.size()) return {};
  auto const& node = tree.nodes[nid];
  for (std::size_t i = 0; i < node.output_names.size(); ++i) {
    if (node.output_names[i] == channel) return node.output_paths[i];
  }
  return {};
}

NodeId consumer_of_path(std::string const& path, PlanTree const& tree)
{
  for (NodeId i = 1; i < tree.nodes.size(); ++i) {
    if (tree.nodes[i].input_path == path) return i;
  }
  return static_cast<NodeId>(tree.nodes.size());
}

bool ensure_path_decoded(std::string const& path, TreeDecodeCtx& ctx, std::string* error_out);

bool decode_bitjoin_node(NodeId nid, TreeDecodeCtx& ctx, std::string* error_out)
{
  PlanTree const& tree = ctx.compound.tree;
  PlanNode const& node = tree.nodes[nid];
  if (!node.attrs.bitjoin.has_value()) {
    if (error_out) *error_out = "decode_bitjoin_node: missing bitjoin attrs";
    return false;
  }
  if (node.output_paths.size() != 1) {
    if (error_out) *error_out = "bitjoin decompression expects exactly 1 output path";
    return false;
  }
  std::string const& packed_path = node.output_paths[0];
  if (ctx.decompressed.find(packed_path) == ctx.decompressed.end()) {
    auto const* repr = rep_at_path(packed_path, nid, tree);
    if (!repr) repr = node_rep(nid, tree);
    if (!repr) {
      if (error_out) *error_out = "bitjoin decode: missing rep at '" + packed_path + "'";
      return false;
    }
    auto col = repr->decompress(ctx.stream, ctx.mr);
    if (!col) {
      if (error_out) *error_out = "bitjoin decode: failed to decompress packed leaf";
      return false;
    }
    ctx.decompressed.emplace(packed_path, std::move(col));
  }

  std::vector<std::string> input_paths;
  std::vector<std::optional<bit_range>> input_ranges;
  input_paths.reserve(node.attrs.bitjoin->inputs.size());
  input_ranges.reserve(node.attrs.bitjoin->inputs.size());
  for (auto const& ref : node.attrs.bitjoin->inputs) {
    std::string path = path_for_node_channel(ref.node, ref.channel, tree);
    if (path.empty()) {
      if (error_out) *error_out = "bitjoin decode: unknown input channel";
      return false;
    }
    input_paths.push_back(std::move(path));
    input_ranges.push_back(ref.range);
  }

  bitjoin_layout layout;
  if (!resolve_bitjoin_layout(node.op, input_paths, input_ranges, &layout, error_out)) {
    return false;
  }

  auto packed_it                = ctx.decompressed.find(packed_path);
  cudf::column_view packed_view = packed_it->second->view();
  int64_t n_elements            = static_cast<int64_t>(packed_view.size());

  struct field_ref {
    uint32_t width, src_lo, dst_lo;
  };
  std::unordered_map<std::string, std::vector<field_ref>> by_input;
  std::vector<std::string> input_order;
  for (size_t fi = 0; fi < input_paths.size(); ++fi) {
    auto& vec = by_input[input_paths[fi]];
    if (vec.empty()) input_order.push_back(input_paths[fi]);
    vec.push_back({layout.widths[fi], layout.src_los[fi], layout.dst_los[fi]});
  }

  for (auto const& path : input_order) {
    auto const& refs     = by_input.at(path);
    uint32_t max_src_top = 0;
    for (auto const& r : refs) {
      uint32_t top = r.src_lo + r.width;
      if (top > max_src_top) max_src_top = top;
    }
    cudf::type_id field_type_id = (max_src_top <= 8)    ? cudf::type_id::UINT8
                                  : (max_src_top <= 16) ? cudf::type_id::UINT16
                                  : (max_src_top <= 32) ? cudf::type_id::UINT32
                                                        : cudf::type_id::UINT64;
    auto field_col              = cudf::make_fixed_width_column(cudf::data_type(field_type_id),
                                                   static_cast<cudf::size_type>(n_elements),
                                                   cudf::mask_state::UNALLOCATED,
                                                   ctx.stream,
                                                   ctx.mr);
    cudaMemsetAsync(
      field_col->mutable_view().head<void>(),
      0,
      static_cast<size_t>(n_elements) * static_cast<size_t>(cudf::size_of(field_col->type())),
      ctx.stream.value());
    for (auto const& r : refs) {
      launch_bitjoin_field(field_col->mutable_view(),
                           packed_view,
                           static_cast<int>(r.dst_lo),
                           static_cast<int>(r.src_lo),
                           r.width,
                           ctx.stream.value());
    }
    ctx.decompressed[path] = std::move(field_col);
  }
  cudaStreamSynchronize(ctx.stream.value());
  ctx.decompressed.erase(packed_path);
  return true;
}

bool decode_node(NodeId nid, TreeDecodeCtx& ctx, std::string* error_out)
{
  if (ctx.skipped.count(nid)) return true;

  PlanTree const& tree = ctx.compound.tree;
  PlanNode const& node = tree.nodes[nid];

  for (auto const& e : node.children) {
    if (!decode_node(e.child, ctx, error_out)) return false;
  }

  if (node.attrs.bitjoin.has_value()) { return decode_bitjoin_node(nid, ctx, error_out); }

  if (is_codegen_compressor(node.op)) {
    if (is_fusion_interior(nid, tree)) {
      ctx.skipped.insert(nid);
      return true;
    }
    std::string const& target = node.input_path;
    if (target.empty()) {
      if (error_out)
        *error_out = "codegen decode: empty target path at node " + std::to_string(nid);
      return false;
    }
    if (ctx.decompressed.find(target) != ctx.decompressed.end()) {
      mark_subtree_nodes_skipped(nid, tree, ctx.skipped);
      return true;
    }
    auto col = dispatch_codegen_subtree(nid, tree, ctx.stream, ctx.mr, error_out);
    if (!col) return false;
    ctx.decompressed.emplace(target, std::move(col));
    mark_subtree_nodes_skipped(nid, tree, ctx.skipped);
    return true;
  }

  if (!node.output_paths.empty()) {
    for (auto const& opath : node.output_paths) {
      if (ctx.decompressed.find(opath) != ctx.decompressed.end()) continue;
      auto chit = tree.nodes[nid].channels.find(opath);
      if (chit != tree.nodes[nid].channels.end() && chit->second) {
        auto col = chit->second->decompress(ctx.stream, ctx.mr);
        if (!col) {
          if (error_out) *error_out = "failed to decompress terminal channel '" + opath + "'";
          return false;
        }
        ctx.decompressed.emplace(opath, std::move(col));
        continue;
      }
      if (!ensure_path_decoded(opath, ctx, error_out)) return false;
    }

    std::vector<std::unique_ptr<cudf::column>> outputs;
    outputs.reserve(node.output_paths.size());
    for (auto const& opath : node.output_paths) {
      auto it = ctx.decompressed.find(opath);
      if (it == ctx.decompressed.end()) {
        if (error_out) *error_out = "missing decoded output '" + opath + "'";
        return false;
      }
      outputs.push_back(std::move(it->second));
      ctx.decompressed.erase(it);
    }

    std::unique_ptr<compressed_representation> repr;
    {
      nvtx3::scoped_range r_build{"build_repr:" + node.op};
      repr = reconstruct_representation(
        node.op, node.output_names, std::move(outputs), ctx.stream, ctx.mr, error_out, node.meta);
    }
    if (!repr) return false;

    std::unique_ptr<cudf::column> col;
    {
      nvtx3::scoped_range r_final{"final_decompress:" + node.op};
      col = repr->decompress(ctx.stream, ctx.mr);
    }
    ctx.kept_reprs.push_back(std::move(repr));
    if (!col) {
      if (error_out) *error_out = "failed to decompress '" + node.input_path + "'";
      return false;
    }
    ctx.decompressed.emplace(node.input_path, std::move(col));
    return true;
  }

  if (auto const* repr = node_rep(nid, tree)) {
    std::string out_path = node.rep_path.empty() ? node.input_path : node.rep_path;
    if (ctx.decompressed.find(out_path) != ctx.decompressed.end()) return true;
    std::unique_ptr<cudf::column> col;
    {
      nvtx3::scoped_range r_leaf{"leaf_decompress:" + out_path};
      col = repr->decompress(ctx.stream, ctx.mr);
    }
    if (!col) {
      if (error_out) *error_out = "failed to decompress leaf '" + out_path + "'";
      return false;
    }
    ctx.decompressed.emplace(out_path, std::move(col));
  }
  return true;
}

bool ensure_path_decoded(std::string const& path, TreeDecodeCtx& ctx, std::string* error_out)
{
  if (ctx.decompressed.find(path) != ctx.decompressed.end()) return true;
  PlanTree const& tree  = ctx.compound.tree;
  NodeId const consumer = consumer_of_path(path, tree);
  if (consumer < tree.nodes.size()) { return decode_node(consumer, ctx, error_out); }
  // Path has no consumer in the tree — search all nodes for a direct rep.
  for (NodeId nid = 0; nid < tree.nodes.size(); ++nid) {
    auto const& tnode = tree.nodes[nid];
    if (tnode.rep && tnode.rep_path == path) {
      auto col = tnode.rep->decompress(ctx.stream, ctx.mr);
      if (!col) {
        if (error_out) *error_out = "failed to decompress leaf '" + path + "'";
        return false;
      }
      ctx.decompressed.emplace(path, std::move(col));
      return true;
    }
    auto cit = tnode.channels.find(path);
    if (cit != tnode.channels.end() && cit->second) {
      auto col = cit->second->decompress(ctx.stream, ctx.mr);
      if (!col) {
        if (error_out) *error_out = "failed to decompress leaf '" + path + "'";
        return false;
      }
      ctx.decompressed.emplace(path, std::move(col));
      return true;
    }
  }
  if (error_out) *error_out = "decode: could not resolve path '" + path + "'";
  return false;
}

bool decode_column_tree(TreeDecodeCtx& ctx, std::string* error_out)
{
  PlanTree const& tree = ctx.compound.tree;
  if (!tree.nodes[0].children.empty()) {
    for (auto const& e : tree.nodes[0].children) {
      if (!decode_node(e.child, ctx, error_out)) return false;
    }
    return true;
  }
  // Fallback when structural edges are missing on the input root (the stored
  // PlanTree from compress can be rep-only).  Seed decode from every op that
  // consumes the column value directly.
  bool progress = false;
  for (NodeId i = 1; i < tree.nodes.size(); ++i) {
    if (tree.nodes[i].input_path != "input") continue;
    if (!decode_node(i, ctx, error_out)) return false;
    progress = true;
  }
  if (!progress && error_out) { *error_out = "decompress: no op consumes path 'input'"; }
  return progress;
}

// resolve_channel_bytes_node defined after all helpers it calls.
std::unique_ptr<cudf::column> resolve_channel_bytes_node(NodeId nid,
                                                         PlanTree const& tree,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  PlanNode const& node = tree.nodes[nid];
  if (node.rep) { return node.rep->decompress(stream, mr); }
  std::vector<std::string> names;
  std::vector<std::unique_ptr<cudf::column>> resolved;
  for (auto const& e : node.children) {
    auto col = resolve_channel_bytes_node(e.child, tree, stream, mr);
    if (!col) return nullptr;
    names.push_back(e.channel);
    resolved.push_back(std::move(col));
  }
  for (auto const& [path, rep] : node.channels) {
    if (!rep) return nullptr;
    auto col = rep->decompress(stream, mr);
    if (!col) return nullptr;
    names.push_back(port_for_output_path(node, path));
    resolved.push_back(std::move(col));
  }
  std::string err;
  auto rep =
    reconstruct_representation(node.op, names, std::move(resolved), stream, mr, &err, node.meta);
  if (!rep) return nullptr;
  return rep->decompress(stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> decompress_column(plan_compound const& compound,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr,
                                                std::string* error_out)
{
  nvtx3::scoped_range r_decompress{"decompress_column"};

  PlanTree const& tree = compound.tree;
  if (tree.nodes.empty() || tree.nodes[0].op != "input") {
    if (error_out) *error_out = "decompress: compound.tree missing input root";
    return nullptr;
  }

  TreeDecodeCtx ctx{compound, {}, {}, {}, stream, mr};
  if (!decode_column_tree(ctx, error_out)) return nullptr;

  auto root_it = ctx.decompressed.find("input");
  if (root_it == ctx.decompressed.end()) {
    if (error_out) *error_out = "decompression completed but 'input' column not reconstructed";
    return nullptr;
  }
  cudaStreamSynchronize(stream.value());
  auto result = std::move(root_it->second);
  if (error_out) error_out->clear();
  return result;
}

// ---------------------------------------------------------------------------
// plan_compound_from_leaves
// ---------------------------------------------------------------------------
// Reconstruct a plan_compound from a DSL string and a pre-built path→rep map.
// Used by the IO read path after it has deserialized and allocated each rep.

std::unique_ptr<plan_compound> plan_compound_from_leaves(
  std::string plan_dsl,
  std::unordered_map<std::string, std::unique_ptr<compressed_representation>> leaves,
  std::string* error_out)
{
  auto compound      = std::make_unique<plan_compound>();
  compound->plan_dsl = std::move(plan_dsl);

  std::string err;
  PlanPathMap path_map;
  auto tree = plan_tree_from_dsl(compound->plan_dsl, &err, &path_map);
  if (!tree) {
    if (error_out) *error_out = "plan_compound_from_leaves: " + err;
    return nullptr;
  }
  compound->tree = std::move(*tree);

  // Assign each leaf to its owning node slot in two passes so that a path
  // which is simultaneously an input_path of one node AND an output_path of
  // its parent (e.g. "for.deltas" consumed by bitpack but listed in for's
  // output_paths) is always claimed as node.rep of the consuming node first,
  // rather than being stolen into the parent's channels.
  //
  // Pass 1: node.rep ← rep keyed by node.input_path (consuming path).
  for (auto& node : compound->tree.nodes) {
    auto it = leaves.find(node.input_path);
    if (it != leaves.end() && it->second) {
      node.rep      = std::move(it->second);
      node.rep_path = node.input_path;
      node.meta     = node.rep->describe_meta();
    }
  }
  // Pass 2: node.channels ← reps keyed by output_paths (terminal leaves only;
  // paths already claimed by a consuming node's rep are gone from the map).
  for (auto& node : compound->tree.nodes) {
    for (auto const& out_path : node.output_paths) {
      auto ch_it = leaves.find(out_path);
      if (ch_it != leaves.end() && ch_it->second) {
        node.channels.emplace(out_path, std::move(ch_it->second));
      }
    }
  }
  return compound;
}

}  // namespace simpatico
