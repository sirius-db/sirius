// SPDX-License-Identifier: Apache-2.0
#include "codegen/bridge/fused_tree_build.hpp"
#include "codegen/codegen_bridge.hpp"
#include "codegen/decode/masked_launch.hpp"
#include "codegen/plan/bitjoin_layout.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "decode/decode_session.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>

// The high-level JIT decode bridge is defined here alongside DecodeWalk because
// buffer binding may recursively materialize entropy tails. Its low-level
// launch_decode_fused_tree counterpart lives in codegen_runtime.cpp.

namespace simpatico {
// The compress driver (the recursive CompressWalk, compress_column) lives
// in plan/compress.cpp.
// This file owns the decode driver
// (decompress_column). The two halves share the bitjoin_layout helpers
// (including copy_column_view{,_as_uint8}) and reconstruct_representation
// (plan/representation_factory.cpp).

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
    case cudf::type_id::INT8: return "int8";
    case cudf::type_id::UINT8: return "uint8";
    case cudf::type_id::INT16: return "int16";
    case cudf::type_id::UINT16: return "uint16";
    case cudf::type_id::INT32: return "int32";
    case cudf::type_id::INT64: return "int64";
    case cudf::type_id::UINT32: return "uint32";
    case cudf::type_id::UINT64: return "uint64";
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

// The frame owns the structural memo; this visitor never owns pending device dependencies.
class DecodeWalk {
 public:
  DecodeWalk(PlanTree const& tree,
             decode_frame& frame,
             std::string* error_out,
             decode_predicate const* pred,
             decode_selection const* sel);
  cudf::column const* materialize(NodeId nid);
  void run(decode_column_slot output);

 private:
  void materialize_fused_node(NodeId nid,
                              decode_selection const* node_sel,
                              decode_column_slot output);
  [[nodiscard]] bool predicate_applies_to(NodeId nid) const;
  [[nodiscard]] bool selection_applies_to(NodeId nid) const;

  PlanTree const& tree;
  decode_frame& frame;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  std::string* error_out;
  decode_predicate const* pred = nullptr;
  decode_selection const* sel  = nullptr;
  NodeId sel_target;
  bool predicate_resolved = false;
};

std::string value_label(ValueId v)
{
  return "(" + std::to_string(v.node) + "," + std::to_string(v.channel) + ")";
}

// Transfers a memoised value to the inverse of its producer. Shared values are copied until their
// last consumer; a sole/last consumer takes ownership. A null entry is deliberately retained after
// a move so any accidental re-request is a deterministic runtime error rather than a silent
// re-decode.
decode_column_slot consume_memo_value(ValueId value, decode_frame& frame)
{
  auto const key     = value_id_key(value);
  auto const* column = frame.find_memo(key);
  if (!column)
    throw std::runtime_error("decode: unresolved or consumed memo value " + value_label(value));
  auto count_it = frame.remaining_consumers.find(key);
  if (count_it == frame.remaining_consumers.end() || count_it->second == 0) {
    throw std::runtime_error("decode: memo value has no remaining consumer " + value_label(value));
  }
  --count_it->second;
  if (count_it->second == 0) return frame.memo_column(key);
  return frame.keep_column(
    std::make_unique<cudf::column>(column->view(), frame.stream(), frame.mr()));
}

// Returns the rep for node nid, or nullptr if the node has none.
compressed_representation const* node_rep(NodeId nid, PlanTree const& tree)
{
  if (nid < tree.nodes.size()) return tree.nodes[nid].rep.get();
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

// name → view map of a rep's channels, for repeated per-slot lookups.
std::unordered_map<std::string, cudf::column_view> channels_by_name(
  compressed_representation const& rep, rmm::cuda_stream_view stream)
{
  std::unordered_map<std::string, cudf::column_view> by_name;
  for (auto const& o : rep.named_channels(stream))
    by_name.emplace(o.name, o.view);
  return by_name;
}

// Bind the ``data``/``offsets`` slots of a synthesized Raw passthrough leaf
// at preorder *node_id*. The Raw leaf has no PlanTree op of its own; its
// bytes live in a RawFused rep parked on the parent node's ``channels``.
//
// Two cases depending on whether the channel was entropy-tail-routed at encode:
//
//   Terminal (no downstream consumer): the RawFused rep holds both ``data``
//   and ``offsets``; bind them directly.
//
//   Entropy-tail (data was routed to a downstream non-fused op, e.g. ans):
//   the RawFused rep holds only ``offsets``; ``data`` is resolved by calling
//   materialize on the downstream PlanTree child node (the non-fused op that
//   compressed the raw bytes). The result is a view into the shared memo, which
//   owns it through the decode launch.
//
// Element size for the data slot:
//   rle.runs  -> always sizeof(int32_t) (run counts are int32 regardless of
//               the column's original type).
//   all others -> element_size (original column element size).
bool bind_raw_passthrough_buffers(std::int32_t node_id,
                                  NodeId parent_id,
                                  std::string const& parent_op,
                                  std::string const& parent_channel,
                                  PlanTree const& tree,
                                  std::size_t element_size,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr,
                                  codegen::jit::LabeledBuffers& labeled,
                                  decode_materialize_fn const& materialize,
                                  std::string* error_out)
{
  // The fused-tree builder always records the materialized channel name on the
  // raw-passthrough origin (differences / runs / values / deltas).
  std::string const& channel_name = parent_channel;

  // run-count elements are always int32, regardless of the original column type.
  const std::size_t data_elem_size = (channel_name == "runs") ? sizeof(std::int32_t) : element_size;

  // Locate the RawFused rep in the parent's channels.
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

  auto by_name = channels_by_name(*rep, stream);

  for (auto const& slot : consumed_slots("RawFused")) {
    if (slot == "data" && by_name.find(slot) == by_name.end()) {
      // Entropy-tail: data was stripped from the rep at encode time and
      // compressed by a downstream non-fused op.  Find that child node and
      // resolve (decompress) its bytes back to the raw element array.
      if (parent_id >= tree.nodes.size()) {
        if (error_out)
          *error_out = "codegen decode: invalid parent_id for entropy-tail data resolve";
        return false;
      }
      auto const& parent_node = tree.nodes[parent_id];
      NodeId child_id         = static_cast<NodeId>(tree.nodes.size());
      for (auto const& e : parent_node.children) {
        if (e.channel == channel_name) {
          child_id = e.child;
          break;
        }
      }
      if (child_id >= tree.nodes.size()) {
        if (error_out)
          *error_out = "codegen decode: no child edge '" + channel_name + "' on parent " +
                       std::to_string(parent_id) + " for entropy-tail resolve";
        return false;
      }
      cudf::column const* resolved = materialize(child_id);
      if (!resolved) {
        if (error_out && error_out->empty())
          *error_out = "codegen decode: entropy-tail resolve failed for RawFused channel '" +
                       channel_name + "'";
        return false;
      }
      cudf::column_view dv                               = resolved->view();
      labeled[codegen::jit::buffer_key(node_id, "data")] = {
        dv.head<void>(), static_cast<std::size_t>(dv.size()), data_elem_size};
      continue;
    }
    auto bit = by_name.find(slot);
    if (bit == by_name.end()) {
      if (error_out) *error_out = "codegen decode: RawFused leaf missing slot '" + slot + "'";
      return false;
    }
    labeled[codegen::jit::buffer_key(node_id, slot)] = {
      bit->second.head<void>(),
      static_cast<std::size_t>(bit->second.size()),
      elem_size_for_slot(slot, data_elem_size)};
  }
  return true;
}

// Bind the device buffers for ONE real fused op node (bitpack / delta / rle)
// at preorder *node_id* into *labeled*. Buffers come from the node's
// rep ``named_channels()`` in per-op CONSUMED-slot order (``consumed_slots``);
// every rep is dense, so decode always uses the Compact gather.
//
// Entropy-tail-routed channels — a CONSUMED slot consumed downstream by
// another op (e.g. ``…packed -> snappy``, ``…packed -> bitcomp -> ans``, or a
// codegen tail ``…chunk_min -> zigzag``), detected as a child edge — are
// RESOLVED here via ``materialize`` (the downstream subtree), which returns a
// view into the shared memo that owns it through completion of the decode
// launch. An identity NO-OP terminal (``…chunk_min -> identity``) leaves the
// bytes inside THIS rep and is bound directly.
bool bind_real_node_buffers(std::int32_t node_id,
                            NodeId plan_node,
                            PlanTree const& tree,
                            std::size_t element_size,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr,
                            codegen::jit::LabeledBuffers& labeled,
                            decode_materialize_fn const& materialize,
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

  auto by_name = channels_by_name(*repr, stream);

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
      // Slot lives directly in this node's rep — bind it straight.
      ptr = bit->second.head<void>();
      len = static_cast<std::size_t>(bit->second.size());
    } else if (has_edge) {
      // Tail-routed slot (a downstream codegen region OR non-codegen rep
      // consumes it): materialize the downstream's output — a view into the
      // shared memo, which owns it through completion of the launch. One path for
      // both, no empty-map special case (e.g. …bitpack -> chunk_min -> zigzag
      // resolves the nested codegen tail through the same memo).
      cudf::column const* col = materialize(eit->second);
      if (!col) {
        if (error_out && error_out->empty())
          *error_out = "codegen decode: failed to resolve tail slot '" + slot + "' at node " +
                       std::to_string(plan_node);
        return false;
      }
      auto v = col->view();
      ptr    = v.head<void>();
      len    = static_cast<std::size_t>(v.size());
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

// Bind every node's device buffers from an already-built fused subtree in the
// builder's DFS-preorder (preorder index == rendered kernel node_id). The
// structural shape (op kinds, children, node-id order) is the builder's
// responsibility — shared with the encode bridge — so this binder only sources
// the per-node reps/buffers. Decode is Compact-only.
bool bind_fused_subtree(BuiltFusedTree const& built,
                        PlanTree const& tree,
                        std::size_t element_size,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr,
                        codegen::jit::LabeledBuffers& labeled,
                        decode_materialize_fn const& materialize,
                        std::string* error_out)
{
  for (std::int32_t node_id = 0; node_id < static_cast<std::int32_t>(built.preorder.size());
       ++node_id) {
    auto const& origin = built.preorder[node_id];
    // Transformer-mode ZigZag stores nothing (it rewrites the lane value
    // inline and recurses); the decode renderer emits no params for it, so
    // there is no buffer to bind. Skip it — the child binds its own buffers.
    if (origin.node != nullptr && origin.node->op == codegen::OpKind::Zigzag &&
        !origin.node->children.empty()) {
      continue;
    }
    if (origin.is_raw_passthrough) {
      if (!bind_raw_passthrough_buffers(node_id,
                                        origin.parent_node,
                                        origin.parent_op,
                                        origin.parent_channel,
                                        tree,
                                        element_size,
                                        stream,
                                        mr,
                                        labeled,
                                        materialize,
                                        error_out)) {
        return false;
      }
    } else {
      if (!bind_real_node_buffers(node_id,
                                  origin.plan_node,
                                  tree,
                                  element_size,
                                  stream,
                                  mr,
                                  labeled,
                                  materialize,
                                  error_out)) {
        return false;
      }
    }
  }
  return true;
}

// The shared prologue of every fused-region decode launch (plain, mask-out,
// mask-consume): the built FusedTree, the region's decoded metadata, and the
// bound device buffers.
struct bound_fused_region {
  BuiltFusedTree built;
  cudf::data_type root_type{cudf::type_id::EMPTY};
  cudf::size_type num_rows = 0;
  const char* dtype        = nullptr;
  codegen::jit::LabeledBuffers labeled;
};

// Build one codegen-fused subtree, resolve its metadata, and bind its device
// buffers (keyed by DFS-preorder node_id) directly from the node-owned reps.
// Entropy-tail-routed channels are materialized into the caller's shared memo,
// which owns them through completion of the launch that follows.
//
// Some intermediate fuse nodes store nothing and own no rep; their children
// own the reps. In that case, the first non-null rep in the fused preorder
// provides the decoded type and num_rows.
std::optional<bound_fused_region> bind_fused_region(PlanTree const& tree,
                                                    NodeId root_nid,
                                                    decode_materialize_fn const& materialize,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref const& mr,
                                                    std::string* error_out)
{
  auto built = build_fused_tree(tree, root_nid);
  if (!built) {
    if (error_out)
      *error_out =
        "codegen decode: no valid fusable region rooted at node " + std::to_string(root_nid);
    return std::nullopt;
  }

  compressed_representation const* root_repr = node_rep(root_nid, tree);
  if (root_repr == nullptr) {
    // Transformer-mode root (e.g. ZigZag with a codegen child) stores no rep;
    // find the first non-null rep in the already-built preorder for metadata.
    for (auto const& origin : built->preorder) {
      if (!origin.is_raw_passthrough && origin.plan_node < tree.nodes.size()) {
        root_repr = node_rep(origin.plan_node, tree);
        if (root_repr) break;
      }
    }
    if (root_repr == nullptr) {
      if (error_out)
        *error_out = "codegen decompress: no rep at root node " + std::to_string(root_nid);
      return std::nullopt;
    }
  }

  bound_fused_region region;
  region.root_type = root_repr->decoded_type();
  region.num_rows  = root_repr->num_rows;
  region.dtype     = codegen_dtype_str_for(region.root_type);
  if (region.dtype == nullptr) {
    if (error_out) *error_out = "codegen decompress: unsupported root dtype";
    return std::nullopt;
  }

  const std::size_t element_size = static_cast<std::size_t>(cudf::size_of(region.root_type));
  std::string bind_err;
  if (!bind_fused_subtree(
        *built, tree, element_size, stream, mr, region.labeled, materialize, &bind_err)) {
    if (error_out) {
      *error_out = bind_err.empty() ? "codegen decompress: incomplete fused subtree" : bind_err;
    }
    return std::nullopt;
  }
  region.built = std::move(*built);
  return region;
}

// Bind one region and submit into a preregistered owner. Every variant uses the same frame.
void decode_fused_subtree_impl(PlanTree const& tree,
                               NodeId root_nid,
                               decode_materialize_fn const& materialize,
                               decode_frame& frame,
                               decode_column_slot output,
                               std::string* error_out,
                               decode_selection const* sel = nullptr)
{
  auto region =
    bind_fused_region(tree, root_nid, materialize, frame.stream(), frame.mr(), error_out);
  if (!region) throw std::runtime_error(error_out ? *error_out : "decode region binding failed");
  auto const num_rows = region->num_rows;
  bool const masked   = sel && sel->active();
  if (masked && sel->survivor_count > num_rows) {
    throw std::runtime_error("decode selection exceeds the column row count");
  }
  auto const out_rows = masked ? static_cast<cudf::size_type>(sel->survivor_count) : num_rows;
  output.adopt(cudf::make_fixed_width_column(
    region->root_type, out_rows, cudf::mask_state::UNALLOCATED, frame.stream(), frame.mr()));
  if (out_rows == 0) return;
  auto& built       = region->built;
  auto& labeled     = region->labeled;
  auto const* dtype = region->dtype;
  if (!masked) {
    launch_decode_fused_tree(
      *built.tree, labeled, dtype, num_rows, output->mutable_view().head<void>(), frame);
    return;
  }
  if (sel->rows) {
    if (!sel->rows->valid() || sel->rows->num_survivors != sel->survivor_count ||
        sel->rows->num_rows != num_rows) {
      throw std::runtime_error("decode row set does not match the selected column");
    }
    sirius::codegen::selection_mask const hollow{nullptr, num_rows, sel->survivor_count, nullptr};
    launch_decode_fused_tree_compacted(*built.tree,
                                       labeled,
                                       dtype,
                                       num_rows,
                                       hollow,
                                       row_enumeration{nullptr, sel->rows},
                                       output->mutable_view().head<void>(),
                                       frame);
    return;
  }
  if (!sel->mask->chunk_offsets || sel->mask->survivor_count != sel->survivor_count) {
    throw std::runtime_error("decode selection mask has no matching count/offsets");
  }
  bool const by_index =
    sel->enumerate_by_index && sel->route == sirius::codegen::decode_route::bitpack_mask &&
    tree.nodes[root_nid].op == "bitpack" && sel->survivor_indices.size() == sel->survivor_count;
  launch_decode_fused_tree_compacted(
    *built.tree,
    labeled,
    dtype,
    num_rows,
    *sel->mask,
    row_enumeration{by_index ? sel->survivor_indices.data<std::int32_t>() : nullptr, nullptr},
    output->mutable_view().head<void>(),
    frame);
}

// Rep holding a bitjoin node's own (packed) output leaf: its node rep, or the
// terminal channel parked for output port 0.
compressed_representation const* bitjoin_packed_rep(PlanNode const& node)
{
  if (node.rep) return node.rep.get();
  if (!node.output_paths.empty()) {
    auto cit = node.channels.find(node.output_paths[0]);
    if (cit != node.channels.end() && cit->second) return cit->second.get();
  }
  return nullptr;
}

// Split a bitjoin node's packed leaf back into its input field values, keyed in
// `memo` by each input's structural ValueId. Fields sharing a source value are
// OR-ed into one column (a source may receive several bit ranges).
void decode_bitjoin(NodeId nid, PlanTree const& tree, decode_frame& frame)
{
  auto const& node = tree.nodes[nid];
  auto const* rep  = bitjoin_packed_rep(node);
  if (!node.attrs.bitjoin || !rep)
    throw std::runtime_error("bitjoin decode: missing layout or packed rep");
  auto packed = frame.make_column();
  decode_standalone(*rep, frame, packed);
  auto const packed_view = packed.view();
  std::vector<std::optional<bit_range>> input_ranges;
  for (auto const& ref : node.attrs.bitjoin->inputs)
    input_ranges.push_back(ref.range);
  bitjoin_layout layout;
  std::string error;
  if (!resolve_bitjoin_layout(node.op, node.input_sources.size(), input_ranges, &layout, &error)) {
    throw std::runtime_error(error);
  }
  struct field_ref {
    std::uint32_t width, src_lo, dst_lo;
  };
  std::unordered_map<std::uint64_t, std::vector<field_ref>> by_src;
  for (std::size_t i = 0; i < node.input_sources.size(); ++i) {
    by_src[value_id_key(node.input_sources[i])].push_back(
      {layout.widths[i], layout.src_los[i], layout.dst_los[i]});
  }
  for (auto const& [key, refs] : by_src) {
    std::uint32_t top = 0;
    for (auto const& ref : refs)
      top = std::max(top, ref.src_lo + ref.width);
    auto const type = top <= 8    ? cudf::type_id::UINT8
                      : top <= 16 ? cudf::type_id::UINT16
                      : top <= 32 ? cudf::type_id::UINT32
                                  : cudf::type_id::UINT64;
    auto output     = frame.memo_column(key);
    output.adopt(cudf::make_fixed_width_column(cudf::data_type{type},
                                               packed_view.size(),
                                               cudf::mask_state::UNALLOCATED,
                                               frame.stream(),
                                               frame.mr()));
    auto status =
      cudaMemsetAsync(output->mutable_view().head<void>(),
                      0,
                      static_cast<std::size_t>(packed_view.size()) * cudf::size_of(output->type()),
                      frame.stream().value());
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    for (auto const& ref : refs) {
      launch_bitjoin_field(output->mutable_view(),
                           packed_view,
                           static_cast<int>(ref.dst_lo),
                           static_cast<int>(ref.src_lo),
                           ref.width,
                           frame.stream().value());
    }
  }
}

void DecodeWalk::materialize_fused_node(NodeId nid,
                                        decode_selection const* node_sel,
                                        decode_column_slot output)
{
  decode_materialize_fn resolve = [this](NodeId dependency) { return materialize(dependency); };
  decode_fused_subtree_impl(tree, nid, resolve, frame, output, error_out, node_sel);
}

cudf::column const* DecodeWalk::materialize(NodeId nid)
{
  auto const& node   = tree.nodes.at(nid);
  auto const primary = node.input_sources.empty() ? ValueId{nid, 0} : node.input_sources.front();
  auto const key     = value_id_key(primary);
  if (frame.contains_memo(key)) {
    auto* value = frame.find_memo(key);
    if (!value)
      throw std::runtime_error("decode: memo value already consumed " + value_label(primary));
    return value;
  }
  if (node.attrs.bitjoin) {
    decode_bitjoin(nid, tree, frame);
    return frame.find_memo(key);
  }
  auto output = frame.memo_column(key);
  if (is_codegen_compressor(node.op)) {
    materialize_fused_node(nid, selection_applies_to(nid) ? sel : nullptr, output);
  } else if (node.rep) {
    if (predicate_applies_to(nid)) {
      if (auto const* dict =
            dynamic_cast<dictionary_compressed_representation const*>(node.rep.get())) {
        predicate_resolved = dict->decompress_predicate(*pred, frame, output);
      }
    }
    if (!output) decode_standalone(*node.rep, frame, output);
  } else {
    std::vector<std::string> names;
    std::vector<decode_column_slot> outputs;
    names.reserve(node.output_names.size());
    outputs.reserve(node.output_names.size());
    for (std::size_t i = 0; i < node.output_names.size(); ++i) {
      auto const& name = node.output_names[i];
      auto child       = std::find_if(node.children.begin(),
                                node.children.end(),
                                [&](PlanEdge const& edge) { return edge.channel == name; });
      if (child != node.children.end()) {
        ValueId const value{nid, static_cast<ChannelId>(i)};
        if (!frame.contains_memo(value_id_key(value))) materialize(child->child);
        names.push_back(name);
        outputs.push_back(consume_memo_value(value, frame));
      } else {
        auto channel = node.channels.find(node.output_paths[i]);
        if (channel == node.channels.end()) continue;
        if (!channel->second) throw std::runtime_error("decode: missing terminal channel");
        auto slot = frame.memo_column(value_id_key(ValueId{nid, static_cast<ChannelId>(i)}));
        if (!slot) decode_standalone(*channel->second, frame, slot);
        names.push_back(name);
        outputs.push_back(slot);
      }
    }
    auto& rep = reconstruct_decode_representation(node.op, names, outputs, node.meta, frame);
    if (predicate_applies_to(nid)) {
      if (auto const* dict = dynamic_cast<dictionary_compressed_representation const*>(&rep)) {
        predicate_resolved = dict->decompress_predicate(*pred, frame, output);
      }
    }
    if (!output) decode_standalone(rep, frame, output);
  }
  if (!output) throw std::runtime_error("decode: leaf returned no column");
  return &output.get();
}

DecodeWalk::DecodeWalk(PlanTree const& tree,
                       decode_frame& frame,
                       std::string* error_out,
                       decode_predicate const* pred,
                       decode_selection const* sel)
  : tree(tree),
    frame(frame),
    stream(frame.stream()),
    mr(frame.mr()),
    error_out(error_out),
    pred(pred && pred->active() ? pred : nullptr),
    sel(sel && sel->active() ? sel : nullptr),
    sel_target(static_cast<NodeId>(tree.nodes.size()))
{
  for (auto const& node : tree.nodes)
    for (auto const& source : node.input_sources)
      ++frame.remaining_consumers[value_id_key(source)];
  if (!this->pred && (!this->sel || this->sel->compacted())) {
    frame.terminal_memo(value_id_key(ValueId{0, 0}));
  }
  if (this->sel && this->sel->compacted()) {
    for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
      auto const& sources = tree.nodes[nid].input_sources;
      if (sources.empty() || !(sources.front() == ValueId{0, 0})) continue;
      if (this->sel->route != sirius::codegen::decode_route::dict_codes)
        sel_target = nid;
      else {
        for (auto const& edge : tree.nodes[nid].children) {
          if (edge.channel == "indices") {
            sel_target = edge.child;
            break;
          }
        }
      }
      break;
    }
  }
}

bool DecodeWalk::predicate_applies_to(NodeId nid) const
{
  if (pred == nullptr || predicate_resolved) { return false; }
  // Only the node that produces the column's final value (node 0, port 0) may
  // answer the predicate; an inner node's output is an intermediate channel.
  auto const& sources = tree.nodes[nid].input_sources;
  return !sources.empty() && sources.front() == ValueId{0, 0};
}

bool DecodeWalk::selection_applies_to(NodeId nid) const
{
  return sel != nullptr && nid == sel_target;
}

void DecodeWalk::run(decode_column_slot output)
{
  auto const key = value_id_key(ValueId{0, 0});
  for (auto const& edge : tree.nodes[0].children) {
    if (frame.find_memo(key)) break;
    materialize(edge.child);
  }
  if (!frame.find_memo(key)) {
    for (NodeId nid = 1; nid < tree.nodes.size() && !frame.find_memo(key); ++nid) {
      for (auto const& source : tree.nodes[nid].input_sources) {
        if (source == ValueId{0, 0}) {
          materialize(nid);
          break;
        }
      }
    }
  }
  auto const* value = frame.find_memo(key);
  if (!value) throw std::runtime_error("decode: input column was not reconstructed");
  if (pred && !predicate_resolved) {
    auto mask            = frame.make_column();
    auto const bool_type = cudf::data_type{cudf::type_id::BOOL8};
    for (auto const& text : pred->equals_any) {
      auto const& needle = frame.keep_scalar(
        std::make_unique<cudf::string_scalar>(text, true, stream, mr), text.size() + 256);
      auto hit = frame.make_column();
      hit.adopt(cudf::binary_operation(
        value->view(), needle, cudf::binary_operator::EQUAL, bool_type, stream, mr));
      if (!mask)
        mask.adopt(frame.release(hit));
      else {
        auto combined = frame.make_column();
        combined.adopt(cudf::binary_operation(
          mask.view(), hit.view(), cudf::binary_operator::LOGICAL_OR, bool_type, stream, mr));
        mask = combined;
      }
    }
    if (!mask) throw std::runtime_error("decode: predicate has no values");
    output.adopt(frame.release(mask));
  } else {
    output.adopt(frame.release_memo(key));
  }
  if (error_out) error_out->clear();
}

}  // namespace

namespace {

void validate_plan(PlanTree const& tree)
{
  if (tree.nodes.empty() || tree.nodes.front().op != "input") {
    throw std::invalid_argument("decode: tree missing input root");
  }
  for (auto const& node : tree.nodes) {
    if (node.output_names.size() != node.output_paths.size()) {
      throw std::invalid_argument("decode: output names and paths differ");
    }
    for (auto const& edge : node.children)
      if (edge.child >= tree.nodes.size())
        throw std::invalid_argument("decode: child out of range");
    for (auto const& source : node.input_sources) {
      if (source.node >= tree.nodes.size())
        throw std::invalid_argument("decode: source out of range");
      auto const channels =
        source.node == 0 ? std::size_t{1} : tree.nodes[source.node].output_names.size();
      if (source.channel >= channels)
        throw std::invalid_argument("decode: source channel out of range");
    }
  }
  std::vector<std::uint8_t> visited(tree.nodes.size());
  std::function<void(NodeId)> visit = [&](NodeId nid) {
    if (visited[nid] == 1) throw std::invalid_argument("decode: cyclic plan");
    if (visited[nid] == 2) return;
    visited[nid] = 1;
    for (auto const& edge : tree.nodes[nid].children)
      visit(edge.child);
    visited[nid] = 2;
  };
  for (NodeId nid = 0; nid < tree.nodes.size(); ++nid)
    visit(nid);
}

}  // namespace

namespace {
// Defined alongside probe_column below; used by decompress_column's route
// checks and the dictionary fast path.
NodeId root_value_producer(PlanTree const& tree);
bool mask_consume_selection_root(PlanTree const& tree);

struct str_split_shape {
  compressed_representation const* chars_rep = nullptr;
  NodeId offsets_nid                         = 0;
};
std::optional<str_split_shape> locate_str_split_shape(PlanTree const& tree);

// The specialized dictionary char-emit (launch_decode_fused_tree_dict_gather):
// constant-width, null-free keys with identity-stored key channels; compressed
// or variable-width keys take the general route. The caller owns key-width
// measurement, keys_chars extraction, the analytic offsets (j * width), and
// the strings assembly — the kernel itself emits only the compacted chars.
// Returns false only when this shape is unsupported. Execution failures throw;
// nothing shared is mutated before a semantic decline.
bool try_dict_gather_fast_path(PlanTree const& tree,
                               decode_selection const& sel,
                               DecodeWalk& walk,
                               decode_frame& frame,
                               decode_column_slot output)
{
  auto const dict_nid = root_value_producer(tree);
  if (dict_nid >= tree.nodes.size()) return false;
  auto const& node = tree.nodes[dict_nid];
  std::optional<decode_column_slot> offsets;
  std::optional<decode_column_slot> chars;
  for (std::size_t i = 0; i < node.output_names.size(); ++i) {
    auto const& name = node.output_names[i];
    if (name != "keys_offsets" && name != "keys_chars") continue;
    if (std::any_of(node.children.begin(), node.children.end(), [&](auto const& edge) {
          return edge.channel == name;
        }))
      return false;
    auto it = node.channels.find(node.output_paths[i]);
    if (it == node.channels.end() || !it->second) return false;
    auto slot = frame.memo_column(value_id_key(ValueId{dict_nid, static_cast<ChannelId>(i)}));
    if (!slot) decode_standalone(*it->second, frame, slot);
    (name == "keys_offsets" ? offsets : chars) = slot;
  }
  if (!offsets || !chars || (*offsets)->type().id() != cudf::type_id::INT32 ||
      (*offsets)->size() < 2 || (*offsets)->null_count() != 0 ||
      (*chars)->type().id() != cudf::type_id::UINT8 || (*chars)->null_count() != 0)
    return false;
  auto host = frame.host_array<std::int32_t>((*offsets)->size());
  frame.read_bytes(host.data(), offsets->view().head<void>(), host.size_bytes());
  auto const width = host[1] - host[0];
  if (width <= 0) return false;
  for (std::size_t i = 2; i < host.size(); ++i)
    if (host[i] - host[i - 1] != width) return false;
  auto codes_nid = static_cast<NodeId>(tree.nodes.size());
  for (auto const& edge : node.children)
    if (edge.channel == "indices") codes_nid = edge.child;
  if (codes_nid >= tree.nodes.size()) return false;
  std::string error;
  decode_materialize_fn resolve = [&walk](NodeId nid) { return walk.materialize(nid); };
  auto region = bind_fused_region(tree, codes_nid, resolve, frame.stream(), frame.mr(), &error);
  if (!region) throw std::runtime_error(error);
  auto const survivors = static_cast<cudf::size_type>(sel.survivor_count);
  auto const& init     = frame.keep_scalar(
    std::make_unique<cudf::numeric_scalar<std::int32_t>>(0, true, frame.stream(), frame.mr()), 512);
  auto const& step = frame.keep_scalar(
    std::make_unique<cudf::numeric_scalar<std::int32_t>>(width, true, frame.stream(), frame.mr()),
    512);
  auto out_offsets = frame.make_column();
  out_offsets.adopt(cudf::sequence(survivors + 1, init, step, frame.stream(), frame.mr()));
  auto& out_chars = frame.allocate_output_buffer(static_cast<std::size_t>(survivors) * width);
  auto* char_data = out_chars.data();
  struct assembly_owners {
    std::vector<std::unique_ptr<cudf::column>> children{1};
    rmm::device_buffer chars;
  } owned;
  struct failed_assembly_drain {
    rmm::cuda_stream_view stream;
    int exceptions = std::uncaught_exceptions();
    ~failed_assembly_drain()
    {
      if (std::uncaught_exceptions() > exceptions) stream.synchronize_no_throw();
    }
  } drain{frame.stream()};
  owned.children[0] = frame.release(out_offsets);
  owned.chars       = std::move(out_chars);
  output.adopt(std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                              survivors,
                                              std::move(owned.chars),
                                              rmm::device_buffer{},
                                              0,
                                              std::move(owned.children)));
  if (survivors != 0) {
    launch_decode_fused_tree_dict_gather(*region->built.tree,
                                         region->labeled,
                                         region->dtype,
                                         region->num_rows,
                                         *sel.mask,
                                         row_enumeration{},
                                         chars->view().head<void>(),
                                         width,
                                         char_data,
                                         frame);
  }
  return true;
}

// Masked str_split decode for `str_split -> {offsets: bitpack, chars: raw}`
// plans (deep offsets chains and entropy-coded chars stay on the `full` route
// via the probe). Variable-width pattern:
//   phase 1 (launch_decode_fused_tree_str_split_meta): masked offsets-
//     subtree decode emitting per-survivor byte lengths + int64 source char
//     starts, compacted by rank;
//   exclusive-sum scan over the lengths column (one extra zeroed tail
//     slot makes the scan output the full cudf offsets layout directly —
//     n+1 entries, last = total survivor chars); ONE D2H of that total sizes
//     the count-first chars buffer — FULL char width is never materialized;
//   phase 2 (launch_masked_char_copy): survivor byte ranges copied from
//     the RAW parked chars buffer into the compacted chars at the scan's
//     destination offsets; the caller assembles via cudf::make_strings_column, with
//     the scan output doubling as the strings offsets column.
void decode_str_split_selected(PlanTree const& tree,
                               decode_selection const& sel,
                               DecodeWalk& walk,
                               decode_frame& frame,
                               decode_column_slot output)
{
  auto const shape = locate_str_split_shape(tree);
  if (!shape) throw std::invalid_argument("decode: unsupported selected str_split shape");
  auto const channels = shape->chars_rep->named_channels(frame.stream());
  if (channels.empty() || channels.front().view.type().id() != cudf::type_id::UINT8) {
    throw std::invalid_argument("decode: selected str_split requires raw UINT8 chars");
  }
  auto const chars = channels.front().view;
  std::string error;
  decode_materialize_fn resolve = [&walk](NodeId nid) { return walk.materialize(nid); };
  auto region =
    bind_fused_region(tree, shape->offsets_nid, resolve, frame.stream(), frame.mr(), &error);
  if (!region) throw std::runtime_error(error);
  if (region->num_rows != sel.mask->num_rows + 1) {
    throw std::invalid_argument("decode: selected string mask does not match the row domain");
  }
  auto const survivors = static_cast<cudf::size_type>(sel.survivor_count);
  auto lengths         = frame.make_column();
  lengths.adopt(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                              survivors + 1,
                                              cudf::mask_state::UNALLOCATED,
                                              frame.stream(),
                                              frame.mr()));
  auto& source_offsets =
    frame.allocate_buffer(static_cast<std::size_t>(survivors) * sizeof(std::int64_t));
  auto status = cudaMemsetAsync(lengths->mutable_view().data<std::int32_t>() + survivors,
                                0,
                                sizeof(std::int32_t),
                                frame.stream().value());
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  if (survivors > 0) {
    launch_decode_fused_tree_str_split_meta(*region->built.tree,
                                            region->labeled,
                                            region->dtype,
                                            sel.mask->num_rows,
                                            *sel.mask,
                                            row_enumeration{},
                                            static_cast<std::int64_t*>(source_offsets.data()),
                                            lengths->mutable_view().data<std::int32_t>(),
                                            frame);
  }
  auto offsets = frame.make_column();
  offsets.adopt(cudf::scan(lengths.view(),
                           *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::EXCLUSIVE,
                           cudf::null_policy::EXCLUDE,
                           frame.stream(),
                           frame.mr()));
  auto const total_chars = frame.read_scalar(offsets.view().data<std::int32_t>() + survivors);
  if (total_chars < 0) throw std::runtime_error("decode: selected string size overflow");
  auto const* offset_data = offsets.view().data<std::int32_t>();
  auto& output_chars      = frame.allocate_output_buffer(static_cast<std::size_t>(total_chars));
  auto* destination       = output_chars.data();
  output.adopt(cudf::make_strings_column(
    survivors, frame.release(offsets), std::move(output_chars), 0, rmm::device_buffer{}));
  if (survivors > 0 && total_chars > 0) {
    launch_masked_char_copy(chars.head<void>(),
                            static_cast<std::int64_t const*>(source_offsets.data()),
                            offset_data,
                            survivors,
                            destination,
                            frame);
  }
}

}  // namespace

validated_selection::validated_selection(PlanTree const& plan, decode_selection const& selection)
  : selection_(selection)
{
  validate_plan(plan);
  namespace sc = sirius::codegen;
  if (!selection.active() || (selection.mask != nullptr) == (selection.rows != nullptr)) {
    throw std::invalid_argument("decode: selection requires exactly one active row source");
  }
  if (selection.route != sc::decode_route::full &&
      selection.route != probe_column(plan).compact_route) {
    throw std::invalid_argument("decode: requested route does not match the plan");
  }
  if (selection.rows) {
    if (selection.route != sc::decode_route::bitpack_mask || selection.rows->num_rows < 0 ||
        selection.rows->num_rows > std::numeric_limits<cudf::size_type>::max() ||
        !selection.rows->valid() || selection.rows->num_survivors != selection.survivor_count) {
      throw std::invalid_argument("decode: invalid compacted chunk row set");
    }
  } else {
    if (selection.mask->num_rows < 0 ||
        selection.mask->num_rows > std::numeric_limits<cudf::size_type>::max() ||
        (selection.mask->num_rows > 0 && !selection.mask->words) ||
        selection.mask->survivor_count != selection.survivor_count ||
        selection.survivor_count > selection.mask->num_rows ||
        (selection.compacted() && !selection.mask->chunk_offsets)) {
      throw std::invalid_argument("decode: inconsistent selection count or offsets");
    }
    auto const root = root_value_producer(plan);
    bool const uses_indices =
      !selection.compacted() ||
      (selection.enumerate_by_index && selection.route == sc::decode_route::bitpack_mask &&
       selection.survivor_count > 0 && root < plan.nodes.size() &&
       plan.nodes[root].op == "bitpack" &&
       selection.survivor_indices.size() == selection.survivor_count);
    if (uses_indices &&
        (selection.survivor_indices.type().id() != cudf::type_id::INT32 ||
         selection.survivor_indices.size() != selection.survivor_count ||
         selection.survivor_indices.null_count() != 0 ||
         (selection.survivor_count > 0 && !selection.survivor_indices.data<std::int32_t>()))) {
      throw std::invalid_argument("decode: survivor indices violate type/shape/null policy");
    }
  }
}

void decode_request(column_decode_request const& request, decode_frame& frame)
{
  auto const* predicate = std::get_if<predicate_result>(&request.result);
  auto const* selection = request.selection ? &request.selection->get() : nullptr;
  auto const* pred      = predicate ? &predicate->predicate : nullptr;
  auto output           = frame.output();
  if (auto const* standalone =
        std::get_if<std::reference_wrapper<standalone_compressed_representation const>>(
          &request.source)) {
    if (predicate || selection)
      throw std::invalid_argument("standalone request cannot substitute or select");
    standalone->get().decompress(frame, output);
    return;
  }
  auto const& tree = std::get<std::reference_wrapper<PlanTree const>>(request.source).get();
  validate_plan(tree);
  namespace sc = sirius::codegen;
  if (predicate && !pred->active()) throw std::invalid_argument("decode: empty predicate request");
  if (predicate && selection && selection->route != sc::decode_route::dict_codes &&
      selection->route != sc::decode_route::full) {
    throw std::invalid_argument("decode: predicate selection requires dict_codes or full route");
  }
  std::string error;
  DecodeWalk walk{tree, frame, &error, pred, selection};
  if (selection && selection->route == sc::decode_route::str_split) {
    decode_str_split_selected(tree, *selection, walk, frame, output);
  } else {
    bool const emitted = selection && selection->route == sc::decode_route::dict_codes &&
                         !predicate &&
                         try_dict_gather_fast_path(tree, *selection, walk, frame, output);
    if (!emitted) {
      if (selection && !selection->compacted()) {
        auto full = frame.make_column();
        walk.run(full);
        if (full->size() != selection->mask->num_rows)
          throw std::invalid_argument("decode: full selection mask does not match the row domain");
        if (full->null_count() != 0)
          throw std::invalid_argument("decode: selected nullable values unsupported");
        auto gathered = cudf::gather(cudf::table_view{{full.view()}},
                                     selection->survivor_indices,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     frame.stream(),
                                     frame.mr());
        auto columns  = gathered->release();
        output.adopt(std::move(columns.front()));
      } else {
        walk.run(output);
      }
    }
  }
  if (selection && (output->size() != selection->survivor_count || output->null_count() != 0)) {
    throw std::runtime_error("decode: selected output has wrong size or unsupported nulls");
  }
  if (predicate) {
    if (output->type().id() != cudf::type_id::BOOL8)
      throw std::runtime_error("decode: predicate is not BOOL8");
    if (predicate->ballot) {
      auto const& destination = *predicate->ballot;
      if (output->size() != destination.num_rows || output->null_count() != 0 ||
          (destination.num_rows > 0 && !destination.words)) {
        throw std::invalid_argument("decode: predicate ballot shape/null policy mismatch");
      }
      sc::mask_from_bool8(output.view().data<std::uint8_t>(),
                          destination.num_rows,
                          destination.words,
                          frame.stream());
    }
  }
}

std::unique_ptr<cudf::column> decompress_column(PlanTree const& tree,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr,
                                                std::string* error_out,
                                                decode_predicate const* pred,
                                                decode_selection const* sel)
{
  nvtx3::scoped_range range{"simpatico::decompress_column"};
  column_decode_request request{std::cref(tree)};
  // Translate only documented host validation failures; submitted execution failures propagate.
  try {
    validate_plan(tree);
    if (pred && pred->active()) request.result = predicate_result{*pred, std::nullopt};
    if (sel && sel->active()) {
      request.selection.emplace(tree, *sel);
      if (pred && pred->active() && sel->route != sirius::codegen::decode_route::dict_codes &&
          sel->route != sirius::codegen::decode_route::full) {
        throw std::invalid_argument(
          "decode: predicate selection requires dict_codes or full route");
      }
    }
  } catch (std::invalid_argument const& e) {
    if (error_out) *error_out = e.what();
    return nullptr;
  }
  std::array const streams{stream};
  decode_session session{streams, mr};
  session.append(std::move(request));
  auto columns = session.finish();
  if (error_out) error_out->clear();
  return std::move(columns.front());
}

namespace {

bool dictionary_value_root(PlanTree const& tree)
{
  if (tree.nodes.empty() || tree.nodes[0].op != "input") { return false; }
  // The producer of the column's final value is whichever node consumes (0,0).
  // Only `dictionary` can answer a predicate off its compressed form.
  for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
    auto const& sources = tree.nodes[nid].input_sources;
    if (!sources.empty() && sources.front() == ValueId{0, 0}) {
      return tree.nodes[nid].op == "dictionary";
    }
  }
  return false;
}

// The node producing the column's final value: whichever consumes (0,0).
NodeId root_value_producer(PlanTree const& tree)
{
  if (tree.nodes.empty() || tree.nodes[0].op != "input") {
    return static_cast<NodeId>(tree.nodes.size());
  }
  for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
    auto const& sources = tree.nodes[nid].input_sources;
    if (!sources.empty() && sources.front() == ValueId{0, 0}) { return nid; }
  }
  return static_cast<NodeId>(tree.nodes.size());
}

// The column's final value is produced by a bitpack region, so the masked
// ballot / mask-walk render variants apply to the root region directly.
bool bitpack_selection_root(PlanTree const& tree)
{
  NodeId const nid = root_value_producer(tree);
  return nid < tree.nodes.size() && tree.nodes[nid].op == "bitpack";
}

// A delta root whose `differences` child is bitpack. The mask_consume
// launcher renders this shape too — the per-chunk prefix-sum reconstruction
// still runs, only the stores are masked/compacted.
bool delta_selection_root(PlanTree const& tree)
{
  NodeId const nid = root_value_producer(tree);
  if (nid >= tree.nodes.size() || tree.nodes[nid].op != "delta") { return false; }
  for (auto const& e : tree.nodes[nid].children) {
    if (e.channel == "differences") {
      return e.child < tree.nodes.size() && tree.nodes[e.child].op == "bitpack";
    }
  }
  return false;  // raw-passthrough differences: not a rendered mask_consume shape
}

// Any root region the mask_consume launcher renders.
bool mask_consume_selection_root(PlanTree const& tree)
{
  return bitpack_selection_root(tree) || delta_selection_root(tree);
}

std::optional<str_split_shape> locate_str_split_shape(PlanTree const& tree)
{
  NodeId const nid = root_value_producer(tree);
  if (nid >= tree.nodes.size() || tree.nodes[nid].op != "str_split") { return std::nullopt; }
  PlanNode const& node = tree.nodes[nid];
  for (auto const& name : node.output_names) {
    if (name == "null_mask") { return std::nullopt; }
  }
  str_split_shape shape;
  shape.offsets_nid = static_cast<NodeId>(tree.nodes.size());
  bool chars_ok     = false;
  bool offsets_ok   = false;
  for (std::size_t i = 0; i < node.output_names.size(); ++i) {
    std::string const& name = node.output_names[i];
    if (name == "chars") {
      bool has_edge = false;
      for (auto const& e : node.children) {
        if (e.channel == name) {
          has_edge = true;
          break;
        }
      }
      if (has_edge) { return std::nullopt; }
      auto it = node.channels.find(node.output_paths[i]);
      if (it != node.channels.end() && it->second && it->second->kind() == OpId::Identity &&
          it->second->decoded_type().id() == cudf::type_id::UINT8) {
        chars_ok        = true;
        shape.chars_rep = it->second.get();
      }
    } else if (name == "offsets") {
      for (auto const& e : node.children) {
        if (e.channel == name) {
          offsets_ok = e.child < tree.nodes.size() &&
                       (tree.nodes[e.child].op == "bitpack" || tree.nodes[e.child].op == "delta");
          shape.offsets_nid = e.child;
          break;
        }
      }
    }
  }
  if (!chars_ok || !offsets_ok) { return std::nullopt; }
  return shape;
}

bool str_split_selection_root(PlanTree const& tree)
{
  return locate_str_split_shape(tree).has_value();
}

bool dict_codes_selection_root(PlanTree const& tree)
{
  if (tree.nodes.empty() || tree.nodes[0].op != "input") { return false; }
  for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
    auto const& sources = tree.nodes[nid].input_sources;
    if (sources.empty() || !(sources.front() == ValueId{0, 0})) { continue; }
    PlanNode const& node = tree.nodes[nid];
    if (node.op != "dictionary") { return false; }
    // Nullable dictionary plans carry a trailing `null_mask` output channel;
    // iteration-1 selection has no null model — refuse (never corrupt).
    for (auto const& name : node.output_names) {
      if (name == "null_mask") { return false; }
    }
    // The mask consumer is the codes region: the `indices` channel must be
    // routed to a bitpack child so it can decode compacted.
    for (auto const& e : node.children) {
      if (e.channel == "indices") {
        return e.child < tree.nodes.size() && tree.nodes[e.child].op == "bitpack";
      }
    }
    return false;  // indices stored inline (identity) — no fused region to mask
  }
  return false;
}

}  // namespace

column_decode_caps probe_column(PlanTree const& tree)
{
  namespace sc = sirius::codegen;
  column_decode_caps caps;
  // The shapes are mutually exclusive: a plan has exactly one (0,0)-producer,
  // so the route is a classification, not a set of overlapping flags.
  if (bitpack_selection_root(tree)) {
    caps.compact_route = sc::decode_route::bitpack_mask;
  } else if (delta_selection_root(tree)) {
    caps.compact_route = sc::decode_route::delta_mask;
  } else if (dict_codes_selection_root(tree)) {
    caps.compact_route = sc::decode_route::dict_codes;
  } else if (str_split_selection_root(tree)) {
    caps.compact_route = sc::decode_route::str_split;
  }
  caps.can_answer_equality = dictionary_value_root(tree);
  return caps;
}

mask_source_status decode_request(mask_decode_request const& request, decode_frame& frame)
{
  validate_plan(request.plan);
  auto const destination = request.destination;
  if (destination.num_rows < 0 || (destination.num_rows > 0 && !destination.words)) {
    throw std::invalid_argument("decode: invalid mask destination");
  }
  std::string error;
  DecodeWalk walk{request.plan, frame, &error, nullptr, nullptr};
  if (auto const* range = std::get_if<sirius::codegen::range_predicate>(&request.source)) {
    if (!probe_column(request.plan).can_produce_mask()) {
      throw std::invalid_argument("decode: plan cannot produce a range ballot");
    }
    auto const root               = root_value_producer(request.plan);
    decode_materialize_fn resolve = [&walk](NodeId dependency) {
      return walk.materialize(dependency);
    };
    auto region =
      bind_fused_region(request.plan, root, resolve, frame.stream(), frame.mr(), &error);
    if (!region) throw std::runtime_error(error);
    if (destination.num_rows != region->num_rows)
      throw std::invalid_argument("decode: ballot row count mismatch");
    sirius::codegen::selection_mask mask{destination.words, destination.num_rows, -1, nullptr};
    if (destination.num_rows > 0) {
      launch_decode_fused_tree_mask_out(
        *region->built.tree, region->labeled, region->dtype, region->num_rows, *range, mask, frame);
    }
    return mask_source_status::ACCEPTED;
  }
  auto const& source = std::get<membership_source>(request.source);
  if (!source.probe) throw std::invalid_argument("decode: empty membership probe");
  auto keys = frame.make_column();
  walk.run(keys);
  if (keys->size() != destination.num_rows) {
    throw std::invalid_argument("decode: membership key row count mismatch");
  }
  auto key_view = keys.view();
  if (key_view.type() != source.stored_type && cudf::is_fixed_width(key_view.type()) &&
      cudf::is_fixed_width(source.stored_type) &&
      cudf::size_of(key_view.type()) == cudf::size_of(source.stored_type)) {
    key_view = cudf::column_view{source.stored_type,
                                 key_view.size(),
                                 key_view.head<void>(),
                                 key_view.null_mask(),
                                 key_view.null_count(),
                                 key_view.offset()};
  }
  auto flags = frame.make_column();
  flags.adopt(source.probe(key_view, frame.stream(), frame.mr()));
  if (!flags) {
    auto const bytes = static_cast<std::size_t>(
                         sirius::codegen::selection_mask::AllocWordsFor(destination.num_rows)) *
                       sizeof(std::uint32_t);
    auto status = cudaMemsetAsync(destination.words, 0xff, bytes, frame.stream().value());
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return mask_source_status::DECLINED;
  }
  if (flags->type().id() != cudf::type_id::BOOL8 || flags->size() != destination.num_rows ||
      flags->null_count() != 0) {
    throw std::invalid_argument("decode: membership flags violate shape/type/null policy");
  }
  sirius::codegen::mask_from_bool8(
    flags.view().data<std::uint8_t>(), destination.num_rows, destination.words, frame.stream());
  return mask_source_status::ACCEPTED;
}

bool decompress_column_selection_mask(PlanTree const& tree,
                                      sirius::codegen::range_predicate pred,
                                      std::uint32_t* mask_words,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr,
                                      std::string* error_out)
{
  compressed_representation const* rep = nullptr;
  try {
    validate_plan(tree);
    if (!mask_words || !probe_column(tree).can_produce_mask()) {
      throw std::invalid_argument("decode: plan or destination cannot serve range ballot");
    }
    auto const root = root_value_producer(tree);
    rep             = node_rep(root, tree);
    if (!rep) throw std::invalid_argument("decode: range ballot root has no representation");
  } catch (std::invalid_argument const& e) {
    if (error_out) *error_out = e.what();
    return false;
  }
  std::array const streams{stream};
  decode_session session{streams, mr};
  session.append(mask_decode_request{tree, pred, {mask_words, rep->num_rows}});
  session.finish();
  if (error_out) error_out->clear();
  return true;
}

std::unique_ptr<cudf::table> compact_scan_filter_output(
  std::vector<std::unique_ptr<cudf::column>>&& columns,
  sirius::codegen::scan_filter_result const& result,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out)
{
  if (!result.applied) {
    // Unfiltered decode: every column is full width already; just assemble.
    return std::make_unique<cudf::table>(std::move(columns));
  }
  if (result.routes.size() != columns.size()) {
    if (error_out) *error_out = "compact_scan_filter_output: routes/columns arity mismatch";
    return nullptr;
  }
  if (result.survivor_count < 0) {
    if (error_out) *error_out = "compact_scan_filter_output: survivor_count not counted";
    return nullptr;
  }
  auto const survivors = static_cast<cudf::size_type>(result.survivor_count);

  std::vector<std::size_t> full_positions;
  std::vector<cudf::column_view> full_views;
  for (std::size_t i = 0; i < columns.size(); ++i) {
    if (!columns[i]) {
      if (error_out) *error_out = "compact_scan_filter_output: null column";
      return nullptr;
    }
    if (columns[i]->null_count() != 0) {
      // Selection targets NOT NULL columns only; refuse rather than risk a
      // mask/null interaction the selection wave has not modeled.
      if (error_out) {
        *error_out =
          "compact_scan_filter_output: selection on a null-masked column is not "
          "supported";
      }
      return nullptr;
    }
    if (result.routes[i] != sirius::codegen::decode_route::full) {
      // Any compacted route: the decode already emitted survivor rows.
      if (columns[i]->size() != survivors) {
        if (error_out) {
          *error_out = "compact_scan_filter_output: compacted-route column is not survivor-sized";
        }
        return nullptr;
      }
      continue;
    }
    // A `full`-route column arrives in one of two shapes depending on the
    // wave-2 routing: already survivor-sized (the in-call decode_selection
    // gather compacted it per column) — pass through; or full width — collected for the single
    // batch-level gather below. When survivors == num_rows the two are
    // indistinguishable, and the ascending all-rows gather is the identity,
    // so passing through is correct either way.
    if (columns[i]->size() == survivors) { continue; }
    if (static_cast<std::int64_t>(columns[i]->size()) != result.num_rows) {
      if (error_out) {
        *error_out =
          "compact_scan_filter_output: full-route column is neither full width nor survivor-sized";
      }
      return nullptr;
    }
    full_positions.push_back(i);
    full_views.push_back(columns[i]->view());
  }

  if (!full_positions.empty()) {
    if (survivors == 0) {
      for (auto const pos : full_positions) {
        columns[pos] = cudf::empty_like(columns[pos]->view());
      }
    } else {
      if (result.row_indices.size() < static_cast<std::size_t>(survivors) * sizeof(std::int32_t)) {
        if (error_out) {
          *error_out = "compact_scan_filter_output: row_indices smaller than survivor_count";
        }
        return nullptr;
      }
      cudf::column_view const gather_map{
        cudf::data_type{cudf::type_id::INT32}, survivors, result.row_indices.data(), nullptr, 0};
      // ONE gather compacts every full-width column of the batch; the indices come
      // from the mask→indices kernel and are in-bounds by construction.
      std::unique_ptr<cudf::table> gathered;
      std::vector<std::unique_ptr<cudf::column>> gathered_columns;
      try {
        gathered         = cudf::gather(cudf::table_view{full_views},
                                gather_map,
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                stream,
                                mr);
        gathered_columns = gathered->release();
        // Complete this cross-column gather phase before replacing its source owners.
        stream.synchronize();
      } catch (...) {
        stream.synchronize_no_throw();
        throw;
      }
      for (std::size_t k = 0; k < full_positions.size(); ++k) {
        columns[full_positions[k]] = std::move(gathered_columns[k]);
      }
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace simpatico
