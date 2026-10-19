// SPDX-License-Identifier: Apache-2.0
#include "codegen/bridge/fused_tree_build.hpp"
#include "codegen/codegen_bridge.hpp"
#include "codegen/decode/masked_launch.hpp"
#include "codegen/plan/bitjoin_layout.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/util/cuda_check.hpp"
#include "codegen/util/nvtx.hpp"
#include "decode/decode_session.hpp"
#include "operators/constant_width_offsets.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <optional>
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

// Interprets one request's plan and owns its structural memo, keyed by ValueId. Only values keyed
// by codegen nodes are bound into fused launches, and they stay in the memo until the walk ends. A
// non-codegen node moves the values it rebuilds from, including its stored terminal channels, into
// its rebuilt representation, which releases them on the frame's stream once the decode from it has
// been queued; bind() refuses such values. run() moves the final value out.
class DecodeWalk {
 public:
  DecodeWalk(PlanTree const& tree,
             decode_frame& frame,
             decode_predicate const* pred,
             decode_selection const* sel);
  // Materialize `consumer` and return the decoded `value` it consumes, for a fused region to bind.
  cudf::column const* bind(NodeId consumer, ValueId value);
  std::unique_ptr<cudf::column> run();

 private:
  // Decode the values that `nid` encoded into the memo: every input of a bitjoin, otherwise its
  // single input.
  void materialize(NodeId nid);
  [[nodiscard]] cudf::column const* find_memo(std::uint64_t key) const;
  cudf::column const* store(std::uint64_t key, std::unique_ptr<cudf::column> value);
  // The memo entry of a terminal channel stored in the plan, decoded unless it is already there, so
  // a specialization that declines after inspecting it leaves it for the general route.
  std::unique_ptr<cudf::column>& terminal_entry(ValueId value,
                                                compressed_representation const& rep);
  std::unique_ptr<cudf::column> consume(ValueId value);
  std::unique_ptr<cudf::column> decode_leaf(compressed_representation const& rep, NodeId nid);
  void decode_bitjoin(NodeId nid);
  std::unique_ptr<cudf::column> materialize_fused_node(NodeId nid,
                                                       decode_selection const* node_sel);

  /// True when @p nid produces the column's final value and a predicate is
  /// pending — the one place a rep may answer the predicate instead of decoding.
  [[nodiscard]] bool predicate_applies_to(NodeId nid) const;

  /// True when @p nid is THE node whose fused decode consumes the pending
  /// selection: the (0,0)-producing bitpack region, or — for the
  /// dictionary-gather mode — the bitpack region producing the dictionary's `indices`
  /// value. Inner fused subtrees (entropy tails, dictionary keys_offsets, ...)
  /// hold metadata that is NOT row-aligned with the column and must decode
  /// full; the precomputed @c sel_target pins the exact consumer.
  [[nodiscard]] bool selection_applies_to(NodeId nid) const;

  PlanTree const& tree;
  decode_frame& frame;
  /// Borrowed; null when the caller wants the column itself.
  decode_predicate const* pred = nullptr;
  /// Borrowed decode-time row selection; null on the default path.
  decode_selection const* sel = nullptr;
  /// The one NodeId selection_applies_to accepts; tree.nodes.size() = none.
  NodeId sel_target;
  /// Set once a rep has answered `pred`, so run() knows not to compare again.
  bool predicate_resolved = false;
  // A present entry holding no column was consumed; requesting it again is an error.
  std::unordered_map<std::uint64_t, std::unique_ptr<cudf::column>> memo;
  std::unordered_map<std::uint64_t, std::size_t> remaining_consumers;
};

std::string value_label(ValueId v)
{
  return "(" + std::to_string(v.node) + "," + std::to_string(v.channel) + ")";
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
  compressed_representation const& rep, ::cuda::stream_ref stream)
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
//   the RawFused rep holds only ``offsets``; ``data`` is resolved through ``materialize`` for the
//   downstream PlanTree child node (the non-fused op that compressed the raw bytes) and the
//   parent's output value. The result is a view into the shared memo, which keeps it until the walk
//   ends.
//
// Element size for the data slot:
//   rle.runs  -> always sizeof(int32_t) (run counts are int32 regardless of
//               the column's original type).
//   all others -> element_size (original column element size).
void bind_raw_passthrough_buffers(std::int32_t node_id,
                                  NodeId parent_id,
                                  std::string const& parent_op,
                                  std::string const& parent_channel,
                                  PlanTree const& tree,
                                  std::size_t element_size,
                                  ::cuda::stream_ref stream,
                                  codegen::jit::LabeledBuffers& labeled,
                                  DecodeWalk& walk)
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
    throw std::runtime_error("codegen decode: RawFused passthrough rep missing at " + parent_op +
                             " node " + std::to_string(parent_id) + " (channel '" + channel_name +
                             "')");
  }

  auto by_name = channels_by_name(*rep, stream);

  for (auto const& slot : consumed_slots("RawFused")) {
    if (slot == "data" && by_name.find(slot) == by_name.end()) {
      // Entropy-tail: data was stripped from the rep at encode time and
      // compressed by a downstream non-fused op.  Find that child node and
      // resolve (decompress) its bytes back to the raw element array.
      if (parent_id >= tree.nodes.size()) {
        throw std::runtime_error("codegen decode: invalid parent_id for entropy-tail data resolve");
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
        throw std::runtime_error("codegen decode: no child edge '" + channel_name + "' on parent " +
                                 std::to_string(parent_id) + " for entropy-tail resolve");
      }
      auto const port = output_port(tree, parent_id, channel_name);
      if (!port) {
        throw std::runtime_error("codegen decode: node " + std::to_string(parent_id) +
                                 " does not output routed channel '" + channel_name + "'");
      }
      cudf::column_view dv = walk.bind(child_id, ValueId{parent_id, *port})->view();
      labeled[codegen::jit::buffer_key(node_id, "data")] = {
        dv.head<void>(), static_cast<std::size_t>(dv.size()), data_elem_size};
      continue;
    }
    auto bit = by_name.find(slot);
    if (bit == by_name.end()) {
      throw std::runtime_error("codegen decode: RawFused leaf missing slot '" + slot + "'");
    }
    labeled[codegen::jit::buffer_key(node_id, slot)] = {
      bit->second.head<void>(),
      static_cast<std::size_t>(bit->second.size()),
      elem_size_for_slot(slot, data_elem_size)};
  }
}

// Bind the device buffers for ONE real fused op node (bitpack / delta / rle)
// at preorder *node_id* into *labeled*. Buffers come from the node's
// rep ``named_channels()`` in per-op CONSUMED-slot order (``consumed_slots``);
// every rep is dense, so decode always uses the Compact gather.
//
// Entropy-tail-routed channels — a CONSUMED slot consumed downstream by another op (e.g. ``…packed
// -> snappy``, ``…packed -> bitcomp -> ans``, or a codegen tail ``…chunk_min -> zigzag``), detected
// as a child edge — are RESOLVED here via ``materialize`` (the downstream subtree), which returns a
// view of this node's output value in the shared memo, kept there until the walk ends. An identity
// NO-OP terminal (``…chunk_min -> identity``) leaves the bytes inside THIS rep and is bound
// directly.
void bind_real_node_buffers(std::int32_t node_id,
                            NodeId plan_node,
                            PlanTree const& tree,
                            std::size_t element_size,
                            ::cuda::stream_ref stream,
                            codegen::jit::LabeledBuffers& labeled,
                            DecodeWalk& walk)
{
  PlanNode const& node                  = tree.nodes[plan_node];
  compressed_representation const* repr = node_rep(plan_node, tree);

  std::string kind = codegen_kind_for_compressor(node.op);
  if (kind.empty()) {
    throw std::runtime_error("codegen decode: non-codegen op at node " + std::to_string(plan_node) +
                             " ('" + node.op + "')");
  }
  if (repr == nullptr) {
    throw std::runtime_error("codegen decode: missing rep at node " + std::to_string(plan_node));
  }

  auto by_name = channels_by_name(*repr, stream);

  std::unordered_map<std::string, NodeId> edge_by_channel;
  edge_by_channel.reserve(node.children.size());
  for (auto const& e : node.children)
    edge_by_channel.emplace(e.channel, e.child);

  auto const slots = consumed_slots(kind);
  if (slots.empty())
    throw std::runtime_error("codegen decode: no consumed-slot list for '" + kind + "'");
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
      // Tail-routed slot (a downstream codegen region OR non-codegen rep consumes it): materialize
      // the downstream's output — a view into the shared memo, which keeps it until the walk ends.
      // One path for both, no empty-map special case (e.g. …bitpack -> chunk_min -> zigzag resolves
      // the nested codegen tail through the same memo).
      auto const port = output_port(tree, plan_node, slot);
      if (!port) {
        throw std::runtime_error("codegen decode: node " + std::to_string(plan_node) +
                                 " does not output routed slot '" + slot + "'");
      }
      auto const v = walk.bind(eit->second, ValueId{plan_node, *port})->view();
      ptr          = v.head<void>();
      len          = static_cast<std::size_t>(v.size());
    } else {
      throw std::runtime_error("codegen decode: missing buffer for slot '" + slot + "' at node " +
                               std::to_string(plan_node));
    }
    labeled[codegen::jit::buffer_key(node_id, slot)] = {
      ptr, len, elem_size_for_slot(slot, element_size)};
  }
}

// Bind every node's device buffers from an already-built fused subtree in the
// builder's DFS-preorder (preorder index == rendered kernel node_id). The
// structural shape (op kinds, children, node-id order) is the builder's
// responsibility — shared with the encode bridge — so this binder only sources
// the per-node reps/buffers. Decode is Compact-only.
void bind_fused_subtree(BuiltFusedTree const& built,
                        PlanTree const& tree,
                        std::size_t element_size,
                        ::cuda::stream_ref stream,
                        codegen::jit::LabeledBuffers& labeled,
                        DecodeWalk& walk)
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
      bind_raw_passthrough_buffers(node_id,
                                   origin.parent_node,
                                   origin.parent_op,
                                   origin.parent_channel,
                                   tree,
                                   element_size,
                                   stream,
                                   labeled,
                                   walk);
    } else {
      bind_real_node_buffers(node_id, origin.plan_node, tree, element_size, stream, labeled, walk);
    }
  }
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

// Build one codegen-fused subtree, resolve its metadata, and bind its device buffers (keyed by
// DFS-preorder node_id) directly from the node-owned reps. Entropy-tail-routed channels are
// materialized into the caller's shared memo, which keeps them until the walk ends.
//
// Some intermediate fuse nodes store nothing and own no rep; their children
// own the reps. In that case, the first non-null rep in the fused preorder
// provides the decoded type and num_rows.
bound_fused_region bind_fused_region(PlanTree const& tree,
                                     NodeId root_nid,
                                     DecodeWalk& walk,
                                     ::cuda::stream_ref stream)
{
  auto built = build_fused_tree(tree, root_nid);
  if (!built) {
    throw std::runtime_error("codegen decode: no valid fusable region rooted at node " +
                             std::to_string(root_nid));
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
      throw std::runtime_error("codegen decompress: no rep at root node " +
                               std::to_string(root_nid));
    }
  }

  bound_fused_region region;
  region.root_type = root_repr->decoded_type();
  region.num_rows  = root_repr->num_rows;
  region.dtype     = codegen_dtype_str_for(region.root_type);
  if (region.dtype == nullptr) {
    throw std::runtime_error("codegen decompress: unsupported root dtype");
  }

  const std::size_t element_size = static_cast<std::size_t>(cudf::size_of(region.root_type));
  bind_fused_subtree(*built, tree, element_size, stream, region.labeled, walk);
  region.built = std::move(*built);
  return region;
}

bool uses_index_walk(PlanTree const& tree, decode_selection const& sel);

// Bind one region, allocate its output, and queue the decode launch.
std::unique_ptr<cudf::column> decode_fused_subtree_impl(PlanTree const& tree,
                                                        NodeId root_nid,
                                                        DecodeWalk& walk,
                                                        decode_frame& frame,
                                                        decode_selection const* sel)
{
  auto region         = bind_fused_region(tree, root_nid, walk, frame.stream());
  auto const num_rows = region.num_rows;
  bool const masked   = sel && sel->active();
  if (masked && sel->survivor_count > num_rows) {
    throw std::runtime_error("decode selection exceeds the column row count");
  }
  auto const out_rows = masked ? static_cast<cudf::size_type>(sel->survivor_count) : num_rows;
  auto output         = cudf::make_fixed_width_column(
    region.root_type, out_rows, cudf::mask_state::UNALLOCATED, frame.stream(), frame.mr());
  if (out_rows == 0) return output;
  auto const& built   = region.built;
  auto const& labeled = region.labeled;
  auto const* dtype   = region.dtype;
  auto* const out     = output->mutable_view().head<void>();
  if (!masked) {
    launch_decode_fused_tree(*built.tree, labeled, dtype, num_rows, out, frame);
    return output;
  }
  // validated_selection checked the selection's own consistency; only the column's row count is
  // new here.
  if (sel->rows) {
    if (sel->rows->num_rows != num_rows) {
      throw std::runtime_error("decode row set does not match the selected column");
    }
    sirius::codegen::selection_mask const hollow{nullptr, num_rows, sel->survivor_count, nullptr};
    launch_decode_fused_tree_compacted(*built.tree,
                                       labeled,
                                       dtype,
                                       num_rows,
                                       hollow,
                                       row_enumeration{nullptr, sel->rows},
                                       out,
                                       frame);
    return output;
  }
  bool const by_index = uses_index_walk(tree, *sel);
  launch_decode_fused_tree_compacted(
    *built.tree,
    labeled,
    dtype,
    num_rows,
    *sel->mask,
    row_enumeration{by_index ? sel->survivor_indices.data<std::int32_t>() : nullptr, nullptr},
    out,
    frame);
  return output;
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

cudf::column const* DecodeWalk::find_memo(std::uint64_t key) const
{
  auto const found = memo.find(key);
  return found == memo.end() ? nullptr : found->second.get();
}

cudf::column const* DecodeWalk::store(std::uint64_t key, std::unique_ptr<cudf::column> value)
{
  if (!value) throw std::runtime_error("decode: leaf returned no column");
  auto const [entry, inserted] = memo.try_emplace(key, std::move(value));
  if (!inserted) throw std::runtime_error("decode: memo value produced twice");
  return entry->second.get();
}

// Transfers a memoised value to the inverse of its producer. Shared values are copied until their
// last consumer; a sole/last consumer takes ownership. An empty entry is deliberately retained
// after a move so any accidental re-request is a deterministic runtime error rather than a silent
// re-decode.
std::unique_ptr<cudf::column> DecodeWalk::consume(ValueId value)
{
  auto const key   = value_id_key(value);
  auto const found = memo.find(key);
  if (found == memo.end() || !found->second) {
    throw std::runtime_error("decode: unresolved or consumed memo value " + value_label(value));
  }
  auto count_it = remaining_consumers.find(key);
  if (count_it == remaining_consumers.end() || count_it->second == 0) {
    throw std::runtime_error("decode: memo value has no remaining consumer " + value_label(value));
  }
  if (--count_it->second == 0) return std::move(found->second);
  return std::make_unique<cudf::column>(found->second->view(), frame.stream(), frame.mr());
}

std::unique_ptr<cudf::column>& DecodeWalk::terminal_entry(ValueId value,
                                                          compressed_representation const& rep)
{
  auto const key = value_id_key(value);
  if (auto const found = memo.find(key); found != memo.end()) {
    if (!found->second)
      throw std::runtime_error("decode: memo value already consumed " + value_label(value));
    return found->second;
  }
  auto decoded = decode_standalone(rep, frame);
  if (!decoded) throw std::runtime_error("decode: leaf returned no column");
  return memo.emplace(key, std::move(decoded)).first->second;
}

cudf::column const* DecodeWalk::bind(NodeId consumer, ValueId value)
{
  if (value.node >= tree.nodes.size() || !is_codegen_compressor(tree.nodes[value.node].op)) {
    throw std::logic_error("decode: fused region binds a value it does not produce " +
                           value_label(value));
  }
  // Look the value up first: materializing a bitjoin checks only its first input, which that
  // input's producer may already have taken over.
  auto const key = value_id_key(value);
  if (!memo.contains(key)) materialize(consumer);
  if (auto const* column = find_memo(key)) return column;
  if (memo.contains(key))
    throw std::runtime_error("decode: memo value already consumed " + value_label(value));
  throw std::runtime_error("decode: node " + std::to_string(consumer) +
                           " does not consume bound value " + value_label(value));
}

// A dictionary producing the column's final value may answer the predicate from its keys.
std::unique_ptr<cudf::column> DecodeWalk::decode_leaf(compressed_representation const& rep,
                                                      NodeId nid)
{
  if (predicate_applies_to(nid)) {
    if (auto const* dict = dynamic_cast<dictionary_compressed_representation const*>(&rep)) {
      if (auto hits = dict->decompress_predicate(*pred, frame)) {
        predicate_resolved = true;
        return hits;
      }
    }
  }
  return decode_standalone(rep, frame);
}

// Split a bitjoin node's packed leaf back into its input field values, keyed in
// `memo` by each input's structural ValueId. Fields sharing a source value are
// OR-ed into one column (a source may receive several bit ranges).
void DecodeWalk::decode_bitjoin(NodeId nid)
{
  auto const& node = tree.nodes[nid];
  auto const* rep  = bitjoin_packed_rep(node);
  if (!node.attrs.bitjoin || !rep)
    throw std::runtime_error("bitjoin decode: missing layout or packed rep");
  auto const packed      = decode_standalone(*rep, frame);
  auto const packed_view = packed->view();
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

  // Group the fields by the source value each targets (a source may collect
  // several bit ranges), keyed structurally by ValueId.
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
    auto output     = cudf::make_fixed_width_column(cudf::data_type{type},
                                                packed_view.size(),
                                                cudf::mask_state::UNALLOCATED,
                                                frame.stream(),
                                                frame.mr());
    throw_if_cuda_error(
      cudaMemsetAsync(output->mutable_view().head<void>(),
                      0,
                      static_cast<std::size_t>(packed_view.size()) * cudf::size_of(output->type()),
                      frame.stream().get()),
      "bitjoin decode: clear field");
    for (auto const& ref : refs) {
      launch_bitjoin_field(output->mutable_view(),
                           packed_view,
                           static_cast<int>(ref.dst_lo),
                           static_cast<int>(ref.src_lo),
                           ref.width,
                           frame.stream().get());
    }
    store(key, std::move(output));
  }
}

std::unique_ptr<cudf::column> DecodeWalk::materialize_fused_node(NodeId nid,
                                                                 decode_selection const* node_sel)
{
  return decode_fused_subtree_impl(tree, nid, *this, frame, node_sel);
}

void DecodeWalk::materialize(NodeId nid)
{
  auto const& node   = tree.nodes.at(nid);
  auto const primary = node.input_sources.empty() ? ValueId{nid, 0} : node.input_sources.front();
  auto const key     = value_id_key(primary);
  if (auto const found = memo.find(key); found != memo.end()) {
    if (!found->second)
      throw std::runtime_error("decode: memo value already consumed " + value_label(primary));
    return;
  }
  // A bitjoin decodes every input value it consumes, not only the first.
  if (node.attrs.bitjoin) {
    decode_bitjoin(nid);
    return;
  }
  if (is_codegen_compressor(node.op)) {
    store(key, materialize_fused_node(nid, selection_applies_to(nid) ? sel : nullptr));
    return;
  }
  if (node.rep) {
    store(key, decode_leaf(*node.rep, nid));
    return;
  }
  std::vector<std::string> names;
  std::vector<std::unique_ptr<cudf::column>> channels;
  names.reserve(node.output_names.size());
  channels.reserve(node.output_names.size());
  for (std::size_t i = 0; i < node.output_names.size(); ++i) {
    auto const& name = node.output_names[i];
    ValueId const value{nid, static_cast<ChannelId>(i)};
    auto child = std::find_if(node.children.begin(),
                              node.children.end(),
                              [&](PlanEdge const& edge) { return edge.channel == name; });
    if (child != node.children.end()) {
      if (!memo.contains(value_id_key(value))) materialize(child->child);
      names.push_back(name);
      channels.push_back(consume(value));
    } else {
      auto channel = node.channels.find(node.output_paths[i]);
      if (channel == node.channels.end()) continue;
      if (!channel->second) throw std::runtime_error("decode: missing terminal channel");
      names.push_back(name);
      channels.push_back(std::move(terminal_entry(value, *channel->second)));
    }
  }
  auto const rep = reconstruct_decode_representation(node, names, std::move(channels), frame);
  store(key, decode_leaf(*rep, nid));
}

DecodeWalk::DecodeWalk(PlanTree const& tree,
                       decode_frame& frame,
                       decode_predicate const* pred,
                       decode_selection const* sel)
  : tree(tree),
    frame(frame),
    pred(pred && pred->active() ? pred : nullptr),
    sel(sel && sel->active() ? sel : nullptr),
    sel_target(static_cast<NodeId>(tree.nodes.size()))
{
  for (auto const& node : tree.nodes)
    for (auto const& source : node.input_sources)
      ++remaining_consumers[value_id_key(source)];
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

std::unique_ptr<cudf::column> DecodeWalk::run()
{
  auto const key = value_id_key(ValueId{0, 0});
  for (auto const& edge : tree.nodes[0].children) {
    if (find_memo(key)) break;
    materialize(edge.child);
  }
  if (!find_memo(key)) {
    for (NodeId nid = 1; nid < tree.nodes.size() && !find_memo(key); ++nid) {
      for (auto const& source : tree.nodes[nid].input_sources) {
        if (source == ValueId{0, 0}) {
          materialize(nid);
          break;
        }
      }
    }
  }
  auto const* value = find_memo(key);
  if (!value) throw std::runtime_error("decode: input column was not reconstructed");
  std::unique_ptr<cudf::column> result;
  if (pred && !predicate_resolved) {
    auto const bool_type = cudf::data_type{cudf::type_id::BOOL8};
    for (auto const& text : pred->equals_any) {
      // Each needle waits for the stream while cuDF allocates its pinned validity buffer. Its
      // bytes upload from `text`, which the retained request copy owns.
      cudf::string_scalar const needle(text, true, frame.stream(), frame.mr());
      auto hit = cudf::binary_operation(
        value->view(), needle, cudf::binary_operator::EQUAL, bool_type, frame.stream(), frame.mr());
      if (!result) {
        result = std::move(hit);
      } else {
        result = cudf::binary_operation(result->view(),
                                        hit->view(),
                                        cudf::binary_operator::LOGICAL_OR,
                                        bool_type,
                                        frame.stream(),
                                        frame.mr());
      }
    }
    if (!result) throw std::runtime_error("decode: predicate has no values");
  } else {
    result = std::move(memo.at(key));
  }
  return result;
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

// A compacted bitpack decode enumerates survivors from the index list rather than the mask bits.
bool uses_index_walk(PlanTree const& tree, decode_selection const& sel)
{
  auto const root = root_value_producer(tree);
  return sel.enumerate_by_index && sel.route == sirius::codegen::decode_route::bitpack_mask &&
         sel.survivor_count > 0 && root < tree.nodes.size() && tree.nodes[root].op == "bitpack" &&
         sel.survivor_indices.size() == sel.survivor_count;
}

struct str_split_shape {
  compressed_representation const* chars_rep = nullptr;
  NodeId offsets_nid                         = 0;
};
std::optional<str_split_shape> locate_str_split_shape(PlanTree const& tree);

// The specialized dictionary char-emit (launch_decode_fused_tree_dict_gather): null-free keys whose
// uniform width the plan node publishes, whose keys_chars is an identity-stored channel bound by
// view without a copy, and whose identity-stored keys_offsets count confirms the chars extent that
// width implies. An unknown or variable width, compressed key channels, and a width that
// contradicts the extent take the general route, which measures or rejects. The caller owns the
// analytic offsets (j * width) and the strings assembly; the kernel itself emits only the compacted
// chars. Returns nullptr only when this shape is unsupported. Execution failures throw; nothing
// shared is mutated before a semantic decline.
std::unique_ptr<cudf::column> try_dict_gather_fast_path(PlanTree const& tree,
                                                        decode_selection const& sel,
                                                        DecodeWalk& walk,
                                                        decode_frame& frame)
{
  auto const dict_nid = root_value_producer(tree);
  if (dict_nid >= tree.nodes.size()) return nullptr;
  auto const& node = tree.nodes[dict_nid];
  auto const width = node.dictionary_key_width_hint;
  if (width <= 0) return nullptr;
  auto const* const chars_column   = terminal_identity_channel(node, "keys_chars");
  auto const* const offsets_column = terminal_identity_channel(node, "keys_offsets");
  if (!chars_column || !offsets_column) return nullptr;
  auto const chars = chars_column->view();
  if (chars.type().id() != cudf::type_id::UINT8 || chars.null_count() != 0) return nullptr;
  // The gather addresses the chars with the width, so the extent it implies is checked against host
  // metadata first, the property reconstruct_decode_representation enforces on the general route.
  auto const keys = static_cast<std::int64_t>(offsets_column->size()) - 1;
  if (keys < 0 || static_cast<std::int64_t>(chars.size()) != keys * width) return nullptr;
  // The analytic offsets are INT32; a larger output takes the general route.
  if (sel.survivor_count * width > std::numeric_limits<cudf::size_type>::max()) return nullptr;
  auto codes_nid = static_cast<NodeId>(tree.nodes.size());
  for (auto const& edge : node.children)
    if (edge.channel == "indices") codes_nid = edge.child;
  if (codes_nid >= tree.nodes.size()) return nullptr;
  auto const region    = bind_fused_region(tree, codes_nid, walk, frame.stream());
  auto const survivors = static_cast<cudf::size_type>(sel.survivor_count);
  std::vector<std::unique_ptr<cudf::column>> children;
  auto const key_width = static_cast<std::int32_t>(width);
  // Built on the device: each cuDF scalar would wait for the stream and upload from the host.
  children.push_back(make_constant_width_offsets(survivors, key_width, frame.stream(), frame.mr()));
  rmm::device_buffer out_chars(
    static_cast<std::size_t>(survivors) * static_cast<std::size_t>(key_width),
    frame.stream(),
    frame.mr());
  auto* const char_data = out_chars.data();
  auto output           = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::STRING},
    survivors,
    std::move(out_chars),
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, frame.stream(), frame.mr()),
    0,
    std::move(children));
  if (survivors != 0) {
    launch_decode_fused_tree_dict_gather(*region.built.tree,
                                         region.labeled,
                                         region.dtype,
                                         region.num_rows,
                                         *sel.mask,
                                         row_enumeration{},
                                         chars.data<std::uint8_t>(),
                                         key_width,
                                         char_data,
                                         frame);
  }
  return output;
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
std::unique_ptr<cudf::column> decode_str_split_selected(PlanTree const& tree,
                                                        decode_selection const& sel,
                                                        DecodeWalk& walk,
                                                        decode_frame& frame)
{
  auto const shape = locate_str_split_shape(tree);
  if (!shape) throw std::invalid_argument("decode: unsupported selected str_split shape");
  auto const channels = shape->chars_rep->named_channels(frame.stream());
  if (channels.empty() || channels.front().view.type().id() != cudf::type_id::UINT8) {
    throw std::invalid_argument("decode: selected str_split requires raw UINT8 chars");
  }
  auto const chars  = channels.front().view;
  auto const region = bind_fused_region(tree, shape->offsets_nid, walk, frame.stream());
  if (region.num_rows != sel.mask->num_rows + 1) {
    throw std::invalid_argument("decode: selected string mask does not match the row domain");
  }
  auto const survivors = static_cast<cudf::size_type>(sel.survivor_count);
  auto lengths         = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                               survivors + 1,
                                               cudf::mask_state::UNALLOCATED,
                                               frame.stream(),
                                               frame.mr());
  rmm::device_buffer source_offsets(
    static_cast<std::size_t>(survivors) * sizeof(std::int64_t), frame.stream(), frame.mr());
  throw_if_cuda_error(cudaMemsetAsync(lengths->mutable_view().data<std::int32_t>() + survivors,
                                      0,
                                      sizeof(std::int32_t),
                                      frame.stream().get()),
                      "selected str_split: clear length tail");
  if (survivors > 0) {
    launch_decode_fused_tree_str_split_meta(*region.built.tree,
                                            region.labeled,
                                            region.dtype,
                                            sel.mask->num_rows,
                                            *sel.mask,
                                            row_enumeration{},
                                            static_cast<std::int64_t*>(source_offsets.data()),
                                            lengths->mutable_view().data<std::int32_t>(),
                                            frame);
  }
  auto offsets           = cudf::scan(lengths->view(),
                            *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                            cudf::scan_type::EXCLUSIVE,
                            cudf::null_policy::EXCLUDE,
                            frame.stream(),
                            frame.mr());
  auto const total_chars = frame.read_scalar(offsets->view().data<std::int32_t>() + survivors);
  if (total_chars < 0) throw std::runtime_error("decode: selected string size overflow");
  auto const* offset_data = offsets->view().data<std::int32_t>();
  rmm::device_buffer output_chars(
    static_cast<std::size_t>(total_chars), frame.stream(), frame.mr());
  auto* destination = output_chars.data();
  // The output takes the offsets before the copy below reads them through `offset_data`.
  auto output = cudf::make_strings_column(
    survivors,
    std::move(offsets),
    std::move(output_chars),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, frame.stream(), frame.mr()));
  if (survivors > 0 && total_chars > 0) {
    launch_masked_char_copy(chars.head<void>(),
                            static_cast<std::int64_t const*>(source_offsets.data()),
                            offset_data,
                            survivors,
                            destination,
                            frame);
  }
  return output;
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
    bool const uses_indices = !selection.compacted() || uses_index_walk(plan, selection);
    if (uses_indices &&
        (selection.survivor_indices.type().id() != cudf::type_id::INT32 ||
         selection.survivor_indices.size() != selection.survivor_count ||
         selection.survivor_indices.null_count() != 0 ||
         (selection.survivor_count > 0 && !selection.survivor_indices.data<std::int32_t>()))) {
      throw std::invalid_argument("decode: survivor indices violate type/shape/null policy");
    }
  }
}

namespace {

// The host checks of a column request, before any device work.
void validate_request(column_decode_request const& request)
{
  auto const* predicate = std::get_if<predicate_result>(&request.result);
  auto const* tree      = std::get_if<std::reference_wrapper<PlanTree const>>(&request.source);
  if (!tree) {
    if (predicate || request.selection)
      throw std::invalid_argument("standalone request cannot substitute or select");
    return;
  }
  validate_plan(tree->get());
  if (predicate && !predicate->predicate.active())
    throw std::invalid_argument("decode: empty predicate request");
  if (predicate && request.selection &&
      request.selection->get().route != sirius::codegen::decode_route::dict_codes &&
      request.selection->get().route != sirius::codegen::decode_route::full) {
    throw std::invalid_argument("decode: predicate selection requires dict_codes or full route");
  }
}

// The host checks of a mask request, before any device work.
void validate_request(mask_decode_request const& request)
{
  validate_plan(request.plan);
  auto const destination = request.destination;
  if (destination.num_rows < 0 || (destination.num_rows > 0 && !destination.words)) {
    throw std::invalid_argument("decode: invalid mask destination");
  }
  if (std::holds_alternative<sirius::codegen::range_predicate>(request.source)) {
    if (!probe_column(request.plan).can_produce_mask()) {
      throw std::invalid_argument("decode: plan cannot produce a range ballot");
    }
  } else if (!std::get<membership_source>(request.source).probe) {
    throw std::invalid_argument("decode: empty membership probe");
  }
}

}  // namespace

std::unique_ptr<cudf::column> decode_request(column_decode_request const& request,
                                             decode_frame& frame)
{
  validate_request(request);
  auto const* predicate = std::get_if<predicate_result>(&request.result);
  auto const* selection = request.selection ? &request.selection->get() : nullptr;
  auto const* pred      = predicate ? &predicate->predicate : nullptr;
  if (auto const* standalone =
        std::get_if<std::reference_wrapper<standalone_compressed_representation const>>(
          &request.source)) {
    return standalone->get().decompress(frame);
  }
  auto const& tree = std::get<std::reference_wrapper<PlanTree const>>(request.source).get();
  namespace sc     = sirius::codegen;
  DecodeWalk walk{tree, frame, pred, selection};
  std::unique_ptr<cudf::column> output;
  if (selection && selection->route == sc::decode_route::str_split) {
    output = decode_str_split_selected(tree, *selection, walk, frame);
  } else {
    if (selection && selection->route == sc::decode_route::dict_codes && !predicate) {
      output = try_dict_gather_fast_path(tree, *selection, walk, frame);
    }
    if (!output && selection && !selection->compacted()) {
      auto const full = walk.run();
      if (full->size() != selection->mask->num_rows)
        throw std::invalid_argument("decode: full selection mask does not match the row domain");
      if (full->null_count() != 0)
        throw unsupported_nullable_selection("decode: selected nullable values unsupported");
      auto gathered = cudf::gather(cudf::table_view{{full->view()}},
                                   selection->survivor_indices,
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   frame.stream(),
                                   frame.mr());
      output        = std::move(gathered->release().front());
    } else if (!output) {
      output = walk.run();
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
      if (output->size() != destination.num_rows ||
          (destination.num_rows > 0 && !destination.words)) {
        throw std::invalid_argument("decode: predicate ballot shape mismatch");
      }
      if (output->null_count() != 0)
        throw unsupported_nullable_selection("decode: nullable predicate ballot unsupported");
      sc::mask_from_bool8(output->view().data<std::uint8_t>(),
                          destination.num_rows,
                          destination.words,
                          frame.stream());
    }
  }
  return output;
}

std::unique_ptr<cudf::column> decompress_column(PlanTree const& tree,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr,
                                                std::string* error_out,
                                                decode_predicate const* pred,
                                                decode_selection const* sel)
{
  nvtx_scoped_range range{"simpatico::decompress_column"};
  column_decode_request request{std::cref(tree)};
  // Translate only documented host validation failures; submitted execution failures propagate.
  try {
    if (pred && pred->active()) request.result = predicate_result{*pred, std::nullopt};
    if (sel && sel->active()) request.selection.emplace(tree, *sel);
    validate_request(request);
  } catch (std::invalid_argument const& e) {
    if (error_out) *error_out = e.what();
    return nullptr;
  }
  auto column = decode_one(request, stream, mr);
  if (error_out) error_out->clear();
  return column;
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
  validate_request(request);
  auto const destination = request.destination;
  DecodeWalk walk{request.plan, frame, nullptr, nullptr};
  if (auto const* range = std::get_if<sirius::codegen::range_predicate>(&request.source)) {
    auto const region =
      bind_fused_region(request.plan, root_value_producer(request.plan), walk, frame.stream());
    if (destination.num_rows != region.num_rows)
      throw std::invalid_argument("decode: ballot row count mismatch");
    sirius::codegen::selection_mask mask{destination.words, destination.num_rows, -1, nullptr};
    if (destination.num_rows > 0) {
      launch_decode_fused_tree_mask_out(
        *region.built.tree, region.labeled, region.dtype, region.num_rows, *range, mask, frame);
    }
    return mask_source_status::ACCEPTED;
  }
  auto const& source = std::get<membership_source>(request.source);
  auto const keys    = walk.run();
  if (keys->size() != destination.num_rows) {
    throw std::invalid_argument("decode: membership key row count mismatch");
  }
  auto key_view = keys->view();
  if (key_view.type() != source.stored_type &&
      cudf::is_bit_castable(key_view.type(), source.stored_type)) {
    key_view = cudf::column_view{source.stored_type,
                                 key_view.size(),
                                 key_view.head<void>(),
                                 key_view.null_mask(),
                                 key_view.null_count(),
                                 key_view.offset()};
  }
  auto const flags = source.probe(key_view, source.prior_mask_words, frame.stream(), frame.mr());
  if (!flags) {
    auto const bytes = static_cast<std::size_t>(
                         sirius::codegen::selection_mask::AllocWordsFor(destination.num_rows)) *
                       sizeof(std::uint32_t);
    throw_if_cuda_error(cudaMemsetAsync(destination.words, 0xff, bytes, frame.stream().get()),
                        "membership decline: fill mask");
    return mask_source_status::DECLINED;
  }
  if (flags->type().id() != cudf::type_id::BOOL8 || flags->size() != destination.num_rows ||
      flags->null_count() != 0) {
    throw std::invalid_argument("decode: membership flags violate shape/type/null policy");
  }
  sirius::codegen::mask_from_bool8(
    flags->view().data<std::uint8_t>(), destination.num_rows, destination.words, frame.stream());
  return mask_source_status::ACCEPTED;
}

bool decompress_column_selection_mask(PlanTree const& tree,
                                      sirius::codegen::range_predicate pred,
                                      std::uint32_t* mask_words,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr,
                                      std::string* error_out)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_column_selection_mask"};

  std::optional<mask_decode_request> request;
  try {
    validate_plan(tree);
    if (!mask_words || !probe_column(tree).can_produce_mask()) {
      throw std::invalid_argument("decode: plan or destination cannot serve range ballot");
    }
    auto const* rep = node_rep(root_value_producer(tree), tree);
    if (!rep) throw std::invalid_argument("decode: range ballot root has no representation");
    request.emplace(tree, pred, mask_destination{mask_words, rep->num_rows});
    validate_request(*request);
  } catch (std::invalid_argument const& e) {
    if (error_out) *error_out = e.what();
    return false;
  }
  (void)decode_one(*request, stream, mr);
  if (error_out) error_out->clear();
  return true;
}

}  // namespace simpatico
