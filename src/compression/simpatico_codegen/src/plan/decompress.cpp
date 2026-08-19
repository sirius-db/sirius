// SPDX-License-Identifier: Apache-2.0
#include "codegen/bridge/fused_tree_build.hpp"
#include "codegen/codegen_bridge.hpp"
#include "codegen/decode/masked_launch.hpp"
#include "codegen/plan/bitjoin_layout.hpp"
#include "codegen/plan/plan_interpreter.hpp"

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

// Decode memoises each reconstructed value by its structural identity — the
// (node, port) it is produced on, packed into a key. A value is computed once
// and shared by every consumer (and by the codegen tail binders). Consumer
// counts let reconstruction move a value into its sole/last consumer while
// preserving shared values. `kept` holds reconstructed reps alive for the
// walk's duration.
struct DecodeMemo {
  std::unordered_map<std::uint64_t, std::unique_ptr<cudf::column>> values;
  std::unordered_map<std::uint64_t, std::size_t> remaining_consumers;
  std::vector<std::unique_ptr<compressed_representation>> kept;
};

// Reverse plan traversal. CompressWalk emits values forward into consumers;
// DecodeWalk materializes producer inputs backward from stored representations.
// The memo and all temporary reconstructed reps live for the full walk.
class DecodeWalk {
 public:
  DecodeWalk(PlanTree const& tree,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref const& mr,
             std::string* error_out,
             decode_predicate const* pred,
             decode_selection const* sel);

  cudf::column const* materialize(NodeId nid);
  std::unique_ptr<cudf::column> run();

 private:
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
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  std::string* error_out;
  DecodeMemo memo;
  /// Borrowed; null when the caller wants the column itself.
  decode_predicate const* pred = nullptr;
  /// Borrowed decode-time row selection; null on the default path.
  decode_selection const* sel = nullptr;
  /// The one NodeId selection_applies_to accepts; tree.nodes.size() = none.
  NodeId sel_target;
  /// Set once a rep has answered `pred`, so run() knows not to compare again.
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
std::unique_ptr<cudf::column> consume_memo_value(ValueId value,
                                                 DecodeMemo& memo,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr,
                                                 std::string* error_out)
{
  auto const key = value_id_key(value);
  auto value_it  = memo.values.find(key);
  if (value_it == memo.values.end()) {
    if (error_out) *error_out = "decode: unresolved memo value " + value_label(value);
    return nullptr;
  }
  if (!value_it->second) {
    if (error_out) {
      *error_out = "decode: memo value " + value_label(value) + " was already consumed";
    }
    return nullptr;
  }

  auto count_it = memo.remaining_consumers.find(key);
  if (count_it == memo.remaining_consumers.end() || count_it->second == 0) {
    if (error_out) {
      *error_out = "decode: memo value " + value_label(value) + " has no remaining consumer";
    }
    return nullptr;
  }

  if (count_it->second == 1) {
    count_it->second = 0;
    return std::move(value_it->second);
  }

  auto copy = std::make_unique<cudf::column>(value_it->second->view(), stream, mr);
  --count_it->second;
  return copy;
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
// view into the shared memo that owns it through the (synchronous) decode
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
      // shared memo, which owns it through the synchronous launch. One path for
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
// which owns them through the (synchronous) launch that follows.
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

// High-level decode bridge implementation: bind the fused region and launch
// into a fresh output column. The launch is synchronous, so there is no
// cross-call binding state to cache; the JIT compile itself is cached
// process-wide in KernelCache.
//
// Returns nullptr + *error_out on failure.
std::unique_ptr<cudf::column> decode_fused_subtree_impl(PlanTree const& tree,
                                                        NodeId root_nid,
                                                        decode_materialize_fn const& materialize,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref const& mr,
                                                        std::string* error_out,
                                                        decode_selection const* sel = nullptr)
{
  auto region = bind_fused_region(tree, root_nid, materialize, stream, mr, error_out);
  if (!region) { return nullptr; }
  cudf::data_type const root_type = region->root_type;
  cudf::size_type const num_rows  = region->num_rows;
  const char* dtype               = region->dtype;
  auto& built                     = region->built;
  auto& labeled                   = region->labeled;

  // Compacted route: the combine + CNT wave already fixed the
  // survivor count, so the compacted output is allocated count-first instead
  // of full width (the whole point of the mask-consuming decode).
  const bool masked = sel != nullptr && sel->active();
  if (masked && (sel->survivor_count > static_cast<std::int64_t>(num_rows))) {
    if (error_out) {
      *error_out = "codegen decompress: selection survivor_count exceeds the column's row count";
    }
    return nullptr;
  }
  cudf::size_type const out_rows =
    masked ? static_cast<cudf::size_type>(sel->survivor_count) : num_rows;
  auto out_col =
    cudf::make_fixed_width_column(root_type, out_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  if (!out_col) {
    if (error_out) *error_out = "codegen decompress: output column alloc failed";
    return nullptr;
  }
  // A zero-survivor chunk has nothing to decode, and a 0-row column's data
  // pointer is null — launching against it would look like an allocation
  // failure to the kernel and get refused, forcing a needless full-width
  // fallback decode of a chunk that was already known to produce no rows.
  if (masked && sel->survivor_count == 0) { return out_col; }
  if (masked) {
    // Mask walk: decode over all num_rows input rows with the mask
    // words + chunk offsets as kernel arguments, writing only survivor rows
    // compacted into out_col (chunk_offsets[c] is chunk c's output base).
    if (sel->mask->chunk_offsets == nullptr) {
      if (error_out)
        *error_out =
          "codegen decompress: selection mask has no chunk_offsets (CNT wave did not run)";
      return nullptr;
    }
    if (sel->mask->survivor_count != sel->survivor_count) {
      if (error_out)
        *error_out = "codegen decompress: decode_selection and mask disagree on survivor_count";
      return nullptr;
    }
    // Runtime pick: the index walk decodes only the listed survivor rows by
    // random access into the packed bits — preferred by the orchestrator at low
    // selectivity. Bitpack LEAF roots only: a delta root keeps the mask walk
    // (the index walk rejects it at render), as does the dictionary codes
    // region by contract. Any anomaly — indices absent, or a count that
    // disagrees with the mask — silently keeps the mask walk, since the pick is
    // an optimization and the mask walk is always renderable here.
    bool const use_index_decode =
      sel->enumerate_by_index && sel->route == sirius::codegen::decode_route::bitpack_mask &&
      sel->survivor_count > 0 && root_nid < tree.nodes.size() &&
      tree.nodes[root_nid].op == "bitpack" &&
      static_cast<std::int64_t>(sel->survivor_indices.size()) == sel->survivor_count;
    bool const masked_ok = launch_decode_fused_tree_compacted(
      *built.tree,
      labeled,
      dtype,
      static_cast<std::int64_t>(num_rows),
      *sel->mask,
      use_index_decode ? sel->survivor_indices.data<std::int32_t>() : nullptr,
      out_col->mutable_view().head<void>(),
      stream);
    if (!masked_ok) {
      if (error_out) {
        *error_out = use_index_decode ? "codegen decompress: compacted (index walk) decode failed"
                                      : "codegen decompress: compacted (mask walk) decode failed";
      }
      return nullptr;
    }
    return out_col;
  }
  bool ok = launch_decode_fused_tree(*built.tree,
                                     labeled,
                                     dtype,
                                     static_cast<std::int64_t>(num_rows),
                                     out_col->mutable_view().head<void>(),
                                     stream);
  if (!ok) {
    if (error_out) *error_out = "codegen decompress: decode failed";
    return nullptr;
  }
  return out_col;
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
bool decode_bitjoin(NodeId nid,
                    PlanTree const& tree,
                    DecodeMemo& memo,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr,
                    std::string* error_out)
{
  PlanNode const& node = tree.nodes[nid];
  if (!node.attrs.bitjoin.has_value()) {
    if (error_out) *error_out = "decode_bitjoin: missing bitjoin attrs";
    return false;
  }

  compressed_representation const* repr = bitjoin_packed_rep(node);
  if (!repr) {
    if (error_out) *error_out = "bitjoin decode: missing packed rep at node " + std::to_string(nid);
    return false;
  }
  auto packed = decompress_standalone_representation(repr, stream, mr, error_out);
  if (!packed) {
    if (error_out) *error_out = "bitjoin decode: failed to decompress packed leaf";
    return false;
  }
  cudf::column_view packed_view = packed->view();
  int64_t n_elements            = static_cast<int64_t>(packed_view.size());

  std::vector<std::optional<bit_range>> input_ranges;
  input_ranges.reserve(node.attrs.bitjoin->inputs.size());
  for (auto const& ref : node.attrs.bitjoin->inputs)
    input_ranges.push_back(ref.range);

  bitjoin_layout layout;
  if (!resolve_bitjoin_layout(
        node.op, node.input_sources.size(), input_ranges, &layout, error_out)) {
    return false;
  }

  // Group the fields by the source value each targets (a source may collect
  // several bit ranges), keyed structurally by ValueId.
  struct field_ref {
    uint32_t width, src_lo, dst_lo;
  };
  std::unordered_map<std::uint64_t, std::vector<field_ref>> by_src;
  std::vector<ValueId> order;
  for (size_t fi = 0; fi < node.input_sources.size(); ++fi) {
    std::uint64_t const k = value_id_key(node.input_sources[fi]);
    auto& vec             = by_src[k];
    if (vec.empty()) order.push_back(node.input_sources[fi]);
    vec.push_back({layout.widths[fi], layout.src_los[fi], layout.dst_los[fi]});
  }

  for (auto const& src : order) {
    auto const& refs     = by_src.at(value_id_key(src));
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
                                                   stream,
                                                   mr);
    cudaMemsetAsync(
      field_col->mutable_view().head<void>(),
      0,
      static_cast<size_t>(n_elements) * static_cast<size_t>(cudf::size_of(field_col->type())),
      stream.value());
    for (auto const& r : refs) {
      launch_bitjoin_field(field_col->mutable_view(),
                           packed_view,
                           static_cast<int>(r.dst_lo),
                           static_cast<int>(r.src_lo),
                           r.width,
                           stream.value());
    }
    memo.values[value_id_key(src)] = std::move(field_col);
  }
  cudaStreamSynchronize(stream.value());
  return true;
}

std::unique_ptr<cudf::column> DecodeWalk::materialize_fused_node(NodeId nid,
                                                                 decode_selection const* node_sel)
{
  decode_materialize_fn resolve = [this](NodeId dependency) { return materialize(dependency); };
  return decode_fused_subtree_impl(tree, nid, resolve, stream, mr, error_out, node_sel);
}

// The single decode resolver. Reconstructs and memoises the value(s) node `nid`
// produces on decode — its input_source(s) — and returns the primary one, keyed
// by structural (node, port) identity so every consumer and the codegen tail
// binders share one memo without matching path strings.
cudf::column const* DecodeWalk::materialize(NodeId nid)
{
  PlanNode const& node  = tree.nodes[nid];
  ValueId const primary = node.input_sources.empty() ? ValueId{nid, 0} : node.input_sources.front();
  std::uint64_t const pk = value_id_key(primary);
  if (auto it = memo.values.find(pk); it != memo.values.end()) {
    if (!it->second) {
      if (error_out) {
        *error_out = "decode: memo value " + value_label(primary) + " was already consumed";
      }
      return nullptr;
    }
    return it->second.get();
  }

  // bitjoin recovers all its input values from one packed leaf.
  if (node.attrs.bitjoin.has_value()) {
    if (!decode_bitjoin(nid, tree, memo, stream, mr, error_out)) return nullptr;
    auto it = memo.values.find(pk);
    return it != memo.values.end() ? it->second.get() : nullptr;
  }

  std::unique_ptr<cudf::column> col;
  if (is_codegen_compressor(node.op)) {
    // Fused op (bitpack/delta/rle/for/zigzag): one JIT kernel inverts the whole
    // region; tail slots resolve via materialize. Only the region producing
    // the column's final value may consume the selection.
    col = materialize_fused_node(nid, selection_applies_to(nid) ? sel : nullptr);
  } else if (node.rep) {
    col = decompress_standalone_representation(node.rep.get(), stream, mr, error_out);
  } else {
    // Multi-output non-codegen op (alp/alp_rd/dictionary/bitextract): gather its
    // outputs in port order (reconstruct matches by name). An output routed to a
    // child edge is resolved and shared via the memo; a terminal output
    // decompresses in place.
    std::vector<std::string> names;
    std::vector<std::unique_ptr<cudf::column>> outputs;
    for (std::size_t i = 0; i < node.output_names.size(); ++i) {
      std::string const& name = node.output_names[i];
      auto child_it           = std::find_if(node.children.begin(),
                                   node.children.end(),
                                   [&](PlanEdge const& e) { return e.channel == name; });
      if (child_it != node.children.end()) {
        ValueId const output_value{nid, static_cast<ChannelId>(i)};
        // Recurse only when the value isn't yet in the memo
        if (!memo.values.count(value_id_key(output_value))) {
          if (!materialize(child_it->child)) return nullptr;
        }
        auto output = consume_memo_value(output_value, memo, stream, mr, error_out);
        if (!output) return nullptr;
        names.push_back(name);
        outputs.push_back(std::move(output));
        continue;
      }
      auto ch_it = node.channels.find(node.output_paths[i]);
      if (ch_it == node.channels.end()) continue;  // not produced here
      if (!ch_it->second) return nullptr;
      auto c = decompress_standalone_representation(ch_it->second.get(), stream, mr, error_out);
      if (!c) return nullptr;
      names.push_back(name);
      outputs.push_back(std::move(c));
    }
    std::string err;
    auto rep =
      reconstruct_representation(node.op, names, std::move(outputs), stream, mr, &err, node.meta);
    if (!rep) {
      if (error_out) *error_out = err;
      return nullptr;
    }
    // The dictionary rep is fully formed here — keys and indices both — and its
    // decompress() is the gather we want to skip. Answer the predicate straight
    // off the keys instead; a nullptr means the rep declined (unexpected index
    // type, null keys) and we fall through to the ordinary decode + compare.
    if (predicate_applies_to(nid)) {
      if (auto const* dict = dynamic_cast<dictionary_compressed_representation const*>(rep.get())) {
        col = dict->decompress_predicate(*pred, stream, mr);
        if (col) { predicate_resolved = true; }
      }
    }
    if (!col) { col = decompress_standalone_representation(rep.get(), stream, mr, error_out); }
    memo.kept.push_back(std::move(rep));
  }
  if (!col) return nullptr;
  auto [it, inserted] = memo.values.emplace(pk, std::move(col));
  (void)inserted;
  return it->second.get();
}

DecodeWalk::DecodeWalk(PlanTree const& tree,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref const& mr,
                       std::string* error_out,
                       decode_predicate const* pred,
                       decode_selection const* sel)
  : tree(tree),
    stream(stream),
    mr(mr),
    error_out(error_out),
    pred(pred != nullptr && pred->active() ? pred : nullptr),
    sel(sel != nullptr && sel->active() ? sel : nullptr),
    sel_target(static_cast<NodeId>(tree.nodes.size()))
{
  for (auto const& node : tree.nodes) {
    for (auto const& src : node.input_sources) {
      ++memo.remaining_consumers[value_id_key(src)];
    }
  }
  if (this->sel != nullptr && this->sel->compacted()) {
    // Pin the one node whose fused decode consumes the mask. The producer of
    // the column's final value is whichever node consumes (0,0).
    for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
      auto const& sources = tree.nodes[nid].input_sources;
      if (sources.empty() || !(sources.front() == ValueId{0, 0})) { continue; }
      if (this->sel->route != sirius::codegen::decode_route::dict_codes) {
        sel_target = nid;  // the root region itself.
      } else {
        // The dictionary route's consumer is the bitpack region producing the
        // dictionary's `indices` value, never the dictionary node itself.
        for (auto const& e : tree.nodes[nid].children) {
          if (e.channel == "indices") {
            sel_target = e.child;
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
  // The result is the input value (node 0, port 0), produced by whichever op(s)
  // consume it. Materialize those; fall back to a scan when the root carries no
  // structural edges (a rep-only tree).
  std::uint64_t const input_key = value_id_key(ValueId{0, 0});
  for (auto const& e : tree.nodes[0].children) {
    if (memo.values.count(input_key)) break;
    if (!materialize(e.child)) return nullptr;
  }
  if (!memo.values.count(input_key)) {
    for (NodeId nid = 1; nid < tree.nodes.size() && !memo.values.count(input_key); ++nid) {
      bool consumes_input = false;
      for (auto const& src : tree.nodes[nid].input_sources)
        if (src.node == 0 && src.channel == 0) consumes_input = true;
      if (consumes_input && !materialize(nid)) return nullptr;
    }
  }

  auto root_it = memo.values.find(input_key);
  if (root_it == memo.values.end() || !root_it->second) {
    if (error_out) *error_out = "decompression completed but 'input' column not reconstructed";
    return nullptr;
  }
  cudaStreamSynchronize(stream.value());
  auto result = std::move(root_it->second);

  // Generic fallback: a predicate was requested but no rep could answer it off
  // its compressed form, so the column was decoded in full. Compare here so the
  // "directive ⇒ BOOL8 result" contract holds for every plan shape — callers
  // rewrite their filter expression on the strength of it.
  if (pred != nullptr && !predicate_resolved && result) {
    auto const bool_t = cudf::data_type{cudf::type_id::BOOL8};
    std::unique_ptr<cudf::column> mask;
    for (auto const& value : pred->equals_any) {
      cudf::string_scalar const needle(value, true, stream);
      auto hit = cudf::binary_operation(
        result->view(), needle, cudf::binary_operator::EQUAL, bool_t, stream, mr);
      mask = mask
               ? cudf::binary_operation(
                   mask->view(), hit->view(), cudf::binary_operator::LOGICAL_OR, bool_t, stream, mr)
               : std::move(hit);
    }
    if (!mask) {
      if (error_out) *error_out = "decompress: predicate directive carried no values";
      return nullptr;
    }
    cudaStreamSynchronize(stream.value());
    result = std::move(mask);
  }

  if (error_out) error_out->clear();
  return result;
}

}  // namespace

std::unique_ptr<cudf::column> decode_fused_subtree(PlanTree const& tree,
                                                   NodeId start_node,
                                                   decode_materialize_fn const& materialize,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref const& mr,
                                                   std::string* error_out)
{
  return decode_fused_subtree_impl(tree, start_node, materialize, stream, mr, error_out);
}

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
// Returns nullptr when the fast path does not apply or the launch declines;
// nothing shared is mutated, so the caller falls through to the general route.
std::unique_ptr<cudf::column> try_dict_gather_fast_path(PlanTree const& tree,
                                                        decode_selection const& sel,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  NodeId const dict_nid = root_value_producer(tree);
  if (dict_nid >= tree.nodes.size()) { return nullptr; }
  PlanNode const& dict_node = tree.nodes[dict_nid];

  // Key channels must be terminal (identity-stored): a child edge on either
  // means compressed keys — general route.
  compressed_representation const* keys_offsets_rep = nullptr;
  compressed_representation const* keys_chars_rep   = nullptr;
  for (std::size_t i = 0; i < dict_node.output_names.size(); ++i) {
    std::string const& name = dict_node.output_names[i];
    if (name != "keys_offsets" && name != "keys_chars") { continue; }
    for (auto const& e : dict_node.children) {
      if (e.channel == name) { return nullptr; }
    }
    auto it = dict_node.channels.find(dict_node.output_paths[i]);
    if (it == dict_node.channels.end() || !it->second) { return nullptr; }
    (name == "keys_offsets" ? keys_offsets_rep : keys_chars_rep) = it->second.get();
  }
  if (keys_offsets_rep == nullptr || keys_chars_rep == nullptr) { return nullptr; }

  std::string err;
  auto keys_offsets = decompress_standalone_representation(keys_offsets_rep, stream, mr, &err);
  if (!keys_offsets || keys_offsets->type().id() != cudf::type_id::INT32 ||
      keys_offsets->size() < 2 || keys_offsets->null_count() != 0) {
    return nullptr;
  }
  auto keys_chars = decompress_standalone_representation(keys_chars_rep, stream, mr, &err);
  if (!keys_chars || keys_chars->type().id() != cudf::type_id::UINT8 ||
      keys_chars->null_count() != 0) {
    return nullptr;
  }

  // Key-width measurement: D2H the K+1 offsets (small — a dictionary's key
  // count) on the decode stream, then require a constant width. The
  // unfiltered path pays an equivalent lazy probe inside dictionary decompress
  // (constant_key_width), so this adds no new sync class; a plan-side cache is
  // a follow-up (needs a mutable PlanNode slot).
  auto const n_offsets = static_cast<std::size_t>(keys_offsets->size());
  std::vector<std::int32_t> host_offsets(n_offsets);
  if (cudaMemcpyAsync(host_offsets.data(),
                      keys_offsets->view().head<void>(),
                      n_offsets * sizeof(std::int32_t),
                      cudaMemcpyDeviceToHost,
                      stream.value()) != cudaSuccess ||
      cudaStreamSynchronize(stream.value()) != cudaSuccess) {
    return nullptr;
  }
  std::int32_t const width = host_offsets[1] - host_offsets[0];
  if (width < 1) { return nullptr; }
  for (std::size_t i = 2; i < n_offsets; ++i) {
    if (host_offsets[i] - host_offsets[i - 1] != width) { return nullptr; }
  }

  // Bind the codes (indices) region; entropy tails resolve through a local
  // walk whose memo owns them across the synchronous launch.
  NodeId codes_nid = static_cast<NodeId>(tree.nodes.size());
  for (auto const& e : dict_node.children) {
    if (e.channel == "indices") {
      codes_nid = e.child;
      break;
    }
  }
  if (codes_nid >= tree.nodes.size()) { return nullptr; }
  DecodeWalk tail_walk{tree, stream, mr, &err, nullptr, nullptr};
  decode_materialize_fn resolve = [&tail_walk](NodeId dependency) {
    return tail_walk.materialize(dependency);
  };
  auto region = bind_fused_region(tree, codes_nid, resolve, stream, mr, &err);
  if (!region) { return nullptr; }

  auto const survivors = sel.survivor_count;
  rmm::device_buffer out_chars(
    static_cast<std::size_t>(survivors) * static_cast<std::size_t>(width), stream, mr);
  if (!launch_decode_fused_tree_dict_gather(*region->built.tree,
                                            region->labeled,
                                            region->dtype,
                                            static_cast<std::int64_t>(region->num_rows),
                                            *sel.mask,
                                            keys_chars->view().head<void>(),
                                            width,
                                            out_chars.data(),
                                            stream)) {
    return nullptr;  // render/launch declined — general route still serves this batch
  }

  // Analytic offsets (j * width) + zero-copy chars wrap. The launcher synced;
  // sync again after the sequence so the caller may free/rebind immediately.
  cudf::numeric_scalar<std::int32_t> const init(0, true, stream);
  cudf::numeric_scalar<std::int32_t> const step(width, true, stream);
  auto offsets =
    cudf::sequence(static_cast<cudf::size_type>(survivors + 1), init, step, stream, mr);
  auto col = cudf::make_strings_column(static_cast<cudf::size_type>(survivors),
                                       std::move(offsets),
                                       std::move(out_chars),
                                       0,
                                       rmm::device_buffer(0, stream, mr));
  cudaStreamSynchronize(stream.value());
  return col;
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
std::unique_ptr<cudf::column> try_str_split_path(PlanTree const& tree,
                                                 decode_selection const& sel,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr,
                                                 std::string* error_out)
{
  auto const shape = locate_str_split_shape(tree);
  if (!shape) {
    if (error_out) *error_out = "decompress: str_split plan shape not supported for masked decode";
    return nullptr;
  }
  compressed_representation const* chars_rep = shape->chars_rep;
  NodeId const offsets_nid                   = shape->offsets_nid;
  auto const chars_channels                  = chars_rep->named_channels(stream);
  if (chars_channels.empty() || chars_channels.front().view.type().id() != cudf::type_id::UINT8) {
    if (error_out) *error_out = "decompress: str_split chars channel is not raw UINT8";
    return nullptr;
  }
  cudf::column_view const chars_view = chars_channels.front().view;

  // Bind the offsets subtree; entropy tails resolve through a local walk
  // whose memo owns them across the synchronous launches.
  std::string tail_err;
  DecodeWalk tail_walk{tree, stream, mr, &tail_err, nullptr, nullptr};
  decode_materialize_fn resolve = [&tail_walk](NodeId dependency) {
    return tail_walk.materialize(dependency);
  };
  auto region = bind_fused_region(tree, offsets_nid, resolve, stream, mr, error_out);
  if (!region) { return nullptr; }

  auto const survivors       = sel.survivor_count;
  auto const num_string_rows = sel.mask->num_rows;

  auto lengths = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                               static_cast<cudf::size_type>(survivors + 1),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
  rmm::device_buffer src_offsets(
    static_cast<std::size_t>(survivors) * sizeof(std::int64_t), stream, mr);
  if (survivors > 0) {
    // Phase 1 writes [0, survivors); the scan's tail slot must read as zero.
    cudaMemsetAsync(lengths->mutable_view().data<std::int32_t>() + survivors,
                    0,
                    sizeof(std::int32_t),
                    stream.value());
    if (!launch_decode_fused_tree_str_split_meta(*region->built.tree,
                                                 region->labeled,
                                                 region->dtype,
                                                 num_string_rows,
                                                 *sel.mask,
                                                 static_cast<std::int64_t*>(src_offsets.data()),
                                                 lengths->mutable_view().data<std::int32_t>(),
                                                 stream)) {
      if (error_out)
        *error_out = "decompress: masked str_split phase 1 (str_split_meta) launch failed";
      return nullptr;
    }
  } else {
    cudaMemsetAsync(lengths->mutable_view().head<void>(), 0, sizeof(std::int32_t), stream.value());
  }

  // Exclusive-sum scan -> destination offsets; doubles as the strings
  // offsets column (cudf layout: survivors+1 entries, last = total chars).
  auto out_offsets         = cudf::scan(lengths->view(),
                                *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                                cudf::scan_type::EXCLUSIVE,
                                cudf::null_policy::EXCLUDE,
                                stream,
                                mr);
  std::int32_t total_chars = 0;
  if (cudaMemcpyAsync(&total_chars,
                      out_offsets->view().data<std::int32_t>() + survivors,
                      sizeof(std::int32_t),
                      cudaMemcpyDeviceToHost,
                      stream.value()) != cudaSuccess ||
      cudaStreamSynchronize(stream.value()) != cudaSuccess) {
    if (error_out) *error_out = "decompress: str_split offsets readback failed";
    return nullptr;
  }

  rmm::device_buffer out_chars(static_cast<std::size_t>(total_chars), stream, mr);
  if (survivors > 0 && total_chars > 0) {
    if (!launch_masked_char_copy(chars_view.head<void>(),
                                 static_cast<std::int64_t const*>(src_offsets.data()),
                                 out_offsets->view().data<std::int32_t>(),
                                 survivors,
                                 out_chars.data(),
                                 stream)) {
      if (error_out) *error_out = "decompress: masked str_split phase 2 (char copy) launch failed";
      return nullptr;
    }
  }
  auto col = cudf::make_strings_column(static_cast<cudf::size_type>(survivors),
                                       std::move(out_offsets),
                                       std::move(out_chars),
                                       0,
                                       rmm::device_buffer(0, stream, mr));
  // The phase-1 lengths column and src_offsets free on return; the launches
  // synced above, and make_strings_column launched nothing — sync once more
  // for the same caller-may-free discipline as the other compacted routes.
  cudaStreamSynchronize(stream.value());
  return col;
}
}  // namespace

std::unique_ptr<cudf::column> decompress_column(PlanTree const& tree,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr,
                                                std::string* error_out,
                                                decode_predicate const* pred,
                                                decode_selection const* sel)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_column"};

  if (tree.nodes.empty() || tree.nodes[0].op != "input") {
    if (error_out) *error_out = "decompress: tree missing input root";
    return nullptr;
  }

  namespace sc = sirius::codegen;

  bool const selecting    = sel != nullptr && sel->active();
  bool const substituting = pred != nullptr && pred->active();
  // The requested route must be the one this plan actually supports: a
  // mismatch would silently decode full width where the caller sized the
  // output from the survivor count.
  if (selecting && sel->route != probe_column(tree).compact_route) {
    if (error_out) {
      *error_out = "decompress: requested decode route does not match the plan's shape";
    }
    return nullptr;
  }
  if (selecting && substituting && sel->route != sc::decode_route::dict_codes &&
      sel->route != sc::decode_route::full) {
    // A predicate answer at a mask-source slot composes only where the
    // predicate has a compacted meaning: the dictionary route answers it over
    // the compacted codes, and `full` produces full-width BOOL8 that the
    // survivor gather compacts. A write-skipping route never materializes the
    // value to compare, so asking for both is a scheduling bug.
    if (error_out) {
      *error_out = "decompress: decode_predicate composes only with the dict_codes or full route";
    }
    return nullptr;
  }
  if (selecting && sel->route == sc::decode_route::str_split) {
    // str_split has NO generic fallback: compacted offsets cannot feed the
    // ordinary str_split reconstruct, so a declined dedicated route must error
    // (the orchestrator re-runs the batch unfiltered) rather than fall through
    // to a full-width walk the compacted() belt below would reject anyway.
    auto col = try_str_split_path(tree, *sel, stream, mr, error_out);
    if (!col) {
      if (error_out && error_out->empty()) {
        *error_out = "decompress: masked str_split decode declined";
      }
      return nullptr;
    }
    if (static_cast<std::int64_t>(col->size()) != sel->survivor_count || col->null_count() != 0) {
      if (error_out) {
        *error_out =
          "decompress: masked str_split decode returned a non-survivor-sized or null-masked column";
      }
      return nullptr;
    }
    return col;
  }

  std::unique_ptr<cudf::column> col;
  if (selecting && sel->route == sc::decode_route::dict_codes && !substituting) {
    // Constant-width fast path: the dictionary char-emit kernel replaces the
    // compacted-codes intermediate + cudf key gather with one launch. A
    // nullptr (not applicable / launch declined) falls through to the general
    // dict route — nothing shared was mutated. Skipped under dual delivery:
    // the caller wants a BOOL8 answer, not strings.
    col = try_dict_gather_fast_path(tree, *sel, stream, mr);
  }
  if (!col) {
    // A predicate answer composes here without special cases: the dictionary
    // route reconstructs its rep over the compacted codes, so the existing
    // answer (decompress_predicate — or run()'s generic decode-and-compare
    // fallback) is already SURVIVOR-SIZED BOOL8; the `full` route produces
    // full-width BOOL8 and the gather below compacts it.
    DecodeWalk walk{tree, stream, mr, error_out, pred, sel};
    col = walk.run();
  }

  if (col && selecting && substituting && col->type().id() != cudf::type_id::BOOL8) {
    // Belt: a predicate directive promises "BOOL8 result" to the
    // filter-expression rewrite; anything else must fail loudly here rather
    // than let a strings column meet a bare boolean reference downstream.
    if (error_out) {
      *error_out = "decompress: predicate directive under selection returned a non-BOOL8 column";
    }
    return nullptr;
  }

  if (col && selecting && sel->compacted()) {
    // Belt and braces for the same hazard: whatever comes back from a
    // compacted-route request must already be survivor-sized (directly from
    // the mask or index walk; the dictionary route via the key gather over the
    // compacted codes) and — per the null policy — carry no null mask.
    if (static_cast<std::int64_t>(col->size()) != sel->survivor_count) {
      if (error_out) {
        *error_out = "decompress: compacted decode returned a non-survivor-sized column";
      }
      return nullptr;
    }
    if (col->null_count() != 0) {
      if (error_out) {
        *error_out = "decompress: selection on a null-masked column is not supported";
      }
      return nullptr;
    }
  }

  // The `full` route: the walk decoded the column full width exactly as today;
  // compact it to the batch's survivor rows with one gather over the shared
  // mask→indices buffer. (Every other route came back survivor-sized from the
  // masked decode and skips this.)
  if (col && selecting && !sel->compacted()) {
    if (col->null_count() != 0) {
      // Selection targets NOT NULL columns only; refuse rather than risk a
      // mask/null interaction the selection wave has not modeled.
      if (error_out) {
        *error_out = "decompress: selection on a null-masked column is not supported";
      }
      return nullptr;
    }
    if (static_cast<std::int64_t>(sel->survivor_indices.size()) != sel->survivor_count) {
      if (error_out) {
        *error_out = "decompress: survivor_indices size does not match survivor_count";
      }
      return nullptr;
    }
    // Indices come from the mask→indices kernel and are in-bounds by
    // construction; skip the bounds pass.
    auto gathered = cudf::gather(cudf::table_view{{col->view()}},
                                 sel->survivor_indices,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
    col           = std::move(gathered->release().front());
    // Same discipline as run(): the caller may free inputs / rebind buffers as
    // soon as we return, so the gather must have completed.
    cudaStreamSynchronize(stream.value());
  }
  return col;
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
      if (it != node.channels.end() && it->second &&
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

bool decompress_column_selection_mask(PlanTree const& tree,
                                      sirius::codegen::range_predicate pred,
                                      std::uint32_t* mask_words,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr,
                                      std::string* error_out)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_column_selection_mask"};

  if (tree.nodes.empty() || tree.nodes[0].op != "input") {
    if (error_out) *error_out = "decompress: tree missing input root";
    return false;
  }
  if (mask_words == nullptr) {
    if (error_out) *error_out = "decompress: selection mask words buffer is null";
    return false;
  }
  // Locate the root-value producer; only a bitpack-rooted region renders the ballot.
  NodeId root = tree.nodes.size();
  for (NodeId nid = 1; nid < tree.nodes.size(); ++nid) {
    auto const& sources = tree.nodes[nid].input_sources;
    if (!sources.empty() && sources.front() == ValueId{0, 0}) {
      root = nid;
      break;
    }
  }
  if (root >= tree.nodes.size() || tree.nodes[root].op != "bitpack") {
    if (error_out) {
      *error_out = "decompress: plan is not bitpack-rooted; cannot decode a selection mask";
    }
    return false;
  }

  // The walk only resolves entropy-tail channels here (no column decode); its
  // memo owns them through the synchronous mask launch below.
  DecodeWalk walk{tree, stream, mr, error_out, nullptr, nullptr};
  decode_materialize_fn resolve = [&walk](NodeId dependency) {
    return walk.materialize(dependency);
  };
  auto region = bind_fused_region(tree, root, resolve, stream, mr, error_out);
  if (!region) { return false; }

  sirius::codegen::selection_mask mask{};
  mask.words    = mask_words;
  mask.num_rows = static_cast<std::int64_t>(region->num_rows);
  bool const ok = launch_decode_fused_tree_mask_out(*region->built.tree,
                                                    region->labeled,
                                                    region->dtype,
                                                    static_cast<std::int64_t>(region->num_rows),
                                                    pred,
                                                    mask,
                                                    stream);
  if (!ok && error_out) { *error_out = "decompress: masked (mask-out) decode failed"; }
  return ok;
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
      auto gathered         = cudf::gather(cudf::table_view{full_views},
                                   gather_map,
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   stream,
                                   mr);
      auto gathered_columns = gathered->release();
      // The full-width sources are replaced (freed) right below; their
      // stream-ordered deallocation is only safe once the gather has read them.
      cudaStreamSynchronize(stream.value());
      for (std::size_t k = 0; k < full_positions.size(); ++k) {
        columns[full_positions[k]] = std::move(gathered_columns[k]);
      }
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace simpatico
