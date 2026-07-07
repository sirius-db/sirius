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

// The C++-native JIT codegen decode entry point (``simpatico::decode_fused_subtree``,
// defined in codegen_runtime.cpp) is declared in codegen_bridge.hpp above. One
// call does the whole pipeline: build tree → compile → bind buffers → launch →
// sync.

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

// Decode memoises each reconstructed value by its structural identity — the
// (node, port) it is produced on, packed into a key. A value is computed once
// and shared by every consumer (and by the codegen tail binders). `kept` holds
// the reconstructed reps alive for the walk's duration.
struct DecodeMemo {
  std::unordered_map<std::uint64_t, std::unique_ptr<cudf::column>> values;
  std::vector<std::unique_ptr<compressed_representation>> kept;
};

std::uint64_t value_key(ValueId v)
{
  return (static_cast<std::uint64_t>(v.node) << 16) | v.channel;
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

// Reconstructs node `nid`'s produced value into the shared memo and returns a
// non-owning view the memo keeps alive. One resolver serves the top-level walk
// and fused-region tail binding, so both share a single memo. Defined below.
cudf::column const* materialize(NodeId nid,
                                PlanTree const& tree,
                                DecodeMemo& memo,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr,
                                std::string* error_out);

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
                                  DecodeMemo& decompressed,
                                  std::string* error_out)
{
  // Use the explicit channel name. Fall back to the old heuristic for
  // forward-compatibility with any preorder that pre-dates parent_channel.
  std::string const channel_name =
    parent_channel.empty() ? ((parent_op == "for") ? "deltas" : "values") : parent_channel;

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

  std::vector<compressible_output> nb = rep->named_channels(stream);
  std::unordered_map<std::string, cudf::column_view> by_name;
  for (auto const& o : nb)
    by_name.emplace(o.name, o.view);

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
      cudf::column const* resolved =
        materialize(child_id, tree, decompressed, stream, mr, error_out);
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
                            DecodeMemo& decompressed,
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

  std::vector<compressible_output> nb = repr->named_channels(stream);
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
      // Slot lives directly in this node's rep — bind it straight.
      ptr = bit->second.head<void>();
      len = static_cast<std::size_t>(bit->second.size());
    } else if (has_edge) {
      // Tail-routed slot (a downstream codegen region OR non-codegen rep
      // consumes it): materialize the downstream's output — a view into the
      // shared memo, which owns it through the synchronous launch. One path for
      // both, no empty-map special case (e.g. …bitpack -> chunk_min -> zigzag
      // resolves the nested codegen tail through the same memo).
      cudf::column const* col = materialize(eit->second, tree, decompressed, stream, mr, error_out);
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

// Build the fused subtree rooted at *root_nid* via the shared structural
// builder (``build_fused_tree``), then bind every node's device buffers in the
// builder's DFS-preorder (preorder index == rendered kernel node_id). The
// structural shape (op kinds, children, node-id order) is the builder's
// responsibility — shared with the encode bridge — so this binder only sources
// the per-node reps/buffers. Decode uses ``fixed_stride=false`` (Compact).
//
// Returns the built tree, or nullptr + *error_out on failure.
std::shared_ptr<codegen::jit::FusedTree> bind_fused_subtree(NodeId root_nid,
                                                            PlanTree const& tree,
                                                            std::size_t element_size,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr,
                                                            codegen::jit::LabeledBuffers& labeled,
                                                            DecodeMemo& decompressed,
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
    // Transformer-mode ZigZag stores nothing (it rewrites the lane value
    // inline and recurses); the decode renderer emits no params for it, so
    // there is no buffer to bind. Skip it — the child binds its own buffers.
    if (origin.node != nullptr && origin.node->op == codegen::OpKind::Zigzag &&
        !origin.node->children.empty()) {
      continue;
    }
    if (origin.is_raw_passthrough) {
      if (!bind_raw_passthrough_buffers(node_id,
                                        origin.parent_rle,
                                        origin.parent_op,
                                        origin.parent_channel,
                                        tree,
                                        element_size,
                                        stream,
                                        mr,
                                        labeled,
                                        decompressed,
                                        error_out)) {
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
                                  decompressed,
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
// Some intermediate fuse nodes store nothing and own no rep;
// their children own the reps. In that case, the first non-null rep in the
// fused preorder provides the decoded type and num_rows.
//
// Returns nullptr + *error_out on failure.
std::unique_ptr<cudf::column> dispatch_codegen_subtree(NodeId root_nid,
                                                       PlanTree const& tree,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr,
                                                       DecodeMemo& decompressed,
                                                       std::string* error_out)
{
  compressed_representation const* root_repr = node_rep(root_nid, tree);
  if (root_repr == nullptr) {
    // Transformer-mode root (e.g. ZigZag with a codegen child) stores no rep;
    // find the first non-null rep in the fused preorder for metadata.
    auto maybe_built = build_fused_tree(tree, root_nid, /*fixed_stride=*/false);
    if (maybe_built) {
      for (auto const& origin : maybe_built->preorder) {
        if (!origin.is_raw_passthrough && origin.plan_node < tree.nodes.size()) {
          root_repr = node_rep(origin.plan_node, tree);
          if (root_repr) break;
        }
      }
    }
    if (root_repr == nullptr) {
      if (error_out)
        *error_out = "codegen decompress: no rep at root node " + std::to_string(root_nid);
      return nullptr;
    }
  }
  cudf::data_type root_type = root_repr->decoded_type();
  cudf::size_type num_rows  = root_repr->num_rows;
  const char* dtype         = codegen_dtype_str_for(root_type);
  if (dtype == nullptr) {
    if (error_out) *error_out = "codegen decompress: unsupported root dtype";
    return nullptr;
  }

  // Build the FusedTree + bind the real device buffers (keyed by DFS-preorder
  // node_id) directly from the node-owned reps.  Entropy-tail-routed channels
  // are materialized into the shared `decompressed` memo, which owns them
  // through the (synchronous) decode launch below.
  codegen::jit::LabeledBuffers labeled;
  const std::size_t element_size = static_cast<std::size_t>(cudf::size_of(root_type));
  std::string bind_err;
  auto fused =
    bind_fused_subtree(root_nid, tree, element_size, stream, mr, labeled, decompressed, &bind_err);
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
  auto packed = repr->decompress(stream, mr);
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
  if (!resolve_bitjoin_layout(node.op, node.input_paths, input_ranges, &layout, error_out)) {
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
    std::uint64_t const k = value_key(node.input_sources[fi]);
    auto& vec             = by_src[k];
    if (vec.empty()) order.push_back(node.input_sources[fi]);
    vec.push_back({layout.widths[fi], layout.src_los[fi], layout.dst_los[fi]});
  }

  for (auto const& src : order) {
    auto const& refs     = by_src.at(value_key(src));
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
    memo.values[value_key(src)] = std::move(field_col);
  }
  cudaStreamSynchronize(stream.value());
  return true;
}

// The single decode resolver. Reconstructs and memoises the value(s) node `nid`
// produces on decode — its input_source(s) — and returns the primary one, keyed
// by structural (node, port) identity so every consumer and the codegen tail
// binders share one memo without matching path strings.
cudf::column const* materialize(NodeId nid,
                                PlanTree const& tree,
                                DecodeMemo& memo,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr,
                                std::string* error_out)
{
  PlanNode const& node  = tree.nodes[nid];
  ValueId const primary = node.input_sources.empty() ? ValueId{nid, 0} : node.input_sources.front();
  std::uint64_t const pk = value_key(primary);
  if (auto it = memo.values.find(pk); it != memo.values.end() && it->second)
    return it->second.get();

  // bitjoin recovers all its input values from one packed leaf.
  if (node.attrs.bitjoin.has_value()) {
    if (!decode_bitjoin(nid, tree, memo, stream, mr, error_out)) return nullptr;
    auto it = memo.values.find(pk);
    return it != memo.values.end() ? it->second.get() : nullptr;
  }

  std::unique_ptr<cudf::column> col;
  if (is_codegen_compressor(node.op)) {
    // Fused op (bitpack/delta/rle/for/zigzag): one JIT kernel inverts the whole
    // region (its rep's decompress() throws); tail slots resolve via materialize.
    col = dispatch_codegen_subtree(nid, tree, stream, mr, memo, error_out);
  } else if (node.rep) {
    col = node.rep->decompress(stream, mr);
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
        if (!materialize(child_it->child, tree, memo, stream, mr, error_out)) return nullptr;
        auto it = memo.values.find(value_key(ValueId{nid, static_cast<ChannelId>(i)}));
        if (it == memo.values.end() || !it->second) {
          if (error_out && error_out->empty())
            *error_out = "decode: output '" + name + "' unresolved at node " + std::to_string(nid);
          return nullptr;
        }
        names.push_back(name);
        outputs.push_back(std::make_unique<cudf::column>(it->second->view(), stream, mr));
        continue;
      }
      auto ch_it = node.channels.find(node.output_paths[i]);
      if (ch_it == node.channels.end()) continue;  // not produced here
      if (!ch_it->second) return nullptr;
      auto c = ch_it->second->decompress(stream, mr);
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
    col = rep->decompress(stream, mr);
    memo.kept.push_back(std::move(rep));
  }
  if (!col) return nullptr;
  auto [it, inserted] = memo.values.emplace(pk, std::move(col));
  (void)inserted;
  return it->second.get();
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

  // The result is the input value (node 0, port 0), produced by whichever op(s)
  // consume it. Materialize those; fall back to a scan when the root carries no
  // structural edges (a rep-only tree).
  DecodeMemo memo;
  std::uint64_t const input_key = value_key(ValueId{0, 0});
  for (auto const& e : tree.nodes[0].children) {
    if (memo.values.count(input_key)) break;
    if (!materialize(e.child, tree, memo, stream, mr, error_out)) return nullptr;
  }
  if (!memo.values.count(input_key)) {
    for (NodeId nid = 1; nid < tree.nodes.size() && !memo.values.count(input_key); ++nid) {
      bool consumes_input = false;
      for (auto const& src : tree.nodes[nid].input_sources)
        if (src.node == 0 && src.channel == 0) consumes_input = true;
      if (consumes_input && !materialize(nid, tree, memo, stream, mr, error_out)) return nullptr;
    }
  }

  auto root_it = memo.values.find(input_key);
  if (root_it == memo.values.end() || !root_it->second) {
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
}  // namespace simpatico
