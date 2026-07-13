// SPDX-License-Identifier: Apache-2.0
//
// Compress driver: a single-pass forward walk over the canonical PlanTree that
// mirrors the decode driver in plan/decompress.cpp. The tree is built up front
// from the parsed plan; starting from the "input" column the walk runs the
// consuming op (bitjoin / codegen-fused / generic), produces the child column
// views, places each produced rep directly onto its owning PlanTree node, and
// recurses into every produced output path that has a real downstream consumer.
// Recursion is keyed by column path via `consumer_by_input` (path -> the node
// that consumes it). There is no bulk re-home pass: reps land on nodes as they
// are created.
//
// The parsed step list is only a parse-time artifact of the front-end (parse ->
// canonical render -> PlanTree); the walk and the JIT encode bridge are entirely
// tree-native. The two halves (compress here, decode in plan/decompress.cpp)
// share the bitjoin/column-copy helpers and reconstruct_representation.
#include "codegen/codegen_bridge.hpp"
#include "codegen/plan/bitjoin_layout.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/plan/representation.hpp"

#include <cudf/column/column_factories.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// The C++-native JIT codegen encode entry points (``jit_encode_subtree``,
// ``extract_fusable_subtree``) are declared in codegen_bridge.hpp above.
// make_compressor (operator_registry.hpp), and the bitjoin_layout /
// copy_column_view{,_as_uint8} helpers (bitjoin_layout.hpp) are shared with decode.

namespace simpatico {
namespace {

using ValueColumnMap  = std::unordered_map<ValueId, cudf::column_view, ValueIdHash>;
using ValueConsumeMap = std::unordered_map<ValueId, NodeId, ValueIdHash>;

// The DSL output-path string a value is produced under (matching the producing
// node's `output_paths`). Node 0 is the column input. Used only where the walk
// still needs a path string: keys_chars byte-copy detection and the node
// `channels` map, whose string keys are the shared encode/decode buffer
// contract.
std::string path_for_value(PlanTree const& tree, ValueId v)
{
  if (v.node == 0 || v.node >= tree.nodes.size()) return "input";
  PlanNode const& np = tree.nodes[v.node];
  if (v.channel < np.output_paths.size()) return np.output_paths[v.channel];
  return {};
}

// Place one produced rep onto its owning node in `tree`, keyed structurally by
// the value id `v` it is produced on.
//
// Placement rule (see plan_tree.hpp PlanNode docs):
//   * If value `v` is consumed by a real op C, the rep is C's own representation
//     -> `consumer_by_input[v]`.rep.
//   * Otherwise `v` is a terminal/identity OUTPUT of `producer` (the node the
//     caller is currently emitting) -> `tree.nodes[producer]`.channels[out_path]
//     (channels stay keyed by the output PATH string: the buffer contract decode
//     reads against).
void place_rep_on_node(PlanTree& tree,
                       ValueConsumeMap const& consumer_by_input,
                       NodeId producer,
                       ValueId v,
                       std::string const& out_path,
                       std::unique_ptr<compressed_representation> rep)
{
  if (!rep) return;
  auto cit = consumer_by_input.find(v);
  if (cit != consumer_by_input.end()) {
    auto& node = tree.nodes[cit->second];
    node.rep   = std::move(rep);
    return;
  }
  // Terminal output (nothing consumes `v`): park on the producing node's
  // channels. `producer` is the node the caller is currently emitting, which
  // produces this output path.
  tree.nodes[producer].channels.emplace(out_path, std::move(rep));
}

// Move each NodeId-keyed leaf rep produced by a jit_encode_subtree() call onto
// its owning PlanTree node.
void place_fused_leaves(PlanTree& tree, plan_compound_builder& builder)
{
  for (auto& [nodeid, rep] : builder.leaves) {
    auto& nd = tree.nodes[nodeid];
    nd.rep   = std::move(rep);
    nd.meta  = nd.rep->describe_meta();
  }
}

// Dictionary keys_chars must round-trip as raw UINT8 bytes (see
// representation_factory.cpp), so an identity leaf backing that path needs a
// type-converting copy; every other path takes a plain same-type copy.
bool is_keys_chars_path(std::string const& path)
{
  return path.find("keys_chars") != std::string::npos;
}

std::unique_ptr<cudf::column> copy_identity_leaf(cudf::column_view const& view,
                                                 std::string const& path,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  return is_keys_chars_path(path) ? copy_column_view_as_uint8(view, stream, mr)
                                  : copy_column_view(view, stream, mr);
}

// Forward, pre-order recursion over the PlanTree keyed by column path. The
// structural mirror of decode_node (which is post-order): given a path whose
// column view is live in `columns`, find the node consuming it and run that op,
// then recurse into every produced output path that has a real (non-synthetic)
// downstream consumer.
//
// Each produced rep is placed directly onto its owning node via
// place_rep_on_node. Intermediate reps whose channels feed downstream ops live
// in `reprs_by_input` to keep their device buffers alive for the views in
// `columns`. They are freed EAGERLY: as soon as the last consumed output a rep
// backs is itself consumed (release_column), the rep's device buffers are
// returned to the allocator — so peak GPU memory tracks the live frontier of
// the walk, not the whole cascade. `col_to_repr_key`/`repr_pending` implement
// the refcount (every path is consumed by exactly one op, so a path is fully
// done the moment its consumer fires).
//
// Single-stream: all work runs on `stream`; cross-column parallelism is the
// caller's concern (see compress_column).
struct CompressWalk {
  PlanTree& tree;
  ValueConsumeMap const& consumer_by_input;
  ValueColumnMap& columns;
  std::unordered_map<ValueId, std::unique_ptr<compressed_representation>, ValueIdHash>&
    reprs_by_input;
  std::vector<bool>& visited;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  std::string* error_out;
  bool failed = false;

  // Eager-release bookkeeping: which reprs_by_input key owns each live column,
  // and how many of a key's consumed outputs are still referenced in `columns`.
  // A repr key is the producing value id of the rep's owning node (e.g. {n,0}
  // for an op's own rep) — never one of its inputs, so it can never alias a
  // downstream op's key.
  std::unordered_map<ValueId, ValueId, ValueIdHash> col_to_repr_key;
  std::unordered_map<ValueId, size_t, ValueIdHash> repr_pending;

  void set_error(std::string msg)
  {
    if (failed) return;
    failed = true;
    if (error_out) *error_out = std::move(msg);
  }

  void place(NodeId producer,
             ValueId v,
             std::string const& out_path,
             std::unique_ptr<compressed_representation> rep)
  {
    place_rep_on_node(tree, consumer_by_input, producer, v, out_path, std::move(rep));
  }

  // Drop the live column view for value `v`; if it was the last consumed output
  // of its producing rep, free that rep (stream-ordered) so its device memory is
  // reclaimed immediately rather than at the end of the walk.
  void release_column(ValueId v)
  {
    columns.erase(v);
    auto it = col_to_repr_key.find(v);
    if (it == col_to_repr_key.end()) return;
    ValueId const key = it->second;
    col_to_repr_key.erase(it);
    auto cnt = repr_pending.find(key);
    if (cnt != repr_pending.end() && --(cnt->second) == 0) {
      reprs_by_input.erase(key);
      repr_pending.erase(cnt);
    }
  }

  // Release every unique input value of `node` (a value may repeat across
  // bitjoin fields). Inputs are consumed exactly once, so this is safe to call
  // as soon as the op's kernels are enqueued on `stream`.
  void release_node_inputs(PlanNode const& node)
  {
    std::unordered_set<ValueId, ValueIdHash> released;
    for (auto const& src : node.input_sources) {
      if (released.insert(src).second) release_column(src);
    }
  }

  void emit_path(ValueId v);
  void emit_bitjoin_node(NodeId n);
  bool emit_fused_node(NodeId n, cudf::column_view col);
  void emit_generic_node(NodeId n, cudf::column_view col);
};

void CompressWalk::emit_path(ValueId v)
{
  if (failed) return;
  auto it = consumer_by_input.find(v);
  if (it == consumer_by_input.end()) return;  // terminal value, no consumer
  NodeId const n = it->second;
  if (visited[n]) return;
  PlanNode const& node = tree.nodes[n];

  // For a multi-input bitjoin, only fire once every field column is live; the
  // branch that produces the last input will re-enter here and proceed.
  for (auto const& src : node.input_sources) {
    if (columns.find(src) == columns.end()) return;
  }
  visited[n] = true;

  if (node.input_sources.size() > 1) {
    emit_bitjoin_node(n);
    return;
  }

  auto col_it = columns.find(node.input_sources[0]);
  if (is_codegen_compressor(node.op)) {
    if (emit_fused_node(n, col_it->second)) return;
  }
  emit_generic_node(n, col_it->second);
}

// bitjoin: multiple field inputs → one packed output column.
void CompressWalk::emit_bitjoin_node(NodeId n)
{
  PlanNode const& node = tree.nodes[n];
  // Bit ranges come from the bitjoin attrs (per input, in field order) — the
  // same source decode reads from.
  std::vector<std::optional<bit_range>> input_ranges;
  if (node.attrs.bitjoin.has_value()) {
    input_ranges.reserve(node.attrs.bitjoin->inputs.size());
    for (auto const& ref : node.attrs.bitjoin->inputs)
      input_ranges.push_back(ref.range);
  }
  bitjoin_layout layout;
  if (!resolve_bitjoin_layout(
        node.op, node.input_sources.size(), input_ranges, &layout, error_out)) {
    failed = true;
    return;
  }

  int64_t const n_elements = static_cast<int64_t>(columns.at(node.input_sources[0]).size());
  auto out_col             = cudf::make_fixed_width_column(layout.output_type,
                                               static_cast<cudf::size_type>(n_elements),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
  cudaMemsetAsync(
    out_col->mutable_view().head<void>(),
    0,
    static_cast<size_t>(n_elements) * static_cast<size_t>(cudf::size_of(layout.output_type)),
    stream.value());

  for (size_t fi = 0; fi < node.input_sources.size(); ++fi) {
    launch_bitjoin_field(out_col->mutable_view(),
                         columns.at(node.input_sources[fi]),
                         static_cast<int>(layout.src_los[fi]),
                         static_cast<int>(layout.dst_los[fi]),
                         layout.widths[fi],
                         stream.value());
  }
  bitjoin_warn_on_truncation(columns, layout, node.input_sources, node.op, stream.value());

  // Route the output: terminal outputs are placed straight onto the tree;
  // outputs consumed by a downstream op stay in reprs_by_input (keeping their
  // buffers alive for the recursion). A bitjoin with no declared output stores
  // its packed leaf as the node's own rep (out value == its input source, which
  // it consumes); a declared output is produced on port 0.
  bool const has_output         = !node.output_paths.empty();
  ValueId const out_val         = has_output ? ValueId{n, 0} : node.input_sources[0];
  std::string const out_path    = has_output ? node.output_paths[0] : std::string{};
  bool const output_is_terminal = !has_output || !consumer_by_input.count(ValueId{n, 0});
  columns.emplace(out_val, out_col->view());
  if (output_is_terminal) {
    place(n,
          out_val,
          out_path,
          std::make_unique<identity_compressed_representation>(std::move(out_col)));
    release_node_inputs(node);
  } else {
    ValueId const repr_key = ValueId{n, 0};  // the bitjoin's own output value
    reprs_by_input.emplace(
      repr_key, std::make_unique<identity_compressed_representation>(std::move(out_col)));
    col_to_repr_key[out_val] = repr_key;
    repr_pending[repr_key]   = 1;
    release_node_inputs(node);
    emit_path(out_val);
  }
}

// codegen-fused region rooted at node `n`. Returns false (without setting an
// error) if the region is not fusable — the caller then runs the generic path.
bool CompressWalk::emit_fused_node(NodeId n, cudf::column_view col)
{
  PlanNode const& root   = tree.nodes[n];
  ValueId const head_val = root.input_sources[0];
  plan_compound_builder builder;
  std::string jit_err;
  CodegenHead head;
  if (!jit_encode_subtree(tree, n, col, stream, mr, builder, &jit_err, &head)) {
    // A codegen-only op (bitpack/delta/rle/...) has no generic compressor, so a
    // real encode failure would otherwise surface as a misleading "unknown
    // compressor" from the generic fallback. Report the actual reason.
    if (!jit_err.empty()) set_error("fused encode of '" + root.op + "': " + jit_err);
    return false;
  }
  place_fused_leaves(tree, builder);

  // Build covered set early so the raw-passthrough loop can use it.
  std::unordered_set<NodeId> const covered(head.covered_nodes.begin(), head.covered_nodes.end());
  std::vector<ValueId> to_recurse;

  // Route each raw-passthrough leaf.
  //
  // When a downstream non-fused op consumes the channel value (entropy-tail):
  //   1. Strip the `data` column from the RawFused rep (keep only `offsets` for
  //      the decode binder, which resolves data via the downstream rep's bytes).
  //   2. Keep the data column alive via reprs_by_input until the downstream op
  //      fires and calls release_column.
  //   3. Seed columns[channel_val] with a view of the raw data so emit_path can
  //      route it to the downstream op (e.g. ans, bitcomp).
  //   4. Add channel_val to to_recurse.
  //
  // When no downstream consumer exists (terminal): park the whole RawFused.
  for (auto& leaf : builder.raw_passthrough_leaves) {
    auto& parent_nd = tree.nodes[leaf.parent_id];
    // Locate the parent's output port that carries this channel: its value id is
    // {parent_id, port}, its channels-map key is output_paths[port].
    ChannelId port  = 0;
    bool found_port = false;
    for (std::size_t i = 0; i < parent_nd.output_names.size(); ++i) {
      if (parent_nd.output_names[i] == leaf.channel_name) {
        port       = static_cast<ChannelId>(i);
        found_port = true;
        break;
      }
    }
    if (!found_port) continue;  // shouldn't happen
    ValueId const channel_val             = ValueId{leaf.parent_id, port};
    std::string const channel_output_path = parent_nd.output_paths[port];

    auto cit                  = consumer_by_input.find(channel_val);
    bool const has_downstream = (cit != consumer_by_input.end() && !covered.count(cit->second));

    if (has_downstream) {
      auto* raw_rep = static_cast<codegen_fused_representation*>(leaf.rep.get());
      std::unique_ptr<cudf::column> data_col;
      for (auto& [bname, bcol] : raw_rep->buffers) {
        if (bname == "data") {
          data_col = std::move(bcol);
          break;
        }
      }
      // Erase the now-null entry so named_channels() returns only offsets.
      raw_rep->buffers.erase(std::remove_if(raw_rep->buffers.begin(),
                                            raw_rep->buffers.end(),
                                            [](auto const& p) { return p.second == nullptr; }),
                             raw_rep->buffers.end());
      if (data_col) {
        cudf::column_view data_view = data_col->view();
        // Key the stripped data rep by the channel's own value id. The
        // downstream op keys its rep by ITS own output ({consumer,0}), never by
        // channel_val, so the two never alias.
        ValueId const repr_key = channel_val;
        reprs_by_input.emplace(
          repr_key, std::make_unique<identity_compressed_representation>(std::move(data_col)));
        col_to_repr_key[channel_val] = repr_key;
        repr_pending[repr_key]       = 1;
        columns.emplace(channel_val, data_view);
        to_recurse.push_back(channel_val);
      }
    }
    // Park the (possibly data-stripped) RawFused rep in the parent's channels
    // so the decode binder can find its offsets buffer.
    parent_nd.channels.emplace(channel_output_path, std::move(leaf.rep));
  }

  // The region's head input has now been fully consumed by the JIT kernel; free
  // it (boundary outputs below are backed by node-owned reps, not by head_val).
  release_column(head_val);

  // The fused region covers several PlanTree nodes; mark them done and recurse
  // into each region output that feeds a real downstream op (e.g. a bitpack
  // `.packed` channel consumed by `deflate`). Interior region edges (whose
  // consumer is itself covered), terminal channels (drained synthetically /
  // owned by the rep), and channels already routed by the raw-passthrough loop
  // above are skipped.
  if (head.covered_nodes.empty()) return true;
  for (NodeId cn : head.covered_nodes) {
    visited[cn]                    = true;
    PlanNode const& cnode          = tree.nodes[cn];
    compressed_representation* rep = cnode.rep.get();
    for (size_t k = 0; k < cnode.output_names.size(); ++k) {
      ValueId const output_val = ValueId{cn, static_cast<ChannelId>(k)};
      auto cit                 = consumer_by_input.find(output_val);
      if (cit == consumer_by_input.end()) continue;  // terminal (owned by rep)
      if (covered.count(cit->second)) continue;      // interior region edge
      // Already routed by the raw-passthrough loop above (data_col was seeded
      // into columns and the value is in to_recurse).
      if (columns.count(output_val)) continue;

      if (!rep) {
        set_error("fused boundary: missing rep at node " + std::to_string(cn));
        return true;
      }
      auto chans                    = rep->named_channels(stream);
      cudf::column_view const* view = nullptr;
      for (auto const& c : chans) {
        if (c.name == cnode.output_names[k]) {
          view = &c.view;
          break;
        }
      }
      if (!view) {
        set_error("fused boundary: rep at node " + std::to_string(cn) + " has no channel '" +
                  cnode.output_names[k] + "'");
        return true;
      }
      columns.emplace(output_val, *view);
      to_recurse.push_back(output_val);
    }
  }
  for (auto const& op : to_recurse) {
    if (failed) return true;
    emit_path(op);
  }
  return true;
}

// generic single-input op: run the registered compressor and route its named
// output channels (terminal → identity leaf; consumed → recurse downstream).
void CompressWalk::emit_generic_node(NodeId n, cudf::column_view col)
{
  PlanNode const& node    = tree.nodes[n];
  ValueId const input_val = node.input_sources[0];
  std::string const path  = path_for_value(tree, input_val);
  auto compressor         = make_compressor(node.op);
  if (!compressor) {
    set_error("unknown compressor '" + node.op + "'");
    return;
  }

  // For identity compressor on keys_chars, convert to UINT8 first.
  cudf::column_view col_to_compress = col;
  std::unique_ptr<cudf::column> temp_col;
  if (node.op == "identity" && is_keys_chars_path(path)) {
    temp_col = copy_column_view_as_uint8(col, stream, mr);
    if (!temp_col) {
      set_error("failed to convert keys_chars to UINT8");
      return;
    }
    col_to_compress = temp_col->view();
  }

  auto repr = compressor->compress(col_to_compress, stream, mr);
  // Single-stream mode: sync so async compressors complete before we read
  // output column views/sizes.
  cudaStreamSynchronize(stream.value());
  if (!repr) {
    set_error("compressor '" + node.op + "' returned null representation");
    return;
  }
  // Capture decode metadata now, before the rep is moved or dropped.
  // This is the only safe window — for ops like ANS/Bitcomp the repr is
  // NOT placed onto the tree node; it stays in reprs_by_input until its
  // consumers release it and then is freed. node.meta persists and carries
  // uncompressed_size / original_type_id / algorithm to the decode path.
  tree.nodes[n].meta = repr->describe_meta();

  if (node.output_names.empty()) {
    release_column(input_val);
    // No declared output: the whole rep becomes this node's own rep (place sees
    // the node consuming input_val and stores it on tree.nodes[n].rep).
    place(n, input_val, path, std::move(repr));
    return;
  }

  auto outputs = repr->named_channels(stream);
  std::unordered_map<std::string, cudf::column_view> output_by_name;
  output_by_name.reserve(outputs.size());
  for (auto const& output : outputs)
    output_by_name.emplace(output.name, output.view);

  // Guard: every channel the rep marks required must be routed by the plan, or
  // data (e.g. a nullable column's validity via str_split's null_mask) would be
  // silently dropped. Runs only when some outputs are declared (a bare terminal
  // `input -> str_split` stored the whole rep above and is safe).
  for (auto const& req : repr->required_channels()) {
    bool declared = false;
    for (auto const& out_name : node.output_names) {
      if (req == out_name) {
        declared = true;
        break;
      }
    }
    if (!declared) {
      set_error("compressor '" + node.op + "' requires output '" + req +
                "' to be routed by the plan (nullable input)");
      return;
    }
  }

  size_t pending = 0;
  std::vector<ValueId> to_recurse;
  // The rep's owner key is this node's own representative output value {n,0} —
  // never an input, so it can't alias a downstream op's key.
  ValueId const repr_key = ValueId{n, 0};
  for (size_t idx = 0; idx < node.output_names.size(); ++idx) {
    auto const& out_name = node.output_names[idx];
    auto out_it          = output_by_name.find(out_name);
    if (out_it == output_by_name.end()) {
      set_error("compressor '" + node.op + "' does not expose output '" + out_name + "'");
      return;
    }
    std::string const& output_path = node.output_paths[idx];
    ValueId const output_val       = ValueId{n, static_cast<ChannelId>(idx)};
    columns.emplace(output_val, out_it->second);

    if (!consumer_by_input.count(output_val)) {
      // Terminal leaf — copy the channel into an identity leaf (reps stay whole
      // on the PlanTree; no destructuring).
      auto leaf_col = copy_identity_leaf(out_it->second, output_path, stream, mr);
      if (!leaf_col) {
        set_error("failed to get column for identity leaf '" + output_path + "'");
        return;
      }
      place(n,
            output_val,
            output_path,
            std::make_unique<identity_compressed_representation>(std::move(leaf_col)));
    } else {
      ++pending;
      col_to_repr_key[output_val] = repr_key;
      to_recurse.push_back(output_val);
    }
  }

  // Keep the repr alive (its channels back the views in `columns`) only while a
  // consumed output still needs it; otherwise it frees here. The refcount is the
  // number of consumed outputs; each is decremented as its consumer releases it.
  if (pending > 0) {
    reprs_by_input.emplace(repr_key, std::move(repr));
    repr_pending[repr_key] = pending;
  }
  // The input column is fully consumed (compress already ran + synced and all
  // output views derive from `repr`, not the input); release it now so an
  // upstream rep it backed can be freed before we descend.
  release_column(input_val);
  for (auto const& output_val : to_recurse) {
    if (failed) return;
    emit_path(output_val);
  }
}

}  // namespace

std::unique_ptr<PlanTree> compress_column(cudf::column_view input,
                                          std::string_view plan_dsl,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr,
                                          std::string* error_out)
{
  // Single-stream per column: all work runs on `stream`. Cross-column
  // parallelism is the caller's job (one column per worker thread, each on its
  // own stream). Intermediate device buffers are freed eagerly by the walk.

  // Front-end: parse the DSL into a flat step list, canonicalize it, and build
  // the canonical PlanTree + producer path map. The step list is only a
  // transient parse artifact here; the walk below is entirely tree-native.
  std::vector<plan_step> steps;
  std::string parse_error;
  if (!parse_plan_dsl(plan_dsl, &steps, &parse_error)) {
    if (error_out) *error_out = parse_error;
    return nullptr;
  }
  auto compound = std::make_unique<PlanTree>();
  {
    std::string tree_err;
    auto tree = plan_tree_from_steps(steps, &tree_err);
    if (!tree) {
      if (error_out) *error_out = "plan-tree build: " + tree_err;
      return nullptr;
    }
    *compound = std::move(*tree);
  }
  PlanTree& tree = *compound;

  // consumer_by_input[V] = the node that consumes the value produced at V,
  // derived structurally from each node's input_sources (the (node, port) each
  // input comes from). First-consumer-wins, matching parse_plan_dsl. Every value
  // is consumed by at most one op, so this key uniquely identifies its consumer;
  // terminal (unconsumed) values are absent and fall to the producer's channels
  // in place_rep_on_node.
  ValueConsumeMap consumer_by_input;
  for (NodeId i = 1; i < tree.nodes.size(); ++i) {
    for (auto const& src : tree.nodes[i].input_sources) {
      consumer_by_input.emplace(src, i);
    }
  }

  // Single recursive walk from "input", placing reps as it goes. When the
  // whole plan fuses into one JIT kernel, emit_fused_node (called from
  // emit_path via CompressWalk below) performs that single jit_encode_subtree
  // call itself and handles both the all-terminal case and the entropy-tail
  // case (a raw-passthrough or boundary channel with a downstream consumer)
  // uniformly — so there is no separate fast path to maintain here.
  ValueColumnMap columns;
  columns.emplace(ValueId{0, 0}, input);  // the column input is value (node 0, port 0)
  std::unordered_map<ValueId, std::unique_ptr<compressed_representation>, ValueIdHash>
    reprs_by_input;
  std::vector<bool> visited(tree.nodes.size(), false);

  CompressWalk walk{
    tree, consumer_by_input, columns, reprs_by_input, visited, stream, mr, error_out};
  walk.emit_path(ValueId{0, 0});
  if (walk.failed) return nullptr;

  // Every real op node must have been reached by the walk; a missed node would
  // silently drop data. (All PlanTree nodes are non-synthetic ops; synthetic
  // identity drains have no node and are covered by their producing op's
  // terminal-output handling.)
  for (NodeId i = 1; i < tree.nodes.size(); ++i) {
    if (visited[i]) continue;
    if (error_out) {
      PlanNode const& nd   = tree.nodes[i];
      std::string in_label = dotted_label(tree, i);
      if (in_label.empty()) in_label = "?";
      *error_out = "plan node[" + std::to_string(i) + "] '" + in_label + " -> " + nd.op +
                   "' was not resolved (missing inputs or cycle)";
    }
    return nullptr;
  }

  if (error_out) error_out->clear();
  return compound;
}

namespace {

// Which output channels a fused preprocessing op MATERIALISES as Raw passthrough
// streams (delta.differences, rle.runs/values, for.deltas) versus which live
// directly on the op node's own rep as BOUNDARY outputs (for.references).
// Empty for ops whose rep already exposes canonical channels (bitpack, zigzag).
struct fused_op_channels {
  std::vector<std::string> materialized;  // drained to an identity leaf to store
  std::vector<std::string> boundary;      // already carried on the op node rep
  bool empty() const { return materialized.empty() && boundary.empty(); }
};

fused_op_channels canonical_fused_channels(std::string const& op)
{
  if (op == "delta") return {{"differences"}, {}};
  if (op == "rle") return {{"runs", "values"}, {}};
  if (op == "for") return {{"deltas"}, {"references"}};
  return {};  // bitpack / zigzag: rep->named_channels() are already canonical
}

// Trial result for a single fused op: exposes the op's canonical output channels
// by their DSL names (so the explorer/sweep emit valid cascades and chain onto
// the real transformed streams) while owning the whole compound so the channel
// views stay valid and byte accounting stays complete.
struct single_op_representation : compressed_representation {
  std::unique_ptr<PlanTree> compound;
  std::vector<compressible_output> chans;

  single_op_representation(cudf::data_type t, cudf::size_type n) : compressed_representation(t, n)
  {
  }

  std::vector<compressible_output> named_channels(rmm::cuda_stream_view) const override
  {
    return chans;
  }

  // Full stored size: every node rep + parked channel, so the op's aux buffers
  // (delta_first / for references / rle run offsets) are counted too.
  size_t compressed_size_bytes(rmm::cuda_stream_view stream) const override
  {
    size_t total = 0;
    if (!compound) return total;
    for (auto const& node : compound->nodes) {
      if (node.rep) total += node.rep->compressed_size_bytes(stream);
      for (auto const& [path, rep] : node.channels) {
        if (rep) total += rep->compressed_size_bytes(stream);
      }
    }
    return total;
  }

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref) const override
  {
    // Not a standalone decodable leaf; callers use named_channels() /
    // compressed_size_bytes() only.
    return nullptr;
  }
};

}  // namespace

std::unique_ptr<compressed_representation> compress_single_op(std::string const& op_name,
                                                              cudf::column_view input,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr,
                                                              std::string* error_out)
{
  if (error_out) error_out->clear();

  // Non-fused path: use the compressor factory directly.
  if (!is_codegen_compressor(op_name)) {
    auto comp = make_compressor(op_name);
    if (!comp) {
      if (error_out) *error_out = "compress_single_op: unknown op '" + op_name + "'";
      return nullptr;
    }
    return comp->compress(input, stream, mr);
  }

  // Fused path.  A bare "input -> op" is not enough: for delta/rle/for the
  // transformed output stream (differences / runs+values / deltas) is only
  // MATERIALISED when the channel has a real downstream consumer — otherwise the
  // op node keeps just its reconstruction aux (delta_first / run offsets /
  // per-chunk references) on `rep`, which is NOT the canonical DSL channel the
  // explorer/sweep must chain onto.  So we spell the canonical outputs
  // explicitly and drain each materialised channel into an `identity` leaf (a
  // real consumer — also required so rle's mandatory `runs` edge exists), then
  // surface those stored streams under their canonical names.
  auto const fused = canonical_fused_channels(op_name);

  std::string dsl = "input -> " + op_name;
  if (!fused.empty()) {
    dsl += " -> ";
    bool first = true;
    for (auto const& c : fused.materialized) {
      dsl += (first ? "" : ", ") + c;
      first = false;
    }
    for (auto const& c : fused.boundary) {
      dsl += (first ? "" : ", ") + c;
      first = false;
    }
    for (auto const& c : fused.materialized)
      dsl += "\n" + op_name + "." + c + " -> identity";
  }

  auto compound = compress_column(input, dsl, stream, mr, error_out);
  if (!compound) return nullptr;

  // Locate the op node (index 1 for a single-step plan).
  PlanNode* op_node  = nullptr;
  NodeId op_node_idx = 0;
  for (std::size_t i = 1; i < compound->nodes.size(); ++i) {
    if (compound->nodes[i].op == op_name) {
      op_node     = &compound->nodes[i];
      op_node_idx = static_cast<NodeId>(i);
      break;
    }
  }
  if (!op_node) {
    if (error_out && error_out->empty())
      *error_out = "compress_single_op: op node '" + op_name + "' not found";
    return nullptr;
  }

  // bitpack / zigzag: the op rep already exposes canonical channels — return it
  // (or, defensively, the first parked channel rep) as before.
  if (fused.empty()) {
    if (op_node->rep) return std::move(op_node->rep);
    for (auto& [path, ch_rep] : op_node->channels)
      if (ch_rep) return std::move(ch_rep);
    if (error_out && error_out->empty())
      *error_out = "compress_single_op: no rep found for op '" + op_name + "'";
    return nullptr;
  }

  // Resolve each canonical output (in canonical channel order — parse_plan_dsl's
  // canonicalize_output_order() already made the DSL author's textual order
  // irrelevant) to a transformed-stream view:
  //   1) the identity leaf that stored a materialised stream, else
  //   2) a boundary channel carried on the op node rep (e.g. for.references), else
  //   3) a terminal Raw passthrough parked on the op node (its `data` buffer).
  auto result = std::make_unique<single_op_representation>(input.type(), input.size());
  for (std::size_t i = 0; i < op_node->output_names.size(); ++i) {
    std::string const& chan_name = op_node->output_names[i];
    std::string const& chan_path = op_node->output_paths[i];
    // The value this channel produces, resolved structurally.
    ValueId const chan_val = ValueId{op_node_idx, static_cast<ChannelId>(i)};

    bool found = false;

    for (std::size_t j = 1; j < compound->nodes.size() && !found; ++j) {
      auto& nj = compound->nodes[j];
      if (nj.op != "identity" || nj.input_sources.empty()) continue;
      if (nj.input_sources[0] != chan_val || !nj.rep) continue;
      auto leaf_chans = nj.rep->named_channels(stream);
      if (!leaf_chans.empty()) {
        result->chans.push_back({chan_name, leaf_chans.front().view});
        found = true;
      }
    }

    if (!found && op_node->rep) {
      for (auto const& c : op_node->rep->named_channels(stream)) {
        if (c.name == chan_name) {
          result->chans.push_back({chan_name, c.view});
          found = true;
          break;
        }
      }
    }

    if (!found) {
      auto ch_it = op_node->channels.find(chan_path);
      if (ch_it != op_node->channels.end() && ch_it->second) {
        auto pch                   = ch_it->second->named_channels(stream);
        cudf::column_view const* v = nullptr;
        for (auto const& c : pch) {
          if (c.name == "data") {
            v = &c.view;
            break;
          }
        }
        if (!v && !pch.empty()) v = &pch.front().view;
        if (v) {
          result->chans.push_back({chan_name, *v});
          found = true;
        }
      }
    }

    if (!found) {
      if (error_out)
        *error_out = "compress_single_op: canonical channel '" + chan_name + "' (" + chan_path +
                     ") not produced for op '" + op_name + "'";
      return nullptr;
    }
  }

  result->compound = std::move(compound);
  return result;
}

}  // namespace simpatico
