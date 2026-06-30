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
#include "codegen/plan/column_copy.hpp"
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
// make_compressor (operator_registry.hpp), the bitjoin_layout helpers, and
// copy_column_view{,_as_uint8} (column_copy.hpp) are shared with decode.

namespace simpatico {
namespace {

// Place one produced rep onto its owning node in `tree`, keyed by the DSL
// `path`.
//
// Placement rule (total over all leaf paths — see plan_tree.hpp PlanNode docs):
//   * If path P is consumed by a real (non-synthetic) op C, the rep is C's own
//     representation (keyed by its input path) -> `consumer_by_input[P]`.rep.
//     (Synthetic identity drains have no tree node, so a terminal output
//     consumed only by a synthetic drain falls through to the channels branch.)
//   * Otherwise P is a terminal/identity OUTPUT of its producing node ->
//     `path_map.node[P]`.channels[P].
// A leaf that matches neither (should not happen for a well-formed plan) is
// parked on the root node's channels so it is never leaked/lost.
void place_rep_on_node(PlanTree& tree,
                       PlanPathMap const& path_map,
                       std::unordered_map<std::string, NodeId> const& consumer_by_input,
                       std::string const& path,
                       std::unique_ptr<compressed_representation> rep)
{
  if (!rep) return;
  auto cit = consumer_by_input.find(path);
  if (cit != consumer_by_input.end()) {
    auto& node    = tree.nodes[cit->second];
    node.rep      = std::move(rep);
    node.rep_path = path;
    return;
  }
  auto pit           = path_map.node.find(path);
  NodeId const owner = (pit != path_map.node.end()) ? pit->second : 0;
  tree.nodes[owner].channels.emplace(path, std::move(rep));
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
  std::unordered_map<std::string, NodeId> const& consumer_by_input;
  std::unordered_map<std::string, cudf::column_view>& columns;
  std::unordered_map<std::string, std::unique_ptr<compressed_representation>>& reprs_by_input;
  PlanPathMap const& path_map;
  std::vector<bool>& visited;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  std::string const& canon_plan_dsl;
  std::string* error_out;
  bool failed = false;

  // Eager-release bookkeeping: which reprs_by_input key owns each live column,
  // and how many of a key's consumed outputs are still referenced in `columns`.
  std::unordered_map<std::string, std::string> col_to_repr_key;
  std::unordered_map<std::string, size_t> repr_pending;

  void set_error(std::string msg)
  {
    if (failed) return;
    failed = true;
    if (error_out) *error_out = std::move(msg);
  }

  void place(std::string const& path, std::unique_ptr<compressed_representation> rep)
  {
    place_rep_on_node(tree, path_map, consumer_by_input, path, std::move(rep));
  }

  // Drop the live column view at `path`; if it was the last consumed output of
  // its producing rep, free that rep (stream-ordered) so its device memory is
  // reclaimed immediately rather than at the end of the walk.
  void release_column(std::string const& path)
  {
    columns.erase(path);
    auto it = col_to_repr_key.find(path);
    if (it == col_to_repr_key.end()) return;
    std::string const key = it->second;
    col_to_repr_key.erase(it);
    auto cnt = repr_pending.find(key);
    if (cnt != repr_pending.end() && --(cnt->second) == 0) {
      reprs_by_input.erase(key);
      repr_pending.erase(cnt);
    }
  }

  // Release every unique input column of `node` (a path may repeat across
  // bitjoin fields). Inputs are consumed exactly once, so this is safe to call
  // as soon as the op's kernels are enqueued on `stream`.
  void release_node_inputs(PlanNode const& node)
  {
    std::unordered_set<std::string> released;
    for (auto const& ipath : node.input_paths) {
      if (released.insert(ipath).second) release_column(ipath);
    }
  }

  void emit_path(std::string const& path);
  void emit_bitjoin_node(NodeId n);
  bool emit_fused_node(NodeId n, cudf::column_view col);
  void emit_generic_node(NodeId n, cudf::column_view col);
};

void CompressWalk::emit_path(std::string const& path)
{
  if (failed) return;
  auto it = consumer_by_input.find(path);
  if (it == consumer_by_input.end()) return;  // terminal column, no consumer
  NodeId const n = it->second;
  if (visited[n]) return;
  PlanNode const& node = tree.nodes[n];

  // For a multi-input bitjoin, only fire once every field column is live; the
  // branch that produces the last input will re-enter here and proceed.
  for (auto const& ipath : node.input_paths) {
    if (columns.find(ipath) == columns.end()) return;
  }
  visited[n] = true;

  if (node.input_paths.size() > 1) {
    emit_bitjoin_node(n);
    return;
  }

  auto col_it = columns.find(node.input_paths[0]);
  if (is_codegen_compressor(node.op)) {
    if (emit_fused_node(n, col_it->second)) return;
  }
  emit_generic_node(n, col_it->second);
}

// bitjoin: multiple field inputs → one packed output column.
void CompressWalk::emit_bitjoin_node(NodeId n)
{
  PlanNode const& node = tree.nodes[n];
  bitjoin_layout layout;
  if (!resolve_bitjoin_layout(node.op, node.input_paths, node.input_ranges, &layout, error_out)) {
    failed = true;
    return;
  }

  int64_t const n_elements = static_cast<int64_t>(columns.at(node.input_paths[0]).size());
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

  for (size_t fi = 0; fi < node.input_paths.size(); ++fi) {
    launch_bitjoin_field(out_col->mutable_view(),
                         columns.at(node.input_paths[fi]),
                         static_cast<int>(layout.src_los[fi]),
                         static_cast<int>(layout.dst_los[fi]),
                         layout.widths[fi],
                         stream.value());
  }
  bitjoin_warn_on_truncation(columns, layout, node.input_paths, node.op, stream.value());

  // Route the output: terminal outputs are placed straight onto the tree;
  // outputs consumed by a downstream op stay in reprs_by_input (keeping their
  // buffers alive for the recursion).
  std::string const out_key =
    node.output_paths.empty() ? node.input_paths[0] : node.output_paths[0];
  bool const output_is_terminal =
    node.output_paths.empty() || !consumer_by_input.count(node.output_paths[0]);
  columns.emplace(out_key, out_col->view());
  if (output_is_terminal) {
    place(out_key, std::make_unique<identity_compressed_representation>(std::move(out_col)));
    release_node_inputs(node);
  } else {
    std::string const repr_key = node.input_paths[0];
    reprs_by_input.emplace(
      repr_key, std::make_unique<identity_compressed_representation>(std::move(out_col)));
    col_to_repr_key[out_key] = repr_key;
    repr_pending[repr_key]   = 1;
    // Release inputs AFTER registering the output owner: an input path may equal
    // repr_key, but col_to_repr_key keys the upstream owner of that input, never
    // the just-stored output, so the two never collide.
    release_node_inputs(node);
    emit_path(out_key);
  }
}

// codegen-fused region rooted at node `n`. Returns false (without setting an
// error) if the region is not fusable — the caller then runs the generic path.
bool CompressWalk::emit_fused_node(NodeId n, cudf::column_view col)
{
  PlanNode const& root        = tree.nodes[n];
  std::string const head_path = root.input_paths[0];
  plan_compound_builder builder;
  std::string jit_err;
  CodegenHead head;
  if (!jit_encode_subtree(tree, n, col, stream, mr, builder, &jit_err, &head)) { return false; }
  // Place NodeId-keyed reps directly onto their owning nodes.
  for (auto& [nodeid, rep] : builder.leaves) {
    auto& nd    = tree.nodes[nodeid];
    nd.rep      = std::move(rep);
    nd.rep_path = nd.input_path;
    nd.meta     = nd.rep->describe_meta();
  }

  // Build covered set early so the raw-passthrough loop can use it.
  std::unordered_set<NodeId> const covered(head.covered_nodes.begin(), head.covered_nodes.end());
  std::vector<std::string> to_recurse;

  // Route each raw-passthrough leaf.
  //
  // When a downstream non-fused op consumes the channel path (entropy-tail):
  //   1. Strip the `data` column from the RawFused rep (keep only `offsets` for
  //      the decode binder, which resolves data via the downstream rep's bytes).
  //   2. Keep the data column alive via reprs_by_input until the downstream op
  //      fires and calls release_column.
  //   3. Seed columns[channel_path] with a view of the raw data so emit_path can
  //      route it to the downstream op (e.g. ans, bitcomp).
  //   4. Add channel_path to to_recurse.
  //
  // When no downstream consumer exists (terminal): park the whole RawFused.
  for (auto& leaf : builder.raw_passthrough_leaves) {
    auto& parent_nd = tree.nodes[leaf.parent_id];
    std::string channel_output_path;
    for (std::size_t i = 0; i < parent_nd.output_names.size(); ++i) {
      if (parent_nd.output_names[i] == leaf.channel_name) {
        channel_output_path = parent_nd.output_paths[i];
        break;
      }
    }
    if (channel_output_path.empty()) continue;  // shouldn't happen

    auto cit                  = consumer_by_input.find(channel_output_path);
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
        std::string repr_key        = channel_output_path + ":raw_passthrough";
        reprs_by_input.emplace(
          repr_key, std::make_unique<identity_compressed_representation>(std::move(data_col)));
        col_to_repr_key[channel_output_path] = repr_key;
        repr_pending[repr_key]               = 1;
        columns.emplace(channel_output_path, data_view);
        to_recurse.push_back(channel_output_path);
      }
    }
    // Park the (possibly data-stripped) RawFused rep in the parent's channels
    // so the decode binder can find its offsets buffer.
    parent_nd.channels.emplace(channel_output_path, std::move(leaf.rep));
  }

  // The region's head input has now been fully consumed by the JIT kernel; free
  // it (boundary outputs below are backed by node-owned reps, not by head_path).
  release_column(head_path);

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
      std::string const& output_path = cnode.output_paths[k];
      auto cit                       = consumer_by_input.find(output_path);
      if (cit == consumer_by_input.end()) continue;  // terminal (owned by rep)
      if (covered.count(cit->second)) continue;      // interior region edge
      // Already routed by the raw-passthrough loop above (data_col was seeded
      // into columns and path is in to_recurse).
      if (columns.count(output_path)) continue;

      if (!rep) {
        set_error("fused boundary: missing rep at node " + std::to_string(cn));
        return true;
      }
      auto chans                    = rep->named_channels();
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
      columns.emplace(output_path, *view);
      to_recurse.push_back(output_path);
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
  PlanNode const& node   = tree.nodes[n];
  std::string const path = node.input_paths[0];
  auto compressor        = make_compressor(node.op);
  if (!compressor) {
    set_error("unknown compressor '" + node.op + "'");
    return;
  }

  // For identity compressor on keys_chars, convert to UINT8 first.
  cudf::column_view col_to_compress = col;
  std::unique_ptr<cudf::column> temp_col;
  if (node.op == "identity" && path.find("keys_chars") != std::string::npos) {
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
    release_column(path);
    place(path, std::move(repr));
    return;
  }

  auto outputs = repr->named_channels();
  std::unordered_map<std::string, cudf::column_view> output_by_name;
  output_by_name.reserve(outputs.size());
  for (auto const& output : outputs)
    output_by_name.emplace(output.name, output.view);

  size_t pending = 0;
  std::vector<std::string> to_recurse;
  std::string const repr_key = path;
  for (size_t idx = 0; idx < node.output_names.size(); ++idx) {
    auto const& out_name = node.output_names[idx];
    auto out_it          = output_by_name.find(out_name);
    if (out_it == output_by_name.end()) {
      set_error("compressor '" + node.op + "' does not expose output '" + out_name + "'");
      return;
    }
    std::string const& output_path = node.output_paths[idx];
    columns.emplace(output_path, out_it->second);

    if (!consumer_by_input.count(output_path)) {
      // Terminal leaf — copy the channel into an identity leaf (reps stay whole
      // on the PlanTree; no release_output destructuring).
      std::unique_ptr<cudf::column> leaf_col;
      if (output_path.find("keys_chars") != std::string::npos) {
        leaf_col = copy_column_view_as_uint8(out_it->second, stream, mr);
      } else {
        leaf_col = copy_column_view(out_it->second, stream, mr);
      }
      if (!leaf_col) {
        set_error("failed to get column for identity leaf '" + output_path + "'");
        return;
      }
      place(output_path, std::make_unique<identity_compressed_representation>(std::move(leaf_col)));
    } else {
      ++pending;
      col_to_repr_key[output_path] = repr_key;
      to_recurse.push_back(output_path);
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
  release_column(path);
  for (auto const& output_path : to_recurse) {
    if (failed) return;
    emit_path(output_path);
  }
}

}  // namespace

std::unique_ptr<plan_compound> compress_column(cudf::column_view input,
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
  std::string canon_plan_dsl = render_plan_steps(steps);

  auto compound      = std::make_unique<plan_compound>();
  compound->plan_dsl = canon_plan_dsl;
  PlanPathMap path_map;
  {
    std::string tree_err;
    auto tree = plan_tree_from_steps(steps, &tree_err, &path_map);
    if (!tree) {
      if (error_out) *error_out = "plan-tree build: " + tree_err;
      return nullptr;
    }
    compound->tree = std::move(*tree);
  }
  PlanTree& tree = compound->tree;

  // consumer_by_input[P] = the node that consumes path P, derived straight from
  // the tree's input metadata. First-consumer-wins, matching parse_plan_dsl. Only
  // real ops are tree nodes, so synthetic-drain-only paths are absent — those
  // fall to the producer's channels in place_rep_on_node.
  std::unordered_map<std::string, NodeId> consumer_by_input;
  for (NodeId i = 1; i < tree.nodes.size(); ++i) {
    for (auto const& ip : tree.nodes[i].input_paths) {
      consumer_by_input.emplace(ip, i);
    }
  }

  // Fast path: the whole plan fuses from "input" into one JIT kernel.  The
  // head op is the consumer of "input"; if jit_encode_subtree succeeds AND
  // every raw-passthrough channel is terminal (no downstream non-fused op), the
  // compound is complete and we return early.  If any raw-passthrough channel
  // has a downstream consumer (entropy-tail routing needed), the fast path is
  // skipped so the general CompressWalk can handle it via emit_fused_node.
  auto head_it = consumer_by_input.find("input");
  if (head_it != consumer_by_input.end()) {
    // Pre-check: inspect the fused region structure without running the GPU
    // kernel.  If any Raw leaf has a downstream consumer, skip the fast path.
    bool fast_path_ok = true;
    {
      auto maybe_built = build_fused_tree(tree,
                                          head_it->second,
                                          /*fixed_stride=*/true);
      if (maybe_built) {
        // Bail if any Raw passthrough leaf has a downstream consumer
        // (entropy-tail routing needed — general walk handles it).
        for (auto const& origin : maybe_built->preorder) {
          if (!origin.is_raw_passthrough || origin.parent_channel.empty()) continue;
          auto const& parent_nd = tree.nodes[origin.parent_rle];
          for (std::size_t i = 0; i < parent_nd.output_names.size(); ++i) {
            if (parent_nd.output_names[i] == origin.parent_channel) {
              if (consumer_by_input.count(parent_nd.output_paths[i])) { fast_path_ok = false; }
              break;
            }
          }
          if (!fast_path_ok) break;
        }

        // Also bail if any covered node has a boundary output feeding a
        // second codegen (JIT-fused) region.  The fast path's single
        // jit_encode_subtree call covers only the primary region; the
        // secondary region must be handled by the general CompressWalk.
        if (fast_path_ok) {
          std::unordered_set<NodeId> covered_set;
          for (auto const& origin : maybe_built->preorder) {
            if (!origin.is_raw_passthrough && origin.plan_node < tree.nodes.size()) {
              covered_set.insert(origin.plan_node);
            }
          }
          for (auto const& origin : maybe_built->preorder) {
            if (origin.is_raw_passthrough || !fast_path_ok) continue;
            if (origin.plan_node >= tree.nodes.size()) continue;
            for (auto const& e : tree.nodes[origin.plan_node].children) {
              if (!covered_set.count(e.child) && e.child < tree.nodes.size() &&
                  is_codegen_compressor(tree.nodes[e.child].op)) {
                fast_path_ok = false;
                break;
              }
            }
          }
        }
      }
    }

    if (fast_path_ok) {
      plan_compound_builder builder;
      std::string jit_err;
      if (jit_encode_subtree(tree, head_it->second, input, stream, mr, builder, &jit_err)) {
        // Place NodeId-keyed reps directly onto their owning nodes.
        for (auto& [nodeid, rep] : builder.leaves) {
          auto& nd    = tree.nodes[nodeid];
          nd.rep      = std::move(rep);
          nd.rep_path = nd.input_path;
          nd.meta     = nd.rep->describe_meta();
        }
        // All raw-passthrough reps are terminal: park in parent channels.
        for (auto& leaf : builder.raw_passthrough_leaves) {
          auto& parent_nd = tree.nodes[leaf.parent_id];
          for (std::size_t i = 0; i < parent_nd.output_names.size(); ++i) {
            if (parent_nd.output_names[i] == leaf.channel_name) {
              parent_nd.channels.emplace(parent_nd.output_paths[i], std::move(leaf.rep));
              break;
            }
          }
        }
        if (error_out) error_out->clear();
        return compound;
      }
      if (!jit_err.empty() && error_out) *error_out = jit_err;
    }
  }

  // General path: single recursive walk from "input", placing reps as it goes.
  std::unordered_map<std::string, cudf::column_view> columns;
  columns.emplace("input", input);
  std::unordered_map<std::string, std::unique_ptr<compressed_representation>> reprs_by_input;
  std::vector<bool> visited(tree.nodes.size(), false);

  CompressWalk walk{tree,
                    consumer_by_input,
                    columns,
                    reprs_by_input,
                    path_map,
                    visited,
                    stream,
                    mr,
                    canon_plan_dsl,
                    error_out};
  walk.emit_path("input");
  if (walk.failed) return nullptr;

  // Every real op node must have been reached by the walk; a missed node would
  // silently drop data. (All PlanTree nodes are non-synthetic ops; synthetic
  // identity drains have no node and are covered by their producing op's
  // terminal-output handling.)
  for (NodeId i = 1; i < tree.nodes.size(); ++i) {
    if (visited[i]) continue;
    if (error_out) {
      PlanNode const& nd = tree.nodes[i];
      *error_out         = "plan node[" + std::to_string(i) + "] '" +
                   (nd.input_paths.empty() ? "?" : nd.input_paths[0]) + " -> " + nd.op +
                   "' was not resolved (missing inputs or cycle)";
    }
    return nullptr;
  }

  if (error_out) error_out->clear();
  return compound;
}

}  // namespace simpatico
