// SPDX-License-Identifier: Apache-2.0
//
// Beam-search compression explorer for simpatico_codegen.
//
// Uses:
//   • `simpatico::make_compressor` for non-fused ops
//   • `simpatico::compress_column` + `simpatico::decompress_column` for
//     fused ops (delta / rle / bitpack / for / zigzag) and for the rerank pass.

#include "explore/compression_explorer.hpp"

#include "codegen/plan/bitjoin_layout.hpp"  // copy_column_view
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"
#include "explore/operator_catalog.hpp"

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace simpatico {
namespace {

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
//
// The operator catalog (`all_compressor_names`), the terminal/preprocessing
// classification, `format_output_names`, and the single-op `try_operator`
// primitive live in explore/operator_catalog.{hpp,cpp} so this explorer and the
// operator-sweep test share one source of truth for operator applicability and
// channel production.

template <typename Fn>
double time_cuda_ms(cudaStream_t stream, Fn&& fn)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  fn();
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(ms);
}

inline bool weights_need_timing(double const w[3]) { return w[1] != 0.0 || w[2] != 0.0; }

// Relative decode cost per codec (beam-retention proxy), accumulated as cost x bytes processed.
inline double op_decode_cost(std::string_view op)
{
  if (op == "identity" || op == "str_split") return 0.15;
  if (op == "deflate" || op == "lz4" || op == "snappy" || op == "cascaded") return 5.0;
  if (op == "dictionary") return 3.0;
  if (op == "ans" || op == "bitcomp" || op == "alp" || op == "alp_rd") return 2.0;
  return 1.0;  // bitpack, rle, delta, zigzag, for, bitextract — fast
}

inline double weighted_score(double ratio, double comp, double decomp, double const w[3])
{
  return std::pow(std::max(ratio, 1e-12), w[0]) * std::pow(std::max(comp, 1e-12), w[1]) *
         std::pow(std::max(decomp, 1e-12), w[2]);
}

// ---------------------------------------------------------------------------
// BFS state types
// ---------------------------------------------------------------------------

struct pending_item {
  std::string path;
  cudf::column_view column;
  size_t size_bytes;
};

struct bfs_step_record {
  std::string dsl_line;
  std::string primary_input_path;
  std::vector<std::string> output_paths;
  size_t input_size;
  size_t output_size_total;
  bool shrank() const { return output_size_total < input_size; }
};

/// A compression candidate in the BFS beam.
struct bfs_candidate {
  std::string dsl;
  double compression_ratio = 1.0;
  size_t total_leaf_bytes  = 0;
  double decode_cost       = 0.0;  // Σ per-op decode cost × bytes processed (speed proxy)

  std::vector<pending_item> pending;
  std::vector<pending_item> leaves;

  // Owned reps — keep column_views from named_channels() valid
  std::vector<std::shared_ptr<compressed_representation>> owned_reprs;

  std::vector<bfs_step_record> steps;
  std::unordered_map<std::string, size_t> path_size;
};

struct ranked_candidate {
  std::string plan_dsl;
  double compression_ratio;
  size_t compressed_size_bytes;
  double compress_throughput_gbps;
  double decompress_throughput_gbps;
};

// ---------------------------------------------------------------------------
// Round-trip timing for the rerank pass
// ---------------------------------------------------------------------------

// Sum of every stored leaf's compressed bytes — the same accounting
// `benchmark`'s `compound_compressed_bytes` uses, so the two never drift.
size_t compound_compressed_bytes(PlanTree const& tree, rmm::cuda_stream_view stream)
{
  size_t total = 0;
  for (auto const& node : tree.nodes) {
    if (node.rep) total += node.rep->compressed_size_bytes(stream);
    for (auto const& [path, rep] : node.channels) {
      if (rep) total += rep->compressed_size_bytes(stream);
    }
  }
  return total;
}

// One real (untimed) compress through the fused production path
// (`compress_column`) — the only way to learn a cascade's true compressed
// size. The BFS's own per-op `try_operator` estimate cannot substitute, and
// the gap is NOT a bug: bitpack is chunked (kChunkSize elems/chunk) and a
// fused region derives num_chunks = ceil(num_rows / kChunkSize) ONCE from the
// region's ROOT input row count, which every node in the region inherits. So
// a bitpack fused behind a row-count-changing op (rle/delta/for) chunks by the
// ORIGINAL row count — e.g. rle->bitpack over 6M rows gives bitpack
// ceil(6M/kChunkSize) chunks, one per original-row tile, each packing that
// tile's local run-values. The same bitpack run standalone (what
// `try_operator`/`compress_single_op` does) sees the already-materialized,
// reduced values array and chunks it ceil(num_values / kChunkSize) ways. Same
// values, different chunk partition => different per-chunk min/bit-width =>
// different packed bytes (both roundtrip correctly; the fused layout is what
// production stores). Chains of 2+ contiguous codegen ops can therefore have a
// materially different true size than the BFS's unfused per-op sum suggests.
bool measure_compressed_bytes(cudf::column_view input,
                              std::string_view plan_dsl,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr,
                              size_t& compressed_bytes_out,
                              std::string* err_out)
{
  std::string err;
  auto compound = compress_column(input, plan_dsl, stream, mr, &err);
  if (!compound) {
    if (err_out) *err_out = "compress_column: " + err;
    return false;
  }
  compressed_bytes_out = compound_compressed_bytes(*compound, stream);
  return true;
}

bool round_trip_time_rr(cudf::column_view input,
                        std::string_view plan_dsl,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr,
                        double& compress_ms_out,
                        double& decompress_ms_out,
                        size_t& compressed_bytes_out,
                        std::string* err_out)
{
  std::unique_ptr<PlanTree> compound;
  std::string err;
  compress_ms_out = time_cuda_ms(
    stream.value(), [&] { compound = compress_column(input, plan_dsl, stream, mr, &err); });
  if (!compound) {
    if (err_out) *err_out = "compress_column: " + err;
    return false;
  }
  compressed_bytes_out = compound_compressed_bytes(*compound, stream);

  std::unique_ptr<cudf::column> decompressed;
  decompress_ms_out = time_cuda_ms(
    stream.value(), [&] { decompressed = decompress_column(*compound, stream, mr, &err); });
  if (!decompressed) {
    if (err_out) *err_out = "decompress_column: " + err;
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Post-BFS dead-step trimming (verbatim from cudf-sys explorer)
// ---------------------------------------------------------------------------

void trim_trailing_dead_steps(bfs_candidate& c)
{
  bool changed = true;
  while (changed) {
    changed = false;
    std::unordered_set<std::string> consumed;
    for (auto const& s : c.steps)
      consumed.insert(s.primary_input_path);
    for (size_t i = c.steps.size(); i-- > 0;) {
      auto const& s = c.steps[i];
      bool is_dead  = true;
      for (auto const& op : s.output_paths)
        if (consumed.count(op)) {
          is_dead = false;
          break;
        }
      if (is_dead && !s.shrank()) {
        c.steps.erase(c.steps.begin() + (std::ptrdiff_t)i);
        changed = true;
        break;
      }
    }
  }
  // Rebuild DSL
  std::ostringstream os;
  for (size_t i = 0; i < c.steps.size(); ++i) {
    if (i > 0) os << "\n";
    os << c.steps[i].dsl_line;
  }
  c.dsl = os.str();
  // Recompute total_leaf_bytes
  std::unordered_set<std::string> consumed_paths, produced_paths;
  for (auto const& s : c.steps) {
    consumed_paths.insert(s.primary_input_path);
    for (auto const& op : s.output_paths)
      produced_paths.insert(op);
  }
  size_t total = 0;
  auto add     = [&](std::string const& p) {
    auto it = c.path_size.find(p);
    if (it != c.path_size.end()) total += it->second;
  };
  if (!consumed_paths.count("input")) add("input");
  for (auto const& p : produced_paths)
    if (!consumed_paths.count(p)) add(p);
  c.total_leaf_bytes  = total;
  auto oit            = c.path_size.find("input");
  size_t orig         = oit != c.path_size.end() ? oit->second : 0;
  c.compression_ratio = (double)orig / std::max<size_t>(total, 1);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

size_t column_size_bytes_ex(cudf::column_view const& col, rmm::cuda_stream_view stream)
{
  if (col.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view scv(col);
    return (size_t)(col.size() + 1) * sizeof(int32_t) + scv.chars_size(stream);
  }
  return (size_t)col.size() * cudf::size_of(col.type());
}

// ---------------------------------------------------------------------------
// BFS Explorer
// ---------------------------------------------------------------------------

// Beam retention: reserve half the slots for top-ratio candidates (so a pure-
// ratio search is never starved of its best subtree), and fill the rest from the
// (ratio, estimated decode speed) Pareto frontier of the remainder — thinned by
// NSGA-II crowding distance when it overflows, since dictionary spawns many
// near-identical frontier points and a ratio trim would starve the fast/low-
// ratio end (e.g. str_split before its channels are compressed). Dropped
// candidates free their GPU reprs on return.
void prune_beam(std::vector<std::unique_ptr<bfs_candidate>>& beam,
                size_t cap,
                size_t original_size)
{
  size_t const n = beam.size();
  if (n <= cap) return;
  std::vector<double> r(n), s(n);
  for (size_t i = 0; i < n; ++i) {
    r[i] = beam[i]->compression_ratio;
    s[i] = (double)original_size / std::max(beam[i]->decode_cost, 1.0);
  }

  std::vector<size_t> idx(n);
  std::iota(idx.begin(), idx.end(), size_t{0});
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return r[a] > r[b]; });

  size_t const ratio_slots = cap / 2;
  std::vector<size_t> keep(idx.begin(), idx.begin() + ratio_slots);
  std::vector<size_t> pool(idx.begin() + ratio_slots, idx.end());
  size_t const want = cap - keep.size();

  std::vector<char> dominated(n, 0);
  for (size_t i : pool)
    for (size_t j : pool)
      if (j != i && r[j] >= r[i] && s[j] >= s[i] && (r[j] > r[i] || s[j] > s[i])) {
        dominated[i] = 1;
        break;
      }
  std::vector<size_t> front, rest;
  for (size_t i : pool) (dominated[i] ? rest : front).push_back(i);

  if (front.size() <= want) {
    keep.insert(keep.end(), front.begin(), front.end());
    for (size_t i = 0; i < rest.size() && keep.size() < cap; ++i)
      keep.push_back(rest[i]);  // rest is already ratio-ordered via idx
  } else {
    // Thin an over-large frontier by crowding distance (boundary points = inf).
    std::vector<double> crowd(n, 0.0);
    for (double const* obj : {r.data(), s.data()}) {
      auto by_obj = front;
      std::sort(by_obj.begin(), by_obj.end(), [&](size_t a, size_t b) { return obj[a] < obj[b]; });
      double span = std::max(obj[by_obj.back()] - obj[by_obj.front()], 1e-12);
      crowd[by_obj.front()] = crowd[by_obj.back()] = std::numeric_limits<double>::infinity();
      for (size_t k = 1; k + 1 < by_obj.size(); ++k)
        crowd[by_obj[k]] += (obj[by_obj[k + 1]] - obj[by_obj[k - 1]]) / span;
    }
    std::sort(front.begin(), front.end(), [&](size_t a, size_t b) { return crowd[a] > crowd[b]; });
    keep.insert(keep.end(), front.begin(), front.begin() + want);
  }

  std::vector<std::unique_ptr<bfs_candidate>> kept;
  kept.reserve(keep.size());
  for (size_t i : keep) kept.push_back(std::move(beam[i]));
  beam.swap(kept);  // old beam (with the dropped tail) freed on return
}

exploration_result explore_column_compression(cudf::column_view input,
                                              exploration_config const& config,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  size_t const full_size = column_size_bytes_ex(input, stream);

  // Head-prefix of `input` (rows). Strings are materialized — a zero-copy slice
  // still views the parent's full offsets/chars, which the byte codecs reject.
  auto prefix = [&](cudf::size_type rows,
                    std::unique_ptr<cudf::column>& owned) -> cudf::column_view {
    auto head = cudf::slice(input, {0, rows}).front();
    if (input.type().id() != cudf::type_id::STRING) return head;
    owned = copy_column_view(head, stream, mr);
    return owned->view();
  };

  // max_explore_bytes caps the measurement column too, so a huge column never
  // reaches compress at full size; the reported ratio is measured on this view.
  std::unique_ptr<cudf::column> measure_owned;
  cudf::column_view measure_col = input;
  if (config.max_explore_bytes > 0 && full_size > config.max_explore_bytes && input.size() > 1) {
    auto const cap_rows = std::max<cudf::size_type>(
      1,
      static_cast<cudf::size_type>(static_cast<double>(input.size()) *
                                   static_cast<double>(config.max_explore_bytes) / full_size));
    if (cap_rows < input.size()) measure_col = prefix(cap_rows, measure_owned);
  }
  size_t const measure_size = column_size_bytes_ex(measure_col, stream);

  // The throwaway BFS ranking may run on an even smaller prefix (sample_rows).
  std::unique_ptr<cudf::column> bfs_owned;
  cudf::column_view bfs_input = measure_col;
  if (config.sample_rows > 0 && config.sample_rows < static_cast<size_t>(measure_col.size()))
    bfs_input = prefix(static_cast<cudf::size_type>(config.sample_rows), bfs_owned);
  size_t original_size  = column_size_bytes_ex(bfs_input, stream);
  size_t max_candidates = config.beam_width > 0 ? config.beam_width : 100;

  if (config.verbose)
    std::cerr << "[explore] Input size: " << original_size
              << " bytes, max_candidates=" << max_candidates << ", max_depth=" << config.max_depth
              << "\n";

  std::string best_dsl;
  double best_ratio      = 1.0;
  size_t best_leaf_bytes = original_size;
  size_t best_depth      = 0;

  // Completed plans (pending.empty()) at any depth get evicted from the beam
  // once deeper cascades with better ratios arrive. Save the best ones here so
  // they always reach the timed rerank pass, giving throughput-efficient
  // lighter-weight plans a fair shot against deep but slow cascades.
  struct shallow_candidate {
    std::string dsl;
    double compression_ratio;
    size_t total_leaf_bytes;
    size_t num_steps;
  };
  std::vector<shallow_candidate> saved_shallow;

  auto const& all_ops = all_compressor_names();

  // Initial beam
  std::vector<std::unique_ptr<bfs_candidate>> beam;
  {
    auto init               = std::make_unique<bfs_candidate>();
    init->dsl               = "";
    init->compression_ratio = 1.0;
    init->total_leaf_bytes  = original_size;
    init->pending.push_back({"input", bfs_input, original_size});
    init->path_size.emplace("input", original_size);
    beam.push_back(std::move(init));
  }

  for (size_t depth = 0; depth < config.max_depth; ++depth) {
    if (config.verbose)
      std::cerr << "[explore] Depth " << depth << ": " << beam.size()
                << " candidates, best=" << best_ratio << "\n";

    // Children buffer, pruned back to max_candidates at 2x so losers are freed
    // as the depth progresses rather than all held to the end.
    std::vector<std::unique_ptr<bfs_candidate>> next_beam;

    for (auto& candidate : beam) {
      if (candidate->pending.empty()) {
        if (candidate->compression_ratio > best_ratio) {
          best_ratio      = candidate->compression_ratio;
          best_dsl        = candidate->dsl;
          best_leaf_bytes = candidate->total_leaf_bytes;
          best_depth      = depth;
        }
        continue;
      }

      for (size_t pi = 0; pi < candidate->pending.size(); ++pi) {
        auto const& pend = candidate->pending[pi];

        for (auto const& op_name : all_ops) {
          auto trial = try_operator(op_name, pend.column, stream, mr);
          if (!trial.success) continue;

          if (config.verbose && depth == 0) {
            double tr = (double)pend.size_bytes / std::max<size_t>(trial.output_bytes, 1);
            std::cerr << "[explore]   " << op_name << ": " << trial.output_bytes << "/"
                      << pend.size_bytes << " (" << std::fixed << std::setprecision(3) << tr
                      << "x)\n";
          }

          bool is_pre = is_preprocessing_compressor(op_name);
          if (trial.output_bytes >= pend.size_bytes && !is_pre) continue;

          auto child = std::make_unique<bfs_candidate>();

          auto output_names = format_output_names(trial.outputs);
          std::ostringstream dsl_line;
          dsl_line << pend.path << " -> " << op_name;
          if (!output_names.empty() && !is_terminal_compressor(op_name))
            dsl_line << " -> " << output_names;

          child->dsl =
            candidate->dsl.empty() ? dsl_line.str() : candidate->dsl + "\n" + dsl_line.str();
          child->steps     = candidate->steps;
          child->path_size = candidate->path_size;

          for (size_t j = 0; j < candidate->pending.size(); ++j)
            if (j != pi) child->pending.push_back(candidate->pending[j]);
          child->leaves = candidate->leaves;

          bool is_term = is_terminal_compressor(op_name);
          std::vector<std::string> step_out_paths;
          size_t step_out_total = 0;
          for (auto const& out : trial.outputs) {
            std::string out_path =
              (pend.path == "input") ? op_name + "." + out.name : pend.path + "." + out.name;
            size_t out_sz = column_size_bytes_ex(out.view, stream);
            pending_item po{out_path, out.view, out_sz};
            if (is_term)
              child->leaves.push_back(po);
            else
              child->pending.push_back(po);
            child->path_size[out_path] = out_sz;
            step_out_paths.push_back(out_path);
            step_out_total += out_sz;
          }
          child->steps.push_back({dsl_line.str(),
                                  pend.path,
                                  std::move(step_out_paths),
                                  pend.size_bytes,
                                  step_out_total});

          if (trial.repr) child->owned_reprs.push_back(trial.repr);
          for (auto const& r : candidate->owned_reprs)
            child->owned_reprs.push_back(r);

          child->total_leaf_bytes = 0;
          for (auto const& l : child->leaves)
            child->total_leaf_bytes += l.size_bytes;
          for (auto const& p : child->pending)
            child->total_leaf_bytes += p.size_bytes;
          child->compression_ratio = (double)original_size / child->total_leaf_bytes;
          child->decode_cost =
            candidate->decode_cost + op_decode_cost(op_name) * (double)pend.size_bytes;

          if (child->compression_ratio > best_ratio) {
            best_ratio      = child->compression_ratio;
            best_dsl        = child->dsl;
            best_leaf_bytes = child->total_leaf_bytes;
            best_depth      = depth + 1;
          }
          // Every BFS candidate is a valid complete plan: any uncovered sub-columns
          // are implicitly stored as identity.  Save all candidates (terminal or
          // not) so that simple, fast plans (e.g. bare bitpack) reach the rerank
          // pass even when deep cascades beat them on ratio.
          if (saved_shallow.size() < max_candidates)
            saved_shallow.push_back(
              {child->dsl, child->compression_ratio, child->total_leaf_bytes, child->steps.size()});
          // Buffer children and prune to the hybrid (top-ratio + Pareto
          // frontier) beam at 2x cap, so low-ratio-but-fast branches (e.g. a
          // bare str_split) aren't evicted before their channels are compressed.
          next_beam.push_back(std::move(child));
          if (next_beam.size() >= 2 * max_candidates)
            prune_beam(next_beam, max_candidates, original_size);
        }
      }
    }

    if (next_beam.empty()) break;

    prune_beam(next_beam, max_candidates, original_size);

    std::sort(next_beam.begin(), next_beam.end(), [](auto const& a, auto const& b) {
      return a->compression_ratio > b->compression_ratio;
    });

    cudaStreamSynchronize(stream.value());
    beam.clear();
    beam.swap(next_beam);
    cudaStreamSynchronize(stream.value());

    bool has_pending = false;
    for (auto const& c2 : beam)
      if (!c2->pending.empty()) {
        has_pending = true;
        break;
      }
    if (!has_pending) break;
  }

  // Trim dead steps
  for (auto& c2 : beam)
    if (c2) trim_trailing_dead_steps(*c2);
  for (auto& c2 : beam) {
    if (c2 && c2->compression_ratio > best_ratio) {
      best_ratio      = c2->compression_ratio;
      best_dsl        = c2->dsl;
      best_leaf_bytes = c2->total_leaf_bytes;
    }
  }

  // Collect finalists
  struct ranked_candidate {
    std::string plan_dsl;
    double compression_ratio;
    size_t compressed_size_bytes;
    double compress_throughput_gbps;
    double decompress_throughput_gbps;
    double decode_cost = 0.0;  // proxy for pareto_beam finalist selection
  };
  std::vector<ranked_candidate> finalists;
  auto add_finalist = [&](std::string dsl, double ratio, size_t bytes, double dc) {
    if (dsl.empty()) return;
    for (auto const& f : finalists)
      if (f.plan_dsl == dsl) return;
    finalists.push_back({std::move(dsl), ratio, bytes, 0.0, 0.0, dc});
  };
  // Beam first so the real decode_cost is recorded; best_dsl/identity dedup to it.
  for (auto& c2 : beam)
    if (c2) add_finalist(c2->dsl, c2->compression_ratio, c2->total_leaf_bytes, c2->decode_cost);
  if (!best_dsl.empty()) add_finalist(best_dsl, best_ratio, best_leaf_bytes, 0.0);
  if (best_ratio <= 1.0 || config.rerank_mode == score_mode::Pareto)
    add_finalist("input -> identity", 1.0, original_size, 0.0);
  std::sort(finalists.begin(), finalists.end(), [](auto const& a, auto const& b) {
    return a.compression_ratio > b.compression_ratio;
  });

  cudaStreamSynchronize(stream.value());
  beam.clear();
  cudaDeviceSynchronize();

  size_t const rerank_top       = config.rerank_top > 0 ? config.rerank_top : 8;
  size_t const simplicity_slots = config.simplicity_slots;

  // Trim to rerank_top: half the slots go to the top-ratio finalists (so the
  // ratio-best plan is always timed), the rest to a (ratio, proxy-speed)
  // crowding-diverse subset so throughput-friendly mid-ratio plans (e.g.
  // str_split cascades) get timed too. Identity is rescued if evicted.
  if (finalists.size() > rerank_top) {
    size_t const m          = finalists.size();
    size_t const keep_ratio = std::max<size_t>(1, rerank_top / 2);
    // finalists are ratio-sorted: reserve the head, diversify over the tail.
    std::vector<ranked_candidate> kept(finalists.begin(), finalists.begin() + keep_ratio);
    std::vector<size_t> pool(m - keep_ratio);
    std::iota(pool.begin(), pool.end(), keep_ratio);

    std::vector<double> rr(m), ss(m), crowd(m, 0.0);
    for (size_t i = 0; i < m; ++i) {
      rr[i] = finalists[i].compression_ratio;
      ss[i] = finalists[i].decode_cost > 0.0
                ? (double)original_size / finalists[i].decode_cost
                : std::numeric_limits<double>::infinity();
    }
    for (double const* obj : {rr.data(), ss.data()}) {
      auto by = pool;
      std::sort(by.begin(), by.end(), [&](size_t a, size_t b) { return obj[a] < obj[b]; });
      double span = std::max(obj[by.back()] - obj[by.front()], 1e-12);
      crowd[by.front()] = crowd[by.back()] = std::numeric_limits<double>::infinity();
      for (size_t k = 1; k + 1 < by.size(); ++k)
        crowd[by[k]] += (obj[by[k + 1]] - obj[by[k - 1]]) / span;
    }
    std::sort(pool.begin(), pool.end(), [&](size_t a, size_t b) { return crowd[a] > crowd[b]; });
    for (size_t k = 0; k < pool.size() && kept.size() < rerank_top; ++k)
      kept.push_back(finalists[pool[k]]);

    auto is_id = [](ranked_candidate const& f) { return f.plan_dsl == "input -> identity"; };
    auto id_it = std::find_if(finalists.begin(), finalists.end(), is_id);
    if (id_it != finalists.end() && std::none_of(kept.begin(), kept.end(), is_id))
      kept.back() = *id_it;
    finalists = std::move(kept);
  }

  // Inject the top simplicity_slots completed plans per step-count level so
  // every completion horizon gets fair representation in the timed rerank pass.
  // Completed plans are evicted from the beam once deeper cascades arrive —
  // saved_shallow preserves them across all depths.
  if (simplicity_slots > 0 && !saved_shallow.empty()) {
    // Sort by (step_count ASC, ratio DESC) so we iterate depth-by-depth,
    // best-ratio-first within each depth.
    std::sort(saved_shallow.begin(), saved_shallow.end(), [](auto const& a, auto const& b) {
      return a.num_steps != b.num_steps ? a.num_steps < b.num_steps
                                        : a.compression_ratio > b.compression_ratio;
    });
    size_t per_depth_count = 0;
    size_t cur_steps       = 0;
    bool first             = true;
    for (auto const& s : saved_shallow) {
      if (first || s.num_steps != cur_steps) {
        cur_steps       = s.num_steps;
        per_depth_count = 0;
        first           = false;
      }
      if (per_depth_count >= simplicity_slots) continue;
      bool already = std::any_of(
        finalists.begin(), finalists.end(), [&](auto const& f) { return f.plan_dsl == s.dsl; });
      if (!already) {
        finalists.push_back({s.dsl, s.compression_ratio, s.total_leaf_bytes, 0.0, 0.0});
        ++per_depth_count;
      }
    }
  }

  if (config.verbose) {
    std::cerr << "[explore] Rerank pool (" << finalists.size() << " candidates):\n";
    for (auto const& f : finalists)
      std::cerr << "[explore]   ratio=" << std::fixed << std::setprecision(3) << f.compression_ratio
                << "  steps=" << std::count(f.plan_dsl.begin(), f.plan_dsl.end(), '\n') + 1 << "  "
                << f.plan_dsl.substr(0, f.plan_dsl.find('\n')) << "\n";
  }

  bool use_pareto      = config.rerank_mode == score_mode::Pareto;
  bool need_throughput = use_pareto || weights_need_timing(config.rerank_weights);

  // Always re-measure every finalist's compressed size through the real
  // fused `compress_column` path — never report the BFS's own unfused
  // byte estimate, which can be off by several x for cascades chaining 2+
  // codegen ops (see measure_compressed_bytes). Measured on `measure_col`
  // (the full column, or its byte-capped prefix).
  if (!finalists.empty() && !need_throughput) {
    double const orig = (double)measure_size;
    std::vector<ranked_candidate> sized;
    for (auto const& f : finalists) {
      size_t bytes = f.compressed_size_bytes;
      if (!measure_compressed_bytes(measure_col, f.plan_dsl, stream, mr, bytes, nullptr)) continue;
      sized.push_back({f.plan_dsl, orig / std::max<size_t>(bytes, 1), bytes, 0.0, 0.0});
    }
    cudaDeviceSynchronize();
    if (!sized.empty()) finalists = std::move(sized);
  } else if (!finalists.empty()) {
    {
      double a, b;
      size_t c2;
      round_trip_time_rr(measure_col, finalists.front().plan_dsl, stream, mr, a, b, c2, nullptr);
      cudaDeviceSynchronize();
    }

    static constexpr size_t kPasses = 3;
    struct acc_t {
      std::string plan_dsl;
      double best_comp_ms   = std::numeric_limits<double>::infinity();
      double best_decomp_ms = std::numeric_limits<double>::infinity();
      size_t bytes          = 0;
      bool any_ok           = false;
    };
    std::vector<acc_t> accs;
    for (auto const& f : finalists)
      accs.push_back({f.plan_dsl,
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      f.compressed_size_bytes,
                      false});

    for (size_t pass = 0; pass < kPasses; ++pass) {
      for (size_t i = 0; i < finalists.size(); ++i) {
        double comp_ms = 0, decomp_ms = 0;
        size_t bytes = accs[i].bytes;
        std::string err;
        bool ok = round_trip_time_rr(
          measure_col, finalists[i].plan_dsl, stream, mr, comp_ms, decomp_ms, bytes, &err);
        cudaDeviceSynchronize();
        if (!ok) continue;
        accs[i].any_ok = true;
        accs[i].bytes  = bytes;
        if (comp_ms < accs[i].best_comp_ms) accs[i].best_comp_ms = comp_ms;
        if (decomp_ms < accs[i].best_decomp_ms) accs[i].best_decomp_ms = decomp_ms;
      }
    }
    double const orig = (double)measure_size;
    std::vector<ranked_candidate> timed;
    for (auto const& a : accs) {
      if (!a.any_ok) continue;
      timed.push_back({a.plan_dsl,
                       orig / std::max<size_t>(a.bytes, 1),
                       a.bytes,
                       a.best_comp_ms > 0 ? orig / a.best_comp_ms / 1.0e6 : 0.0,
                       a.best_decomp_ms > 0 ? orig / a.best_decomp_ms / 1.0e6 : 0.0});
    }
    if (!timed.empty()) finalists = std::move(timed);
  }

  // Local refinement: hill-climb around the measured finalists. The beam lands
  // near an optimum but not necessarily on it (greedy expansion + a crude decode
  // proxy), so measure one-op neighbors of each finalist — swap a terminal leaf's
  // codec, or drop a terminal leaf line (that channel is then stored identity) —
  // and let the final selection pick across seeds and neighbors alike.
  if (!finalists.empty()) {
    static constexpr size_t kMaxNeighborMeasurements = 24;
    static constexpr std::string_view kSwapOps[]     = {
      "ans", "bitcomp", "lz4", "snappy", "deflate"};

    auto measure_dsl = [&](std::string const& dsl, ranked_candidate& out) -> bool {
      size_t bytes      = 0;
      double const orig = (double)measure_size;
      if (need_throughput) {
        double cms = 0, dms = 0;
        if (!round_trip_time_rr(measure_col, dsl, stream, mr, cms, dms, bytes, nullptr))
          return false;
        out = {dsl,
               orig / std::max<size_t>(bytes, 1),
               bytes,
               cms > 0 ? orig / cms / 1.0e6 : 0.0,
               dms > 0 ? orig / dms / 1.0e6 : 0.0};
      } else {
        if (!measure_compressed_bytes(measure_col, dsl, stream, mr, bytes, nullptr)) return false;
        out = {dsl, orig / std::max<size_t>(bytes, 1), bytes, 0.0, 0.0};
      }
      return true;
    };
    auto known = [&](std::string const& dsl) {
      return std::any_of(
        finalists.begin(), finalists.end(), [&](auto const& f) { return f.plan_dsl == dsl; });
    };

    // Refine best-scoring seeds first so the measurement budget goes where it counts.
    auto seeds = finalists;
    if (!use_pareto)
      std::sort(seeds.begin(), seeds.end(), [&](auto const& a, auto const& b) {
        return weighted_score(a.compression_ratio,
                              a.compress_throughput_gbps,
                              a.decompress_throughput_gbps,
                              config.rerank_weights) > weighted_score(b.compression_ratio,
                                                                      b.compress_throughput_gbps,
                                                                      b.decompress_throughput_gbps,
                                                                      config.rerank_weights);
      });

    // Split the budget across seeds so one many-leaved plan can't starve the rest.
    size_t const per_seed =
      std::max<size_t>(4, kMaxNeighborMeasurements / std::max<size_t>(seeds.size(), 1));
    size_t measured = 0, seed_left = 0;
    auto try_dsl = [&](std::string const& dsl) {
      if (dsl.empty() || seed_left == 0 || measured >= kMaxNeighborMeasurements || known(dsl))
        return;
      --seed_left;
      ++measured;
      ranked_candidate rc;
      bool ok = false;
      try {  // a neighbor may pair a codec with an incompatible channel; skip it
        ok = measure_dsl(dsl, rc);
      } catch (std::exception const&) {
      }
      if (ok) finalists.push_back(std::move(rc));
    };

    for (auto const& seed : seeds) {
      if (measured >= kMaxNeighborMeasurements) break;
      if (seed.plan_dsl == "input -> identity") continue;
      seed_left = per_seed;
      std::vector<std::string> lines;
      {
        std::istringstream is(seed.plan_dsl);
        for (std::string l; std::getline(is, l);)
          if (!l.empty()) lines.push_back(l);
      }

      // Parse structure: LHS paths and every declared child channel.
      std::vector<std::string> lhs_paths, children;
      for (auto const& line : lines) {
        auto a1 = line.find(" -> ");
        if (a1 == std::string::npos) continue;
        std::string lhs = line.substr(0, a1);
        lhs_paths.push_back(lhs);
        std::string tail = line.substr(a1 + 4);
        auto a2          = tail.find(" -> ");
        if (a2 == std::string::npos) continue;
        std::string op            = tail.substr(0, a2);
        std::string outs          = tail.substr(a2 + 4);
        std::string const& prefix = (lhs == "input") ? op : lhs;
        for (size_t pos = 0; pos < outs.size();) {
          auto comma      = outs.find(", ", pos);
          std::string out = outs.substr(pos, comma == std::string::npos ? comma : comma - pos);
          children.push_back(prefix + "." + out);
          pos = comma == std::string::npos ? outs.size() : comma + 2;
        }
      }

      // Move 1 — cover an uncovered channel with a terminal codec (an implicit
      // identity leaf may be the plan's biggest remaining win, e.g. raw chars).
      for (auto const& ch : children) {
        if (std::find(lhs_paths.begin(), lhs_paths.end(), ch) != lhs_paths.end()) continue;
        for (auto alt : kSwapOps)
          try_dsl(seed.plan_dsl + "\n" + ch + " -> " + std::string(alt));
      }

      // Moves 2 & 3 — swap a terminal leaf's codec / drop a terminal leaf line
      // (its channel is then stored identity).
      for (size_t li = 0; li < lines.size() && measured < kMaxNeighborMeasurements; ++li) {
        auto arrow     = lines[li].find(" -> ");
        std::string op = lines[li].substr(arrow + 4);
        if (op.find(" -> ") != std::string::npos) continue;  // has outputs: not a leaf
        if (!is_terminal_compressor(op)) continue;

        auto rebuild = [&](std::string const& replacement_line) {
          std::string dsl;
          for (size_t j = 0; j < lines.size(); ++j) {
            if (j == li && replacement_line.empty()) continue;  // drop the line
            dsl += (dsl.empty() ? "" : "\n") + (j == li ? replacement_line : lines[j]);
          }
          return dsl;
        };
        for (auto alt : kSwapOps)
          if (alt != op) try_dsl(rebuild(lines[li].substr(0, arrow + 4) + std::string(alt)));
        if (lines.size() > 1) try_dsl(rebuild(""));
      }
    }
    cudaDeviceSynchronize();
  }

  // Pareto frontier
  constexpr double kEps = 0.01;
  auto ge_eps           = [](double a, double b) { return a >= b * (1.0 - kEps); };
  auto gt_eps           = [](double a, double b) { return a > b * (1.0 + kEps); };
  auto near_eq          = [](double a, double b) {
    double d = std::max(std::abs(a), std::abs(b));
    return d == 0.0 || std::abs(a - b) <= d * kEps;
  };
  std::vector<ranked_candidate> frontier;
  if (use_pareto && need_throughput) {
    for (auto const& c2 : finalists) {
      bool dominated = false;
      for (auto const& o : finalists) {
        if (&o == &c2) continue;
        bool ge = ge_eps(o.compression_ratio, c2.compression_ratio) &&
                  ge_eps(o.compress_throughput_gbps, c2.compress_throughput_gbps) &&
                  ge_eps(o.decompress_throughput_gbps, c2.decompress_throughput_gbps);
        bool gt = gt_eps(o.compression_ratio, c2.compression_ratio) ||
                  gt_eps(o.compress_throughput_gbps, c2.compress_throughput_gbps) ||
                  gt_eps(o.decompress_throughput_gbps, c2.decompress_throughput_gbps);
        if (ge && gt) {
          dominated = true;
          break;
        }
      }
      if (dominated) continue;
      bool dup = false;
      for (auto const& f : frontier) {
        if (near_eq(f.compression_ratio, c2.compression_ratio) &&
            near_eq(f.compress_throughput_gbps, c2.compress_throughput_gbps) &&
            near_eq(f.decompress_throughput_gbps, c2.decompress_throughput_gbps)) {
          dup = true;
          break;
        }
      }
      if (!dup) frontier.push_back(c2);
    }
  }
  auto& pool = (use_pareto && need_throughput) ? frontier : finalists;

  cudaDeviceSynchronize();

  exploration_result result;
  result.original_size_bytes = measure_size;  // ratio/compressed measured on measure_col
  result.cascade_depth       = best_depth;

  if (pool.empty()) {
    result.plan_dsl              = "input -> identity";
    result.compression_ratio     = 1.0;
    result.compressed_size_bytes = original_size;
  } else {
    auto pick_score = [&](ranked_candidate const& c2) -> double {
      if (use_pareto && need_throughput) return c2.compression_ratio;
      return weighted_score(c2.compression_ratio,
                            c2.compress_throughput_gbps,
                            c2.decompress_throughput_gbps,
                            config.rerank_weights);
    };
    auto best_it = std::max_element(
      pool.begin(), pool.end(), [&](ranked_candidate const& a, ranked_candidate const& b) {
        return pick_score(a) < pick_score(b);
      });
    result.plan_dsl                   = best_it->plan_dsl;
    result.compression_ratio          = best_it->compression_ratio;
    result.compressed_size_bytes      = best_it->compressed_size_bytes;
    result.compress_throughput_gbps   = best_it->compress_throughput_gbps;
    result.decompress_throughput_gbps = best_it->decompress_throughput_gbps;
    // Derive depth from the actual winning plan's line count, not from the
    // BFS ratio-tracking variable (which reflects when the beam last improved).
    result.cascade_depth =
      static_cast<size_t>(std::count(result.plan_dsl.begin(), result.plan_dsl.end(), '\n') + 1);
    if (use_pareto && need_throughput) {
      std::ostringstream os;
      size_t alt_n = 0;
      for (auto const& c2 : frontier)
        if (c2.plan_dsl != result.plan_dsl) ++alt_n;
      if (alt_n > 0) {
        os << "\n=== Pareto alternates (" << alt_n << ") ===\n";
        size_t idx = 0;
        for (auto const& c2 : frontier) {
          if (c2.plan_dsl == result.plan_dsl) continue;
          os << "--- alt " << idx++ << " ---\n"
             << "  ratio=" << std::fixed << std::setprecision(2) << c2.compression_ratio
             << "x comp=" << c2.compress_throughput_gbps
             << " GB/s decomp=" << c2.decompress_throughput_gbps << " GB/s\n";
          std::istringstream is(c2.plan_dsl);
          for (std::string line; std::getline(is, line);)
            os << "  " << line << "\n";
        }
        result.pareto_alternates_summary = os.str();
      }
    }
  }

  if (config.verbose)
    std::cerr << "[explore] Final: ratio=" << result.compression_ratio
              << " depth=" << result.cascade_depth << "\nDSL:\n"
              << result.plan_dsl << "\n";

  // The BFS just tried every nvcomp-backed op (ans/bitcomp/cascaded/
  // snappy/lz4/deflate) against this column and its intermediates, which
  // grows each codec's thread-local Manager scratch to a high-water mark
  // sized for the largest of those calls — memory nvcomp never shrinks
  // back down on its own. Release it now so it doesn't sit pinned while
  // exploring whatever column comes next.

  return result;
}

}  // namespace simpatico
