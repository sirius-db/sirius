/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "creator/task_creator.hpp"
#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/order/bounded_sort.hpp"
#include "op/order/gpu_sampled_merge.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cudf/copying.hpp>
#include <cudf/merge.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sirius::op {
namespace sm = sampled_merge;
namespace {
using batches                  = sirius_physical_merge_sort::batches;
constexpr uint64_t native_rows = std::numeric_limits<cudf::size_type>::max();
// Each range owns an equal share of this logical position space; refinement divides a
// node's share among its children. Consumers accept sparse positions.
constexpr std::size_t merge_position_space = std::size_t{1} << 62;

struct contribution_size {
  uint64_t rows;
  uint64_t bytes;
};

std::vector<contribution_size> sizes(batches const& values)
{
  std::vector<contribution_size> result;
  for (auto const& value : values) {
    auto info = get_batch_rows_and_bytes(*value);
    if (!info.rows) { throw internal_exception("Sort batch lacks row metadata"); }
    result.push_back({*info.rows, info.bytes});
  }
  return result;
}

}  // namespace

std::size_t sirius_physical_merge_sort::no_history_peak_memory_estimate(
  input_stats const& stats) const
{
  return merge_cold_peak_estimate(stats.bytes);
}

sirius_physical_merge_sort::work::work(batches values,
                                       step kind,
                                       std::size_t node,
                                       std::vector<uint64_t> ordinals,
                                       uint64_t parameter,
                                       std::optional<std::size_t> position)
  : pipelineable_operator_data(std::move(values), position),
    kind(kind),
    node(node),
    ordinals(std::move(ordinals)),
    parameter(parameter)
{
}
std::unique_ptr<pipelineable_operator_data>
sirius_physical_merge_sort::work::with_batches_preserving_rows(batches values) const
{
  auto copy =
    std::make_unique<work>(std::move(values), kind, node, ordinals, parameter, get_position());
  copy->adopt_row_identity(*this);
  return copy;
}
sirius_physical_merge_sort::result::result(batches values,
                                           step kind,
                                           std::size_t node,
                                           std::vector<uint64_t> ordinals,
                                           std::vector<piece> pieces,
                                           std::optional<std::size_t> position)
  : pipelineable_operator_data(std::move(values), position),
    kind(kind),
    node(node),
    ordinals(std::move(ordinals)),
    pieces(std::move(pieces))
{
}
std::unique_ptr<pipelineable_operator_data>
sirius_physical_merge_sort::result::with_batches_preserving_rows(batches values) const
{
  auto copy =
    std::make_unique<result>(std::move(values), kind, node, ordinals, pieces, get_position());
  copy->adopt_row_identity(*this);
  return copy;
}

void sirius_physical_merge_sort::register_refinement(
  cucascade::shared_data_repository_manager& manager)
{
  auto id = get_operator_id();
  for (auto [name, slot] : {std::pair{"sort_parts", &_parts},
                            std::pair{"sort_samples", &_samples},
                            std::pair{"sort_splitters", &_splitters}}) {
    manager.add_new_repository(id, name, std::make_unique<cucascade::shared_data_repository>());
    *slot = manager.get_repository(id, name).get();
  }
}

// Called once, under the lock, after every range contribution has arrived. Reads row and
// byte metadata only: nothing is restored to a device.
std::pair<std::size_t, std::size_t> sirius_physical_merge_sort::ingest()
{
  _ingested  = true;
  auto& repo = *get_port("default")->repo;
  if (repo.all_empty()) { return {repo.num_partitions(), 0}; }
  if (!_parts || !_samples || !_splitters) {
    throw internal_exception("Merge sort refinement repositories are not registered");
  }
  if (repo.get_ordering() == cucascade::data_repository_ordering::ORDERED) {
    throw internal_exception("Merge sort requires unordered range contributions");
  }
  auto ranges = repo.num_partitions();
  auto width  = merge_position_space / ranges;
  for (std::size_t range = 0; range < ranges; ++range) {
    node n{.lo = range * width, .hi = (range + 1) * width};
    uint64_t ordinal = 0;
    while (auto batch = repo.pop_next_data_batch(range)) {
      auto info = get_batch_rows_and_bytes(*batch);
      if (!info.rows) { throw internal_exception("Sort batch lacks row metadata"); }
      if (!*info.rows) { continue; }
      _parts->add_data_batch_at_position(batch, ordinal, _nodes.size());
      n.parts.push_back({ordinal++, *info.rows, info.bytes});
      n.rows += *info.rows;
      n.bytes += info.bytes;
    }
    if (n.parts.empty()) { continue; }
    n.state = phase::ready;
    _active.emplace(n.lo, _nodes.size());
    _nodes.push_back(std::move(n));
  }
  return {ranges, _nodes.size()};
}

bool sirius_physical_merge_sort::fits(node const& n) const
{
  return n.rows <= native_rows && (n.bytes <= chunk_bytes || n.rows == 1);
}

std::optional<std::size_t> sirius_physical_merge_sort::actionable() const
{
  for (auto [lo, id] : _active) {
    auto const& n = _nodes[id];
    switch (n.state) {
      case phase::ready:
      case phase::plan_ready: return id;
      case phase::sampling:
      case phase::cutting:
      case phase::coalescing:
        if (n.cursor < n.parts.size()) { return id; }
        break;
      default: break;
    }
  }
  return std::nullopt;
}

void sirius_physical_merge_sort::retire(std::size_t id)
{
  auto& n = _nodes[id];
  _active.erase({n.lo, id});
  n.state = phase::done;
  n.parts.clear();
  n.parts.shrink_to_fit();
}

// A group closes only when the next contribution would exceed chunk_bytes, so any two
// adjacent groups together exceed it. After coalescing, regrouping the runs therefore
// yields one run per group: sample and cut tasks each take a single run, and a further
// coalescing pass would not reduce k. This assumes a merged run reports the bytes of its
// inputs; cudf::merge can differ slightly (null masks, string offsets), which may let two
// runs share a group but cannot affect correctness or termination.
std::size_t sirius_physical_merge_sort::group_end(node const& n, std::size_t begin) const
{
  auto end       = begin + 1;
  uint64_t bytes = n.parts[begin].bytes;
  while (end < n.parts.size() && bytes + n.parts[end].bytes <= chunk_bytes) {
    bytes += n.parts[end++].bytes;
  }
  return end;
}

// One greedy group (see group_end). Sampling borrows payload, which stays queued for
// cutting; cutting and coalescing take ownership. Cutting also borrows the splitters.
std::unique_ptr<sirius_physical_merge_sort::work> sirius_physical_merge_sort::claim_group(
  std::size_t id, step kind, bool pop)
{
  auto& n  = _nodes[id];
  auto end = group_end(n, n.cursor);
  batches values;
  std::vector<uint64_t> ordinals;
  for (; n.cursor < end; ++n.cursor) {
    auto const& part = n.parts[n.cursor];
    auto batch       = pop ? _parts->pop_data_batch_by_position(part.ordinal, id)
                           : _parts->get_data_batch_by_position(part.ordinal, id);
    if (!batch) { throw internal_exception("Sort refinement contribution disappeared"); }
    values.push_back(std::move(batch));
    ordinals.push_back(part.ordinal);
  }
  uint64_t parameter = n.stride;
  if (kind == step::cut) {
    auto splitters = _splitters->get_data_batch_by_position(0, id);
    if (!splitters) { throw internal_exception("Sort refinement splitters disappeared"); }
    values.push_back(std::move(splitters));
    parameter = n.first_child;
  } else if (kind == step::coalesce) {
    parameter = n.first_child;
  }
  ++n.outstanding;
  return std::make_unique<work>(std::move(values), kind, id, std::move(ordinals), parameter);
}

// A single contribution keeps its ordinal and moves to the replacement node without a
// task or device work. Multi-contribution groups become coalesce tasks.
std::unique_ptr<sirius_physical_merge_sort::work> sirius_physical_merge_sort::claim_coalesce(
  std::size_t id)
{
  while (_nodes[id].cursor < _nodes[id].parts.size()) {
    auto& n = _nodes[id];
    if (group_end(n, n.cursor) > n.cursor + 1) { return claim_group(id, step::coalesce, true); }
    auto part  = n.parts[n.cursor++];
    auto batch = _parts->pop_data_batch_by_position(part.ordinal, id);
    if (!batch) { throw internal_exception("Sort refinement contribution disappeared"); }
    auto& replacement = _nodes[n.first_child];
    _parts->add_data_batch_at_position(batch, part.ordinal, n.first_child);
    replacement.parts.push_back(part);
    replacement.rows += part.rows;
    replacement.bytes += part.bytes;
  }
  finish_coalesce(id);
  return nullptr;
}

// Once every group has moved or published, the replacement takes over the interval.
void sirius_physical_merge_sort::finish_coalesce(std::size_t id)
{
  auto& n = _nodes[id];
  if (n.cursor < n.parts.size() || n.outstanding) { return; }
  auto replacement_id = n.first_child;
  auto contributions  = n.parts.size();
  retire(id);
  auto& replacement = _nodes[replacement_id];
  if (replacement.parts.size() >= contributions) {
    throw internal_exception("Sort coalescing made no progress");
  }
  std::sort(replacement.parts.begin(), replacement.parts.end(), [](auto const& a, auto const& b) {
    return a.ordinal < b.ordinal;
  });
  replacement.state = phase::ready;
  _active.emplace(replacement.lo, replacement_id);
}

std::unique_ptr<sirius_physical_merge_sort::work> sirius_physical_merge_sort::claim(std::size_t id)
{
  auto& n = _nodes[id];
  switch (n.state) {
    case phase::ready: {
      if (fits(n)) {
        batches values;
        std::vector<uint64_t> ordinals;
        for (auto const& part : n.parts) {
          auto batch = _parts->pop_data_batch_by_position(part.ordinal, id);
          if (!batch) { throw internal_exception("Sort merge contribution disappeared"); }
          values.push_back(std::move(batch));
          ordinals.push_back(part.ordinal);
        }
        n.state = phase::merging;
        ++n.outstanding;
        return std::make_unique<work>(
          std::move(values), step::merge, id, std::move(ordinals), 0, n.lo);
      }
      // Oversized. With R rows per tile and k contributions, the sampled-merge bound
      // gives tiles of at most R rows. Since bytes > chunk_bytes, rows > R, so every
      // child has strictly fewer rows than this node.
      auto average  = 1 + (n.bytes - 1) / n.rows;
      auto limit    = std::min(native_rows, std::max(uint64_t{1}, chunk_bytes / average));
      uint64_t k    = n.parts.size();
      n.stride      = limit / k / 4 + 1;
      n.sample_step = (limit - k * (n.stride - 1)) / n.stride;
      if (!n.sample_step) { throw internal_exception("Sort refinement sample step is zero"); }
      uint64_t sample_rows = 0;
      for (auto const& part : n.parts) {
        sample_rows += part.rows / n.stride;
      }
      // Many small contributions drive the stride toward one and the sample toward every
      // row. Planning should not exceed a tile, so coalesce first if that reduces k.
      if (sample_rows > limit) {
        std::size_t groups = 0;
        for (std::size_t begin = 0; begin < n.parts.size(); begin = group_end(n, begin)) {
          ++groups;
        }
        if (groups < k) {
          SIRIUS_LOG_DEBUG("Merge sort: coalescing contributions={}, groups={}", k, groups);
          auto replacement = _nodes.size();
          _nodes.push_back(node{.lo = n.lo, .hi = n.hi});
          auto& parent       = _nodes[id];  // push_back invalidated n
          parent.state       = phase::coalescing;
          parent.cursor      = 0;
          parent.first_child = replacement;
          parent.child_count = 1;
          return claim_coalesce(id);
        }
      }
      if (sample_rows > native_rows) {
        throw invalid_input_exception("Sort refinement samples exceed native row limit");
      }
      n.state  = phase::sampling;
      n.cursor = 0;
      return claim_group(id, step::sample, false);
    }
    case phase::sampling: return claim_group(id, step::sample, false);
    case phase::coalescing: return claim_coalesce(id);
    case phase::plan_ready: {
      batches values;
      if (id < _samples->num_partitions()) {
        for (auto position : _samples->get_positions(id)) {
          values.push_back(_samples->pop_data_batch_by_position(position, id));
        }
      }
      if (values.empty()) { throw internal_exception("Oversized sort node produced no samples"); }
      n.state = phase::planning;
      ++n.outstanding;
      return std::make_unique<work>(
        std::move(values), step::plan, id, std::vector<uint64_t>{}, n.sample_step);
    }
    case phase::cutting: return claim_group(id, step::cut, true);
    default: throw internal_exception("Sort refinement node is not claimable");
  }
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::get_next_task_input_data()
{
  if (!is_source_pipeline_finished()) { return nullptr; }
  std::optional<std::pair<std::size_t, std::size_t>> routed;
  std::unique_ptr<work> claimed;
  {
    std::lock_guard guard(lock);
    if (!_ingested) { routed = ingest(); }
    // A coalescing claim that only moves single contributions yields no task, but can
    // make its replacement node actionable.
    while (!claimed) {
      auto id = actionable();
      if (!id) { break; }
      claimed = claim(*id);
    }
  }
  if (routed) {
    SIRIUS_LOG_DEBUG("Merge sort: ranges={}, nonempty_ranges={}", routed->first, routed->second);
  }
  return claimed;
}

std::optional<task_creation_hint> sirius_physical_merge_sort::get_next_task_hint()
{
  if (!is_source_pipeline_finished()) { return sirius_physical_operator::get_next_task_hint(); }
  std::lock_guard guard(lock);
  bool ready = _ingested ? actionable().has_value() : !sirius_physical_operator::all_ports_empty();
  if (ready) { return task_creation_hint{TaskCreationHint::READY, this}; }
  return std::nullopt;
}

bool sirius_physical_merge_sort::all_ports_empty()
{
  std::lock_guard guard(lock);
  return _active.empty() && sirius_physical_operator::all_ports_empty();
}

// Publication is the only place refinement state advances, so retries never duplicate
// or skip a transition. New work is not discovered by downstream scheduling; request it.
void sirius_physical_merge_sort::on_task_output_published(operator_data const& output)
{
  auto const& data = dynamic_cast<result const&>(output);
  bool more        = false;
  {
    std::lock_guard guard(lock);
    auto id = data.node;
    if (!_nodes.at(id).outstanding) {
      throw internal_exception("Sort refinement completed without outstanding work");
    }
    --_nodes[id].outstanding;
    switch (data.kind) {
      case step::merge: retire(id); break;
      case step::sample: {
        auto& n = _nodes[id];
        if (n.cursor == n.parts.size() && !n.outstanding) { n.state = phase::plan_ready; }
        break;
      }
      case step::plan: {
        auto tiles = data.ordinals.at(0) + 1;
        auto lo = _nodes[id].lo, hi = _nodes[id].hi;
        auto width = (hi - lo) / tiles;
        if (!width) { throw invalid_input_exception("Sort refinement exhausted output positions"); }
        auto first = _nodes.size();
        for (std::size_t t = 0; t < tiles; ++t) {
          _nodes.push_back(
            node{.lo = lo + t * width, .hi = t + 1 == tiles ? hi : lo + (t + 1) * width});
        }
        auto& n       = _nodes[id];
        n.first_child = first;
        n.child_count = tiles;
        n.state       = phase::cutting;
        n.cursor      = 0;
        break;
      }
      case step::coalesce: {
        auto const& p     = data.pieces.at(0);
        auto& replacement = _nodes.at(p.child);
        replacement.parts.push_back({p.ordinal, p.rows, p.bytes});
        replacement.rows += p.rows;
        replacement.bytes += p.bytes;
        finish_coalesce(id);
        break;
      }
      case step::cut: {
        for (auto const& p : data.pieces) {
          auto& child = _nodes.at(p.child);
          child.parts.push_back({p.ordinal, p.rows, p.bytes});
          child.rows += p.rows;
          child.bytes += p.bytes;
        }
        auto& n = _nodes[id];
        if (n.cursor < n.parts.size() || n.outstanding) { break; }
        // Every producer of the children has published: they are complete.
        _splitters->pop_data_batch_by_position(0, id);
        auto first = n.first_child, count = n.child_count;
        auto rows = n.rows;
        retire(id);
        for (auto c = first; c < first + count; ++c) {
          auto& child = _nodes[c];
          if (child.parts.empty()) {
            child.state = phase::done;
            continue;
          }
          if (child.rows >= rows) { throw internal_exception("Sort refinement made no progress"); }
          std::sort(child.parts.begin(), child.parts.end(), [](auto const& a, auto const& b) {
            return a.ordinal < b.ordinal;
          });
          child.state = phase::ready;
          _active.emplace(child.lo, c);
        }
        break;
      }
    }
    more = actionable().has_value();
  }
  if (more) {
    if (auto pipeline = get_pipeline()) {
      if (auto* creator = pipeline->get_task_creator()) { creator->schedule(this); }
    }
  }
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute(operator_data const& input_data,
                                                                   ::cuda::stream_ref stream)
{
  auto const& input = dynamic_cast<work const&>(input_data);
  switch (input.kind) {
    case step::merge: return execute_merge(input, stream);
    case step::sample: return execute_sample(input, stream);
    case step::plan: return execute_plan(input, stream);
    case step::cut: return execute_cut(input, stream);
    case step::coalesce: return execute_coalesce(input, stream);
  }
  throw internal_exception("Unknown sort refinement step");
}

// Merge one greedy group into a single sorted run, keeping every column: the final
// projection applies only to output leaves.
std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_coalesce(
  work const& input, ::cuda::stream_ref stream)
{
  auto reads  = input.get_read_only_batches();
  auto* space = reads.at(0).get_memory_space();
  auto mr     = space->get_default_allocator();
  std::vector<cudf::table_view> views;
  views.reserve(reads.size());
  for (auto const& read : reads) {
    views.push_back(get_cudf_table_view(read));
  }
  auto run  = make_data_batch(cudf::merge(views,
                                         comparison->key_columns,
                                         comparison->directions,
                                         comparison->null_precedence,
                                         stream,
                                         mr),
                             *space,
                             stream,
                             batch_telemetry());
  auto info = get_batch_rows_and_bytes(*run);
  std::vector<piece> pieces{{input.parameter, input.ordinals.at(0), info.rows.value(), info.bytes}};
  return std::make_unique<result>(batches{std::move(run)},
                                  step::coalesce,
                                  input.node,
                                  std::vector<uint64_t>{},
                                  std::move(pieces));
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_sample(work const& input,
                                                                          ::cuda::stream_ref stream)
{
  auto reads  = input.get_read_only_batches();
  auto* space = reads.at(0).get_memory_space();
  auto mr     = space->get_default_allocator();
  batches samples;
  std::vector<uint64_t> ordinals;
  for (std::size_t i = 0; i < reads.size(); ++i) {
    auto view = get_cudf_table_view(reads[i]);
    if (static_cast<uint64_t>(view.num_rows()) < input.parameter) { continue; }
    samples.push_back(make_data_batch(
      sm::gather_samples(view, input.ordinals.at(i), input.parameter, *comparison, stream, mr),
      *space,
      stream,
      batch_telemetry()));
    ordinals.push_back(input.ordinals[i]);
  }
  return std::make_unique<result>(
    std::move(samples), step::sample, input.node, std::move(ordinals), std::vector<piece>{});
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_plan(work const& input,
                                                                        ::cuda::stream_ref stream)
{
  auto reads  = input.get_read_only_batches();
  auto* space = reads.at(0).get_memory_space();
  auto mr     = space->get_default_allocator();
  std::vector<cudf::table_view> views;
  views.reserve(reads.size());
  for (auto const& read : reads) {
    views.push_back(get_cudf_table_view(read));
  }
  auto splitters = sm::order_sample_splitters(views, input.parameter, *comparison, stream, mr);
  uint64_t rows  = splitters->num_rows();
  if (!rows) { throw internal_exception("Oversized sort node produced no splitters"); }
  return std::make_unique<result>(
    batches{make_data_batch(std::move(splitters), *space, stream, batch_telemetry())},
    step::plan,
    input.node,
    std::vector<uint64_t>{rows},
    std::vector<piece>{});
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_cut(work const& input,
                                                                       ::cuda::stream_ref stream)
{
  auto reads = input.get_read_only_batches();
  if (reads.size() != input.ordinals.size() + 1) {
    throw internal_exception("Sort cut task requires contributions and splitters");
  }
  auto* space    = reads[0].get_memory_space();
  auto mr        = space->get_default_allocator();
  auto splitters = get_cudf_table_view(reads.back());
  batches output;
  std::vector<piece> pieces;
  try {
    for (std::size_t i = 0; i < input.ordinals.size(); ++i) {
      auto run        = get_cudf_table_view(reads[i]);
      auto ordinal    = input.ordinals[i];
      auto cut_column = sm::discover_cuts(run, ordinal, splitters, *comparison, stream, mr);
      std::vector<cudf::size_type> cuts(static_cast<std::size_t>(splitters.num_rows()) + 2);
      CUDF_CUDA_TRY(cudaMemcpyAsync(cuts.data() + 1,
                                    cut_column->view().data<cudf::size_type>(),
                                    splitters.num_rows() * sizeof(cudf::size_type),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      stream.sync();
      cuts.front() = 0;
      cuts.back()  = run.num_rows();
      if (!std::is_sorted(cuts.begin(), cuts.end())) {
        throw internal_exception("Sort cuts do not cover the contribution monotonically");
      }
      for (std::size_t t = 0; t + 1 < cuts.size(); ++t) {
        if (cuts[t] == cuts[t + 1]) { continue; }
        auto view  = cudf::slice(run, {cuts[t], cuts[t + 1]}, stream).front();
        auto batch = make_data_batch(
          std::make_unique<cudf::table>(view, stream, mr), *space, stream, batch_telemetry());
        auto info = get_batch_rows_and_bytes(*batch);
        pieces.push_back({input.parameter + t, ordinal, info.rows.value(), info.bytes});
        output.push_back(std::move(batch));
      }
    }
    stream.sync();
  } catch (...) {
    cudaStreamSynchronize(stream.get());
    throw;
  }
  return std::make_unique<result>(
    std::move(output), step::cut, input.node, std::vector<uint64_t>{}, std::move(pieces));
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute_merge(work const& input,
                                                                         ::cuda::stream_ref stream)
{
  auto reads    = input.get_read_only_batches();
  auto* space   = reads.at(0).get_memory_space();
  auto mr       = space->get_default_allocator();
  uint64_t rows = 0;
  for (auto size : sizes(input.get_data_batches())) {
    rows += size.rows;
  }
  if (rows > native_rows) { throw invalid_input_exception("Sort tile exceeds native row limit"); }
  std::unique_ptr<cudf::table> table;
  if (reads.size() == 1) {
    auto view = get_cudf_table_view(reads.front());
    if (!_final_projections.empty()) {
      view = view.select(_final_projections.begin(), _final_projections.end());
    }
    table = std::make_unique<cudf::table>(view, stream, mr);
  } else {
    std::vector<cudf::table_view> views;
    views.reserve(reads.size());
    for (auto const& read : reads) {
      views.push_back(get_cudf_table_view(read));
    }
    auto columns = cudf::merge(views,
                               comparison->key_columns,
                               comparison->directions,
                               comparison->null_precedence,
                               stream,
                               mr)
                     ->release();
    if (!_final_projections.empty()) {
      std::vector<std::unique_ptr<cudf::column>> projected;
      projected.reserve(_final_projections.size());
      for (auto index : _final_projections) {
        projected.push_back(std::move(columns[index]));
      }
      columns = std::move(projected);
    }
    table = std::make_unique<cudf::table>(std::move(columns));
  }
  auto merged = make_data_batch(std::move(table), *space, stream, batch_telemetry());
  return std::make_unique<result>(batches{std::move(merged)},
                                  step::merge,
                                  input.node,
                                  std::vector<uint64_t>{},
                                  std::vector<piece>{},
                                  input.get_position());
}

// Refinement outputs go only to this operator's own repositories; publication into them
// precedes on_task_output_published, which makes them claimable.
void sirius_physical_merge_sort::sink(operator_data const& output, ::cuda::stream_ref stream)
{
  auto const& data   = dynamic_cast<result const&>(output);
  auto const& values = data.get_data_batches();
  switch (data.kind) {
    case step::merge: sirius_physical_operator::sink(output, stream); return;
    case step::sample:
      for (std::size_t i = 0; i < values.size(); ++i) {
        _samples->add_data_batch_at_position(values[i], data.ordinals.at(i), data.node);
      }
      return;
    case step::plan: _splitters->add_data_batch_at_position(values.at(0), 0, data.node); return;
    case step::cut:
    case step::coalesce:
      for (std::size_t i = 0; i < values.size(); ++i) {
        auto const& p = data.pieces.at(i);
        _parts->add_data_batch_at_position(values[i], p.ordinal, p.child);
      }
      return;
  }
}
}  // namespace sirius::op
