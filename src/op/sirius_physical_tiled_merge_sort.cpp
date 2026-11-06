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

#include "data/data_batch_utils.hpp"
#include "op/order/bounded_sort.hpp"
#include "op/order/gpu_sampled_merge.hpp"
#include "op/order/sampled_merge.hpp"
#include "op/order/tiled_merge_work_queue.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/merge.hpp>

#include <algorithm>
#include <chrono>
#include <format>
#include <numeric>
#include <unordered_set>

namespace sirius::op {
namespace sm = sampled_merge;

// Finite operator-local work and dependencies. All idle device owners live in
// registered repositories; these descriptors contain locations and host metadata.
struct sirius_physical_merge_sort::tiled_state {
  enum storage { INPUTS, SAMPLES, SPLITTERS, PIECES, RESULTS, STORAGE_COUNT };
  enum class phase { NORMALIZE, COALESCE, SAMPLE, ORDER, EXTRACT, TILES, MERGING, DONE };
  enum class kind { NORMALIZE, COALESCE, SAMPLE, ORDER, EXTRACT, COPY_FRAGMENT, MERGE, PUBLISH };
  enum class status { PENDING, CLAIMED, COMPLETE };
  enum class recovery { NONE, SPLIT_GROUP, REFINE_TILE, SPLIT_COPY, COPY_SPANS };
  static char const* name(kind type)
  {
    switch (type) {
      case kind::NORMALIZE: return "normalize";
      case kind::COALESCE: return "coalesce";
      case kind::SAMPLE: return "sample";
      case kind::ORDER: return "order_samples";
      case kind::EXTRACT: return "extract";
      case kind::COPY_FRAGMENT: return "copy_fragment";
      case kind::MERGE: return "merge";
      case kind::PUBLISH: return "publish";
    }
    throw internal_exception("Unknown tiled sort work");
  }
  static constexpr uint64_t window_tiles = 128;
  static constexpr uint64_t native_rows  = std::numeric_limits<cudf::size_type>::max();

  struct slot {
    cucascade::shared_data_repository* repo;
    std::size_t partition, position;
  };
  struct identity {
    cucascade::shared_data_repository* repo;
    std::size_t partition;
    uint64_t id;
    bool operator==(identity const&) const = default;
  };
  struct identity_hash {
    std::size_t operator()(identity const& key) const noexcept
    {
      auto hash = std::hash<cucascade::shared_data_repository*>{}(key.repo);
      hash ^= std::hash<std::size_t>{}(key.partition) + (hash << 6) + (hash >> 2);
      return hash ^ (std::hash<uint64_t>{}(key.id) + (hash << 6) + (hash >> 2));
    }
  };
  struct ref {
    slot location;
    uint64_t id, rows, bytes;
    identity key() const { return {location.repo, location.partition, id}; }
    auto get() const
    {
      auto batch = location.repo->get_data_batch_by_id(id, location.partition);
      if (!batch) { throw internal_exception("Missing tiled sort batch"); }
      return batch;
    }
    void retire() const { location.repo->pop_data_batch_by_id(id, location.partition); }
    static ref describe(slot location, cucascade::data_batch const& batch)
    {
      auto info = get_batch_rows_and_bytes(batch);
      if (!info.rows) { throw invalid_input_exception("Sort input lacks row-count metadata"); }
      return {location, batch.get_batch_id(), *info.rows, info.bytes};
    }
  };
  struct group {
    std::vector<ref> inputs;
    std::vector<std::shared_ptr<group>> children;
    std::optional<ref> output;
  };
  struct piece {
    ref batch;
    std::size_t input;
    uint64_t begin, end;
  };
  struct region;
  struct tile {
    std::size_t partition;
    uint64_t next_position;
    std::vector<piece> pieces;
    uint64_t rows{0};
    std::optional<ref> result;
    std::shared_ptr<region> child;
  };
  struct work {
    kind type;
    status state{status::PENDING};
    std::weak_ptr<region> owner;
    std::vector<ref> sources;
    std::shared_ptr<group> coalescing;
    std::size_t input{0}, tile_index{0};
    uint64_t begin{0}, end{0}, first_splitter{0};
    slot destination{};
  };
  struct region {
    phase stage{phase::NORMALIZE};
    std::vector<ref> inputs;
    tiled_merge_work_queue<work> work_items;
    std::weak_ptr<region> parent;
    std::size_t unfinished_children{0};
    bool enqueued{false};
    std::vector<std::vector<std::pair<uint64_t, ref>>> normalized;
    std::vector<std::shared_ptr<group>> groups;
    std::vector<std::optional<ref>> samples;
    std::vector<ref> pages;
    std::vector<tile> tiles;
    // Refinement retains the unpublished parent's pieces until its transpose
    // commits, even if normalization/coalescing has produced replacements.
    std::vector<ref> retained_parent_sources;
    std::unordered_set<identity, identity_hash> retained_parent_ids;
    std::size_t creator_visits{0};
    sm::parameters sampling{};
    uint64_t row_limit{0}, splitter_count{0}, requested_row_limit{0}, coalescing_inputs{0};
    std::size_t sample_partition{0}, splitter_partition{0};
  };
  struct record {
    slot destination;
    std::size_t tile_index{0};
    uint64_t begin{0}, end{0};
  };
  struct task_data : pipelineable_operator_data {
    std::shared_ptr<work> item;
    std::vector<record> records;
    std::vector<record> spans;
    uint64_t splitter_count{0};
    std::size_t admitted_bytes{0}, attempt_ceiling{0};
    recovery action{recovery::NONE};
    task_data(std::vector<std::shared_ptr<cucascade::data_batch>> batches,
              std::shared_ptr<work> item,
              std::optional<std::size_t> position = std::nullopt)
      : pipelineable_operator_data(std::move(batches), position), item(std::move(item))
    {
    }
    bool is_control_only() const noexcept override { return action != recovery::NONE; }
    std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
      std::vector<std::shared_ptr<cucascade::data_batch>> batches) const override
    {
      auto result     = std::make_unique<task_data>(std::move(batches), item, get_position());
      result->records = records;
      result->spans   = spans;
      result->splitter_count  = splitter_count;
      result->admitted_bytes  = admitted_bytes;
      result->attempt_ceiling = attempt_ceiling;
      result->action          = action;
      result->adopt_row_identity(*this);
      return result;
    }
  };

  static std::string failure(task_data const& data,
                             char const* site,
                             std::size_t configured,
                             char const* reason)
  {
    uint64_t bytes = 0;
    for (auto const& source : data.item->sources) {
      bytes = sm::checked_add(bytes, source.bytes);
    }
    auto page = data.item->type == kind::EXTRACT ? data.item->sources.back().bytes : 0;
    return std::format(
      "Tiled sort {} {}: {}; owning_backing={}, page_bytes={}, granted={}, "
      "effective_ceiling={}, configured_ceiling={}",
      name(data.item->type),
      site,
      reason,
      bytes,
      page,
      data.admitted_bytes,
      data.attempt_ceiling,
      configured);
  }

  std::vector<std::shared_ptr<region>> roots;
  std::deque<std::shared_ptr<region>> ready_regions;
  tiled_merge_work_queue<work> finals;
  std::size_t unfinished_roots{0};
  bool publishing{false}, finished{false};
  std::vector<std::size_t> partitions = std::vector<std::size_t>(STORAGE_COUNT, 0);

  slot allocate(sirius_physical_merge_sort& op, storage which)
  {
    auto& next = partitions.at(which);
    auto index = next;
    next       = sm::checked_add(next, 1);
    auto* repo = op._tile_storage.at(which);
    if (next > repo->num_partitions()) { repo->set_num_partitions(next); }
    return {repo, index, 0};
  }
  static bool complete(region const& r) { return r.work_items.complete_stage(); }
  static bool actionable(region const& r)
  {
    return r.stage != phase::DONE &&
           (!r.work_items.pending.empty() || (complete(r) && !r.unfinished_children));
  }
  void enqueue(std::shared_ptr<region> const& r)
  {
    if (!r->enqueued && actionable(*r)) {
      ready_regions.push_back(r);
      r->enqueued = true;
    }
  }
  bool ready() const
  {
    return !finished && (!ready_regions.empty() || (!unfinished_roots && !publishing) ||
                         !finals.pending.empty());
  }
  void finish(std::shared_ptr<region> const& r)
  {
    if (r->stage == phase::DONE) { throw internal_exception("Sort region completed twice"); }
    SIRIUS_LOG_DEBUG("sort.tile_schedule claims={}, region_visits={}",
                     r->work_items.inspections,
                     r->creator_visits);
    r->stage = phase::DONE;
    if (auto parent = r->parent.lock()) {
      if (!parent->unfinished_children) { throw internal_exception("Sort child count underflow"); }
      --parent->unfinished_children;
      enqueue(parent);
    } else {
      if (!unfinished_roots) { throw internal_exception("Sort root count underflow"); }
      --unfinished_roots;
    }
  }
  static void retire(region const& r, ref const& input)
  {
    if (!r.retained_parent_ids.contains(input.key())) { input.retire(); }
  }
  void refine(sirius_physical_merge_sort& op,
              std::shared_ptr<region> const& parent,
              std::size_t index)
  {
    auto& tile = parent->tiles.at(index);
    if (tile.rows <= 1) {
      throw invalid_input_exception("Cannot further refine a one-row sort tile after task failure");
    }
    auto child                 = std::make_shared<region>();
    child->requested_row_limit = std::max(uint64_t{1}, tile.rows / 2);
    for (auto const& piece : tile.pieces) {
      child->inputs.push_back(piece.batch);
    }
    child->retained_parent_sources = child->inputs;
    for (auto const& input : child->inputs) {
      child->retained_parent_ids.insert(input.key());
    }
    child->parent = parent;
    enqueue(child);
    ++parent->unfinished_children;
    tile.child = child;
    tile.pieces.clear();
    SIRIUS_LOG_DEBUG("sort.tile_refine rows={}, child_R={}", tile.rows, child->requested_row_limit);
  }
  void flatten_results(std::shared_ptr<region> const& r, uint64_t& position)
  {
    for (auto const& tile : r->tiles) {
      if (tile.child) {
        flatten_results(tile.child, position);
        continue;
      }
      if (!tile.result) { continue; }
      auto item                  = std::make_shared<work>();
      item->type                 = kind::PUBLISH;
      item->owner                = r;
      item->sources              = {*tile.result};
      item->destination.position = position;
      position                   = sm::checked_add(position, 1);
      finals.add(item);
    }
  }
  std::shared_ptr<work> add(std::shared_ptr<region> const& r, kind type, std::vector<ref> sources)
  {
    auto item     = std::make_shared<work>();
    item->type    = type;
    item->owner   = r;
    item->sources = std::move(sources);
    r->work_items.add(item);
    enqueue(r);
    return item;
  }
  void coalesce(sirius_physical_merge_sort& op,
                std::shared_ptr<region> const& r,
                std::shared_ptr<group> const& g)
  {
    if (g->inputs.size() == 1) {
      g->output = g->inputs.front();
      return;
    }
    auto item         = add(r, kind::COALESCE, g->inputs);
    item->coalescing  = g;
    item->destination = allocate(op, INPUTS);
  }
  void split_group(sirius_physical_merge_sort& op,
                   std::shared_ptr<region> const& r,
                   std::shared_ptr<group> const& g)
  {
    auto middle = g->inputs.size() / 2;
    if (!middle) { throw invalid_input_exception("Unfit single sort contribution"); }
    for (auto [begin, end] :
         {std::pair{std::size_t{0}, middle}, std::pair{middle, g->inputs.size()}}) {
      auto child = std::make_shared<group>();
      child->inputs.assign(g->inputs.begin() + begin, g->inputs.begin() + end);
      g->children.push_back(child);
      coalesce(op, r, child);
    }
  }
  static void flatten(group const& g, std::vector<ref>& outputs)
  {
    if (g.output) {
      outputs.push_back(*g.output);
      return;
    }
    for (auto const& child : g.children) {
      flatten(*child, outputs);
    }
  }

  void advance(sirius_physical_merge_sort& op, std::shared_ptr<region> const& r)
  {
    while (complete(*r)) {
      if (r->stage == phase::DONE) { return; }
      if (r->stage == phase::NORMALIZE) {
        if (!r->normalized.empty()) {
          std::vector<ref> next;
          for (std::size_t i = 0; i < r->inputs.size(); ++i) {
            auto& replacements = r->normalized[i];
            if (replacements.empty()) {
              next.push_back(r->inputs[i]);
              continue;
            }
            std::sort(replacements.begin(), replacements.end(), [](auto const& a, auto const& b) {
              return a.first < b.first;
            });
            for (auto const& replacement : replacements) {
              next.push_back(replacement.second);
            }
            retire(*r, r->inputs[i]);
          }
          r->inputs = std::move(next);
          r->normalized.clear();
        }
        r->normalized.resize(r->inputs.size());
        for (std::size_t i = 0; i < r->inputs.size(); ++i) {
          auto const& input = r->inputs[i];
          if (input.bytes <= op.chunk_bytes || input.rows <= 1) { continue; }
          for (auto [begin, end] :
               {std::pair{uint64_t{0}, input.rows / 2}, std::pair{input.rows / 2, input.rows}}) {
            auto item         = add(r, kind::NORMALIZE, {input});
            item->input       = i;
            item->begin       = begin;
            item->end         = end;
            item->destination = allocate(op, INPUTS);
          }
        }
        if (!r->work_items.pending.empty()) { return; }
        r->normalized.clear();
        r->stage             = phase::COALESCE;
        r->coalescing_inputs = r->inputs.size();
        std::vector<sm::contribution_size> sizes;
        for (auto const& input : r->inputs) {
          sizes.push_back({input.rows, input.bytes});
        }
        for (auto planned : sm::plan_coalescing(sizes, op.chunk_bytes, native_rows)) {
          auto g = std::make_shared<group>();
          g->inputs.assign(r->inputs.begin() + planned.begin, r->inputs.begin() + planned.end);
          r->groups.push_back(g);
          coalesce(op, r, g);
        }
      } else if (r->stage == phase::COALESCE) {
        r->inputs.clear();
        for (auto const& g : r->groups) {
          flatten(*g, r->inputs);
        }
        r->groups.clear();
        uint64_t rows = 0, bytes = 0;
        for (auto const& input : r->inputs) {
          rows  = sm::checked_add(rows, input.rows);
          bytes = sm::checked_add(bytes, input.bytes);
        }
        if (!rows) {
          finish(r);
          return;
        }
        auto average = bytes ? 1 + (bytes - 1) / rows : uint64_t{1};
        r->row_limit = std::min(native_rows, std::max(uint64_t{1}, op.chunk_bytes / average));
        if (r->requested_row_limit) {
          r->row_limit = std::min(r->row_limit, r->requested_row_limit);
        }
        SIRIUS_LOG_DEBUG("sort.tiles pre_coalesce={}, inputs={}, rows={}, bytes={}, row_limit={}",
                         r->coalescing_inputs,
                         r->inputs.size(),
                         rows,
                         bytes,
                         r->row_limit);
        if (rows <= r->row_limit) {
          auto location = allocate(op, PIECES);
          r->tiles.push_back({location.partition, r->inputs.size(), {}, rows});
          for (std::size_t i = 0; i < r->inputs.size(); ++i) {
            r->tiles.back().pieces.push_back({r->inputs[i], i, 0, r->inputs[i].rows});
          }
          r->stage = phase::TILES;
          return;
        }
        r->sampling         = sm::choose_parameters(r->inputs.size(), r->row_limit);
        r->sample_partition = allocate(op, SAMPLES).partition;
        r->samples.resize(r->inputs.size());
        r->stage = phase::SAMPLE;
        for (std::size_t i = 0; i < r->inputs.size(); ++i) {
          if (!sm::sample_count(r->inputs[i].rows, r->sampling.stride)) { continue; }
          auto item         = add(r, kind::SAMPLE, {r->inputs[i]});
          item->input       = i;
          item->destination = {op._tile_storage[SAMPLES], r->sample_partition, i};
        }
      } else if (r->stage == phase::SAMPLE) {
        std::vector<ref> samples;
        uint64_t rows = 0;
        for (auto const& input : r->samples) {
          if (!input) { continue; }
          samples.push_back(*input);
          rows = sm::checked_add(rows, input->rows);
        }
        if (rows > native_rows) {
          throw invalid_input_exception("Sort sample row count exceeds the native table limit");
        }
        auto item             = add(r, kind::ORDER, std::move(samples));
        item->destination     = allocate(op, SPLITTERS);
        r->splitter_partition = item->destination.partition;
        r->stage              = phase::ORDER;
      } else if (r->stage == phase::ORDER) {
        for (uint64_t t = 0; t <= r->splitter_count; ++t) {
          auto location = allocate(op, PIECES);
          r->tiles.push_back({location.partition, r->inputs.size(), {}, 0});
        }
        for (std::size_t i = 0; i < r->inputs.size(); ++i) {
          for (std::size_t page = 0; page < r->pages.size(); ++page) {
            auto item            = add(r, kind::EXTRACT, {r->inputs[i], r->pages[page]});
            item->input          = i;
            item->begin          = page * window_tiles;
            item->end            = std::min(item->begin + window_tiles, r->splitter_count + 1);
            item->first_splitter = item->begin ? item->begin - 1 : 0;
          }
        }
        for (auto const& sample : r->samples) {
          if (sample) { sample->retire(); }
        }
        r->samples.clear();
        r->stage = phase::EXTRACT;
      } else if (r->stage == phase::EXTRACT) {
        std::vector<std::vector<std::pair<std::size_t, piece>>> coverage(r->inputs.size());
        for (std::size_t t = 0; t < r->tiles.size(); ++t) {
          auto& tile = r->tiles[t];
          std::sort(tile.pieces.begin(), tile.pieces.end(), [](piece const& a, piece const& b) {
            return std::tie(a.input, a.begin) < std::tie(b.input, b.begin);
          });
          for (auto const& piece : tile.pieces) {
            tile.rows = sm::checked_add(tile.rows, piece.batch.rows);
            coverage[piece.input].push_back({t, piece});
          }
          if (tile.rows > r->row_limit) {
            throw internal_exception("Sampled sort tile exceeds row bound");
          }
        }
        for (std::size_t i = 0; i < coverage.size(); ++i) {
          uint64_t cursor = 0;
          for (auto const& [t, piece] : coverage[i]) {
            if (piece.begin != cursor || piece.batch.rows != piece.end - piece.begin) {
              throw internal_exception("Sort extraction lost or duplicated input rows");
            }
            cursor = piece.end;
          }
          if (cursor != r->inputs[i].rows) {
            throw internal_exception("Incomplete sort extraction");
          }
          retire(*r, r->inputs[i]);
        }
        for (auto const& page : r->pages) {
          page.retire();
        }
        r->pages.clear();
        r->inputs.clear();
        for (auto const& parent : r->retained_parent_sources) {
          parent.retire();
        }
        r->retained_parent_sources.clear();
        r->retained_parent_ids.clear();
        r->stage = phase::TILES;
        return;
      } else if (r->stage == phase::TILES) {
        for (std::size_t t = 0; t < r->tiles.size(); ++t) {
          auto& tile = r->tiles[t];
          if (!tile.rows) { continue; }
          std::vector<ref> sources;
          for (auto const& piece : tile.pieces) {
            sources.push_back(piece.batch);
          }
          auto item         = add(r, kind::MERGE, std::move(sources));
          item->tile_index  = t;
          item->destination = allocate(op, RESULTS);
        }
        r->stage = phase::MERGING;
      } else if (r->stage == phase::MERGING) {
        if (r->unfinished_children) { return; }
        finish(r);
        return;
      }
    }
  }
};

std::size_t sirius_physical_merge_sort::no_history_peak_memory_estimate(
  input_stats const& stats) const
{
  return stats.num_batches == 1 ? memory::saturating_add(memory::saturating_mul(stats.bytes, 2),
                                                         merge_cold_estimate_base)
                                : merge_cold_peak_estimate(stats.bytes);
}

void sirius_physical_merge_sort::register_scratch(
  cucascade::shared_data_repository_manager& manager)
{
  if (!_tile_storage.empty()) {
    throw internal_exception("Tiled merge scratch already registered");
  }
  for (auto name : {"sort_tile_inputs",
                    "sort_tile_samples",
                    "sort_tile_splitters",
                    "sort_tile_pieces",
                    "sort_tile_results"}) {
    manager.add_new_repository(
      get_operator_id(), name, std::make_unique<cucascade::shared_data_repository>());
    _tile_storage.push_back(manager.get_repository(get_operator_id(), name).get());
  }
}

bool sirius_physical_merge_sort::all_ports_empty()
{
  std::lock_guard lock(_drain_mutex);
  if (!_tiled_state) { return sirius_physical_operator::all_ports_empty(); }
  return _tiled_state->finished;
}

std::optional<task_creation_hint> sirius_physical_merge_sort::get_next_task_hint()
{
  if (!is_source_pipeline_finished()) { return sirius_physical_operator::get_next_task_hint(); }
  std::lock_guard lock(_drain_mutex);
  if (!_tiled_state ? !sirius_physical_operator::all_ports_empty() : _tiled_state->ready()) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::get_next_task_input_data()
{
  std::lock_guard lock(_drain_mutex);
  if (!is_source_pipeline_finished()) { return nullptr; }
  if (!_tiled_state) {
    if (_tile_storage.size() != tiled_state::STORAGE_COUNT) {
      throw internal_exception("Tiled merge scratch is not registered");
    }
    _tiled_state = std::make_shared<tiled_state>();
    auto* repo   = get_port("default")->repo;
    for (std::size_t p = 0; p < repo->num_partitions(); ++p) {
      auto r   = std::make_shared<tiled_state::region>();
      auto ids = repo->get_batch_ids(p);
      std::sort(ids.begin(), ids.end());
      for (auto id : ids) {
        auto batch = repo->get_data_batch_by_id(id, p);
        auto input = tiled_state::ref::describe({repo, p, 0}, *batch);
        if (input.rows) {
          r->inputs.push_back(input);
        } else {
          input.retire();
        }
      }
      _tiled_state->roots.push_back(r);
      ++_tiled_state->unfinished_roots;
      _tiled_state->enqueue(r);
    }
    SIRIUS_LOG_DEBUG("sort.tile_ranges ranges={}, nonempty_ranges={}",
                     _tiled_state->roots.size(),
                     std::count_if(_tiled_state->roots.begin(),
                                   _tiled_state->roots.end(),
                                   [](auto const& r) { return !r->inputs.empty(); }));
  }
  while (!_tiled_state->ready_regions.empty()) {
    auto r = std::move(_tiled_state->ready_regions.front());
    _tiled_state->ready_regions.pop_front();
    r->enqueued = false;
    ++r->creator_visits;
    _tiled_state->advance(*this, r);
    auto data = r->work_items.claim([](auto const& item) {
      std::vector<std::shared_ptr<cucascade::data_batch>> batches;
      batches.reserve(item->sources.size());
      for (auto const& input : item->sources) {
        batches.push_back(input.get());
      }
      return std::make_unique<tiled_state::task_data>(std::move(batches), item);
    });
    if (data) {
      data->item->state = tiled_state::status::CLAIMED;
      _tiled_state->enqueue(r);
      return data;
    }
    _tiled_state->enqueue(r);
  }
  if (!_tiled_state->unfinished_roots) {
    if (!_tiled_state->publishing) {
      uint64_t position = 0;
      for (auto const& root : _tiled_state->roots) {
        _tiled_state->flatten_results(root, position);
      }
      _tiled_state->publishing = true;
      if (_tiled_state->finals.complete_stage()) { _tiled_state->finished = true; }
    }
    auto task = _tiled_state->finals.claim([](auto const& item) {
      return std::make_unique<tiled_state::task_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{item->sources.front().get()},
        item,
        item->destination.position);
    });
    if (task) {
      task->item->state = tiled_state::status::CLAIMED;
      return task;
    }
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute(operator_data const& input_data,
                                                                   ::cuda::stream_ref stream)
{
  auto started      = std::chrono::steady_clock::now();
  using state       = tiled_state;
  auto const& input = dynamic_cast<state::task_data const&>(input_data);
  if (input.action != state::recovery::NONE) {
    auto output = std::make_unique<state::task_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{}, input.item);
    output->action = input.action;
    return output;
  }
  auto const& item = *input.item;
  auto owner       = item.owner.lock();
  if (!owner) { throw internal_exception("Sort work outlived its region"); }
  auto reads = input.get_read_only_batches();
  if (reads.empty()) { throw internal_exception("Tiled sort task has no inputs"); }
  // INVARIANT: gpu_pipeline_task -> prepare_for_processing -> lock_or_prepare_batch
  // restores exactly this item's inputs onto the task's reservation device.
  auto* space = reads.front().get_memory_space();
  auto mr     = space->get_default_allocator();
  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  auto result = std::make_unique<state::task_data>(std::move(outputs), input.item);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  auto emit = [&](std::unique_ptr<cudf::table> table, state::record record) {
    batches.push_back(make_data_batch(std::move(table), *space, stream, batch_telemetry()));
    result->records.push_back(record);
  };
  std::vector<cudf::table_view> views;
  for (auto const& read : reads) {
    views.push_back(get_cudf_table_view(read));
  }
  nvtx_scoped_range nvtx{"sort.tile_work"};
  try {
    if (item.type == state::kind::PUBLISH) {
      if (_final_projections.empty()) {
        batches = input.get_data_batches();
      } else {
        std::vector<cudf::size_type> projection(_final_projections.begin(),
                                                _final_projections.end());
        batches.push_back(make_data_batch(
          std::make_unique<cudf::table>(views.front().select(projection), stream, mr),
          *space,
          stream,
          batch_telemetry()));
      }
    } else if (item.type == state::kind::MERGE) {
      if (views.size() == 1) {
        batches = input.get_data_batches();
        result->records.push_back({item.destination});
      } else {
        emit(cudf::merge(views,
                         comparison->key_columns,
                         comparison->directions,
                         comparison->null_precedence,
                         stream,
                         mr),
             {item.destination});
      }
    } else if (item.type == state::kind::NORMALIZE || item.type == state::kind::COPY_FRAGMENT) {
      auto view = cudf::slice(
        views.front(),
        {static_cast<cudf::size_type>(item.begin), static_cast<cudf::size_type>(item.end)},
        stream)[0];
      emit(std::make_unique<cudf::table>(view, stream, mr),
           {item.destination, item.tile_index, item.begin, item.end});
    } else if (item.type == state::kind::COALESCE) {
      emit(cudf::merge(views,
                       comparison->key_columns,
                       comparison->directions,
                       comparison->null_precedence,
                       stream,
                       mr),
           {item.destination});
    } else if (item.type == state::kind::SAMPLE) {
      emit(sm::gather_samples(
             views.front(), item.input, owner->sampling.stride, *comparison, stream, mr),
           {item.destination});
    } else if (item.type == state::kind::ORDER) {
      auto splitters =
        sm::order_sample_splitters(views, owner->sampling.sample_step, *comparison, stream, mr);
      result->splitter_count = splitters->num_rows();
      auto count             = result->splitter_count + 1;
      for (uint64_t begin = 0, page = 0; begin < count; begin += state::window_tiles, ++page) {
        auto end             = std::min(begin + state::window_tiles, count);
        auto first           = begin ? begin - 1 : 0;
        auto last            = std::min(end, result->splitter_count);
        auto destination     = item.destination;
        destination.position = page;
        if (!page && end == count) {
          emit(std::move(splitters), {destination});
        } else {
          auto view =
            cudf::slice(splitters->view(),
                        {static_cast<cudf::size_type>(first), static_cast<cudf::size_type>(last)},
                        stream)[0];
          emit(std::make_unique<cudf::table>(view, stream, mr), {destination});
        }
      }
    } else if (item.type == state::kind::EXTRACT) {
      auto cuts = sm::discover_cuts(views[0], item.input, views[1], *comparison, stream, mr);
      std::vector<cudf::size_type> host(cuts->size());
      // One bulk integer readback, including adjacent page boundaries. Keep the
      // host destination alive through the exception fence as well.
      try {
        CUDF_CUDA_TRY(cudaMemcpyAsync(host.data(),
                                      cuts->view().data<cudf::size_type>(),
                                      host.size() * sizeof(cudf::size_type),
                                      cudaMemcpyDeviceToHost,
                                      stream.get()));
        stream.sync();
      } catch (...) {
        cudaStreamSynchronize(stream.get());
        throw;
      }
      for (auto t = item.begin; t < item.end; ++t) {
        auto begin = t ? host.at(t - 1 - item.first_splitter) : 0;
        auto end =
          t == owner->splitter_count ? views[0].num_rows() : host.at(t - item.first_splitter);
        if (begin > end || begin < 0 || end > views[0].num_rows()) {
          throw internal_exception("Nonmonotone tiled sort cuts");
        }
        if (begin == end) { continue; }
        auto const& tile = owner->tiles.at(t);
        result->spans.push_back({{_tile_storage[state::PIECES], tile.partition, item.input},
                                 static_cast<std::size_t>(t),
                                 static_cast<uint64_t>(begin),
                                 static_cast<uint64_t>(end)});
      }
      for (auto const& span : result->spans) {
        auto view = cudf::slice(
          views[0],
          {static_cast<cudf::size_type>(span.begin), static_cast<cudf::size_type>(span.end)},
          stream)[0];
        emit(std::make_unique<cudf::table>(view, stream, mr), span);
      }
      SIRIUS_LOG_TRACE(
        "sort.extract input={}, tile_begin={}, tile_end={}, readbacks=1, waits=1, pieces={}",
        item.input,
        item.begin,
        item.end,
        batches.size());
    }
  } catch (rmm::out_of_memory const&) {
    cudaStreamSynchronize(stream.get());
    // Let ordinary retries grow an underestimated reservation first. Once the
    // configured capacity is granted, only subdivision or a normal error remains.
    if (input.admitted_bytes < input.attempt_ceiling) { throw; }
    // No sink has been entered. Smaller disjoint work can be committed by the
    // ordinary success callback; partial sink publication can never reach here.
    switch (item.type) {
      case state::kind::COALESCE: result->action = state::recovery::SPLIT_GROUP; break;
      case state::kind::MERGE: result->action = state::recovery::REFINE_TILE; break;
      case state::kind::NORMALIZE:
      case state::kind::COPY_FRAGMENT:
        if (item.end - item.begin <= 1) {
          throw invalid_input_exception(
            state::failure(input, "compute", reservation_limit, "one-row owning copy cannot fit"));
        }
        result->action = state::recovery::SPLIT_COPY;
        break;
      case state::kind::EXTRACT:
        if (result->spans.empty()) {
          throw invalid_input_exception(
            state::failure(input,
                           "compute",
                           reservation_limit,
                           "cut search failed before independently copyable spans exist"));
        }
        result->action = state::recovery::COPY_SPANS;
        break;
      default:
        throw invalid_input_exception(state::failure(
          input,
          "compute",
          reservation_limit,
          item.type == state::kind::PUBLISH ? "final projection/output capacity envelope exceeded"
                                            : "supported sample allocation envelope exceeded"));
    }
    SIRIUS_LOG_DEBUG("sort.tile_recovery kind={}, completed_copies={}, spans={}, action={}",
                     state::name(item.type),
                     batches.size(),
                     result->spans.size(),
                     static_cast<int>(result->action));
    batches.clear();
    result->records.clear();
  } catch (...) {
    cudaStreamSynchronize(stream.get());
    throw;
  }
  auto output =
    std::make_unique<state::task_data>(std::move(batches), input.item, input.get_position());
  output->records        = std::move(result->records);
  output->spans          = std::move(result->spans);
  output->splitter_count = result->splitter_count;
  output->action         = result->action;
  uint64_t output_bytes = 0, output_rows = 0;
  for (auto const& batch : output->get_data_batches()) {
    auto metadata = get_batch_rows_and_bytes(*batch);
    output_bytes  = sm::checked_add(output_bytes, metadata.bytes);
    output_rows   = sm::checked_add(output_rows, metadata.rows.value_or(0));
  }
  SIRIUS_LOG_TRACE(
    "sort.tile_work kind={}, inputs={}, outputs={}, output_rows={}, output_bytes={}, splitters={}, "
    "split={}, enqueue_ms={}",
    state::name(item.type),
    item.sources.size(),
    output->get_data_batches().size(),
    output_rows,
    output_bytes,
    output->splitter_count,
    static_cast<int>(output->action),
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
  return output;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::recover_preparation_failure(
  operator_data const& input, task_attempt_context attempt)
{
  auto const& data = dynamic_cast<tiled_state::task_data const&>(input);
  if (attempt.granted_bytes < attempt.effective_ceiling) { return nullptr; }
  auto const& item = *data.item;
  using kind       = tiled_state::kind;
  if (item.type != kind::COALESCE && item.type != kind::MERGE) {
    auto reason = item.type == kind::PUBLISH ? "final-result restoration envelope exceeded"
                                             : "source/page restoration envelope exceeded";
    throw invalid_input_exception(
      tiled_state::failure(data, "preparation", reservation_limit, reason));
  }
  if (item.type == kind::MERGE && item.sources.size() == 1) {
    throw invalid_input_exception(tiled_state::failure(
      data, "preparation", reservation_limit, "individual piece restoration envelope exceeded"));
  }
  auto output = std::make_unique<tiled_state::task_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{}, data.item);
  output->action = item.type == kind::COALESCE ? tiled_state::recovery::SPLIT_GROUP
                                               : tiled_state::recovery::REFINE_TILE;
  return output;
}

void sirius_physical_merge_sort::notify_task_attempt(operator_data& input,
                                                     task_attempt_context attempt)
{
  auto& data           = dynamic_cast<tiled_state::task_data&>(input);
  data.admitted_bytes  = attempt.granted_bytes;
  data.attempt_ceiling = attempt.effective_ceiling;
}

void sirius_physical_merge_sort::sink(operator_data const& output, ::cuda::stream_ref stream)
{
  auto const& data = dynamic_cast<tiled_state::task_data const&>(output);
  if (data.item->type == tiled_state::kind::PUBLISH) {
    sirius_physical_operator::sink(output, stream);
    return;
  }
  if (data.records.size() != data.get_data_batches().size()) {
    throw internal_exception("Sort publication is missing storage positions");
  }
  for (std::size_t i = 0; i < data.records.size(); ++i) {
    auto const& location = data.records[i].destination;
    location.repo->add_data_batch_at_position(
      data.get_data_batches()[i], location.position, location.partition);
  }
}

void sirius_physical_merge_sort::on_task_output_published(operator_data const& output)
{
  SIRIUS_LOG_TRACE("sort.task_wait site=publication_completion, host_waits=1");
  using state      = tiled_state;
  auto const& data = dynamic_cast<state::task_data const&>(output);
  std::lock_guard lock(_drain_mutex);
  auto& item = *data.item;
  if (item.state != state::status::CLAIMED) {
    throw internal_exception("Sort work completed twice");
  }
  auto r = item.owner.lock();
  if (!r) { throw internal_exception("Missing sort region at completion"); }
  if (data.action != state::recovery::NONE) {
    if (data.action == state::recovery::REFINE_TILE) {
      _tiled_state->refine(*this, r, item.tile_index);
    } else if (data.action == state::recovery::SPLIT_GROUP) {
      _tiled_state->split_group(*this, r, item.coalescing);
    } else if (data.action == state::recovery::COPY_SPANS) {
      for (auto const& span : data.spans) {
        auto middle       = span.begin + (span.end - span.begin) / 2;
        auto add_fragment = [&](uint64_t begin, uint64_t end) {
          if (begin == end) { return; }
          auto child   = _tiled_state->add(r, state::kind::COPY_FRAGMENT, {item.sources.front()});
          child->input = item.input;
          child->tile_index  = span.tile_index;
          child->begin       = begin;
          child->end         = end;
          auto& tile         = r->tiles.at(span.tile_index);
          child->destination = {_tile_storage[state::PIECES],
                                tile.partition,
                                static_cast<std::size_t>(tile.next_position)};
          tile.next_position = sm::checked_add(tile.next_position, 1);
        };
        add_fragment(span.begin, middle);
        add_fragment(middle, span.end);
      }
    } else if (data.action == state::recovery::SPLIT_COPY) {
      auto middle = item.begin + (item.end - item.begin) / 2;
      for (auto [begin, end] : {std::pair{item.begin, middle}, std::pair{middle, item.end}}) {
        auto child            = _tiled_state->add(r, item.type, item.sources);
        child->input          = item.input;
        child->tile_index     = item.tile_index;
        child->begin          = begin;
        child->end            = end;
        child->first_splitter = item.first_splitter;
        if (item.type == state::kind::NORMALIZE) {
          child->destination = _tiled_state->allocate(*this, state::INPUTS);
        } else if (item.type == state::kind::COPY_FRAGMENT) {
          auto& tile         = r->tiles.at(item.tile_index);
          child->destination = {_tile_storage[state::PIECES],
                                tile.partition,
                                static_cast<std::size_t>(tile.next_position)};
          tile.next_position = sm::checked_add(tile.next_position, 1);
        }
      }
    }
  } else {
    for (std::size_t i = 0; i < data.records.size(); ++i) {
      auto const& record = data.records[i];
      auto reference     = state::ref::describe(record.destination, *data.get_data_batches()[i]);
      if (item.type == state::kind::NORMALIZE) {
        r->normalized.at(item.input).push_back({record.begin, reference});
      } else if (item.type == state::kind::COALESCE) {
        item.coalescing->output = reference;
        for (auto const& source : item.sources) {
          state::retire(*r, source);
        }
      } else if (item.type == state::kind::SAMPLE) {
        r->samples.at(item.input) = reference;
      } else if (item.type == state::kind::ORDER) {
        r->pages.push_back(reference);
        r->splitter_count = data.splitter_count;
      } else if (item.type == state::kind::MERGE) {
        r->tiles.at(item.tile_index).result = reference;
        for (auto const& source : item.sources) {
          state::retire(*r, source);
        }
        r->tiles.at(item.tile_index).pieces.clear();
      } else {
        r->tiles.at(record.tile_index)
          .pieces.push_back({reference, item.input, record.begin, record.end});
      }
    }
  }
  item.state = state::status::COMPLETE;
  if (item.type != state::kind::PUBLISH) {
    r->work_items.complete();
    _tiled_state->enqueue(r);
  }
  if (item.type == state::kind::PUBLISH) {
    item.sources.front().retire();
    _tiled_state->finals.complete();
    _tiled_state->finished = _tiled_state->finals.complete_stage();
  }
}

}  // namespace sirius::op
