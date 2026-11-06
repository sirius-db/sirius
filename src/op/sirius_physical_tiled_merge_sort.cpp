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
#include "op/sirius_physical_merge_sort.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/merge.hpp>

#include <algorithm>
#include <chrono>
#include <deque>
#include <format>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace sirius::op {
namespace sm = sampled_merge;

namespace {
// Only the tiled operator's finite stage/final-publication lists use this queue.
// Claimed work stays owned by its task, not by a repeatedly scanned stage list.
template <typename Work>
struct tiled_merge_work_queue {
  std::deque<std::shared_ptr<Work>> pending;
  std::size_t outstanding{0};
  std::size_t inspections{0};

  void add(std::shared_ptr<Work> item)
  {
    pending.push_back(std::move(item));
    ++outstanding;
  }
  // Prepare task-owned metadata before committing the claim. Allocation failure
  // leaves both the pending item and its outstanding count intact.
  template <typename Prepare>
  auto claim(Prepare&& prepare)
  {
    using result_type = std::invoke_result_t<Prepare, std::shared_ptr<Work> const&>;
    if (pending.empty()) { return result_type{}; }
    auto result = std::forward<Prepare>(prepare)(pending.front());
    pending.pop_front();
    ++inspections;
    return result;
  }
  void complete()
  {
    if (!outstanding) { throw std::logic_error("Sort work count underflow"); }
    --outstanding;
  }
  bool complete_stage() const { return outstanding == 0; }
};
}  // namespace

// Finite operator-local work and dependencies. All idle device owners live in
// registered repositories; these descriptors contain locations and host metadata.
struct sirius_physical_merge_sort::tiled_state {
  enum storage { INPUTS, SAMPLES, SPLITTERS, PIECES, RESULTS, STORAGE_COUNT };
  enum class phase { NORMALIZE, COALESCE, SAMPLE, ORDER, EXTRACT, TILES, MERGING, DONE };
  enum class kind { NORMALIZE, COALESCE, SAMPLE, ORDER, EXTRACT, MERGE, PUBLISH };
  enum class status { PENDING, CLAIMED, COMPLETE };
  static char const* name(kind type)
  {
    switch (type) {
      case kind::NORMALIZE: return "normalize";
      case kind::COALESCE: return "coalesce";
      case kind::SAMPLE: return "sample";
      case kind::ORDER: return "order_samples";
      case kind::EXTRACT: return "extract";
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
  struct ref {
    slot location;
    uint64_t id, rows, bytes;
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
    std::vector<piece> pieces;
    uint64_t rows{0};
    std::optional<ref> result;
  };
  struct request;
  struct work {
    status state{status::PENDING};
    std::weak_ptr<region> owner;
    std::shared_ptr<const request> execution;
  };
  struct region {
    phase stage{phase::NORMALIZE};
    std::vector<ref> inputs;
    tiled_merge_work_queue<work> work_items;
    bool enqueued{false};
    std::vector<std::vector<std::pair<uint64_t, ref>>> normalized;
    std::vector<std::shared_ptr<group>> groups;
    std::vector<std::optional<ref>> samples;
    std::vector<ref> pages;
    std::vector<tile> tiles;
    std::size_t creator_visits{0};
    sm::parameters sampling{};
    uint64_t row_limit{0}, splitter_count{0}, coalescing_inputs{0};
    std::size_t sample_partition{0};
  };
  // Immutable worker payloads, separate from coordinator bookkeeping.
  struct normalize_request {
    static constexpr kind type = kind::NORMALIZE;
    slot destination;
    uint64_t begin, end;
    std::size_t input;
  };
  struct coalesce_request {
    static constexpr kind type = kind::COALESCE;
    slot destination;
    std::size_t group_index;
  };
  struct sample_request {
    static constexpr kind type = kind::SAMPLE;
    slot destination;
    std::size_t input;
    uint64_t stride;
  };
  struct order_request {
    static constexpr kind type = kind::ORDER;
    slot destination;
    uint64_t sample_step;
  };
  struct extract_request {
    static constexpr kind type = kind::EXTRACT;
    std::size_t input;
    uint64_t begin, end, first_splitter, splitter_count;
    std::vector<slot> destinations;
  };
  struct merge_request {
    static constexpr kind type = kind::MERGE;
    slot destination;
    std::size_t tile_index;
  };
  struct publish_request {
    static constexpr kind type = kind::PUBLISH;
    std::vector<cudf::size_type> projection;
    std::size_t position;
  };
  using operation = std::variant<normalize_request,
                                 coalesce_request,
                                 sample_request,
                                 order_request,
                                 extract_request,
                                 merge_request,
                                 publish_request>;
  struct request {
    std::vector<ref> sources;
    std::shared_ptr<const sort_comparison_config> comparison;
    operation op;
    kind type() const
    {
      return std::visit([](auto const& value) { return value.type; }, op);
    }
  };
  struct produced_batch {
    slot destination;
    std::shared_ptr<cucascade::data_batch> batch;
    ref describe() const { return ref::describe(destination, *batch); }
  };
  struct normalize_result {
    produced_batch output;
    uint64_t begin;
  };
  struct coalesce_result {
    produced_batch output;
  };
  struct sample_result {
    produced_batch output;
  };
  struct order_result {
    std::vector<produced_batch> pages;
    uint64_t splitter_count{0};
  };
  struct extracted_piece {
    produced_batch output;
    std::size_t tile_index;
    uint64_t begin, end;
  };
  struct extract_result {
    std::vector<extracted_piece> pieces;
  };
  struct merge_result {
    produced_batch output;
  };
  struct publish_result {
    std::shared_ptr<cucascade::data_batch> batch;
  };
  using result = std::variant<normalize_result,
                              coalesce_result,
                              sample_result,
                              order_result,
                              extract_result,
                              merge_result,
                              publish_result>;

  // The adapter derives its runtime batch list from these paired results.
  template <typename Result, typename Visit>
  static void visit_batches(Result& result, Visit&& visit)
  {
    std::visit(
      [&](auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, order_result>) {
          for (auto& page : value.pages) {
            visit(page.batch);
          }
        } else if constexpr (std::is_same_v<T, extract_result>) {
          for (auto& piece : value.pieces) {
            visit(piece.output.batch);
          }
        } else if constexpr (std::is_same_v<T, publish_result>) {
          visit(value.batch);
        } else {
          visit(value.output.batch);
        }
      },
      result);
  }
  static std::vector<std::shared_ptr<cucascade::data_batch>> batches(result const& value)
  {
    std::vector<std::shared_ptr<cucascade::data_batch>> out;
    visit_batches(value, [&](auto const& batch) { out.push_back(batch); });
    return out;
  }
  struct task_data : pipelineable_operator_data {
    std::shared_ptr<work> item;
    std::shared_ptr<const request> execution;
    std::optional<result> completed;
    std::size_t admitted_bytes{0}, attempt_ceiling{0};
    task_data(std::vector<std::shared_ptr<cucascade::data_batch>> batches,
              std::shared_ptr<work> item,
              std::shared_ptr<const request> execution,
              std::optional<std::size_t> position = std::nullopt)
      : pipelineable_operator_data(std::move(batches), position),
        item(std::move(item)),
        execution(std::move(execution))
    {
    }
    std::unique_ptr<pipelineable_operator_data> with_batches_preserving_rows(
      std::vector<std::shared_ptr<cucascade::data_batch>> batches) const override
    {
      auto result =
        std::make_unique<task_data>(std::move(batches), item, execution, get_position());
      result->completed = completed;
      if (result->completed) {
        std::size_t index = 0;
        visit_batches(*result->completed,
                      [&](auto& batch) { batch = result->get_data_batches().at(index++); });
        if (index != result->get_data_batches().size()) {
          throw internal_exception("Sort result replacement changed batch count");
        }
      }
      result->admitted_bytes  = admitted_bytes;
      result->attempt_ceiling = attempt_ceiling;
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
    for (auto const& source : data.execution->sources) {
      bytes = sm::checked_add(bytes, source.bytes);
    }
    auto page = data.execution->type() == kind::EXTRACT ? data.execution->sources.back().bytes : 0;
    return std::format(
      "Tiled sort {} {}: {}; owning_backing={}, page_bytes={}, granted={}, "
      "effective_ceiling={}, configured_ceiling={}",
      name(data.execution->type()),
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
    return r.stage != phase::DONE && (!r.work_items.pending.empty() || complete(r));
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
    if (!unfinished_roots) { throw internal_exception("Sort root count underflow"); }
    --unfinished_roots;
  }
  static void retire(region const&, ref const& input) { input.retire(); }
  static std::shared_ptr<work> make_work(sirius_physical_merge_sort const& op,
                                         std::shared_ptr<region> const& r,
                                         std::vector<ref> sources,
                                         operation payload)
  {
    auto item       = std::make_shared<work>();
    item->owner     = r;
    item->execution = std::make_shared<const request>(
      request{std::move(sources), op.comparison, std::move(payload)});
    return item;
  }
  void flatten_results(sirius_physical_merge_sort const& op,
                       std::shared_ptr<region> const& r,
                       uint64_t& position)
  {
    for (auto const& tile : r->tiles) {
      if (!tile.result) { continue; }
      finals.add(make_work(
        op,
        r,
        {*tile.result},
        publish_request{{op._final_projections.begin(), op._final_projections.end()}, position}));
      position = sm::checked_add(position, 1);
    }
  }
  void add(sirius_physical_merge_sort const& op,
           std::shared_ptr<region> const& r,
           std::vector<ref> sources,
           operation payload)
  {
    r->work_items.add(make_work(op, r, std::move(sources), std::move(payload)));
    enqueue(r);
  }
  void coalesce(sirius_physical_merge_sort& op, std::shared_ptr<region> const& r, std::size_t index)
  {
    auto const& g = r->groups.at(index);
    if (g->inputs.size() == 1) {
      g->output = g->inputs.front();
      return;
    }
    add(op, r, g->inputs, coalesce_request{allocate(op, INPUTS), index});
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
            add(op, r, {input}, normalize_request{allocate(op, INPUTS), begin, end, i});
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
          coalesce(op, r, r->groups.size() - 1);
        }
      } else if (r->stage == phase::COALESCE) {
        r->inputs.clear();
        for (auto const& g : r->groups) {
          r->inputs.push_back(g->output.value());
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
        SIRIUS_LOG_DEBUG("sort.tiles pre_coalesce={}, inputs={}, rows={}, bytes={}, row_limit={}",
                         r->coalescing_inputs,
                         r->inputs.size(),
                         rows,
                         bytes,
                         r->row_limit);
        if (rows <= r->row_limit) {
          auto location = allocate(op, PIECES);
          r->tiles.push_back({location.partition, {}, rows});
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
          add(op,
              r,
              {r->inputs[i]},
              sample_request{
                {op._tile_storage[SAMPLES], r->sample_partition, i}, i, r->sampling.stride});
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
        add(op,
            r,
            std::move(samples),
            order_request{allocate(op, SPLITTERS), r->sampling.sample_step});
        r->stage = phase::ORDER;
      } else if (r->stage == phase::ORDER) {
        for (uint64_t t = 0; t <= r->splitter_count; ++t) {
          auto location = allocate(op, PIECES);
          r->tiles.push_back({location.partition, {}, 0});
        }
        for (std::size_t i = 0; i < r->inputs.size(); ++i) {
          for (std::size_t page = 0; page < r->pages.size(); ++page) {
            auto begin = page * window_tiles;
            auto end   = std::min(begin + window_tiles, r->splitter_count + 1);
            std::vector<slot> destinations;
            for (auto t = begin; t < end; ++t) {
              destinations.push_back({op._tile_storage[PIECES], r->tiles[t].partition, i});
            }
            add(
              op,
              r,
              {r->inputs[i], r->pages[page]},
              extract_request{
                i, begin, end, begin ? begin - 1 : 0, r->splitter_count, std::move(destinations)});
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
          add(op, r, std::move(sources), merge_request{allocate(op, RESULTS), t});
        }
        r->stage = phase::MERGING;
      } else if (r->stage == phase::MERGING) {
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
    auto data = r->work_items.claim([this](auto const& item) {
      std::vector<std::shared_ptr<cucascade::data_batch>> batches;
      batches.reserve(item->execution->sources.size());
      for (auto const& input : item->execution->sources) {
        batches.push_back(input.get());
      }
      return std::make_unique<tiled_state::task_data>(std::move(batches), item, item->execution);
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
        _tiled_state->flatten_results(*this, root, position);
      }
      _tiled_state->publishing = true;
      if (_tiled_state->finals.complete_stage()) { _tiled_state->finished = true; }
    }
    auto task = _tiled_state->finals.claim([this](auto const& item) {
      return std::make_unique<tiled_state::task_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{item->execution->sources.front().get()},
        item,
        item->execution,
        std::get<tiled_state::publish_request>(item->execution->op).position);
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
  auto started           = std::chrono::steady_clock::now();
  using state            = tiled_state;
  auto const& input      = dynamic_cast<state::task_data const&>(input_data);
  auto const& request    = *input.execution;
  auto const& comparison = request.comparison;
  auto reads             = input.get_read_only_batches();
  if (reads.empty()) { throw internal_exception("Tiled sort task has no inputs"); }
  // INVARIANT: gpu_pipeline_task -> prepare_for_processing -> lock_or_prepare_batch
  // restores exactly this item's inputs onto the task's reservation device.
  auto* space = reads.front().get_memory_space();
  auto mr     = space->get_default_allocator();
  // Keep partial outputs alive until the exception fence has completed.
  state::result completed;
  auto emit = [&](std::unique_ptr<cudf::table> table, state::slot destination) {
    return state::produced_batch{
      destination, make_data_batch(std::move(table), *space, stream, batch_telemetry())};
  };
  std::vector<cudf::table_view> views;
  for (auto const& read : reads) {
    views.push_back(get_cudf_table_view(read));
  }
  nvtx_scoped_range nvtx{"sort.tile_work"};
  try {
    std::visit(
      [&](auto const& item) {
        if constexpr (std::decay_t<decltype(item)>::type == state::kind::PUBLISH) {
          auto& value = completed.emplace<state::publish_result>();
          if (item.projection.empty()) {
            value.batch = input.get_data_batches().front();
          } else {
            value.batch = make_data_batch(
              std::make_unique<cudf::table>(views.front().select(item.projection), stream, mr),
              *space,
              stream,
              batch_telemetry());
          }
        } else if constexpr (std::decay_t<decltype(item)>::type == state::kind::MERGE) {
          auto& value = completed.emplace<state::merge_result>();
          if (views.size() == 1) {
            value.output = {item.destination, input.get_data_batches().front()};
          } else {
            value.output = emit(cudf::merge(views,
                                            comparison->key_columns,
                                            comparison->directions,
                                            comparison->null_precedence,
                                            stream,
                                            mr),
                                item.destination);
          }
        } else if constexpr (std::decay_t<decltype(item)>::type == state::kind::NORMALIZE) {
          auto& value = completed.emplace<state::normalize_result>();
          value.begin = item.begin;
          auto view   = cudf::slice(
            views.front(),
            {static_cast<cudf::size_type>(item.begin), static_cast<cudf::size_type>(item.end)},
            stream)[0];
          value.output = emit(std::make_unique<cudf::table>(view, stream, mr), item.destination);
        } else if constexpr (std::decay_t<decltype(item)>::type == state::kind::COALESCE) {
          auto& value  = completed.emplace<state::coalesce_result>();
          value.output = emit(cudf::merge(views,
                                          comparison->key_columns,
                                          comparison->directions,
                                          comparison->null_precedence,
                                          stream,
                                          mr),
                              item.destination);
        } else if constexpr (std::decay_t<decltype(item)>::type == state::kind::SAMPLE) {
          auto& value  = completed.emplace<state::sample_result>();
          value.output = emit(
            sm::gather_samples(views.front(), item.input, item.stride, *comparison, stream, mr),
            item.destination);
        } else if constexpr (std::decay_t<decltype(item)>::type == state::kind::ORDER) {
          auto& value = completed.emplace<state::order_result>();
          auto splitters =
            sm::order_sample_splitters(views, item.sample_step, *comparison, stream, mr);
          value.splitter_count = splitters->num_rows();
          auto count           = value.splitter_count + 1;
          value.pages.reserve(1 + (count - 1) / state::window_tiles);
          for (uint64_t begin = 0, page = 0; begin < count; begin += state::window_tiles, ++page) {
            auto end             = std::min(begin + state::window_tiles, count);
            auto first           = begin ? begin - 1 : 0;
            auto last            = std::min(end, value.splitter_count);
            auto destination     = item.destination;
            destination.position = page;
            if (!page && end == count) {
              value.pages.push_back(emit(std::move(splitters), destination));
            } else {
              auto view = cudf::slice(
                splitters->view(),
                {static_cast<cudf::size_type>(first), static_cast<cudf::size_type>(last)},
                stream)[0];
              value.pages.push_back(
                emit(std::make_unique<cudf::table>(view, stream, mr), destination));
            }
          }
        } else if constexpr (std::decay_t<decltype(item)>::type == state::kind::EXTRACT) {
          auto& value = completed.emplace<state::extract_result>();
          value.pieces.reserve(item.end - item.begin);
          auto cuts = sm::discover_cuts(views[0], item.input, views[1], *comparison, stream, mr);
          std::vector<cudf::size_type> host(cuts->size());
          // One bulk integer readback, including adjacent page boundaries. Keep the
          // host destination alive through the exception fence as well.
          try {
            CUDF_CUDA_TRY(cudaMemcpyAsync(host.data(),
                                          cuts->view().template data<cudf::size_type>(),
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
              t == item.splitter_count ? views[0].num_rows() : host.at(t - item.first_splitter);
            if (begin > end || begin < 0 || end > views[0].num_rows()) {
              throw internal_exception("Nonmonotone tiled sort cuts");
            }
            if (begin == end) { continue; }
            auto view = cudf::slice(views[0], {begin, end}, stream)[0];
            value.pieces.push_back({emit(std::make_unique<cudf::table>(view, stream, mr),
                                         item.destinations.at(t - item.begin)),
                                    static_cast<std::size_t>(t),
                                    static_cast<uint64_t>(begin),
                                    static_cast<uint64_t>(end)});
          }
          SIRIUS_LOG_TRACE(
            "sort.extract input={}, tile_begin={}, tile_end={}, readbacks=1, waits=1, pieces={}",
            item.input,
            item.begin,
            item.end,
            value.pieces.size());
        }
      },
      request.op);
  } catch (rmm::out_of_memory const&) {
    cudaStreamSynchronize(stream.get());
    if (input.admitted_bytes < input.attempt_ceiling) { throw; }
    throw invalid_input_exception(
      state::failure(input, "compute", reservation_limit, "allocation envelope exceeded"));
  } catch (...) {
    cudaStreamSynchronize(stream.get());
    throw;
  }
  auto output = std::make_unique<state::task_data>(
    state::batches(completed), input.item, input.execution, input.get_position());
  output->completed     = std::move(completed);
  uint64_t output_bytes = 0, output_rows = 0;
  for (auto const& batch : output->get_data_batches()) {
    auto metadata = get_batch_rows_and_bytes(*batch);
    output_bytes  = sm::checked_add(output_bytes, metadata.bytes);
    output_rows   = sm::checked_add(output_rows, metadata.rows.value_or(0));
  }
  SIRIUS_LOG_TRACE(
    "sort.tile_work kind={}, inputs={}, outputs={}, output_rows={}, output_bytes={}, splitters={}, "
    "enqueue_ms={}",
    state::name(request.type()),
    request.sources.size(),
    output->get_data_batches().size(),
    output_rows,
    output_bytes,
    std::holds_alternative<state::order_result>(*output->completed)
      ? std::get<state::order_result>(*output->completed).splitter_count
      : 0,
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
  return output;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::recover_preparation_failure(
  operator_data const& input, task_attempt_context attempt)
{
  auto const& data = dynamic_cast<tiled_state::task_data const&>(input);
  if (attempt.granted_bytes < attempt.effective_ceiling) { return nullptr; }

  throw invalid_input_exception(tiled_state::failure(
    data, "preparation", reservation_limit, "source/page restoration envelope exceeded"));
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
  if (data.execution->type() == tiled_state::kind::PUBLISH) {
    sirius_physical_operator::sink(output, stream);
    return;
  }
  auto publish = [](tiled_state::produced_batch const& output) {
    auto const& location = output.destination;
    location.repo->add_data_batch_at_position(output.batch, location.position, location.partition);
  };
  std::visit(
    [&](auto const& value) {
      using T = std::decay_t<decltype(value)>;
      if constexpr (std::is_same_v<T, tiled_state::order_result>) {
        for (auto const& page : value.pages) {
          publish(page);
        }
      } else if constexpr (std::is_same_v<T, tiled_state::extract_result>) {
        for (auto const& piece : value.pieces) {
          publish(piece.output);
        }
      } else if constexpr (!std::is_same_v<T, tiled_state::publish_result>) {
        publish(value.output);
      }
    },
    data.completed.value());
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
  std::visit(
    [&](auto const& value) {
      using T = std::decay_t<decltype(value)>;
      if constexpr (std::is_same_v<T, state::normalize_result>) {
        r->normalized.at(std::get<state::normalize_request>(item.execution->op).input)
          .push_back({value.begin, value.output.describe()});
      } else if constexpr (std::is_same_v<T, state::coalesce_result>) {
        r->groups.at(std::get<state::coalesce_request>(item.execution->op).group_index)->output =
          value.output.describe();
        for (auto const& source : item.execution->sources) {
          source.retire();
        }
      } else if constexpr (std::is_same_v<T, state::sample_result>) {
        r->samples.at(std::get<state::sample_request>(item.execution->op).input) =
          value.output.describe();
      } else if constexpr (std::is_same_v<T, state::order_result>) {
        for (auto const& page : value.pages) {
          r->pages.push_back(page.describe());
        }
        r->splitter_count = value.splitter_count;
      } else if constexpr (std::is_same_v<T, state::merge_result>) {
        r->tiles.at(std::get<state::merge_request>(item.execution->op).tile_index).result =
          value.output.describe();
        for (auto const& source : item.execution->sources) {
          source.retire();
        }
        r->tiles.at(std::get<state::merge_request>(item.execution->op).tile_index).pieces.clear();
      } else if constexpr (std::is_same_v<T, state::extract_result>) {
        for (auto const& piece : value.pieces) {
          r->tiles.at(piece.tile_index)
            .pieces.push_back({piece.output.describe(),
                               std::get<state::extract_request>(item.execution->op).input,
                               piece.begin,
                               piece.end});
        }
      }
    },
    data.completed.value());
  item.state = state::status::COMPLETE;
  if (item.execution->type() != state::kind::PUBLISH) {
    r->work_items.complete();
    _tiled_state->enqueue(r);
  }
  if (item.execution->type() == state::kind::PUBLISH) {
    item.execution->sources.front().retire();
    _tiled_state->finals.complete();
    _tiled_state->finished = _tiled_state->finals.complete_stage();
  }
}

}  // namespace sirius::op
