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

  void add(std::shared_ptr<Work> item) { pending.push_back(std::move(item)); }
  // Prepare task-owned metadata before committing the claim. Allocation failure
  // leaves both the pending item and its outstanding count intact.
  template <typename Prepare>
  auto claim(Prepare&& prepare)
  {
    using result_type = std::invoke_result_t<Prepare, std::shared_ptr<Work> const&>;
    if (pending.empty()) { return result_type{}; }
    auto result = std::forward<Prepare>(prepare)(pending.front());
    pending.pop_front();
    ++outstanding;
    return result;
  }
  void complete()
  {
    if (!outstanding) { throw std::logic_error("Sort work count underflow"); }
    --outstanding;
  }
  bool complete_stage() const { return pending.empty() && outstanding == 0; }
};
}  // namespace

// Finite operator-local work and dependencies. All idle device owners live in
// registered repositories; these descriptors contain locations and host metadata.
struct sirius_physical_merge_sort::tiled_state {
  enum storage { INPUTS, SAMPLES, SPLITTERS, PIECES, RESULTS, STORAGE_COUNT };
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
  };
  // Repository operations stay here; handles never own a GPU view or read lock.
  // Repository retirement removes discoverability only after the owning stage's
  // publication/device-completion barrier. Existing readers retain their owners.
  struct storage_access {
    std::vector<cucascade::shared_data_repository*> repositories;
    std::vector<std::size_t> partitions = std::vector<std::size_t>(STORAGE_COUNT, 0);

    explicit storage_access(std::vector<cucascade::shared_data_repository*> repos)
      : repositories(std::move(repos))
    {
    }
    slot allocate(storage which)
    {
      auto& next = partitions.at(which);
      auto index = next;
      next       = sm::checked_add(next, 1);
      auto* repo = repositories.at(which);
      if (next > repo->num_partitions()) { repo->set_num_partitions(next); }
      return {repo, index, 0};
    }
    static ref describe(slot location, cucascade::data_batch const& batch)
    {
      auto info = get_batch_rows_and_bytes(batch);
      if (!info.rows) { throw invalid_input_exception("Sort input lacks row-count metadata"); }
      return {location, batch.get_batch_id(), *info.rows, info.bytes};
    }
    static auto resolve(ref const& input)
    {
      auto batch = input.location.repo->get_data_batch_by_id(input.id, input.location.partition);
      if (!batch) { throw internal_exception("Missing tiled sort batch"); }
      return batch;
    }
    static void publish(slot destination, std::shared_ptr<cucascade::data_batch> const& batch)
    {
      destination.repo->add_data_batch_at_position(
        batch, destination.position, destination.partition);
    }
    static void retire(ref const& input)
    {
      input.location.repo->pop_data_batch_by_id(input.id, input.location.partition);
    }
    static void retire_all(std::vector<ref> const& inputs)
    {
      for (auto const& input : inputs) {
        retire(input);
      }
    }
    // Called only after the replacement has been published and the device work
    // completed. Forwarded batches may retain their id in a new repository.
    static ref commit_replacement(std::vector<ref> const& inputs,
                                  slot destination,
                                  cucascade::data_batch const& output)
    {
      auto replacement = describe(destination, output);
      for (auto const& input : inputs) {
        if (input.location.repo != replacement.location.repo ||
            input.location.partition != replacement.location.partition ||
            input.id != replacement.id) {
          retire(input);
        }
      }
      return replacement;
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
  struct range_state;
  struct tile {
    std::size_t partition;
    std::vector<piece> pieces;
    uint64_t rows{0};
    std::optional<ref> result;
  };
  struct request;
  struct work {
    status state{status::PENDING};
    std::weak_ptr<range_state> owner;
    std::shared_ptr<const request> execution;
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
    ref describe() const { return storage_access::describe(destination, *batch); }
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

  struct stage_work {
    tiled_merge_work_queue<work> queue;
  };
  struct normalize_stage : stage_work {
    std::vector<ref> inputs;
    std::vector<std::vector<std::pair<uint64_t, ref>>> normalized;
    void accept(normalize_result const& value, request const& task)
    {
      normalized.at(std::get<normalize_request>(task.op).input)
        .push_back({value.begin, value.output.describe()});
    }
  };
  struct coalesce_stage : stage_work {
    std::vector<group> groups;
    std::size_t input_count{0};
    void accept(coalesce_result const& value, request const& task)
    {
      groups.at(std::get<coalesce_request>(task.op).group_index).output =
        storage_access::commit_replacement(
          task.sources, value.output.destination, *value.output.batch);
    }
  };
  struct sample_stage : stage_work {
    std::vector<ref> inputs;
    uint64_t row_limit;
    sm::parameters sampling;
    std::vector<std::optional<ref>> samples;
    void accept(sample_result const& value, request const& task)
    {
      samples.at(std::get<sample_request>(task.op).input) = value.output.describe();
    }
  };
  struct order_stage : stage_work {
    std::vector<ref> inputs, samples, pages;
    uint64_t row_limit, splitter_count{0};
    void accept(order_result const& value, request const&)
    {
      for (auto const& page : value.pages) {
        pages.push_back(page.describe());
      }
      splitter_count = value.splitter_count;
    }
  };
  struct extract_stage : stage_work {
    std::vector<ref> inputs, pages;
    std::vector<tile> tiles;
    uint64_t row_limit;
    void accept(extract_result const& value, request const& task)
    {
      for (auto const& piece : value.pieces) {
        tiles.at(piece.tile_index)
          .pieces.push_back({piece.output.describe(),
                             std::get<extract_request>(task.op).input,
                             piece.begin,
                             piece.end});
      }
    }
  };
  struct merge_stage : stage_work {
    std::vector<tile> tiles;
    void accept(merge_result const& value, request const& task)
    {
      auto& tile  = tiles.at(std::get<merge_request>(task.op).tile_index);
      tile.result = storage_access::commit_replacement(
        task.sources, value.output.destination, *value.output.batch);
      tile.pieces.clear();
    }
  };
  struct done_stage {
    std::vector<ref> results;
    void accept(publish_result const&, request const&) {}
  };
  using active_stage = std::variant<normalize_stage,
                                    coalesce_stage,
                                    sample_stage,
                                    order_stage,
                                    extract_stage,
                                    merge_stage,
                                    done_stage>;
  struct range_state {
    active_stage stage{normalize_stage{}};
    bool enqueued{false};
    std::size_t creator_visits{0}, claims{0};
    auto* queue()
    {
      return std::visit(
        [](auto& value) -> tiled_merge_work_queue<work>* {
          if constexpr (std::is_same_v<std::decay_t<decltype(value)>, done_stage>) {
            return nullptr;
          } else {
            return &value.queue;
          }
        },
        stage);
    }
    bool actionable()
    {
      auto* work = queue();
      return work && (!work->pending.empty() || work->complete_stage());
    }
    void accept(result const& output, request const& task)
    {
      std::visit(
        [&](auto& active) {
          std::visit(
            [&](auto const& value) {
              if constexpr (requires { active.accept(value, task); }) {
                active.accept(value, task);
              } else {
                throw internal_exception("Sort result does not match active stage");
              }
            },
            output);
        },
        stage);
    }
  };

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

  std::vector<std::shared_ptr<range_state>> ranges;
  std::deque<std::shared_ptr<range_state>> ready_ranges;
  tiled_merge_work_queue<work> finals;
  std::size_t unfinished_ranges{0};
  bool publishing{false}, finished{false};
  storage_access scratch;
  explicit tiled_state(std::vector<cucascade::shared_data_repository*> repositories)
    : scratch(std::move(repositories))
  {
  }
  void enqueue(std::shared_ptr<range_state> const& r)
  {
    if (!r->enqueued && r->actionable()) {
      ready_ranges.push_back(r);
      r->enqueued = true;
    }
  }
  bool ready() const
  {
    return !finished && (!ready_ranges.empty() || (!unfinished_ranges && !publishing) ||
                         !finals.pending.empty());
  }
  void finish(std::shared_ptr<range_state> const& r, std::vector<ref> results)
  {
    SIRIUS_LOG_DEBUG("sort.tile_schedule claims={}, range_visits={}", r->claims, r->creator_visits);
    r->stage = done_stage{std::move(results)};
    if (!unfinished_ranges) { throw internal_exception("Sort range count underflow"); }
    --unfinished_ranges;
  }
  static std::shared_ptr<work> make_work(sirius_physical_merge_sort const& op,
                                         std::shared_ptr<range_state> const& r,
                                         std::vector<ref> sources,
                                         operation payload)
  {
    auto item       = std::make_shared<work>();
    item->owner     = r;
    item->execution = std::make_shared<const request>(
      request{std::move(sources), op.comparison, std::move(payload)});
    return item;
  }
  void schedule_publication(sirius_physical_merge_sort const& op,
                            std::shared_ptr<range_state> const& r,
                            uint64_t& position)
  {
    for (auto const& result : std::get<done_stage>(r->stage).results) {
      finals.add(make_work(
        op,
        r,
        {result},
        publish_request{{op._final_projections.begin(), op._final_projections.end()}, position}));
      position = sm::checked_add(position, 1);
    }
    std::get<done_stage>(r->stage).results.clear();
  }
  void add(sirius_physical_merge_sort const& op,
           std::shared_ptr<range_state> const& r,
           std::vector<ref> sources,
           operation payload)
  {
    r->queue()->add(make_work(op, r, std::move(sources), std::move(payload)));
    enqueue(r);
  }
  void start_merge(sirius_physical_merge_sort& op,
                   std::shared_ptr<range_state> const& r,
                   std::vector<tile> tiles)
  {
    merge_stage next;
    next.tiles  = std::move(tiles);
    r->stage    = std::move(next);
    auto& stage = std::get<merge_stage>(r->stage);
    for (std::size_t t = 0; t < stage.tiles.size(); ++t) {
      auto const& tile = stage.tiles[t];
      if (!tile.rows) { continue; }
      std::vector<ref> sources;
      for (auto const& piece : tile.pieces) {
        sources.push_back(piece.batch);
      }
      add(op, r, std::move(sources), merge_request{scratch.allocate(RESULTS), t});
    }
  }

  void finish_stage(normalize_stage& stage,
                    sirius_physical_merge_sort& op,
                    std::shared_ptr<range_state> const& r)
  {
    if (!stage.normalized.empty()) {
      std::vector<ref> next;
      for (std::size_t i = 0; i < stage.inputs.size(); ++i) {
        auto& replacements = stage.normalized[i];
        if (replacements.empty()) {
          next.push_back(stage.inputs[i]);
          continue;
        }
        std::sort(replacements.begin(), replacements.end(), [](auto const& a, auto const& b) {
          return a.first < b.first;
        });
        for (auto const& replacement : replacements) {
          next.push_back(replacement.second);
        }
        storage_access::retire(stage.inputs[i]);
      }
      stage.inputs = std::move(next);
      stage.normalized.clear();
    }
    stage.normalized.resize(stage.inputs.size());
    for (std::size_t i = 0; i < stage.inputs.size(); ++i) {
      auto const& input = stage.inputs[i];
      if (input.bytes <= op.chunk_bytes || input.rows <= 1) { continue; }
      for (auto [begin, end] :
           {std::pair{uint64_t{0}, input.rows / 2}, std::pair{input.rows / 2, input.rows}}) {
        add(op, r, {input}, normalize_request{scratch.allocate(INPUTS), begin, end, i});
      }
    }
    if (!stage.queue.complete_stage()) { return; }
    coalesce_stage next;
    next.input_count = stage.inputs.size();
    std::vector<sm::contribution_size> sizes;
    for (auto const& input : stage.inputs) {
      sizes.push_back({input.rows, input.bytes});
    }
    for (auto planned : sm::plan_coalescing(sizes, op.chunk_bytes, native_rows)) {
      group g;
      g.inputs.assign(stage.inputs.begin() + planned.begin, stage.inputs.begin() + planned.end);
      next.groups.push_back(std::move(g));
    }
    r->stage         = std::move(next);
    auto& coalescing = std::get<coalesce_stage>(r->stage);
    for (std::size_t i = 0; i < coalescing.groups.size(); ++i) {
      auto& g = coalescing.groups[i];
      if (g.inputs.size() == 1) {
        g.output = g.inputs.front();
      } else {
        add(op, r, g.inputs, coalesce_request{scratch.allocate(INPUTS), i});
      }
    }
  }

  void finish_stage(coalesce_stage& stage,
                    sirius_physical_merge_sort& op,
                    std::shared_ptr<range_state> const& r)
  {
    std::vector<ref> inputs;
    uint64_t rows = 0, bytes = 0;
    for (auto const& g : stage.groups) {
      inputs.push_back(g.output.value());
      rows  = sm::checked_add(rows, inputs.back().rows);
      bytes = sm::checked_add(bytes, inputs.back().bytes);
    }
    if (!rows) {
      finish(r, {});
      return;
    }
    auto average   = bytes ? 1 + (bytes - 1) / rows : uint64_t{1};
    auto row_limit = std::min(native_rows, std::max(uint64_t{1}, op.chunk_bytes / average));
    SIRIUS_LOG_DEBUG("sort.tiles pre_coalesce={}, inputs={}, rows={}, bytes={}, row_limit={}",
                     stage.input_count,
                     inputs.size(),
                     rows,
                     bytes,
                     row_limit);
    if (rows <= row_limit) {
      tile single{scratch.allocate(PIECES).partition, {}, rows};
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        single.pieces.push_back({inputs[i], i, 0, inputs[i].rows});
      }
      std::vector<tile> tiles;
      tiles.push_back(std::move(single));
      start_merge(op, r, std::move(tiles));
      return;
    }
    sample_stage next;
    next.row_limit = row_limit;
    next.sampling  = sm::choose_parameters(inputs.size(), row_limit);
    next.samples.resize(inputs.size());
    next.inputs    = std::move(inputs);
    r->stage       = std::move(next);
    auto& sampling = std::get<sample_stage>(r->stage);
    auto partition = scratch.allocate(SAMPLES).partition;
    for (std::size_t i = 0; i < sampling.inputs.size(); ++i) {
      if (!sm::sample_count(sampling.inputs[i].rows, sampling.sampling.stride)) { continue; }
      add(
        op,
        r,
        {sampling.inputs[i]},
        sample_request{{scratch.repositories[SAMPLES], partition, i}, i, sampling.sampling.stride});
    }
  }

  void finish_stage(sample_stage& stage,
                    sirius_physical_merge_sort& op,
                    std::shared_ptr<range_state> const& r)
  {
    order_stage next;
    uint64_t rows = 0;
    for (auto const& sample : stage.samples) {
      if (!sample) { continue; }
      next.samples.push_back(*sample);
      rows = sm::checked_add(rows, sample->rows);
    }
    if (rows > native_rows) {
      throw invalid_input_exception("Sort sample row count exceeds the native table limit");
    }
    auto sample_step = stage.sampling.sample_step;
    next.inputs      = std::move(stage.inputs);
    next.row_limit   = stage.row_limit;
    r->stage         = std::move(next);
    add(op,
        r,
        std::get<order_stage>(r->stage).samples,
        order_request{scratch.allocate(SPLITTERS), sample_step});
  }

  void finish_stage(order_stage& stage,
                    sirius_physical_merge_sort& op,
                    std::shared_ptr<range_state> const& r)
  {
    extract_stage next;
    auto splitter_count = stage.splitter_count;
    next.row_limit      = stage.row_limit;
    next.inputs         = std::move(stage.inputs);
    next.pages          = std::move(stage.pages);
    for (uint64_t t = 0; t <= splitter_count; ++t) {
      next.tiles.push_back({scratch.allocate(PIECES).partition, {}, 0});
    }
    storage_access::retire_all(stage.samples);
    r->stage         = std::move(next);
    auto& extracting = std::get<extract_stage>(r->stage);
    for (std::size_t i = 0; i < extracting.inputs.size(); ++i) {
      for (std::size_t page = 0; page < extracting.pages.size(); ++page) {
        auto begin = page * window_tiles;
        auto end   = std::min(begin + window_tiles, splitter_count + 1);
        std::vector<slot> destinations;
        for (auto t = begin; t < end; ++t) {
          destinations.push_back({scratch.repositories[PIECES], extracting.tiles[t].partition, i});
        }
        add(op,
            r,
            {extracting.inputs[i], extracting.pages[page]},
            extract_request{
              i, begin, end, begin ? begin - 1 : 0, splitter_count, std::move(destinations)});
      }
    }
  }

  void finish_stage(extract_stage& stage,
                    sirius_physical_merge_sort& op,
                    std::shared_ptr<range_state> const& r)
  {
    std::vector<std::vector<std::pair<std::size_t, piece>>> coverage(stage.inputs.size());
    for (std::size_t t = 0; t < stage.tiles.size(); ++t) {
      auto& tile = stage.tiles[t];
      std::sort(tile.pieces.begin(), tile.pieces.end(), [](piece const& a, piece const& b) {
        return std::tie(a.input, a.begin) < std::tie(b.input, b.begin);
      });
      for (auto const& piece : tile.pieces) {
        tile.rows = sm::checked_add(tile.rows, piece.batch.rows);
        coverage[piece.input].push_back({t, piece});
      }
      if (tile.rows > stage.row_limit) {
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
      if (cursor != stage.inputs[i].rows) {
        throw internal_exception("Incomplete sort extraction");
      }
    }
    storage_access::retire_all(stage.inputs);
    storage_access::retire_all(stage.pages);
    start_merge(op, r, std::move(stage.tiles));
  }

  void finish_stage(merge_stage& stage,
                    sirius_physical_merge_sort&,
                    std::shared_ptr<range_state> const& r)
  {
    std::vector<ref> results;
    for (auto const& tile : stage.tiles) {
      if (tile.rows) { results.push_back(tile.result.value()); }
    }
    finish(r, std::move(results));
  }
  void finish_stage(done_stage&, sirius_physical_merge_sort&, std::shared_ptr<range_state> const&)
  {
  }

  void advance(sirius_physical_merge_sort& op, std::shared_ptr<range_state> const& r)
  {
    // Empty stages advance synchronously; only pending or in-flight work suspends
    // the range. A transition never scans or wakes unrelated ranges.
    while (auto* queue = r->queue()) {
      if (!queue->complete_stage()) { return; }
      std::visit([&](auto& stage) { finish_stage(stage, op, r); }, r->stage);
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
    _tiled_state = std::make_shared<tiled_state>(_tile_storage);
    auto* repo   = get_port("default")->repo;
    for (std::size_t p = 0; p < repo->num_partitions(); ++p) {
      auto r   = std::make_shared<tiled_state::range_state>();
      auto ids = repo->get_batch_ids(p);
      std::sort(ids.begin(), ids.end());
      for (auto id : ids) {
        auto batch = repo->get_data_batch_by_id(id, p);
        auto input = tiled_state::storage_access::describe({repo, p, 0}, *batch);
        if (input.rows) {
          std::get<tiled_state::normalize_stage>(r->stage).inputs.push_back(input);
        } else {
          tiled_state::storage_access::retire(input);
        }
      }
      _tiled_state->ranges.push_back(r);
      ++_tiled_state->unfinished_ranges;
      _tiled_state->enqueue(r);
    }
    SIRIUS_LOG_DEBUG(
      "sort.tile_ranges ranges={}, nonempty_ranges={}",
      _tiled_state->ranges.size(),
      std::count_if(_tiled_state->ranges.begin(), _tiled_state->ranges.end(), [](auto const& r) {
        return !std::get<tiled_state::normalize_stage>(r->stage).inputs.empty();
      }));
  }
  while (!_tiled_state->ready_ranges.empty()) {
    auto r = std::move(_tiled_state->ready_ranges.front());
    _tiled_state->ready_ranges.pop_front();
    r->enqueued = false;
    ++r->creator_visits;
    _tiled_state->advance(*this, r);
    if (!r->queue()) { continue; }
    auto data = r->queue()->claim([](auto const& item) {
      std::vector<std::shared_ptr<cucascade::data_batch>> batches;
      batches.reserve(item->execution->sources.size());
      for (auto const& input : item->execution->sources) {
        batches.push_back(tiled_state::storage_access::resolve(input));
      }
      return std::make_unique<tiled_state::task_data>(std::move(batches), item, item->execution);
    });
    if (data) {
      data->item->state = tiled_state::status::CLAIMED;
      ++r->claims;
      _tiled_state->enqueue(r);
      return data;
    }
    _tiled_state->enqueue(r);
  }
  if (!_tiled_state->unfinished_ranges) {
    if (!_tiled_state->publishing) {
      uint64_t position = 0;
      for (auto const& root : _tiled_state->ranges) {
        _tiled_state->schedule_publication(*this, root, position);
      }
      _tiled_state->publishing = true;
      if (_tiled_state->finals.complete_stage()) { _tiled_state->finished = true; }
    }
    auto task = _tiled_state->finals.claim([](auto const& item) {
      return std::make_unique<tiled_state::task_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{
          tiled_state::storage_access::resolve(item->execution->sources.front())},
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
    tiled_state::storage_access::publish(output.destination, output.batch);
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
  if (!r) { throw internal_exception("Missing sort range at completion"); }
  r->accept(data.completed.value(), *item.execution);
  item.state = state::status::COMPLETE;
  if (item.execution->type() != state::kind::PUBLISH) {
    r->queue()->complete();
    _tiled_state->enqueue(r);
  }
  if (item.execution->type() == state::kind::PUBLISH) {
    state::storage_access::retire_all(item.execution->sources);
    _tiled_state->finals.complete();
    _tiled_state->finished = _tiled_state->finals.complete_stage();
  }
}

}  // namespace sirius::op
