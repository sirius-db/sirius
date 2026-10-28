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

#pragma once

#include "event/downgraded_batch.hpp"
#include "exec/queue_priority.hpp"
#include "preload/config.hpp"
#include "preload/preload_policy.hpp"
#include "preload/preload_stats.hpp"
#include "query_id.hpp"

#include <cucascade/memory/common.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cucascade {
class data_batch;
class data_repository;
}  // namespace cucascade

namespace sirius::memory {
class device_preload_budget;
}  // namespace sirius::memory

namespace sirius::op {
class sirius_physical_partition_consumer_operator;
}  // namespace sirius::op

namespace sirius::pipeline {
class sirius_pipeline;
}  // namespace sirius::pipeline

namespace sirius::preload {

/**
 * @brief A batch the downgrade executor moved down, which preload may move back up.
 *
 * Holds the batch weakly; @c location is a hint that the worker checks under the batch lock.
 */
struct candidate {
  std::weak_ptr<cucascade::data_batch> batch;
  std::uint64_t batch_id{0};
  query_id_t query{};
  /// Newest conversion sequence applied; records with a smaller one only fill gaps.
  std::uint64_t last_seq{0};
  cucascade::memory::memory_space_id location{cucascade::memory::Tier::SIZE, -1};
  /// First GPU the batch was evicted from (device id).
  std::optional<int> origin_gpu;
  /// HOST the batch lived on before DISK (device id).
  std::optional<int> origin_host;
  /// Largest size over its records; the GPU size, when known, is the best estimate.
  std::size_t est_bytes{0};
  event::downgrade_source source{event::downgrade_source::repository};
  std::uintptr_t repo_token{0};
  std::size_t partition{0};
  std::optional<exec::queue_priority> task_priority;
  std::optional<int> preferred_device;
  std::uint32_t lock_misses{0};
  bool in_flight{false};
  bool unsupported{false};
  bool speculative_banned{false};
  bool gpu_banned{false};
  /// After a lock refusal: the queue epoch at dispatch. Not plannable while it is current.
  std::uint64_t blocked_queue_epoch{~std::uint64_t{0}};
};

/**
 * @brief A data repository of a query, as preload sees it.
 *
 * The pointers are dereferenced only while holding a work lease on the query.
 */
struct repo_info {
  /// Identity token, matching @c downgraded_batch::repository.
  std::uintptr_t token{0};
  cucascade::data_repository* repo{nullptr};
  /// Priority of the pipeline that consumes the repository.
  exec::queue_priority dest_priority{0};
  /// The consuming operator when it drains one partition per task; null otherwise.
  op::sirius_physical_partition_consumer_operator const* partition_consumer{nullptr};
  /// Producers behind a FULL barrier of the consuming operator.
  std::vector<std::weak_ptr<pipeline::sirius_pipeline>> full_sources;
};

/// Per-query state. Only @ref candidate_store::add_query creates it.
struct query_state {
  std::vector<int> active_gpus;
  /// Sorted by ascending @c dest_priority (execution order).
  std::vector<repo_info> repos;
  /// token -> index into @c repos.
  std::unordered_map<std::uintptr_t, std::size_t> repo_index;
  /// Candidates found in each repository (by token).
  std::unordered_map<std::uintptr_t, std::unordered_set<std::uint64_t>> candidates_by_repo;
  /// Promotions of each batch in this query.
  std::unordered_map<std::uint64_t, std::uint32_t> promotions;
  /// (last_seq, batch_id) of every candidate of this query, for eviction, pruning and retire.
  std::set<std::pair<std::uint64_t, std::uint64_t>> by_seq;
  /// Batch ids of this query's ledger entries, so retiring it needs no scan of the ledger.
  std::unordered_set<std::uint64_t> ledger_ids;
  /// Claimed tombstones: a batch a task claimed whose candidate was then removed, mapped to the
  /// conversion sequence it was claimed at. A duplicate or older downgrade record of such a batch
  /// must not recreate an unclaimed candidate; only a strictly newer sequence reopens it. The
  /// board keeps the matching entry claimed until the tombstone is unwatched (retire/forget/
  /// clear/stop).
  std::unordered_map<std::uint64_t, std::uint64_t> claimed_seqs;
  per_query_stats stats;
  /// A FULL barrier of this query may have cleared since the last pass.
  bool readiness_dirty{true};
  /// A failed promotion has already been reported for this query.
  bool reported_failure{false};
};

/// A batch preload moved up that its consumer has not consumed yet.
struct ledger_entry {
  std::weak_ptr<cucascade::data_batch> batch;
  query_id_t query{};
  /// GPU index, or -1 for HOST staging.
  int gpu{-1};
  klass k{klass::A};
  std::size_t bytes{0};
  /// Conversion sequence of the promotion (see @c promotion_result::conversion_seq). A record at
  /// or below it describes a move from before the promotion and is ignored.
  std::uint64_t seq{0};
  /// Generation stamp of the budget charge this entry carries (see @c promotion_job::job_seq).
  /// Releasing the budget acts only on a charge with the same stamp.
  std::uint64_t job_seq{0};
  /// A consumer on another GPU was already counted as a misprediction.
  bool mispredicted{false};
};

/// What @ref candidate_store::intake did with a record.
enum class intake_status : std::uint8_t {
  inserted,               ///< a new candidate
  bounced,                ///< a new candidate for a batch preload had moved up (now banned from B)
  merged,                 ///< merged into an existing candidate
  dropped_unknown_query,  ///< the query is not prepared, or already retired
  dropped_expired,        ///< the batch is gone
  dropped_unsupported,    ///< the batch did not move to HOST or DISK
  dropped_promotion_cap,  ///< the batch was promoted the maximum number of times
  dropped_overflow,       ///< the store is full and this record lost the eviction
  dropped_stale_record    ///< the record predates the promotion in the ledger
};

/// A planned item handed to a lane: which candidate goes where.
struct ready_item {
  std::uint64_t batch_id{0};
  target t{};
  klass k{klass::A};
  /// The target's preload ceiling in consumed bytes.
  std::size_t ceiling{0};
};

/// Which lane ready items an outcome invalidates.
enum class purge_scope : std::uint8_t {
  none,
  target,   ///< every item for the job's target space
  held_off  ///< every item for the job's GPU, and staging to that GPU's NUMA-local host
};

/// Side effects of @ref candidate_store::apply_outcome that the caller carries out.
struct outcome_effect {
  purge_scope purge{purge_scope::none};
  /// The first failed promotion of its query: worth one warning.
  bool first_failure_in_query{false};
  /// The query was retired before the outcome arrived: only statistics were updated.
  bool late{false};
};

/// How the planner classified a live ledger entry while holding the query's lease.
enum class ledger_verdict : std::uint8_t {
  outstanding,  ///< still waiting for its consumer
  hit,          ///< consumed
  reclassify_a  ///< a class-B entry that a task has claimed
};

/// A live ledger entry for the planner to classify.
struct ledger_probe {
  std::uint64_t batch_id{0};
  query_id_t query{};
  std::weak_ptr<cucascade::data_batch> batch;
  klass k{klass::A};
};

/// A repository with candidates, for the planner's class-B scan.
struct repo_view {
  repo_info info;
  /// Candidate batch ids in this repository, sorted.
  std::vector<std::uint64_t> candidate_ids;
};

/// One query, for a planner pass.
struct query_view {
  query_id_t query{};
  std::vector<int> active_gpus;
  bool readiness_dirty{false};
  /// Repositories with candidates, by ascending destination priority.
  std::vector<repo_view> repos;
};

/// What the planner copies under the executor's lock before it scans without it.
struct pass_snapshot {
  bool has_candidates{false};
  std::uint64_t queue_epoch{0};
  /// Per space slot: GPUs by index, then HOST spaces by index.
  std::vector<std::uint64_t> mem_epoch;
  std::vector<std::uint64_t> gate_epoch;
  std::vector<std::size_t> inflight_bytes;
  /// Oldest query first.
  std::vector<query_view> queries;
  /// Live ledger entries to classify under each query's lease.
  std::vector<ledger_probe> ledger;
};

/**
 * @brief The preload executor's bookkeeping: candidates, per-query state, the ledger of preloaded
 *        batches, memory and queue epochs, and in-flight accounting.
 *
 * Not thread-safe: the executor calls every member under its own leaf lock. Nothing here touches
 * a batch (beyond lock-free @c weak_ptr::expired), a repository, a memory space or CUDA.
 *
 * Invariants:
 * - Only @ref add_query creates per-query state. Records, outcomes and verdicts for a query the
 *   store does not know update statistics only, so nothing comes back after @ref retire_query.
 * - A refusal gate is stamped with the epoch the job saw when it was taken, so a memory event
 *   during the attempt always leaves the space ungated.
 * - The device preloading budget is charged per live job at TAKE and released at claim or on any
 *   non-promoting outcome; the ledger records each charge's generation so a later erase releases
 *   exactly it. The ledger itself is kept only for statistics, not for byte accounting.
 */
class candidate_store {
 public:
  /**
   * @param cfg Preload configuration (caps and limits).
   * @param gpu_devices Device ids of the GPU spaces; a GPU's index is its position here.
   * @param host_devices Device ids of the HOST spaces; a HOST's index is its position here.
   * @param counters Executor-wide counters, updated alongside the per-query ones.
   */
  /**
   * @param budget The device preloading budget board, or null when device preloading is off. The
   *        store charges a GPU take, releases on every non-promoting outcome and ledger erase,
   *        reclassifies on B->A, and unwatches an id that leaves both the candidate set and the
   *        ledger. Must outlive the store.
   */
  candidate_store(preload_executor_config cfg,
                  std::vector<int> gpu_devices,
                  std::vector<int> host_devices,
                  preload_counters& counters,
                  memory::device_preload_budget* budget = nullptr);

  // ---- spaces ---------------------------------------------------------------------------------

  [[nodiscard]] std::size_t gpu_count() const noexcept { return _gpu_devices.size(); }
  [[nodiscard]] std::size_t host_count() const noexcept { return _host_devices.size(); }
  [[nodiscard]] std::optional<std::size_t> gpu_index_of(int device) const noexcept;
  [[nodiscard]] std::optional<std::size_t> host_index_of(int device) const noexcept;

  // ---- epochs ---------------------------------------------------------------------------------

  [[nodiscard]] std::uint64_t mem_epoch(target t) const noexcept;
  [[nodiscard]] std::uint64_t gate_epoch(target t) const noexcept;
  /// Whether a worker was refused by @p t since its last memory event.
  [[nodiscard]] bool gated(target t) const noexcept;
  [[nodiscard]] std::uint64_t queue_epoch() const noexcept { return _queue_epoch; }

  void advance_mem_epoch(target t) noexcept;
  void advance_host_mem_epochs() noexcept;
  void advance_all_mem_epochs() noexcept;
  void advance_queue_epoch() noexcept { ++_queue_epoch; }

  // ---- in-flight accounting -------------------------------------------------------------------

  [[nodiscard]] std::size_t inflight_bytes(target t) const noexcept;
  void add_inflight(target t, std::size_t bytes) noexcept;
  void remove_inflight(target t, std::size_t bytes) noexcept;
  /// HOST->GPU tokens in use on GPU index @p gpu.
  [[nodiscard]] std::uint32_t h2g_tokens(std::size_t gpu) const noexcept;
  void acquire_h2g_token(std::size_t gpu) noexcept;
  void release_h2g_token(std::size_t gpu) noexcept;

  // ---- queries --------------------------------------------------------------------------------

  /**
   * @brief Create the state of query @p q. The only way per-query state comes into being.
   *
   * @param repos The query's repositories, in any order; sorted here by destination priority.
   *        A repeated token keeps its first entry in @p repos.
   * @return false (and nothing changes) if @p q is already known.
   */
  bool add_query(query_id_t q, std::vector<int> active_gpus, std::vector<repo_info> repos);

  [[nodiscard]] bool knows_query(query_id_t q) const noexcept { return _queries.contains(q); }
  [[nodiscard]] query_state const* find_query(query_id_t q) const noexcept;
  void mark_readiness_dirty(query_id_t q) noexcept;

  /**
   * @brief Purge everything keyed by @p q: candidates, ledger entries (counted as wasted),
   *        promotion history and query state. Its statistics are kept until @ref forget_query.
   *
   * Idempotent.
   */
  void retire_query(query_id_t q) noexcept;

  /**
   * @brief Purge @p q again, advance every memory epoch (its repositories' memory was freed),
   *        and hand back its statistics.
   *
   * Idempotent: a second call, or a call for a query the store never knew, finds nothing,
   * leaves the epochs alone and returns empty.
   */
  std::optional<per_query_stats> forget_query(query_id_t q) noexcept;

  // ---- intake ---------------------------------------------------------------------------------

  /**
   * @brief Take in one @c batch_downgraded record of query @p q.
   *
   * New batches become candidates; known ones merge: the record with the larger conversion
   * sequence decides the location and repository, and every record can fill the estimate,
   * origins and task hints. A batch still in the ledger bounced when the record is newer than the
   * promotion: its entry is erased and the candidate is banned from speculative promotion. A
   * record older than the promotion is ignored. At @c max_candidates the newest query's most
   * recently downgraded candidate that is not in flight is evicted.
   */
  intake_status intake(query_id_t q, event::downgraded_batch const& rec);

  // ---- candidates -----------------------------------------------------------------------------

  [[nodiscard]] candidate const* find(std::uint64_t batch_id) const noexcept;
  /// Not in flight, not unsupported, and not blocked at the current queue epoch.
  [[nodiscard]] bool plannable(candidate const& c) const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return _by_id.size(); }
  [[nodiscard]] bool has_candidates() const noexcept { return !_by_id.empty(); }
  void drop_candidate_noexcept(std::uint64_t batch_id) noexcept;

  /**
   * @brief Erase up to @p budget expired candidates, resuming where the previous call stopped.
   *
   * @return The number erased.
   */
  std::size_t prune_expired_candidates(std::size_t budget = 256) noexcept;

  // ---- planning and jobs ----------------------------------------------------------------------

  /// Copy what a planner pass needs, and clear the queries' readiness flags.
  [[nodiscard]] pass_snapshot snapshot_phase0();

  /**
   * @brief Turn a ready item into a job and mark its candidate in flight.
   *
   * Stamps the target's memory epoch, the queue epoch and the candidate's sequence. The OOM
   * multiplier applies to GPU targets only.
   *
   * @return Empty when the candidate is gone, not plannable, its query is unknown, or the target
   *         is out of range.
   */
  [[nodiscard]] std::optional<promotion_job> take(ready_item const& item);

  /**
   * @brief Apply the result of a job taken with @ref take.
   *
   * A memory refusal gates the target with the job's epoch (never moving an existing gate back),
   * and a promotion off HOST advances that HOST's epoch, whether or not the query is still known:
   * both describe a space. Everything else changes candidate and ledger state only when the query
   * and the candidate still exist.
   *
   * A promotion to a GPU moves the candidate into the ledger. A DISK->HOST staging adds a ledger
   * entry too but keeps the candidate, now located on that HOST, so a later pass can take it on
   * to its GPU; it is dropped instead once the batch reached @c max_promotions_per_batch.
   */
  outcome_effect apply_outcome(promotion_job const& job, promotion_result const& result);

  // ---- ledger ---------------------------------------------------------------------------------

  [[nodiscard]] ledger_entry const* find_ledger(std::uint64_t batch_id) const noexcept;
  [[nodiscard]] std::size_t ledger_size() const noexcept { return _ledger.size(); }
  /// Preloaded, unclaimed bytes on GPU index @p gpu (all classes, then class B), read from the
  /// budget board. Zero when device preloading is off.
  [[nodiscard]] std::size_t outstanding_bytes(std::size_t gpu) const noexcept;
  [[nodiscard]] std::size_t outstanding_b_bytes(std::size_t gpu) const noexcept;

  /// Erase ledger entries whose batch is gone; each counts as a hit. Needs no lease.
  std::size_t sweep_expired_ledger() noexcept;

  /// Apply the planner's classification of live entries; ids no longer in the ledger are skipped.
  void apply_ledger_verdicts(
    std::span<std::pair<std::uint64_t, ledger_verdict> const> verdicts) noexcept;

  /// A queued task on GPU index @p gpu consumes @p batch_id: count a misprediction, once per
  /// entry, when preload put it on another GPU.
  void note_consumer_device(std::uint64_t batch_id, std::size_t gpu) noexcept;

  /// Drop all candidates, queries, ledger entries and retired statistics. Epochs are kept.
  void clear() noexcept;

 private:
  using candidate_map = std::unordered_map<std::uint64_t, candidate>;

  [[nodiscard]] std::optional<std::size_t> slot(target t) const noexcept;
  void count(query_id_t q, counter c, std::uint64_t n = 1) noexcept;
  void count_outcome(promotion_job const& job, promotion_result const& result) noexcept;
  void merge(candidate& c, query_state& qs, event::downgraded_batch const& rec);
  void erase_candidate(candidate_map::iterator it) noexcept;
  void erase_ledger(std::unordered_map<std::uint64_t, ledger_entry>::iterator it) noexcept;
  /// Drop @p batch_id from the budget board once it is in neither the candidate set nor the
  /// ledger. No-op when the board is absent.
  void maybe_unwatch(std::uint64_t batch_id) noexcept;
  void add_ledger(candidate const& c,
                  query_state& qs,
                  promotion_job const& job,
                  std::size_t bytes,
                  std::uint64_t seq);
  void reindex_seq(candidate& c, query_state& qs, std::uint64_t seq);
  [[nodiscard]] std::optional<std::uint64_t> evict_one() noexcept;

  preload_executor_config _cfg;
  std::vector<int> _gpu_devices;
  std::vector<int> _host_devices;
  preload_counters& _counters;
  /// The device preloading budget board, or null when device preloading is off.
  memory::device_preload_budget* _budget{nullptr};
  /// Monotonic generation stamp handed to each GPU budget charge.
  std::uint64_t _next_job_seq{0};

  candidate_map _by_id;
  /// Ascending query id = oldest first. Only add_query inserts.
  std::map<query_id_t, query_state> _queries;
  /// Statistics of retired queries, until forget_query.
  std::map<query_id_t, per_query_stats> _stats_by_query;
  std::unordered_map<std::uint64_t, ledger_entry> _ledger;

  std::vector<std::uint64_t> _mem_epoch;   ///< per slot
  std::vector<std::uint64_t> _gate_epoch;  ///< per slot
  std::vector<std::size_t> _inflight;      ///< per slot
  std::vector<std::uint32_t> _h2g_tokens;  ///< per GPU
  std::uint64_t _queue_epoch{0};

  query_id_t _prune_query{};
  std::pair<std::uint64_t, std::uint64_t> _prune_key{};
};

}  // namespace sirius::preload
