/*
 * Copyright 2025, Sirius Contributors.
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

#include "exec/queue_priority.hpp"
#include "op/sirius_physical_operator_type.hpp"

#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace sirius::exec {

// =============================================================================
// Secondary-index key types
// =============================================================================

using operator_key = op::SiriusPhysicalOperatorType;
using query_key    = std::uint32_t;
using device_key   = int;

/// Reserved device_key bucket for tasks with no preferred GPU. A task's preferred
/// device is a std::optional<int> (see gpu_pipeline_task::get_preferred_device_id);
/// callers map std::nullopt to this value so "no preference" tasks share a bucket
/// and are reachable with gpu_index{no_preferred_device}. Real device ids are >= 0.
inline constexpr device_key no_preferred_device = -1;

/// Tagged lookup keys used with try_pop_from()/try_pop_back_from(): the struct
/// *type* selects which index dimension to query, its value selects the bucket.
struct operator_index {
  operator_key value;
};
struct query_index {
  query_key value;
};
struct gpu_index {
  device_key id;
};

/// The per-task attributes extracted exactly once at push time. `priority` drives
/// the global ordering (smaller = popped first, matching queue_priority); the rest
/// are the secondary-index keys. `device_id` is the task's preferred GPU, or
/// no_preferred_device (-1) when it has none.
///
/// This queue assumes the scheduler's priority encoding, in which priority is a
/// function of (query, pipeline): all tasks of one pipeline share a single
/// priority, one priority belongs to exactly one pipeline, and a query occupies a
/// contiguous band of priorities. Operator type is constant within a pipeline.
/// Preferred device is *not* — tasks in one pipeline may prefer different GPUs.
struct index_keys {
  queue_priority priority;
  operator_key operator_type;
  query_key query_id;
  device_key device_id;
};

// =============================================================================
// multi_index_priority_queue
// =============================================================================

/**
 * @brief A priority queue whose elements are additionally reachable by operator
 *        type, query id, and preferred GPU device.
 *
 * The structure exploits the scheduler's priority encoding (see index_keys) rather
 * than sorting every task three times over. Its spine is a single ordered map of
 * priority *levels*:
 *   - each level holds the equal-priority tasks of one pipeline in FIFO order;
 *   - the map orders levels by priority, so the global order falls out for free.
 * Because priority, query, and operator are all functions of the same axis, two of
 * the three secondary indexes are just *lookups onto levels* — no per-task sorting:
 *   - query id -> the (contiguous) set of its levels;
 *   - operator -> the set of levels carrying that operator.
 * Preferred device is the only genuinely task-granular axis (tasks within a level
 * may prefer different GPUs), so it keeps a real side index, priority-bucketed and
 * FIFO within a (device, priority) group. Tasks with no preferred device share the
 * gpu_index{no_preferred_device} (-1) bucket.
 *
 * All pops are O(log P) in the number of active priority levels (P, roughly the
 * number of in-flight pipelines) plus O(1) list work — never O(log N) in the task
 * count.
 *
 * The queue is thread-safe: every operation takes an internal mutex, so any mix of
 * producers and consumers may call it concurrently. pop() / pop_back() block until
 * a task is available (or the queue is interrupted, returning nullptr). The
 * non-blocking consumers (try_pop, try_pop_back, and the secondary-index
 * try_pop_from / try_pop_back_from) return std::nullopt when empty; the
 * secondary-index pops never block, since "wait for a task of this
 * operator/query/device" has no well-defined completion.
 *
 * interrupt() wakes every thread blocked in pop()/pop_back(); reactivate() resumes.
 * drain() drops all tasks; drain(query_index) drops one query's tasks.
 *
 * The queue is non-copyable (tasks are uniquely owned) and non-movable: the
 * indexes reference the level lists, so it is owned in place rather than moved.
 *
 * @tparam Task The payload type; owned via std::unique_ptr<Task>.
 */
template <typename Task>
class multi_index_priority_queue {
 public:
  using task_ptr = std::unique_ptr<Task>;
  /// Extracts a task's ordering priority and secondary-index keys. Invoked once
  /// per push, before the task pointer is moved into storage.
  using key_extractor = std::function<index_keys(const Task&)>;

 private:
  struct node;

  /// One priority level: the equal-priority tasks of a single pipeline, in FIFO
  /// (insertion) order. Since priority <-> pipeline is 1:1, a level also names its
  /// (query, operator), which is constant within the pipeline.
  struct level {
    query_key qid{};
    operator_key op{};
    std::list<node> tasks;
  };
  using level_map = std::map<queue_priority, level>;

  /// The per-task record. Lives inside its level's list; caches its own position
  /// in that list and in its device bucket so removal is O(1) on both.
  struct node {
    task_ptr task;
    index_keys keys;
    typename std::list<node>::iterator level_it;
    typename std::list<node*>::iterator device_it;
  };

  /// Per-device side index: priority-bucketed, FIFO within a (device, priority).
  using device_index_map =
    std::unordered_map<device_key, std::map<queue_priority, std::list<node*>>>;

 public:
  /// @param extractor Computes each task's priority and index keys. Must be valid.
  explicit multi_index_priority_queue(key_extractor extractor) : _extract(std::move(extractor))
  {
    assert(_extract && "multi_index_priority_queue requires a valid key extractor");
  }

  multi_index_priority_queue(const multi_index_priority_queue&)            = delete;
  multi_index_priority_queue& operator=(const multi_index_priority_queue&) = delete;
  multi_index_priority_queue(multi_index_priority_queue&&)                 = delete;
  multi_index_priority_queue& operator=(multi_index_priority_queue&&)      = delete;
  ~multi_index_priority_queue()                                            = default;

  /// Inserts a task, computing its priority and secondary-index keys once, then
  /// wakes one thread blocked in pop()/pop_back(). If the queue is interrupted
  /// (i.e. shutting down) the task is dropped rather than enqueued, matching the
  /// teardown contract callers rely on (e.g. returning a task to a closed queue).
  /// Strongly exception-safe: if any allocation throws, the queue is left as it
  /// was (the task is destroyed) and nothing is enqueued.
  void push(task_ptr task)
  {
    assert(task && "cannot push a null task");
    const index_keys keys = _extract(*task);
    {
      std::lock_guard<std::mutex> lock(_mutex);
      // Interrupted (shutdown): drop the task instead of enqueuing it.
      if (!_active) { return; }

      auto lit             = _levels.find(keys.priority);
      const bool new_level = (lit == _levels.end());
      if (new_level) {
        lit = _levels.emplace(keys.priority, level{keys.query_id, keys.operator_type}).first;
      }
      level& lv = lit->second;

      bool node_added   = false;
      bool device_added = false;
      typename std::list<node*>::iterator dev_it;
      try {
        if (new_level) {
          _operator_levels[keys.operator_type].insert(keys.priority);
          _query_levels[keys.query_id].insert(keys.priority);
        }

        lv.tasks.emplace_back();
        node_added = true;
        node& n    = lv.tasks.back();
        n.level_it = std::prev(lv.tasks.end());
        n.keys     = keys;

        auto& dlist = _by_device[keys.device_id][keys.priority];
        dlist.push_back(&n);
        dev_it       = std::prev(dlist.end());
        device_added = true;
        n.device_it  = dev_it;

        n.task = std::move(task);  // noexcept; commit last
        ++_size;
      } catch (...) {
        if (device_added) {
          erase_from_device(keys.device_id, keys.priority, dev_it);
        } else {
          prune_empty_device(keys.device_id, keys.priority);
        }
        if (node_added) { lv.tasks.pop_back(); }
        if (new_level) {
          unset_level(_operator_levels, keys.operator_type, keys.priority);
          unset_level(_query_levels, keys.query_id, keys.priority);
          _levels.erase(lit);
        }
        throw;
      }
    }
    _cv.notify_one();
  }

  /// Blocks until the globally-first (lowest-priority-value) task is available and
  /// returns it, or returns nullptr if the queue is interrupted while empty.
  [[nodiscard]] task_ptr pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_levels.empty() || !_active; });
    if (_levels.empty()) { return nullptr; }
    return extract_node(&_levels.begin()->second.tasks.front());
  }

  /// Like pop(), but blocks for and returns the globally-last (highest-priority-
  /// value) task; returns nullptr if interrupted while empty.
  [[nodiscard]] task_ptr pop_back()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_levels.empty() || !_active; });
    if (_levels.empty()) { return nullptr; }
    return extract_node(&_levels.rbegin()->second.tasks.back());
  }

  /// Removes and returns the globally-first task, or nullopt if empty. Non-blocking.
  [[nodiscard]] std::optional<task_ptr> try_pop()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_levels.empty()) { return std::nullopt; }
    return extract_node(&_levels.begin()->second.tasks.front());
  }

  /// Removes and returns the globally-last task, or nullopt if empty. Non-blocking.
  [[nodiscard]] std::optional<task_ptr> try_pop_back()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_levels.empty()) { return std::nullopt; }
    return extract_node(&_levels.rbegin()->second.tasks.back());
  }

  /// Removes and returns the first (lowest-priority, then FIFO) task with the given
  /// index key, or nullopt. Non-blocking (secondary-index pops never wait).
  [[nodiscard]] std::optional<task_ptr> try_pop_from(const operator_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_levels(_operator_levels, idx.value, /*front=*/true);
  }
  [[nodiscard]] std::optional<task_ptr> try_pop_from(const query_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_levels(_query_levels, idx.value, /*front=*/true);
  }
  /// gpu_index{no_preferred_device} (-1) selects tasks with no preferred device.
  [[nodiscard]] std::optional<task_ptr> try_pop_from(const gpu_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_device(idx.id, /*front=*/true);
  }

  /// Removes and returns the last (highest-priority, then most-recent) task with
  /// the given index key, or nullopt. Non-blocking.
  [[nodiscard]] std::optional<task_ptr> try_pop_back_from(const operator_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_levels(_operator_levels, idx.value, /*front=*/false);
  }
  [[nodiscard]] std::optional<task_ptr> try_pop_back_from(const query_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_levels(_query_levels, idx.value, /*front=*/false);
  }
  /// gpu_index{no_preferred_device} (-1) selects tasks with no preferred device.
  [[nodiscard]] std::optional<task_ptr> try_pop_back_from(const gpu_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_device(idx.id, /*front=*/false);
  }

  /// Removes and returns the first task -- in pop order (lowest priority, then
  /// FIFO) -- for which `pred(const Task&)` is true, or nullopt if none matches.
  /// Non-blocking. `pred` is called under the lock, so it must not touch the queue.
  template <typename Pred>
  [[nodiscard]] std::optional<task_ptr> try_pop_if(Pred pred)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& [prio, lv] : _levels) {
      if (node* n = find_in_list(lv.tasks, pred)) { return extract_node(n); }
    }
    return std::nullopt;
  }

  /// Like try_pop_if(pred), but restricted to one secondary-index bucket, scanned
  /// in the same pop order. gpu_index{no_preferred_device} (-1) selects tasks with
  /// no preferred device.
  template <typename Pred>
  [[nodiscard]] std::optional<task_ptr> try_pop_if(const operator_index& idx, Pred pred)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_if_in_levels(_operator_levels, idx.value, pred);
  }
  template <typename Pred>
  [[nodiscard]] std::optional<task_ptr> try_pop_if(const query_index& idx, Pred pred)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_if_in_levels(_query_levels, idx.value, pred);
  }
  template <typename Pred>
  [[nodiscard]] std::optional<task_ptr> try_pop_if(const gpu_index& idx, Pred pred)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto it = _by_device.find(idx.id);
    if (it == _by_device.end()) { return std::nullopt; }
    for (auto& [prio, bucket] : it->second) {
      for (node* n : bucket) {
        if (pred(static_cast<const Task&>(*n->task))) { return extract_node(n); }
      }
    }
    return std::nullopt;
  }

  /// Removes and returns the first task satisfying `pred`, scanning the whole
  /// queue front-to-back (lowest priority first, FIFO within a level) when
  /// `front_to_back` is true, or back-to-front (highest priority first, most
  /// recent within a level) when false. Unlike try_pop_if, `pred` receives a
  /// *mutable* `Task&`. nullopt if none matches. Non-blocking; `pred` runs under
  /// the lock and must not touch the queue.
  template <typename Pred>
  [[nodiscard]] std::optional<task_ptr> mutable_pop_if(Pred pred, bool front_to_back)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (front_to_back) {
      for (auto& [prio, lv] : _levels) {
        for (node& n : lv.tasks) {
          if (pred(*n.task)) { return extract_node(&n); }
        }
      }
    } else {
      for (auto lit = _levels.rbegin(); lit != _levels.rend(); ++lit) {
        auto& tasks = lit->second.tasks;
        for (auto nit = tasks.rbegin(); nit != tasks.rend(); ++nit) {
          if (pred(*nit->task)) { return extract_node(&(*nit)); }
        }
      }
    }
    return std::nullopt;
  }

  /// Wakes every thread blocked in pop()/pop_back(); while interrupted they return
  /// nullptr once the queue is empty. Idempotent.
  void interrupt()
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _active = false;
    }
    _cv.notify_all();
  }

  /// Resumes normal blocking behavior after interrupt().
  void reactivate()
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _active = true;
    }
    _cv.notify_all();
  }

  /// Drops every queued task (destroying them) and empties all indexes.
  void drain()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _levels.clear();
    _query_levels.clear();
    _operator_levels.clear();
    _by_device.clear();
    _size = 0;
  }

  /// Drops every queued task belonging to the given query, removing them from all
  /// indexes. Tasks of other queries are untouched.
  void drain(const query_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto qit = _query_levels.find(idx.value);
    if (qit == _query_levels.end()) { return; }
    // Snapshot the level set first: drop_level() mutates (and may erase) it.
    const std::vector<queue_priority> levels(qit->second.begin(), qit->second.end());
    for (const queue_priority prio : levels) {
      drop_level(prio);
    }
  }

  [[nodiscard]] bool is_open() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _active;
  }
  [[nodiscard]] bool empty() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _levels.empty();
  }

  /// Visits every queued task in dispatch order (lowest priority level first,
  /// FIFO within a level; reversed when @p front_to_back is false) without
  /// removing anything. The visitor returns false to stop the walk early. Runs
  /// under the queue mutex: the visitor must be lightweight and must not touch
  /// the queue. Used by the downgraded-task prefetcher to snapshot the inputs
  /// of soon-to-run tasks.
  void for_each_mutable(const std::function<bool(Task&)>& visitor, bool front_to_back)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (front_to_back) {
      for (auto& [prio, lvl] : _levels) {
        for (auto& n : lvl.tasks) {
          if (!visitor(*n.task)) { return; }
        }
      }
    } else {
      for (auto it = _levels.rbegin(); it != _levels.rend(); ++it) {
        for (auto n = it->second.tasks.rbegin(); n != it->second.tasks.rend(); ++n) {
          if (!visitor(*n->task)) { return; }
        }
      }
    }
  }
  [[nodiscard]] std::size_t size() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _size;
  }

  /// Number of queued tasks matching a given secondary-index key.
  [[nodiscard]] std::size_t size(const operator_index& idx) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto it = _operator_levels.find(idx.value);
    return it == _operator_levels.end() ? 0 : sum_levels(it->second);
  }
  [[nodiscard]] std::size_t size(const query_index& idx) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto it = _query_levels.find(idx.value);
    return it == _query_levels.end() ? 0 : sum_levels(it->second);
  }
  [[nodiscard]] std::size_t size(const gpu_index& idx) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto it = _by_device.find(idx.id);
    if (it == _by_device.end()) { return 0; }
    std::size_t total = 0;
    for (const auto& [prio, bucket] : it->second) {
      total += bucket.size();
    }
    return total;
  }

  /// Number of distinct non-empty buckets currently held in a dimension.
  [[nodiscard]] std::size_t operator_bucket_count() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _operator_levels.size();
  }
  [[nodiscard]] std::size_t query_bucket_count() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _query_levels.size();
  }
  [[nodiscard]] std::size_t device_bucket_count() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _by_device.size();
  }

  /// Task count for every live bucket in a dimension, keyed by the index value.
  /// Only non-empty buckets appear (a key absent from the map holds zero tasks).
  [[nodiscard]] std::unordered_map<operator_key, std::size_t> operator_bucket_sizes() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::unordered_map<operator_key, std::size_t> sizes;
    sizes.reserve(_operator_levels.size());
    for (const auto& [key, levels] : _operator_levels) {
      sizes.emplace(key, sum_levels(levels));
    }
    return sizes;
  }
  [[nodiscard]] std::unordered_map<query_key, std::size_t> query_bucket_sizes() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::unordered_map<query_key, std::size_t> sizes;
    sizes.reserve(_query_levels.size());
    for (const auto& [key, levels] : _query_levels) {
      sizes.emplace(key, sum_levels(levels));
    }
    return sizes;
  }
  /// Keyed by device id; the no_preferred_device (-1) bucket holds tasks with no
  /// preferred GPU.
  [[nodiscard]] std::unordered_map<device_key, std::size_t> device_bucket_sizes() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::unordered_map<device_key, std::size_t> sizes;
    sizes.reserve(_by_device.size());
    for (const auto& [dev, buckets] : _by_device) {
      std::size_t total = 0;
      for (const auto& [prio, bucket] : buckets) {
        total += bucket.size();
      }
      sizes.emplace(dev, total);
    }
    return sizes;
  }

 private:
  /// Unlinks one task from the level and device indexes, frees it, and returns the
  /// owned task. Removes the level entirely (and its pipeline/query/operator
  /// lookups) if it becomes empty. Reads every field of `*n` before destroying it.
  task_ptr extract_node(node* n)
  {
    const queue_priority prio = n->keys.priority;
    const auto lit            = _levels.find(prio);
    level& lv                 = lit->second;

    erase_from_device(n->keys.device_id, prio, n->device_it);

    task_ptr task = std::move(n->task);
    lv.tasks.erase(n->level_it);  // destroys *n
    --_size;

    if (lv.tasks.empty()) { remove_level(lit); }
    return task;
  }

  /// Pops the front (front=true, lowest priority) or back task of the lowest/
  /// highest level referenced by a level-set index (operator or query).
  template <typename Map, typename Key>
  std::optional<task_ptr> pop_from_levels(Map& map, const Key& key, bool front)
  {
    const auto it = map.find(key);
    if (it == map.end() || it->second.empty()) { return std::nullopt; }
    const queue_priority prio = front ? *it->second.begin() : *it->second.rbegin();
    return extract_node(level_end_node(prio, front));
  }

  /// Returns the first node in `tasks` (FIFO order) satisfying `pred`, or nullptr.
  template <typename Pred>
  node* find_in_list(std::list<node>& tasks, Pred& pred)
  {
    for (node& n : tasks) {
      if (pred(static_cast<const Task&>(*n.task))) { return &n; }
    }
    return nullptr;
  }

  /// Scans a level-set index's levels in ascending-priority order and pops the
  /// first (FIFO within a level) task satisfying `pred`.
  template <typename Map, typename Key, typename Pred>
  std::optional<task_ptr> pop_if_in_levels(Map& map, const Key& key, Pred& pred)
  {
    const auto it = map.find(key);
    if (it == map.end()) { return std::nullopt; }
    for (const queue_priority prio : it->second) {
      if (node* n = find_in_list(_levels.at(prio).tasks, pred)) { return extract_node(n); }
    }
    return std::nullopt;
  }

  /// Pops the front (lowest-priority bucket) or back task preferring a device.
  std::optional<task_ptr> pop_from_device(device_key dev, bool front)
  {
    const auto it = _by_device.find(dev);
    if (it == _by_device.end() || it->second.empty()) { return std::nullopt; }
    const auto pit = front ? it->second.begin() : std::prev(it->second.end());
    node* n        = front ? pit->second.front() : pit->second.back();
    return extract_node(n);
  }

  /// Front or back task of the level at `prio`. The level is never empty here.
  node* level_end_node(queue_priority prio, bool front)
  {
    level& lv = _levels.at(prio);
    return front ? &lv.tasks.front() : &lv.tasks.back();
  }

  /// Erases the whole (now-empty) level and its query/operator lookups.
  void remove_level(typename level_map::iterator lit)
  {
    const level& lv = lit->second;
    unset_level(_operator_levels, lv.op, lit->first);
    unset_level(_query_levels, lv.qid, lit->first);
    _levels.erase(lit);
  }

  /// Drops every task of one level (removing each from its device bucket) and the
  /// level itself. Used by drain(query_index).
  void drop_level(queue_priority prio)
  {
    const auto lit = _levels.find(prio);
    if (lit == _levels.end()) { return; }
    for (node& n : lit->second.tasks) {
      erase_from_device(n.keys.device_id, prio, n.device_it);
      --_size;
    }
    remove_level(lit);
  }

  /// Removes one entry from the device side index, pruning the empty priority
  /// bucket and then the empty device entry so the map stays sparse.
  void erase_from_device(device_key dev,
                         queue_priority prio,
                         typename std::list<node*>::iterator it)
  {
    const auto dit = _by_device.find(dev);
    assert(dit != _by_device.end() && "device bucket missing for a live node");
    auto& buckets  = dit->second;
    const auto pit = buckets.find(prio);
    assert(pit != buckets.end() && "device priority bucket missing for a live node");
    pit->second.erase(it);
    if (pit->second.empty()) { buckets.erase(pit); }
    if (buckets.empty()) { _by_device.erase(dit); }
  }

  /// Prunes an empty (dev, prio) device bucket left behind by a failed push.
  void prune_empty_device(device_key dev, queue_priority prio)
  {
    const auto dit = _by_device.find(dev);
    if (dit == _by_device.end()) { return; }
    const auto pit = dit->second.find(prio);
    if (pit != dit->second.end() && pit->second.empty()) { dit->second.erase(pit); }
    if (dit->second.empty()) { _by_device.erase(dit); }
  }

  /// Removes one priority from a key -> {levels} map, pruning the empty key.
  template <typename Map, typename Key>
  static void unset_level(Map& map, const Key& key, queue_priority prio)
  {
    const auto it = map.find(key);
    if (it == map.end()) { return; }
    it->second.erase(prio);
    if (it->second.empty()) { map.erase(it); }
  }

  /// Total tasks across a set of priority levels.
  std::size_t sum_levels(const std::set<queue_priority>& prios) const
  {
    std::size_t total = 0;
    for (const queue_priority prio : prios) {
      total += _levels.at(prio).tasks.size();
    }
    return total;
  }

  mutable std::mutex _mutex;
  std::condition_variable _cv;
  bool _active{true};  ///< false after interrupt(): blocked pops stop waiting.

  level_map _levels;  ///< Spine: priority -> level.
  std::unordered_map<query_key, std::set<queue_priority>> _query_levels;  ///< query -> its levels.
  std::unordered_map<operator_key, std::set<queue_priority>>
    _operator_levels;           ///< operator -> its levels.
  device_index_map _by_device;  ///< device -> prio -> tasks.
  key_extractor _extract;
  std::size_t _size{0};
};

}  // namespace sirius::exec
