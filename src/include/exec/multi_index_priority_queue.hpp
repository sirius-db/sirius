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

#include "exec/inspectable_priority_queue.hpp"  // queue_priority
#include "op/sirius_physical_operator_type.hpp"
#include "third_party/plf_colony.h"

#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::exec {

// =============================================================================
// Secondary-index key types
// =============================================================================

/// The attribute types the queue is indexed by. "For now" there are three
/// dimensions; adding another means extending index_keys and adding one map +
/// one node iterator + one try_pop_from()/try_pop_back_from() overload below.
using operator_key = op::SiriusPhysicalOperatorType;
using pipeline_key = std::size_t;
using query_key    = std::uint32_t;

/// Tagged lookup keys used with try_pop_from()/try_pop_back_from(): the struct
/// *type* selects which index dimension to query, its `value` selects the bucket.
struct operator_index {
  operator_key value;
};
struct pipeline_index {
  pipeline_key value;
};
struct query_index {
  query_key value;
};

/// The per-task attributes extracted exactly once at push time. `priority` drives
/// the global ordering (smaller = popped first, matching queue_priority); the
/// remaining fields are the secondary-index keys. Captured by value at push time,
/// so mutating a task after enqueue does not move it.
struct index_keys {
  queue_priority priority;
  operator_key operator_type;
  pipeline_key pipeline_id;
  query_key query_id;
};

namespace detail {

// Storage backends, selected by tag. Each backend stores the queue's internal
// nodes and, crucially, keeps them pointer-stable across insert/erase of *other*
// nodes so the ordered indexes can key on raw node pointers. add() returns the
// stable pointer (used as the ordering key) plus an opaque handle used to erase
// that node later.
struct colony_storage {};  ///< Default: cache-friendly, pointer-stable colony.
struct list_storage {};    ///< Fallback: node-based std::list.

template <typename Tag, typename Node>
struct storage_traits;  // primary left undefined: unsupported tag

// plf::colony guarantees pointer/reference stability across insert and erase and
// recovers an iterator from a stable pointer via get_iterator(), so the erase
// handle is simply the node pointer itself.
template <typename Node>
struct storage_traits<colony_storage, Node> {
  using container = plf::colony<Node>;
  using handle    = Node*;

  static std::pair<Node*, handle> add(container& c, Node&& value)
  {
    auto it   = c.insert(std::move(value));
    Node* ptr = &(*it);
    return {ptr, ptr};
  }
  static void erase(container& c, handle h) { c.erase(c.get_iterator(h)); }
};

// std::list nodes are stable too; here the list iterator is the erase handle.
template <typename Node>
struct storage_traits<list_storage, Node> {
  using container = std::list<Node>;
  using handle    = typename container::iterator;

  static std::pair<Node*, handle> add(container& c, Node&& value)
  {
    auto it = c.insert(c.end(), std::move(value));
    return {&(*it), it};
  }
  static void erase(container& c, handle h) { c.erase(h); }
};

}  // namespace detail

// =============================================================================
// multi_index_priority_queue
// =============================================================================

/**
 * @brief A priority queue whose elements are additionally reachable through
 *        secondary indexes on operator type, pipeline id, and query id.
 *
 * Tasks are stored once in a pointer-stable backing container (plf::colony by
 * default) and referenced from four ordered indexes: one global priority order
 * plus one bucket per secondary-index key value. Every index sorts by
 * (priority, insertion-sequence): ascending priority (smaller = popped first,
 * matching queue_priority), FIFO among equal priorities. This lets a caller pop
 * either the globally best/worst task or the best/worst task *restricted to* a
 * given operator type / pipeline / query in O(log n).
 *
 * The queue is thread-safe: every operation takes an internal mutex, so any mix
 * of producers and consumers may call it concurrently. The blocking consumers are
 * the global pops:
 *   - pop() / pop_back() block until a task is available (or the queue is
 *     interrupted, in which case they return nullptr).
 * All other consumers are non-blocking and return std::nullopt when there is
 * nothing to hand back — in particular the secondary-index pops (try_pop_from /
 * try_pop_back_from) never block, since "wait for a task of this operator/pipeline/
 * query" has no well-defined completion.
 *
 * interrupt() wakes every thread blocked in pop()/pop_back() (they return
 * nullptr); reactivate() resumes normal blocking. drain() drops all queued tasks,
 * and drain(query_index) drops just the tasks of one query.
 *
 * The queue is non-copyable (tasks are uniquely owned) and non-movable: the
 * ordered indexes and the backing container reference each other, so the queue is
 * meant to be owned in place rather than moved.
 *
 * @tparam Task       The payload type; owned via std::unique_ptr<Task>.
 * @tparam StorageTag Backing-storage selector: detail::colony_storage (default)
 *                    or detail::list_storage.
 */
template <typename Task, typename StorageTag = detail::colony_storage>
class multi_index_priority_queue {
 public:
  using task_ptr = std::unique_ptr<Task>;
  /// Extracts a task's ordering priority and secondary-index keys. Invoked once
  /// per push, before the task pointer is moved into storage.
  using key_extractor = std::function<index_keys(const Task&)>;

 private:
  struct node;

  using traits       = detail::storage_traits<StorageTag, node>;
  using storage_type = typename traits::container;
  using erase_handle = typename traits::handle;

  /// Orders nodes by (priority, seq): ascending priority, then FIFO among equal
  /// priorities. `seq` is a strictly increasing push counter, so this is a strict
  /// total order and every ordered index below can be a std::set (no duplicates).
  struct node_order {
    bool operator()(const node* a, const node* b) const
    {
      if (a->keys.priority != b->keys.priority) { return a->keys.priority < b->keys.priority; }
      return a->seq < b->seq;
    }
  };
  using order_set = std::set<node*, node_order>;

  struct node {
    task_ptr task;
    index_keys keys;
    std::uint64_t seq;
    erase_handle self;  ///< Handle used to erase this node from _storage.
    // Cached positions in each index, so removing a node is O(log n) per index
    // rather than a lookup by value.
    typename order_set::iterator global_it;
    typename order_set::iterator operator_it;
    typename order_set::iterator pipeline_it;
    typename order_set::iterator query_it;
  };

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
  /// wakes one thread blocked in pop()/pop_back(). push() always enqueues, even
  /// while the queue is interrupted (interrupt() only affects blocking pops); the
  /// enqueued work simply drains out later. Strongly exception-safe: if indexing
  /// throws (only on allocation failure) the task is destroyed and nothing is
  /// enqueued.
  void push(task_ptr task)
  {
    assert(task && "cannot push a null task");
    const index_keys keys = _extract(*task);
    {
      std::lock_guard<std::mutex> lock(_mutex);

      // Value-initialize so the cached-iterator / self-handle members are not
      // indeterminate when the node is moved into storage below.
      node n{};
      n.task = std::move(task);
      n.keys = keys;
      n.seq  = _seq;

      auto [ptr, handle] = traits::add(_storage, std::move(n));
      ptr->self          = handle;

      // Index the node in all four orderings. If any insertion throws (only on
      // allocation failure), unwind so the queue keeps its invariant that a live
      // node is present in *every* index or in none, and reclaim the storage slot.
      try {
        ptr->global_it   = _by_priority.insert(ptr).first;
        ptr->operator_it = _by_operator[keys.operator_type].insert(ptr).first;
        ptr->pipeline_it = _by_pipeline[keys.pipeline_id].insert(ptr).first;
        ptr->query_it    = _by_query[keys.query_id].insert(ptr).first;
      } catch (...) {
        unindex_by_value(ptr, keys);
        traits::erase(_storage, handle);
        throw;
      }
      ++_seq;
    }
    _cv.notify_one();
  }

  /// Blocks until the globally-first (lowest-priority-value) task is available and
  /// returns it, or returns nullptr if the queue is interrupted while empty. If
  /// the queue is interrupted but still holds tasks, those drain out first.
  [[nodiscard]] task_ptr pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_by_priority.empty() || !_active; });
    if (_by_priority.empty()) { return nullptr; }
    return extract_node(*_by_priority.begin());
  }

  /// Like pop(), but blocks for and returns the globally-last (highest-priority-
  /// value) task; returns nullptr if interrupted while empty.
  [[nodiscard]] task_ptr pop_back()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_by_priority.empty() || !_active; });
    if (_by_priority.empty()) { return nullptr; }
    return extract_node(*std::prev(_by_priority.end()));
  }

  /// Removes and returns the globally-first task, or nullopt if empty. Non-blocking.
  [[nodiscard]] std::optional<task_ptr> try_pop()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_by_priority.empty()) { return std::nullopt; }
    return extract_node(*_by_priority.begin());
  }

  /// Removes and returns the globally-last task, or nullopt if empty. Non-blocking.
  [[nodiscard]] std::optional<task_ptr> try_pop_back()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_by_priority.empty()) { return std::nullopt; }
    return extract_node(*std::prev(_by_priority.end()));
  }

  /// Removes and returns the first task with the given operator type, or nullopt.
  /// Non-blocking (secondary-index pops never wait).
  [[nodiscard]] std::optional<task_ptr> try_pop_from(const operator_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_bucket(_by_operator, idx.value, /*front=*/true);
  }
  [[nodiscard]] std::optional<task_ptr> try_pop_from(const pipeline_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_bucket(_by_pipeline, idx.value, /*front=*/true);
  }
  [[nodiscard]] std::optional<task_ptr> try_pop_from(const query_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_bucket(_by_query, idx.value, /*front=*/true);
  }

  /// Removes and returns the last task with the given index key, or nullopt.
  /// Non-blocking.
  [[nodiscard]] std::optional<task_ptr> try_pop_back_from(const operator_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_bucket(_by_operator, idx.value, /*front=*/false);
  }
  [[nodiscard]] std::optional<task_ptr> try_pop_back_from(const pipeline_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_bucket(_by_pipeline, idx.value, /*front=*/false);
  }
  [[nodiscard]] std::optional<task_ptr> try_pop_back_from(const query_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return pop_from_bucket(_by_query, idx.value, /*front=*/false);
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
    _by_priority.clear();
    _by_operator.clear();
    _by_pipeline.clear();
    _by_query.clear();
    _storage.clear();
  }

  /// Drops every queued task belonging to the given query, removing them from all
  /// indexes. Tasks of other queries are untouched.
  void drain(const query_index& idx)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto mit = _by_query.find(idx.value);
    if (mit == _by_query.end()) { return; }
    // Snapshot the bucket first: extract_node() mutates (and may erase) it.
    const std::vector<node*> nodes(mit->second.begin(), mit->second.end());
    for (node* ptr : nodes) {
      extract_node(ptr);  // returned task is dropped (destroyed) here
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
    return _by_priority.empty();
  }
  [[nodiscard]] std::size_t size() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _by_priority.size();
  }

  /// Number of queued tasks matching a given secondary-index key.
  [[nodiscard]] std::size_t size(const operator_index& idx) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return bucket_size(_by_operator, idx.value);
  }
  [[nodiscard]] std::size_t size(const pipeline_index& idx) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return bucket_size(_by_pipeline, idx.value);
  }
  [[nodiscard]] std::size_t size(const query_index& idx) const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return bucket_size(_by_query, idx.value);
  }

  /// Number of distinct non-empty buckets currently held in a dimension (empty
  /// buckets are pruned, so this is the count of live keys).
  [[nodiscard]] std::size_t operator_bucket_count() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _by_operator.size();
  }
  [[nodiscard]] std::size_t pipeline_bucket_count() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _by_pipeline.size();
  }
  [[nodiscard]] std::size_t query_bucket_count() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _by_query.size();
  }

  /// Task count for every live bucket in a dimension, keyed by the index value.
  /// Only non-empty buckets appear (a key absent from the map holds zero tasks).
  [[nodiscard]] std::unordered_map<operator_key, std::size_t> operator_bucket_sizes() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return bucket_sizes(_by_operator);
  }
  [[nodiscard]] std::unordered_map<pipeline_key, std::size_t> pipeline_bucket_sizes() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return bucket_sizes(_by_pipeline);
  }
  [[nodiscard]] std::unordered_map<query_key, std::size_t> query_bucket_sizes() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return bucket_sizes(_by_query);
  }

 private:
  /// Unlinks `ptr` from all four indexes, frees its storage slot, and hands back
  /// the owned task. Reads every field of `*ptr` *before* erasing it from storage.
  task_ptr extract_node(node* ptr)
  {
    _by_priority.erase(ptr->global_it);
    erase_from_bucket(_by_operator, ptr->keys.operator_type, ptr->operator_it);
    erase_from_bucket(_by_pipeline, ptr->keys.pipeline_id, ptr->pipeline_it);
    erase_from_bucket(_by_query, ptr->keys.query_id, ptr->query_it);

    task_ptr task             = std::move(ptr->task);
    const erase_handle handle = ptr->self;
    traits::erase(_storage, handle);
    return task;
  }

  /// Pops the front (front=true) or back task of a secondary-index bucket.
  template <typename Map, typename Key>
  std::optional<task_ptr> pop_from_bucket(Map& map, const Key& key, bool front)
  {
    const auto mit = map.find(key);
    if (mit == map.end() || mit->second.empty()) { return std::nullopt; }
    node* ptr = front ? *mit->second.begin() : *std::prev(mit->second.end());
    return extract_node(ptr);
  }

  /// Erases one entry from a secondary-index bucket, dropping the bucket entirely
  /// when it becomes empty so the maps stay sparse.
  template <typename Map, typename Key>
  void erase_from_bucket(Map& map, const Key& key, typename order_set::iterator it)
  {
    const auto mit = map.find(key);
    assert(mit != map.end() && "index bucket missing for a live node");
    mit->second.erase(it);
    if (mit->second.empty()) { map.erase(mit); }
  }

  template <typename Map, typename Key>
  static std::size_t bucket_size(const Map& map, const Key& key)
  {
    const auto mit = map.find(key);
    return mit == map.end() ? 0 : mit->second.size();
  }

  template <typename Map>
  static std::unordered_map<typename Map::key_type, std::size_t> bucket_sizes(const Map& map)
  {
    std::unordered_map<typename Map::key_type, std::size_t> sizes;
    sizes.reserve(map.size());
    for (const auto& [key, bucket] : map) {
      sizes.emplace(key, bucket.size());
    }
    return sizes;
  }

  /// Removes `ptr` from whichever indexes it did make it into, tolerating nodes
  /// that were only partially indexed. Used to unwind a push() that threw; erase
  /// by value is idempotent, so it is safe regardless of how far indexing got.
  void unindex_by_value(node* ptr, const index_keys& keys)
  {
    _by_priority.erase(ptr);
    erase_value_from_bucket(_by_operator, keys.operator_type, ptr);
    erase_value_from_bucket(_by_pipeline, keys.pipeline_id, ptr);
    erase_value_from_bucket(_by_query, keys.query_id, ptr);
  }

  template <typename Map, typename Key>
  void erase_value_from_bucket(Map& map, const Key& key, node* ptr)
  {
    const auto mit = map.find(key);
    if (mit == map.end()) { return; }
    mit->second.erase(ptr);
    if (mit->second.empty()) { map.erase(mit); }
  }

  mutable std::mutex _mutex;
  std::condition_variable _cv;
  bool _active{true};  ///< false after interrupt(): blocked pops stop waiting.

  storage_type _storage;
  order_set _by_priority;
  std::unordered_map<operator_key, order_set> _by_operator;
  std::unordered_map<pipeline_key, order_set> _by_pipeline;
  std::unordered_map<query_key, order_set> _by_query;
  key_extractor _extract;
  std::uint64_t _seq{0};
};

}  // namespace sirius::exec
