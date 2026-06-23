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

#include "duckdb/common/common.hpp"
#include "helper/logical_type.hpp"
#include "helper/types.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "sirius/exception.hpp"
#include "telemetry-bridge/gen/uuid.rs.h"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace sirius {

namespace op {
class sirius_physical_operator;
}  // namespace op

namespace pipeline {
class sirius_pipeline;
class sirius_pipeline_build_state;
class sirius_meta_pipeline;
}  // namespace pipeline
namespace op {

enum class TaskCreationHint { WAITING_FOR_INPUT_DATA, READY };

enum class MemoryBarrierType { PIPELINE, PARTIAL, FULL };

struct task_creation_hint {
  TaskCreationHint hint{TaskCreationHint::WAITING_FOR_INPUT_DATA};
  sirius_physical_operator* producer{nullptr};
};

/**
 * @brief Tag identifying the concrete operator_data subclass.
 *
 * Returned by operator_data::get_type() so callers can branch on the runtime
 * type without a dynamic_cast. Each subclass overrides get_type() to return
 * its own value; BASE is the default for the unspecialized base class.
 */
enum class operator_data_type : uint8_t {
  BASE,
  PIPELINEABLE,
  PARTITIONED,
  GPU_SCAN,
};

/**
 * @brief Generic base class for operator input/output data.
 *
 * This is an intentionally minimal base class with no opinion on what the
 * data should be. Derived classes define the concrete data representation.
 */
class operator_data {
 public:
  operator_data()          = default;
  virtual ~operator_data() = default;

  operator_data(const operator_data&)            = default;
  operator_data& operator=(const operator_data&) = default;
  operator_data(operator_data&&)                 = default;
  operator_data& operator=(operator_data&&)      = default;

  /**
   * @brief Identify the concrete subclass at runtime.
   *
   * Subclasses override to return their own operator_data_type. The base
   * implementation returns operator_data_type::BASE.
   */
  [[nodiscard]] virtual operator_data_type get_type() const { return operator_data_type::BASE; }

  // Batch ID pre-assigned by sirius_gpu_scan_operator::get_next_task_input_data() while
  // holding the scan-order mutex, so that assignment order matches split-pull order.
  // execute() uses this as the output data_batch's ID instead of calling get_next_batch_id()
  // at completion time, which would be non-deterministic under concurrent execution.
  uint64_t pre_assigned_batch_id = 0;

  /**
   * @brief Whether this operator_data refers to GPU-resident data already paid
   *        for upstream (e.g. a pinned-cache batch). Defaults to false.
   *
   * Used by memory estimation in sirius_gpu_scan_operator to distinguish a
   * pass-through over an already-resident batch from a fresh decode that
   * still needs allocation. Overridden by scan_operator_with_pinned_table_input.
   */
  [[nodiscard]] virtual bool is_resident() const noexcept { return false; }

  /**
   * @brief Per-task preparation hook invoked before the operator consumes this data.
   *
   * Called by the pipeline task machinery after the GPU pipeline executor has acquired
   * a memory reservation for the task, and before the operator's execute() runs. This
   * gives the data object a chance to perform any per-task setup that needs to happen
   * in the context of the task's reserved memory space, including:
   *   - Locking (or converting-then-locking) owned data batches into the requested
   *     memory space — see pipelineable_operator_data::prepare_for_processing.
   *   - Capturing the memory space for later use by execute() in operators that
   *     produce output but own no input batches — e.g. source operators such as
   *     parquet_scan_data, which need a target memory space for their output tables
   *     but have no upstream batch to inherit one from.
   *
   * @param requested_memory_space  The memory space associated with the task's
   *                                reservation. Any locking, conversion, or
   *                                allocation performed during preparation should
   *                                target this space. May be nullptr when the
   *                                caller has no target preference, in which case
   *                                implementations should fall back to each batch's
   *                                current space (see pipelineable_operator_data).
   * @param stream                  CUDA stream available for any data-movement
   *                                kernels triggered by preparation.
   * Throws on failure.
   * pipelineable_operator_data rethrows rmm::out_of_memory after logging,
   * and throws sirius::internal_exception for unrecoverable preparation errors.
   *
   * The default implementation is a no-op appropriate for operator_data subclasses
   * that own no data requiring locking and need no per-task setup.
   * Override when either condition changes.
   */
  virtual void prepare_for_processing(
    const ::cucascade::memory::memory_space* requested_memory_space, rmm::cuda_stream_view stream) {
  };

  /**
   * @brief Estimate the uncompressed GPU memory footprint of this data.
   *
   * Used by the reservation system to size memory reservations before a task
   * executes. The default returns 0, which is appropriate for metadata-only
   * subclasses (e.g. parquet_metadata_input). Subclasses that carry or
   * represent GPU-resident data should override to return a meaningful estimate.
   */
  [[nodiscard]] virtual std::size_t get_estimated_size_in_bytes() const { return 0; }

  /**
   * @brief Record the GPU this data should be processed on.
   *
   * Set upstream of task creation when the producer of this data has already
   * chosen a device — e.g. the scan manager round-robins fresh-read scan
   * splits across the available GPUs and stamps the chosen device here. The
   * task creator reads this back via @ref get_preferred_device_id and forwards
   * it onto the pipeline task so the scheduler dispatches the task to that GPU.
   */
  void set_preferred_device_id(int device_id) { _preferred_device_id = device_id; }

  /**
   * @brief The GPU this data was assigned to, or std::nullopt when no producer
   *        expressed a preference (the task creator then falls back to its own
   *        locality heuristics).
   */
  [[nodiscard]] std::optional<int> get_preferred_device_id() const { return _preferred_device_id; }

 private:
  /// Producer-assigned device preference; nullopt until set_preferred_device_id.
  std::optional<int> _preferred_device_id;
};

/**
 * @brief Operator data carrying data batches for pipeline execution.
 *
 * Unifies idle and read-only-locked batch access in a single container. Holds an
 * optional vector of idle data_batch shared_ptrs and/or an optional vector of
 * read_only_data_batch RAII accessors. Lazy conversion is performed on demand:
 *   - get_data_batches() populates _data_batches from _read_only_data_batches if needed
 *   - get_read_only_batches() populates _read_only_data_batches from _data_batches if needed
 *
 * prepare_for_processing() locks idle batches and stores the result in
 * _read_only_data_batches. remove_read_only_lock() releases all shared locks.
 */
class pipelineable_operator_data : public operator_data {
 public:
  pipelineable_operator_data()
  {
    _data_batches = std::vector<std::shared_ptr<::cucascade::data_batch>>();
  }
  explicit pipelineable_operator_data(
    std::vector<std::shared_ptr<::cucascade::data_batch>> data_batches)
    : _data_batches(std::move(data_batches))
  {
  }
  explicit pipelineable_operator_data(
    std::vector<::cucascade::read_only_data_batch> read_only_data_batches)
    : _read_only_data_batches(std::move(read_only_data_batches))
  {
  }

  [[nodiscard]] operator_data_type get_type() const override
  {
    return operator_data_type::PIPELINEABLE;
  }

  /**
   * @brief Get idle data batch pointers, lazily populating from read-only batches if needed.
   */
  [[nodiscard]] const std::vector<std::shared_ptr<::cucascade::data_batch>>& get_data_batches()
    const;

  /**
   * @brief Get read-only locked batches, lazily populating from idle batches if needed.
   */
  [[nodiscard]] std::vector<::cucascade::read_only_data_batch> get_read_only_batches(
    bool leave_locked = false) const;

  /**
   * @brief Release all read-only locks by resetting _read_only_data_batches.
   */
  void remove_read_only_lock()
  {
    // Releasing the lock means getting rid of any _read_only_data_batches that we may have cached.
    // But we want to make sure we do keep the data alive. So we ensure that the data_batches are
    // populated.
    if (!_data_batches) { auto _ = get_data_batches(); }
    _read_only_data_batches = std::nullopt;
  }

  /**
   * @brief Lock all data batches for processing in the requested memory space.
   *
   * Iterates over all idle batches and locks (or converts then locks) each one,
   * storing the results in _read_only_data_batches. Throws sirius::internal_exception
   * if any batch pointer is null or any batch fails to lock. Propagates rmm::out_of_memory.
   */
  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    std::size_t total = 0;
    auto ro_batches   = get_read_only_batches(false);
    for (auto const& ro : ro_batches) {
      if (ro.get_data()) { total += ro.get_data()->get_uncompressed_data_size_in_bytes(); }
    }
    return total;
  }

 private:
  mutable std::optional<std::vector<std::shared_ptr<::cucascade::data_batch>>> _data_batches;
  mutable std::optional<std::vector<::cucascade::read_only_data_batch>> _read_only_data_batches;
};

/**
 * @brief Operator data with partition index for partitioned pipeline execution.
 *
 * Extends pipelineable_operator_data to include partition index information,
 * used by partition-aware operators (partition, concat, etc.).
 */
class partitioned_operator_data : public pipelineable_operator_data {
 public:
  partitioned_operator_data() = default;
  partitioned_operator_data(std::vector<std::shared_ptr<::cucascade::data_batch>> data_batches,
                            std::size_t partition_idx)
    : pipelineable_operator_data(std::move(data_batches)), _partition_idx(partition_idx)
  {
  }

  [[nodiscard]] operator_data_type get_type() const override
  {
    return operator_data_type::PARTITIONED;
  }

  /**
   * @brief Get the partition index.
   * @return Partition index
   */
  [[nodiscard]] std::size_t get_partition_idx() const { return _partition_idx; }

 private:
  std::size_t _partition_idx = 0;
};

/**
 * @brief Input statistics passed to no_history_peak_memory_estimate().
 *
 * Carries the dimensions that operator overrides typically condition on: the
 * number of input batches, the total uncompressed byte count, and the runtime
 * type of the operator_data feeding the task (so overrides can branch on the
 * concrete input shape without a dynamic_cast).
 */
struct input_stats {
  std::size_t num_batches = 0;
  std::size_t bytes       = 0;
  operator_data_type type = operator_data_type::BASE;
  /// Mirrors @ref operator_data::is_resident on the data that fed this task,
  /// captured at stats-build time. Used by sirius_gpu_scan_operator's memory
  /// estimate to skip the decode-expansion factor on already-resident inputs.
  bool resident = false;
};

//! sirius_physical_operator is the base class of the physical operators present in the
//! execution plan
class sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::INVALID;
  //! Static counter for generating unique operator IDs
  static inline std::atomic<size_t> next_operator_id{0};

 public:
  sirius_physical_operator(SiriusPhysicalOperatorType type,
                           duckdb::vector<sirius::logical_type> types,
                           std::size_t estimated_cardinality)
    : type(type),
      types(std::move(types)),
      estimated_cardinality(estimated_cardinality),
      operator_id(next_operator_id++)
  {
  }
  sirius_physical_operator() : operator_id(next_operator_id++) {}
  virtual ~sirius_physical_operator() {}

  //! The physical operator type
  SiriusPhysicalOperatorType type;
  //! The set of children of the operator (operators that feed data _into_ the current operator)
  duckdb::vector<duckdb::unique_ptr<sirius_physical_operator>> children;
  //! The types returned by this physical operator
  duckdb::vector<sirius::logical_type> types;
  //! The estimated cardinality of this physical operator
  std::size_t estimated_cardinality;
  //! The unique ID of this operator (auto-incremented at creation)
  size_t operator_id;

  //! Lock for concurrent access to operator state
  std::mutex lock;

 public:
  virtual std::string get_name() const;

  virtual std::string params_to_string() const { return ""; }

  virtual std::string to_string() const;

  void print() const;

  virtual duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_children() const;

  //! Return a vector of the types that will be returned by this operator
  const duckdb::vector<sirius::logical_type>& get_types() const { return types; }

  //! Get the unique operator ID
  size_t get_operator_id() const { return operator_id; }

  virtual bool equals(const sirius_physical_operator& other) const { return false; }

  virtual void verify();

 public:
  // Operator interface

  /**
   * @brief Estimate peak GPU memory for this operator when no execution history is available.
   *
   * Called by gpu_pipeline_task::get_estimated_reservation_size_info() on the first task
   * execution for a pipeline (before any history records exist). The default is 2× the
   * input bytes, matching the historical fallback. Overrides may return a lower value for
   * operators that are known to be pass-throughs under certain conditions (e.g. concat with
   * a single batch, or partition with a single partition), or a higher value for operators
   * that expand input significantly (e.g. decompressing parquet).
   *
   * A return value of 0 means "no additional peak memory expected"; the caller will fall
   * back to the 2× default when every operator in the pipeline returns 0.
   *
   * @param stats  Batch count and total input bytes for the task about to run.
   * @return Estimated peak GPU bytes this operator will allocate.
   */
  [[nodiscard]] virtual std::size_t no_history_peak_memory_estimate(const input_stats& stats) const
  {
    return stats.bytes * 2;
  }

  virtual std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                                 rmm::cuda_stream_view stream);

  //! The influence the operator has on order (insertion order means no influence)
  virtual sirius::OrderPreservationType operator_order() const
  {
    return sirius::OrderPreservationType::INSERTION_ORDER;
  }

 public:
  // Source interface
  virtual bool is_source() const { return false; }

  //! The type of order emitted by the operator (as a source)
  virtual sirius::OrderPreservationType source_order() const
  {
    return sirius::OrderPreservationType::INSERTION_ORDER;
  }

 public:
  // Sink interface
  virtual void sink(const operator_data& input_data, rmm::cuda_stream_view stream);

  virtual bool is_sink() const { return false; }

  //! Whether or not the sink operator depends on the order of the input chunks
  //! If this is set to true, we cannot do things like caching intermediate vectors
  virtual bool sink_order_dependent() const { return false; }

 public:
  // Pipeline construction
  virtual duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_sources() const;

  //! Build the pipelines for the operator
  virtual void build_pipelines(pipeline::sirius_pipeline& current,
                               pipeline::sirius_meta_pipeline& meta_pipeline);

  //! Called when the pipeline this operator belongs to finishes. Sets finalized=true, then
  //! dispatches to on_finalize_operator(). Do not override this; override on_finalize_operator().
  void finalize_operator()
  {
    finalized.store(true);
    on_finalize_operator();
  }

  //! True after finalize_operator() has been called on this operator.
  std::atomic<bool> finalized = false;

 protected:
  //! Override this instead of finalize_operator() to perform cleanup when a pipeline finishes.
  virtual void on_finalize_operator() {}

 public:
  template <class TARGET>
  TARGET& Cast()
  {
    // TODO(amin) this is buggy code
    if (TARGET::TYPE != SiriusPhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw internal_exception(
        "Failed to cast physical operator to type - physical operator type mismatch");
    }
    return reinterpret_cast<TARGET&>(*this);
  }

  template <class TARGET>
  const TARGET& Cast() const
  {
    if (TARGET::TYPE != SiriusPhysicalOperatorType::INVALID && type != TARGET::TYPE) {
      throw internal_exception(
        "Failed to cast physical operator to type - physical operator type mismatch");
    }
    return reinterpret_cast<const TARGET&>(*this);
  }

  struct port {
    MemoryBarrierType type;
    /// May be NULL for dependency-only ports that carry no data flow (e.g., "dependency").
    /// Null repos are treated as "empty, not data-gating" by the base-class port handling methods
    /// (get_next_task_hint, get_next_task_input_data, all_ports_empty, push_data_batch).
    ::cucascade::shared_data_repository* repo;
    duckdb::shared_ptr<pipeline::sirius_pipeline> src_pipeline;
    duckdb::shared_ptr<pipeline::sirius_pipeline> dest_pipeline;
    //! A UUID for a port on an operator at the beginning of a
    // pipeline. This port receives data from a prior pipeline,
    // forming an incoming edge from that pipeline.
    uuid::UUID source_port_uuid{uuid::now_v7()};
  };

  /// Describes a downstream operator's port to which data is pushed
  struct next_port_info {
    //! The downstream operator that receives data batches from this sink
    sirius_physical_operator* next_operator;
    //! The port name on the downstream operator to push data into
    std::string_view next_operator_port_name;
    //! A UUID to encode the concept of a pseudo port, to conform to the model of quent,
    // that sits on an operator, at the end of a pipeline, sending data to a downstream
    // pipeline's first operator's receiving port, forming a directed edge from the current
    // operator's pipeline to the next_operator's pipeline:
    // ┌─ pipeline A ──────────────────┐  ┌─ pipeline B ──────────────────┐
    // │          ┌─ last op ───────┐  │  │  ┌─ first op ──────┐          │
    // │ ┌────┐   │ ┌─ pseudo ┐     │  │  │  │     ┌─ port ─┐  │   ┌────┐ │
    // │ │ op │...│ │        *───────────────────────▶      │  │...│ op │ │
    // │ └────┘   │ └─ port ──┘     │  │  │  │     └────────┘  │   └────┘ │
    // │          └─────────────────┘  │  │  └─────────────────┘          │
    // └───────────────────────────────┘  └───────────────────────────────┘
    uuid::UUID pseudo_sink_port_uuid;
  };

  // source pipeline pushed to repo of the ports
  void push_data_batch(std::string_view port_id, std::shared_ptr<::cucascade::data_batch> batch);
  //! Add a port to the operator
  void add_port(std::string_view port_id, std::unique_ptr<port> p);
  //! Get a port from the operator
  port* get_port(std::string_view port_id);
  //! Get all ports from the operator
  std::vector<std::string_view> get_port_ids();
  //! Check if the source pipeline is finished
  bool is_source_pipeline_finished();
  //! Returns true if any FULL-barrier port has src_pipeline == src
  bool has_full_barrier_from(const pipeline::sirius_pipeline* src) const;
  //! Add a next port after sink
  void add_next_port_after_sink(next_port_info port_info);
  //! Get the next ports after sink
  const std::vector<sirius_physical_operator::next_port_info>& get_next_ports_after_sink() const;

  //! Get the next task hint
  virtual std::optional<task_creation_hint> get_next_task_hint();

  /// \brief check if there are more tasks to create
  /// \note not necessarily ready to create at the moment
  /// the function is called
  virtual bool can_create_more_tasks() const
  {
    // WSM TODO implement this
    throw std::runtime_error("can_create_more_tasks not implemented for operator " + get_name());
    return true;
  }

  /// \brief check if this operator has exhausted its limit, allowing the pipeline to finish early
  virtual bool is_limit_exhausted() const { return false; }

  //! Get the input batch
  virtual std::unique_ptr<operator_data> get_next_task_input_data();

  //! Check if all ports are empty
  [[nodiscard]] virtual bool all_ports_empty();
  //! Check if the pipeline is finished
  bool check_pipeline_finished();

  //! Get pipeline
  duckdb::shared_ptr<pipeline::sirius_pipeline> get_pipeline() const noexcept;

  void set_pipeline(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline);

 protected:
  duckdb::shared_ptr<pipeline::sirius_pipeline> _pipeline;
  //! Lookup map: port name -> raw pointer into _ports_list (never owns)
  std::unordered_map<std::string, port*> ports;
  //! Ownership container for ports, kept sorted by src_pipeline->get_pipeline_id().
  //! std::list is used intentionally: its nodes have stable addresses, so raw pointers
  //! in `ports` are never invalidated by insertions.
  std::list<std::unique_ptr<port>> _ports_list;
  //! The next operators to be executed after this operator when it is used as a sink
  std::vector<sirius_physical_operator::next_port_info> next_port_after_sink;
};

}  // namespace op
}  // namespace sirius
