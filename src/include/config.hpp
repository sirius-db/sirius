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

#include <expression_evaluator/expression_evaluator_strategy.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace duckdb {

/// Copy-on-read string for the process-wide mutable Config members below.
///
/// `SET` callbacks reassign these strings while other connections' queries read
/// them concurrently (register E2): a plain `std::string` reassign/read race is
/// a torn read or a use-after-free of the old buffer. Readers therefore receive
/// an immutable snapshot (`get()`, or the implicit `std::string` conversion for
/// source compatibility) and writers swap the backing shared_ptr under a mutex.
class ConfigString {
 public:
  explicit ConfigString(std::string initial)
    : value_(std::make_shared<const std::string>(std::move(initial)))
  {
  }

  ConfigString(const ConfigString&)            = delete;
  ConfigString& operator=(const ConfigString&) = delete;

  /// Immutable snapshot of the current value.
  [[nodiscard]] std::string get() const { return *load(); }

  /// Source-compatible read: `std::string s = Config::LOG_DIR;` still works and
  /// yields a stable copy.
  operator std::string() const { return get(); }  // NOLINT(google-explicit-constructor)

  ConfigString& operator=(std::string value)
  {
    auto next = std::make_shared<const std::string>(std::move(value));
    std::lock_guard<std::mutex> guard(mutex_);
    value_ = std::move(next);
    return *this;
  }

  friend bool operator==(const ConfigString& lhs, const char* rhs) { return *lhs.load() == rhs; }
  friend bool operator==(const ConfigString& lhs, const std::string& rhs)
  {
    return *lhs.load() == rhs;
  }

 private:
  [[nodiscard]] std::shared_ptr<const std::string> load() const
  {
    std::lock_guard<std::mutex> guard(mutex_);
    return value_;
  }

  mutable std::mutex mutex_;
  std::shared_ptr<const std::string> value_;
};

// If you are adding a new field to this struct, then you also need to make the following changes:
// * Specify the default value in config.cpp
// * Add a configuration field associated with Sirius (see InitialGPUConfigs in sirius_extension.cpp
// for examples)
//
// CONCURRENCY (register E2): these static members are written by `SET` callbacks
// while
// queries on other connections read them, with no other serialization — the
// query-lifecycle gate admits several queries and never covered SETs anyway.
// Scalars are therefore std::atomic (reads/writes stay source-compatible via
// the implicit conversions); strings are copy-on-read ConfigString. A SET is
// visible to queries that start after it; a per-plan-consistent value must be
// captured once per query instead of re-read mid-plan (see
// sirius::current_expression_evaluator_strategy below for the strategy knob).
struct Config {
  // For gpu buffer manager
  static std::atomic<bool> USE_PIN_MEM_FOR_CPU_PROCESSING;  // use_pin_memory
  static std::atomic<bool> USE_PIN_MEM_FOR_CACHING;         // use_pin_memory_for_caching

  // For expression executor
  static std::atomic<bool> USE_CUDF_EXPR;  // use_cudf_expr
  // Strategy used by sirius::expression_evaluator.
  // Do not read this directly from plan/execution code: use
  // sirius::current_expression_evaluator_strategy(), which serves the value the
  // active query snapshotted at admission so one plan is internally consistent.
  // TODO: this should eventually be selected adaptively per-call by the executor based on
  // expression shape and operator statistics; the config knob will become a policy override.
  static std::atomic<::sirius::expression_evaluator_strategy>
    EXPRESSION_EVALUATOR_STRATEGY;  // expression_evaluator_strategy

  // For gpu physical top-N
  static std::atomic<bool> USE_CUSTOM_TOP_N;  // use_custom_top_n

  // For gpu physical table scan
  static std::atomic<bool> USE_OPT_TABLE_SCAN;                   // use_opt_table_scan
  static std::atomic<int> OPT_TABLE_SCAN_NUM_CUDA_STREAMS;       // opt_table_scan_num_streams
  static std::atomic<uint64_t> OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE;  // opt_table_scan_memcpy_size

  // For printing gpu table
  static std::atomic<uint64_t> PRINT_GPU_TABLE_MAX_ROWS;

  // For checking whether to fall back to duckdb execution
  static std::atomic<bool> ENABLE_FALLBACK_CHECK;

  // Whether to use special JIT implementation for particular regex evaluation
  static std::atomic<bool> ENABLE_REGEX_JIT_IMPL;

  // Whether to use modified pipeline for the new execution model
  static std::atomic<bool> MODIFIED_PIPELINE;

  // For duckdb scan task:
  //  - the default batch size
  // TODO: probably want to use sirius config for this value
  static std::atomic<uint64_t> DEFAULT_SCAN_TASK_BATCH_SIZE;

  // For sort partitioning:
  //  - max bytes per sort partition (0 = auto based on 33% GPU memory)
  static std::atomic<uint64_t> MAX_SORT_PARTITION_BYTES;

  // Logging configuration
  static ConfigString LOG_BACKEND;
  static ConfigString LOG_LEVEL;
  static ConfigString LOG_DIR;
  static std::atomic<int> LOG_FLUSH_SECONDS;
};

}  // namespace duckdb

namespace sirius {

struct Config {
  static const uint64_t NUM_GPU_EXECUTOR_THREADS         = 2;
  static const uint64_t NUM_PIPELINE_EXECUTOR_THREADS    = 1;
  static const uint64_t NUM_DUCKDB_SCAN_EXECUTOR_THREADS = 2;
  static const uint64_t NUM_DOWNGRADE_EXECUTOR_THREADS   = 1;
  static const uint64_t NUM_GPU                          = 1;
};

/// The expression-evaluator strategy for the CURRENT query: the value the
/// active execution window snapshotted at admission when the calling thread
/// holds one (see sirius::query_config_snapshot in sirius_config.hpp), else the
/// live global `duckdb::Config::EXPRESSION_EVALUATOR_STRATEGY`. Plan
/// generation runs on the window-holding thread, so every operator of one plan
/// captures the same value — a concurrent `SET expression_evaluator_strategy`
/// affects only queries admitted after it (register E2).
expression_evaluator_strategy current_expression_evaluator_strategy() noexcept;

}  // namespace sirius
