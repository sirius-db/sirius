// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "output/parquet_spec.hpp"

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>

namespace sirius::output {
// No GPU or frontend dependencies. Callbacks perform encoding outside the lock.
// The query owner must quiesce tasks before abort(), finish(), or destruction.
class parquet_session {
 public:
  explicit parquet_session(parquet_export_spec spec);
  ~parquet_session();
  parquet_session(parquet_session const&)            = delete;
  parquet_session& operator=(parquet_session const&) = delete;
  using input_id                                     = std::vector<std::uint64_t>;
  using write_part = std::function<void(std::size_t, std::filesystem::path const&)>;
  void write(input_id const& id,
             std::optional<std::size_t> position,
             std::vector<std::uint64_t> const& nonempty_rows,
             write_part const& writer);
  parquet_export_result finish(
    write_part const& write_empty,
    std::function<void(parquet_export_result const&)> const& prepare_completion = {});
  void abort() noexcept;
  parquet_export_spec const& spec() const { return spec_; }

 private:
  struct input_record {
    std::optional<std::size_t> position;
    std::vector<std::uint64_t> rows;
    std::vector<std::filesystem::path> files;
    bool active    = false;
    bool committed = false;
  };
  void start_locked();
  parquet_export_spec spec_;
  void reserve_part_locked(std::filesystem::path const& path);
  void remove_part_locked(std::filesystem::path const& path) noexcept;
  std::set<std::filesystem::path> owned_files_;
  std::mutex mutex_;
  std::optional<bool> positioned_;
  std::set<std::size_t> positions_;
  std::map<input_id, input_record> inputs_;
  std::atomic<std::uint64_t> next_{0};
  bool started_           = false;
  bool created_directory_ = false;
  bool successful_        = false;
  bool aborted_           = false;
};
}  // namespace sirius::output
