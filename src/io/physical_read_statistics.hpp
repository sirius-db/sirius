// Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace sirius::io {

// Opt-in backend observations. These count OS requests, not storage-device traffic.
// The owning datasource chooses the phase; shared ownership follows queued work.
struct physical_read_statistics {
  std::atomic<uint64_t> requests{0}, completions{0}, bytes_requested{0}, bytes_returned{0};
  std::atomic<uint64_t> failures{0}, short_reads{0}, retries{0}, unobserved_completions{0};
};

inline thread_local std::shared_ptr<physical_read_statistics> physical_reads_for_testing;

class scoped_physical_reads {
 public:
  explicit scoped_physical_reads(std::shared_ptr<physical_read_statistics> value) noexcept
    : previous_(std::exchange(physical_reads_for_testing, std::move(value)))
  {
  }
  ~scoped_physical_reads() { physical_reads_for_testing = std::move(previous_); }
  scoped_physical_reads(scoped_physical_reads const&)            = delete;
  scoped_physical_reads& operator=(scoped_physical_reads const&) = delete;

 private:
  std::shared_ptr<physical_read_statistics> previous_;
};

// One slot may retry several physical reads. Only accepted submissions are counted.
class physical_read_attempt {
 public:
  physical_read_attempt() = default;
  ~physical_read_attempt() { reset(); }
  physical_read_attempt(physical_read_attempt const&)            = delete;
  physical_read_attempt& operator=(physical_read_attempt const&) = delete;
  physical_read_attempt(physical_read_attempt&& other) noexcept
    : stats_(std::move(other.stats_)),
      requested_(other.requested_),
      attempts_(other.attempts_),
      pending_(std::exchange(other.pending_, false))
  {
  }
  physical_read_attempt& operator=(physical_read_attempt&&) = delete;
  void submitted(std::shared_ptr<physical_read_statistics> const& stats, size_t bytes) noexcept
  {
    if (!stats) return;
    stats_     = stats;
    requested_ = bytes;
    pending_   = true;
    stats_->requests.fetch_add(1, std::memory_order_relaxed);
    stats_->bytes_requested.fetch_add(bytes, std::memory_order_relaxed);
    if (attempts_++) stats_->retries.fetch_add(1, std::memory_order_relaxed);
  }
  void completed(int64_t result) noexcept
  {
    if (!pending_) return;
    pending_ = false;
    stats_->completions.fetch_add(1, std::memory_order_relaxed);
    if (result < 0)
      stats_->failures.fetch_add(1, std::memory_order_relaxed);
    else {
      stats_->bytes_returned.fetch_add(result, std::memory_order_relaxed);
      if (static_cast<uint64_t>(result) < requested_)
        stats_->short_reads.fetch_add(1, std::memory_order_relaxed);
    }
  }
  void reset() noexcept
  {
    if (pending_) stats_->unobserved_completions.fetch_add(1, std::memory_order_relaxed);
    stats_.reset();
    pending_   = false;
    attempts_  = 0;
    requested_ = 0;
  }

 private:
  std::shared_ptr<physical_read_statistics> stats_;
  size_t requested_{0}, attempts_{0};
  bool pending_{false};
};
}  // namespace sirius::io
