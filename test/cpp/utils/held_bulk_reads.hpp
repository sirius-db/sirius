/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "utils/tracking_local_ioctx.hpp"

#include <rmm/resource_ref.hpp>

#include <cucascade/exec/semi_future.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::test {

struct first_file_error : std::runtime_error {
  first_file_error() : std::runtime_error("first file read failed") {}
};
struct next_file_error : std::runtime_error {
  next_file_error() : std::runtime_error("next file allocation failed") {}
};

// The local backend has already written the real Parquet bytes before we publish a
// held completion. No test, including one against an incorrect caller, writes a late
// sentinel into a freed device allocation.
class held_bulk_io : public tracking_local_ioctx {
  struct request {
    std::size_t bytes;
    std::vector<void*> targets;
    std::optional<cucascade::exec::promise<std::size_t>> terminal{std::in_place};
    std::size_t releases{};
  };

 public:
  enum class setup_fault { none, after_failed_read, after_successful_read };

  explicit held_bulk_io(setup_fault fault) : tracking_local_ioctx(true), fault_(fault) {}

  void back_short_object_with(std::shared_ptr<cucascade::io::io_object const> full)
  {
    full_object_ = std::move(full);
  }

  cucascade::exec::semi_future<std::size_t> mixed_readv_async_io(
    cucascade::io::io_object const& object,
    std::vector<cucascade::io::prepared_io_slice>&& slices) noexcept override
  {
    try {
      bool const clipped = full_object_ && full_object_->object_path() == object.object_path();
      std::vector<void*> targets;
      std::size_t requested = 0, available = 0;
      for (auto const& slice : slices) {
        targets.push_back(slice.d_buffer.data);
        requested += slice.size();
        if (slice.offset() < object.size()) {
          available += std::min(slice.size(), object.size() - slice.offset());
        }
      }
      // The shorter logical object reports EOF, but a caller that ignores the count
      // still decodes complete, valid chunks from the full backing file.
      auto bytes = tracking_local_ioctx::mixed_readv_async_io(clipped ? *full_object_ : object,
                                                              std::move(slices))
                     .get();
      if (clipped) {
        if (bytes != requested) { throw std::logic_error("incomplete test backing file"); }
        bytes = available;
      }
      std::lock_guard lock(mutex_);
      auto& req   = reads_.emplace_back(request{bytes, std::move(targets)});
      auto future = req.terminal->get_semi_future();
      peak_       = std::max(peak_, ++pending_);
      if (fault_ != setup_fault::none && reads_.size() == 1) {
        fail_next_alloc_ = true;
        settle(req, fault_ == setup_fault::after_failed_read);
      } else if (open_) {
        settle(req, false);
      }
      changed_.notify_all();
      return future;
    } catch (...) {
      return cucascade::exec::make_semi_future<std::size_t>(std::current_exception());
    }
  }

  std::size_t wait_issued(std::size_t count, std::chrono::milliseconds bound)
  {
    std::unique_lock lock(mutex_);
    changed_.wait_for(lock, bound, [&] { return reads_.size() >= count; });
    return reads_.size();
  }

  void release(std::size_t i, bool fail = false)
  {
    std::lock_guard lock(mutex_);
    settle(reads_.at(i), fail);
  }

  // Also settle requests that arrive after cleanup starts (in particular on serial main).
  void open()
  {
    std::lock_guard lock(mutex_);
    open_ = true;
    for (auto& req : reads_) {
      if (req.terminal) { settle(req, false); }
    }
  }

  void allocating()
  {
    std::lock_guard lock(mutex_);
    if (fail_next_alloc_) {
      fail_next_alloc_ = false;
      setup_fault_     = true;
      throw next_file_error{};
    }
  }

  void releasing(void* p) noexcept
  {
    std::lock_guard lock(mutex_);
    // Take the most recent registration if an allocator reused the address.
    for (auto it = reads_.rbegin(); it != reads_.rend(); ++it) {
      auto target = std::find(it->targets.begin(), it->targets.end(), p);
      if (target != it->targets.end()) {
        early_release_ |= it->terminal.has_value();
        *target = nullptr;
        ++it->releases;
        return;
      }
    }
  }

  struct snapshot {
    std::size_t calls, peak, sibling_targets, sibling_releases;
    bool early_release, setup_fault;
  };
  snapshot sample(std::size_t sibling = 1) const
  {
    std::lock_guard lock(mutex_);
    return {reads_.size(),
            peak_,
            reads_.size() > sibling ? reads_[sibling].targets.size() : 0,
            reads_.size() > sibling ? reads_[sibling].releases : 0,
            early_release_,
            setup_fault_};
  }

  std::size_t completion_bytes(std::size_t i) const
  {
    std::lock_guard lock(mutex_);
    return reads_.at(i).bytes;
  }

 private:
  void settle(request& req, bool fail)
  {
    if (fail) {
      req.terminal.value().set_exception(first_file_error{});
    } else {
      req.terminal.value().set_value(std::size_t{req.bytes});
    }
    req.terminal.reset();
    --pending_;
  }

  setup_fault const fault_;
  std::shared_ptr<cucascade::io::io_object const> full_object_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<request> reads_;
  std::size_t pending_{}, peak_{};
  bool open_{}, early_release_{}, fail_next_alloc_{}, setup_fault_{};
};

class bulk_target_resource {
 public:
  bulk_target_resource(rmm::device_async_resource_ref upstream, held_bulk_io& io)
    : upstream_(upstream), io_(&io)
  {
  }
  void* allocate(::cuda::stream_ref s, std::size_t n, std::size_t a = alignof(std::max_align_t))
  {
    io_->allocating();
    return upstream_.allocate(s, n, a);
  }
  void deallocate(::cuda::stream_ref s,
                  void* p,
                  std::size_t n,
                  std::size_t a = alignof(std::max_align_t)) noexcept
  {
    io_->releasing(p);
    upstream_.deallocate(s, p, n, a);
  }
  void* allocate_sync(std::size_t n, std::size_t a = alignof(std::max_align_t))
  {
    io_->allocating();
    return upstream_.allocate_sync(n, a);
  }
  void deallocate_sync(void* p, std::size_t n, std::size_t a = alignof(std::max_align_t)) noexcept
  {
    io_->releasing(p);
    upstream_.deallocate_sync(p, n, a);
  }
  bool operator==(bulk_target_resource const& other) const noexcept { return io_ == other.io_; }
  friend void get_property(bulk_target_resource const&, ::cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref upstream_;
  held_bulk_io* io_;
};

}  // namespace sirius::test
