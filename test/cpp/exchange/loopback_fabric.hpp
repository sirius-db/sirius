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

#include "exchange/fabric.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace sirius::test::exchange_utils {

using sirius::exchange::fabric;
using sirius::exchange::fabric_notification;
using sirius::exchange::frame_ring;
using sirius::exchange::rank_t;

/**
 * @brief The ranks of an exchange in one process: a write is a device-to-device copy between
 * their rings, and a notification goes in the peer's inbox once the copy is done.
 */
class loopback_hub {
 public:
  struct options {
    /// Deliver each poll's notifications in a random order.
    bool shuffle = false;
    /// Deliver notifications only every this many polls, so several arrive together.
    unsigned deliver_every = 1;
    /// Fail every write after this many succeeded.
    std::optional<std::uint64_t> fail_after_writes;
  };

  loopback_hub(rank_t world, options opts) : _opts(opts), _inboxes(world), _recv(world, nullptr) {}
  explicit loopback_hub(rank_t world) : loopback_hub(world, options{}) {}

  std::unique_ptr<fabric> attach(rank_t rank, frame_ring& send, frame_ring& recv)
  {
    std::lock_guard lock{_mutex};
    _recv[rank] = &recv;
    return std::make_unique<endpoint>(*this, rank, send);
  }

 private:
  class endpoint : public fabric {
   public:
    endpoint(loopback_hub& hub, rank_t rank, frame_ring& send)
      : _hub(hub), _rank(rank), _send(send), _rng(rank)
    {
    }
    ~endpoint() override
    {
      _stream.synchronize_no_throw();
      for (auto& [id, w] : _writes) {
        static_cast<void>(cudaEventDestroy(w.done));
      }
    }

    rank_t rank() const noexcept override { return _rank; }
    rank_t world_size() const noexcept override
    {
      return static_cast<rank_t>(_hub._inboxes.size());
    }

    std::uint64_t write(rank_t peer,
                        std::size_t src,
                        std::size_t dst,
                        std::size_t bytes,
                        std::string notification) override
    {
      frame_ring* recv = nullptr;
      {
        std::lock_guard lock{_hub._mutex};
        recv = _hub._recv.at(peer);
      }
      pending w{peer, std::move(notification), nullptr, false};
      if (_hub._opts.fail_after_writes && _posted >= *_hub._opts.fail_after_writes) {
        w.failed = true;
      } else {
        RMM_CUDA_TRY(cudaMemcpyAsync(recv->base() + dst,
                                     _send.base() + src,
                                     bytes,
                                     cudaMemcpyDeviceToDevice,
                                     _stream.value()));
        RMM_CUDA_TRY(cudaEventCreateWithFlags(&w.done, cudaEventDisableTiming));
        RMM_CUDA_TRY(cudaEventRecord(w.done, _stream.value()));
      }
      ++_posted;
      _writes.emplace(_next, std::move(w));
      return _next++;
    }

    write_status status(std::uint64_t id) override
    {
      auto it = _writes.find(id);
      if (it->second.failed) {
        _writes.erase(it);
        return write_status::failed;
      }
      if (cudaEventQuery(it->second.done) == cudaErrorNotReady) { return write_status::pending; }
      RMM_CUDA_TRY(cudaEventDestroy(it->second.done));
      _hub.deliver(it->second.peer, {_rank, std::move(it->second.notification)});
      _writes.erase(it);
      return write_status::done;
    }

    void notify(rank_t peer, std::string notification) override
    {
      _hub.deliver(peer, {_rank, std::move(notification)});
    }

    void poll(std::vector<fabric_notification>& out) override
    {
      if (++_polls % _hub._opts.deliver_every != 0) { return; }
      std::vector<fabric_notification> taken;
      {
        std::lock_guard lock{_hub._mutex};
        taken.swap(_hub._inboxes[_rank]);
      }
      if (_hub._opts.shuffle) { std::shuffle(taken.begin(), taken.end(), _rng); }
      std::move(taken.begin(), taken.end(), std::back_inserter(out));
    }

   private:
    struct pending {
      rank_t peer;
      std::string notification;
      cudaEvent_t done;
      bool failed;
    };

    loopback_hub& _hub;
    rank_t _rank;
    frame_ring& _send;
    rmm::cuda_stream _stream;
    std::mt19937 _rng;
    std::map<std::uint64_t, pending> _writes;
    std::uint64_t _next{1};
    std::uint64_t _posted{0};
    std::uint64_t _polls{0};
  };

  void deliver(rank_t to, fabric_notification n)
  {
    std::lock_guard lock{_mutex};
    _inboxes.at(to).push_back(std::move(n));
  }

  options _opts;
  std::mutex _mutex;
  std::vector<std::vector<fabric_notification>> _inboxes;
  std::vector<frame_ring*> _recv;
};

}  // namespace sirius::test::exchange_utils
