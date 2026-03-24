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

#include <cucascade/data/data_batch.hpp>
#include <cuda_runtime_api.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace sirius {
namespace pipeline {

struct release_packet {
  cudaEvent_t event;
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
};

class batch_release_pool {
 public:
  explicit batch_release_pool(int device_id, int num_threads = 2, int initial_events = 64)
    : _device_id(device_id), _running(true)
  {
    cudaSetDevice(_device_id);
    _event_pool.reserve(initial_events);
    for (int i = 0; i < initial_events; ++i) {
      cudaEvent_t event;
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      _event_pool.push_back(event);
    }

    _workers.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      _workers.emplace_back([this]() { worker_loop(); });
    }
  }

  ~batch_release_pool() { stop(); }

  batch_release_pool(const batch_release_pool&)            = delete;
  batch_release_pool& operator=(const batch_release_pool&) = delete;

  cudaEvent_t acquire_event()
  {
    std::lock_guard<std::mutex> lock(_event_mutex);
    if (_event_pool.empty()) {
      cudaEvent_t event;
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      return event;
    }
    cudaEvent_t event = _event_pool.back();
    _event_pool.pop_back();
    return event;
  }

  void submit(release_packet packet)
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _queue.push(std::move(packet));
    }
    _cv.notify_one();
  }

  void stop()
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (!_running) { return; }
      _running = false;
    }
    _cv.notify_all();
    for (auto& worker : _workers) {
      if (worker.joinable()) { worker.join(); }
    }
    // Destroy all pooled events
    std::lock_guard<std::mutex> lock(_event_mutex);
    for (auto& event : _event_pool) {
      cudaEventDestroy(event);
    }
    _event_pool.clear();
  }

 private:
  void return_event(cudaEvent_t event)
  {
    std::lock_guard<std::mutex> lock(_event_mutex);
    _event_pool.push_back(event);
  }

  void worker_loop()
  {
    cudaSetDevice(_device_id);
    while (true) {
      release_packet packet;
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this]() { return !_queue.empty() || !_running; });
        if (!_running && _queue.empty()) { return; }
        packet = std::move(_queue.front());
        _queue.pop();
      }
      cudaEventSynchronize(packet.event);
      packet.batches.clear();
      return_event(packet.event);
    }
  }

  int _device_id;
  bool _running;
  std::mutex _mutex;
  std::condition_variable _cv;
  std::queue<release_packet> _queue;
  std::vector<std::thread> _workers;
  std::mutex _event_mutex;
  std::vector<cudaEvent_t> _event_pool;
};

}  // namespace pipeline
}  // namespace sirius
