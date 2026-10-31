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

#include "sirius/exception.hpp"

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <exception>
#include <functional>
#include <memory>

namespace sirius {
enum class gpu_completion_point { prepare, execute, failure, sink, retry, shutdown, retirement };

// Tests inject a failed observation only after the real CUDA wait succeeds. This permits
// inspection and eventual destruction of retained owners without creating an unsafe callback.
struct gpu_completion_test_hooks {
  std::function<bool(gpu_completion_point)> fail;
  std::function<void()> before_task_owner_allocation;
  std::function<void(void*, void (*)(void*))> retain;
  bool joined = false;
  gpu_completion_test_hooks* previous;
  static inline thread_local gpu_completion_test_hooks* current = nullptr;
  gpu_completion_test_hooks() : previous(current) { current = this; }
  ~gpu_completion_test_hooks() { current = previous; }
  gpu_completion_test_hooks(gpu_completion_test_hooks const&)            = delete;
  gpu_completion_test_hooks& operator=(gpu_completion_test_hooks const&) = delete;
};

inline bool observe_gpu_completion(cudaError_t status, gpu_completion_point point) noexcept
{
  auto* hooks = gpu_completion_test_hooks::current;
  if (hooks) {
    hooks->joined = status == cudaSuccess;
    if (hooks->joined && hooks->fail) {
      try {
        if (hooks->fail(point)) return false;
      } catch (...) {
        return false;
      }
    }
  }
  return status == cudaSuccess;
}
inline bool join_gpu_work(::cuda::stream_ref stream, gpu_completion_point point) noexcept
{
  return observe_gpu_completion(cudaStreamSynchronize(stream.get()), point);
}
inline bool is_unjoined_gpu_work(std::exception_ptr error) noexcept
{
  try {
    if (error) std::rethrow_exception(error);
  } catch (unjoined_gpu_work const&) {
    return true;
  } catch (...) {
  }
  return false;
}

// No allocation on the production failure path. The containing runtime must retain the
// allocator/backing and stream too; this owner is deliberately never destroyed in that case.
template <class T>
void retain_unjoined_owner(std::unique_ptr<T>& owner) noexcept
{
  if (!owner) return;
  auto* hooks = gpu_completion_test_hooks::current;
  if (hooks && hooks->joined && hooks->retain) {
    auto* raw = owner.release();
    try {
      hooks->retain(raw, [](void* value) { delete static_cast<T*>(value); });
    } catch (...) {
    }
    return;
  }
  (void)owner.release();
}
}  // namespace sirius
