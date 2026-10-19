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

#include "exec/exchange_direct.hpp"

#include "sirius/exception.hpp"

#include <rmm/cuda_device.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <utility>

namespace sirius::exec {

direct_exchange::direct_exchange(cucascade::memory::memory_space& gpu)
  : _gpu{gpu}, _device{gpu.get_device_id()}
{
}

std::uint64_t direct_exchange::adopt(std::shared_ptr<cucascade::data_batch> batch,
                                     std::function<void()> on_gone)
{
  if (!batch) { throw sirius::invalid_input_exception("direct exchange: adopt() of a null batch"); }
  std::lock_guard const lock{_mutex};
  require_open();
  auto const token = _next++;
  _held.emplace(token, std::move(batch));
  if (on_gone) { _on_gone.emplace(token, std::move(on_gone)); }
  return token;
}

std::shared_ptr<cucascade::data_batch> direct_exchange::take_batch(std::uint64_t token)
{
  std::shared_ptr<cucascade::data_batch> batch;
  {
    std::lock_guard const lock{_mutex};
    require_open();
    auto const it = _held.find(token);
    if (it == _held.end()) {
      throw sirius::invalid_input_exception("direct exchange: token {} holds no received batch",
                                            token);
    }
    batch = std::move(it->second);
    _held.erase(it);
  }
  gone(token);
  return batch;
}

void direct_exchange::release(std::uint64_t token)
{
  {
    std::lock_guard const lock{_mutex};
    // The batch's buffers are freed on the device they live on.
    rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_device}};
    require_open();
    _held.erase(token);
  }
  gone(token);
}

std::size_t direct_exchange::outstanding() const
{
  std::lock_guard const lock{_mutex};
  require_open();
  return _held.size();
}

void direct_exchange::close()
{
  std::map<std::uint64_t, std::function<void()>> on_gone;
  {
    std::lock_guard const lock{_mutex};
    rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_device}};
    _held.clear();
    on_gone.swap(_on_gone);
    _closed = true;
  }
  for (auto& [token, callback] : on_gone) {
    if (callback) { callback(); }
  }
}

void direct_exchange::gone(std::uint64_t token)
{
  std::function<void()> callback;
  {
    std::lock_guard const lock{_mutex};
    auto const it = _on_gone.find(token);
    if (it == _on_gone.end()) { return; }
    callback = std::move(it->second);
    _on_gone.erase(it);
  }
  if (callback) { callback(); }
}

void direct_exchange::require_open() const
{
  if (_closed) { throw sirius::invalid_input_exception("direct exchange: already closed"); }
}

}  // namespace sirius::exec
