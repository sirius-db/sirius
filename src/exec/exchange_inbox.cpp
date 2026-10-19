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

#include "exec/exchange_inbox.hpp"

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "exec/exchange_staging.hpp"
#include "log/logging.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <format>
#include <utility>

namespace sirius::exec {
namespace {

rmm::cuda_stream stream_on(int device)
{
  rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{device}};
  return rmm::cuda_stream{};
}

}  // namespace

exchange_inbox::exchange_inbox(cucascade::memory::memory_space& gpu)
  : _gpu{gpu},
    _device{gpu.get_device_id()},
    _stream{stream_on(_device)},
    _held{std::make_shared<cucascade::shared_data_repository>()}
{
}

void exchange_inbox::enable_spill(exchange_staging& staging, make_room_fn make_room)
{
  {
    std::lock_guard const lock{_mutex};
    _make_room = std::move(make_room);
  }
  staging.track(_held);
}

std::unique_ptr<cucascade::memory::reservation> exchange_inbox::reserve(std::size_t bytes)
{
  make_room_fn make_room;
  {
    // A closed exchange's memory space may be gone with its Context; refuse before touching it.
    std::lock_guard const lock{_mutex};
    require_open();
    make_room = _make_room;
  }
  auto reservation = _gpu.make_reservation_or_null(bytes);
  if (reservation || !make_room) { return reservation; }
  // A request may fail without having run: closing any query window drains the downgrade
  // executor, which cancels what is queued. Only a request that ran and still left too little
  // is final.
  constexpr int attempts = 3;
  for (int attempt = 1; attempt <= attempts; ++attempt) {
    try {
      make_room(bytes);
      break;
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN("exchange inbox: making room for {} bytes failed (attempt {} of {}): {}",
                      bytes,
                      attempt,
                      attempts,
                      e.what());
    }
  }
  return _gpu.make_reservation_or_null(bytes);
}

void exchange_inbox::bring_to_gpu(cucascade::data_batch& batch)
{
  std::size_t bytes = 0;
  {
    auto const read_only = batch.to_read_only();
    if (read_only.get_current_tier() == cucascade::memory::Tier::GPU) { return; }
    bytes = sirius::peak_materialization_bytes(read_only.get_data());
  }
  // Making room may spill, so it runs without the lock.
  auto reservation = reserve(bytes);
  if (!reservation) {
    throw rmm::out_of_memory(
      std::format("exchange inbox: {} bytes to bring a spilled batch back, {} available",
                  bytes,
                  _gpu.get_available_memory()));
  }
  // The copy and the conversion's sync run on the exchange's stream; the lock keeps another
  // bring_to_gpu() from attaching its reservation to that stream meanwhile. The lock is taken
  // before the batch's.
  std::lock_guard const lock{_mutex};
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_device}};
  require_open();
  // Blocking: whoever holds the batch now (the downgrade executor mid-spill) releases it soon.
  auto mutable_batch = batch.to_mutable();
  if (mutable_batch.get_current_tier() == cucascade::memory::Tier::GPU) { return; }
  auto const stream = _stream.view();
  auto* tracker     = _gpu.get_memory_resource_of<cucascade::memory::Tier::GPU>();
  if (!tracker->attach_reservation_to_tracker(stream, std::move(reservation))) {
    throw sirius::internal_exception("exchange inbox: this thread already tracks a reservation");
  }
  absl::Cleanup detach = [&] { tracker->reset_stream_reservation(stream); };
  mutable_batch.convert_to<cucascade::gpu_table_representation>(
    sirius::converter_registry::get(), &_gpu, stream);
}

std::uint64_t exchange_inbox::adopt(std::shared_ptr<cucascade::data_batch> batch,
                                    std::function<void()> on_gone)
{
  if (!batch) { throw sirius::invalid_input_exception("exchange inbox: adopt() of a null batch"); }
  std::lock_guard const lock{_mutex};
  require_open();
  auto const token = _next++;
  _held_ids.emplace(token, batch->get_batch_id());
  _held->add_data_batch(std::move(batch));
  if (on_gone) { _on_gone.emplace(token, std::move(on_gone)); }
  return token;
}

std::shared_ptr<cucascade::data_batch> exchange_inbox::take_batch(std::uint64_t token)
{
  std::shared_ptr<cucascade::data_batch> batch;
  {
    std::lock_guard const lock{_mutex};
    require_open();
    auto const it = _held_ids.find(token);
    if (it == _held_ids.end()) {
      throw sirius::invalid_input_exception("exchange inbox: token {} holds no received batch",
                                            token);
    }
    batch = _held->pop_data_batch_by_id(it->second);
    _held_ids.erase(it);
  }
  gone(token);
  if (!batch) {
    throw sirius::internal_exception("exchange inbox: token {} lost its batch", token);
  }
  return batch;
}

void exchange_inbox::release(std::uint64_t token)
{
  {
    std::lock_guard const lock{_mutex};
    // The batch's buffers are freed on the device they live on.
    rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_device}};
    require_open();
    if (auto const it = _held_ids.find(token); it != _held_ids.end()) {
      (void)_held->pop_data_batch_by_id(it->second);
      _held_ids.erase(it);
    }
  }
  gone(token);
}

std::size_t exchange_inbox::outstanding() const
{
  std::lock_guard const lock{_mutex};
  require_open();
  return _held_ids.size();
}

void exchange_inbox::close()
{
  std::map<std::uint64_t, std::function<void()>> on_gone;
  {
    std::lock_guard const lock{_mutex};
    rmm::cuda_set_device_raii const device{rmm::cuda_device_id{_device}};
    for (auto const& [token, batch_id] : _held_ids) {
      (void)_held->pop_data_batch_by_id(batch_id);
    }
    _held_ids.clear();
    on_gone.swap(_on_gone);
    _closed = true;
  }
  for (auto& [token, callback] : on_gone) {
    if (callback) { callback(); }
  }
}

void exchange_inbox::gone(std::uint64_t token)
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

void exchange_inbox::require_open() const
{
  if (_closed) { throw sirius::invalid_input_exception("exchange inbox: already closed"); }
}

}  // namespace sirius::exec
