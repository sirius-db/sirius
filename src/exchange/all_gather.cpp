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

#include "exchange/all_gather.hpp"

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "exchange/exchange_service.hpp"
#include "sirius/exception.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <condition_variable>
#include <mutex>
#include <set>

namespace sirius::exchange {
namespace {

/// Collects one table and an end from every rank.
struct collector : inbound {
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::shared_ptr<cucascade::data_batch>> tables;
  std::set<rank_t> ended;
  std::exception_ptr error;

  explicit collector(rank_t world) : tables(world) {}

  void on_batch(rank_t from,
                std::uint64_t /*batch*/,
                std::shared_ptr<cucascade::data_batch> data) override
  {
    std::lock_guard lock{mutex};
    tables.at(from) = std::move(data);
  }
  void on_end(rank_t from) override
  {
    std::lock_guard lock{mutex};
    ended.insert(from);
    changed.notify_all();
  }
  void on_error(std::exception_ptr e) override
  {
    std::lock_guard lock{mutex};
    error = std::move(e);
    changed.notify_all();
  }
};

}  // namespace

std::vector<std::shared_ptr<cucascade::data_batch>> all_gather(
  exchange_service& service,
  stream_id stream,
  std::shared_ptr<cucascade::data_batch> mine,
  cucascade::memory::memory_space& gpu,
  rmm::cuda_stream_view cuda_stream,
  std::chrono::milliseconds timeout)
{
  auto const world = service.world_size();
  auto c           = std::make_shared<collector>(world);
  service.attach(stream, c);
  struct detach_on_exit {
    exchange_service& s;
    stream_id id;
    ~detach_on_exit() { s.detach(id); }
  } const detach{service, stream};

  bool const contributes = mine && get_cudf_table_view(*mine).num_rows() > 0;
  for (rank_t r = 0; r < world; ++r) {
    if (contributes) { service.send(stream, r, mine); }
    service.close(stream, r);
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> tables;
  {
    std::unique_lock lock{c->mutex};
    if (!c->changed.wait_for(lock, timeout, [&] { return c->error || c->ended.size() == world; })) {
      throw sirius::internal_exception("exchange all-gather on stream {}: {} of {} ranks in {} ms",
                                       stream,
                                       c->ended.size(),
                                       world,
                                       timeout.count());
    }
    if (c->error) { std::rethrow_exception(c->error); }
    tables = std::move(c->tables);
  }

  // A table that landed on host is copied back to the GPU.
  for (auto& table : tables) {
    if (!table) { continue; }
    auto ro = table->to_read_only();
    if (ro.get_current_tier() == cucascade::memory::Tier::GPU) { continue; }
    auto const id = sirius::get_next_batch_id();
    auto copy     = ro.clone_to<cucascade::gpu_table_representation>(
      sirius::converter_registry::get(), id, &gpu, cuda_stream);
    table = std::move(copy);
  }
  return tables;
}

void leave_all_gather(exchange_service& service, stream_id stream)
{
  service.detach(stream);
  for (rank_t r = 0; r < service.world_size(); ++r) {
    service.close(stream, r);
  }
}

}  // namespace sirius::exchange
