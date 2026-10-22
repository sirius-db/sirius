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

#include "exchange/transport.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <chrono>
#include <memory>
#include <vector>

namespace cucascade {
class data_batch;
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace sirius::exchange {

class exchange_service;

/**
 * @brief Every rank contributes one small table on @p stream; each gets all of them.
 *
 * Blocks until every rank's table has arrived, so every rank must call it with the same stream
 * id. Meant for small tables, such as sort samples and filters.
 *
 * @param mine This rank's table, on the GPU; null or empty to contribute nothing.
 * @param gpu Where tables that landed elsewhere are brought before they are returned.
 * @return One entry per rank, in rank order, on the GPU; null for a rank that contributed nothing.
 * @throws the exchange's error, or sirius::internal_exception after @p timeout.
 */
[[nodiscard]] std::vector<std::shared_ptr<cucascade::data_batch>> all_gather(
  exchange_service& service,
  stream_id stream,
  std::shared_ptr<cucascade::data_batch> mine,
  cucascade::memory::memory_space& gpu,
  rmm::cuda_stream_view cuda_stream,
  std::chrono::milliseconds timeout = std::chrono::minutes{10});

/**
 * @brief Takes part in @p stream's all-gather with nothing, without waiting for the other ranks.
 *
 * Every rank gets this rank's end; what they send here is dropped. For a rank that would ignore the
 * gathered tables, so that it never waits on a peer that may be waiting on it.
 */
void leave_all_gather(exchange_service& service, stream_id stream);

}  // namespace sirius::exchange
