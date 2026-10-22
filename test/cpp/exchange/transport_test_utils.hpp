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

#include "exchange/exchange_test_utils.hpp"
#include "exchange/transport.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

/// What exchange tests send through a transport, and the checks on what arrived. Batches are
/// derived from (sender, receiver, index), so a rank can check what it received from a peer in
/// another process.
namespace sirius::test::exchange_utils {

using sirius::exchange::rank_t;
using sirius::exchange::stream_id;
using sirius::exchange::transport;

/// What one rank received.
struct received {
  std::mutex mutex;
  std::condition_variable changed;
  std::map<std::pair<rank_t, stream_id>, std::vector<table_digest>> batches;
  std::set<std::pair<rank_t, stream_id>> ended;
  std::exception_ptr error;

  template <typename Predicate>
  bool wait_for(Predicate done, std::chrono::seconds limit = std::chrono::seconds{60})
  {
    std::unique_lock lock{mutex};
    return changed.wait_for(lock, limit, [&] { return done(); });
  }
};

/// Callbacks that record into @p got.
inline transport::callbacks recording(received& got)
{
  auto* g = &got;
  return {[g](stream_id stream, rank_t from, std::shared_ptr<cucascade::data_batch> batch) {
            auto d = digest(batch);
            std::lock_guard lock{g->mutex};
            g->batches[{from, stream}].push_back(std::move(d));
            g->changed.notify_all();
          },
          [g](stream_id stream, rank_t from) {
            std::lock_guard lock{g->mutex};
            REQUIRE(g->ended.insert({from, stream}).second);
            g->changed.notify_all();
          },
          [g](std::exception_ptr error) {
            std::lock_guard lock{g->mutex};
            g->error = error;
            g->changed.notify_all();
          }};
}

/// Batch @p i from @p from to @p to.
inline std::shared_ptr<cucascade::data_batch> exchange_batch(rank_t from,
                                                             rank_t to,
                                                             int i,
                                                             std::size_t rows)
{
  return gpu_batch(sample_table(rows + 97 * (from + to + static_cast<rank_t>(i))));
}

/// Sends @p per_stream batches of about @p rows rows on each stream from @p t to every rank,
/// itself included, then closes every stream.
inline void send_all(transport& t,
                     std::vector<stream_id> const& streams,
                     int per_stream,
                     std::size_t rows)
{
  for (rank_t to = 0; to < t.world_size(); ++to) {
    for (auto const stream : streams) {
      for (int i = 0; i < per_stream; ++i) {
        t.send(stream, to, exchange_batch(t.rank(), to, i, rows));
      }
    }
  }
  for (rank_t to = 0; to < t.world_size(); ++to) {
    for (auto const stream : streams) {
      t.close(stream, to);
    }
  }
}

/// Waits until rank @p to has every end, then checks it received exactly what send_all() sent it.
inline void check_all(received& got,
                      rank_t to,
                      rank_t world,
                      std::vector<stream_id> const& streams,
                      int per_stream,
                      std::size_t rows)
{
  REQUIRE(got.wait_for([&] {
    if (got.error) { return true; }
    for (rank_t from = 0; from < world; ++from) {
      for (auto const stream : streams) {
        if (!got.ended.contains({from, stream})) { return false; }
      }
    }
    return true;
  }));
  std::lock_guard lock{got.mutex};
  REQUIRE_FALSE(got.error);
  auto const by_hashes = [](table_digest const& a, table_digest const& b) {
    return a.row_hashes < b.row_hashes;
  };
  for (rank_t from = 0; from < world; ++from) {
    std::vector<table_digest> want;
    for (int i = 0; i < per_stream; ++i) {
      want.push_back(digest(exchange_batch(from, to, i, rows)));
    }
    std::sort(want.begin(), want.end(), by_hashes);
    for (auto const stream : streams) {
      auto have = got.batches[{from, stream}];
      std::sort(have.begin(), have.end(), by_hashes);
      CHECK(have == want);
    }
  }
}

}  // namespace sirius::test::exchange_utils
