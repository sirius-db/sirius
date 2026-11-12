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

#include "compression/spill_dictionary_registry.hpp"

#include "log/logging.hpp"
#include "telemetry/nvtx.hpp"

#include <exception>
#include <utility>

namespace sirius::compression {

spill_dictionary_registry& spill_dictionary_registry::global()
{
  static spill_dictionary_registry registry;
  return registry;
}

spill_dictionary_registry::~spill_dictionary_registry()
{
  {
    std::lock_guard lock(_mutex);
    _stop = true;
  }
  _wake.notify_all();
  if (_worker.joinable()) { _worker.join(); }
}

std::shared_ptr<simpatico::onpair_dictionary const> spill_dictionary_registry::find(
  spill_column_key const& key) const
{
  std::lock_guard lock(_mutex);
  auto const it = _entries.find(key);
  return it == _entries.end() ? nullptr : it->second.dict;
}

bool spill_dictionary_registry::needs_training(spill_column_key const& key) const
{
  std::lock_guard lock(_mutex);
  return !_entries.contains(key);
}

bool spill_dictionary_registry::train_async(spill_column_key const& key,
                                            simpatico::onpair_training_sample sample,
                                            simpatico::onpair_encoder_config const& config)
{
  {
    std::lock_guard lock(_mutex);
    if (_stop || _entries.contains(key)) { return false; }
    auto const generation = _next_generation++;
    _entries.emplace(key, entry{.generation = generation});
    _jobs.push_back(job{key, generation, std::move(sample), config});
    if (!_worker.joinable()) {
      _worker = std::thread([this] { run(); });
    }
  }
  _wake.notify_one();
  return true;
}

bool spill_dictionary_registry::record_ratio(
  spill_column_key const& key,
  std::shared_ptr<simpatico::onpair_dictionary const> const& dict,
  double ratio,
  double min_fraction)
{
  std::lock_guard lock(_mutex);
  auto const it = _entries.find(key);
  if (it == _entries.end() || it->second.dict != dict) { return true; }
  auto& e = it->second;
  if (e.baseline_ratio == 0.0) {
    e.baseline_ratio = ratio;
    return true;
  }
  if (ratio >= min_fraction * e.baseline_ratio) { return true; }
  SIRIUS_LOG_DEBUG(
    "[spill_compression] column {} compressed {:.2f}x against a {:.2f}x baseline; retraining",
    key.column,
    ratio,
    e.baseline_ratio);
  _entries.erase(it);
  return false;
}

void spill_dictionary_registry::erase_query(query_id_t query)
{
  std::lock_guard lock(_mutex);
  std::erase_if(_entries, [&](auto const& kv) { return kv.first.source.query == query; });
  std::erase_if(_jobs, [&](job const& j) { return j.key.source.query == query; });
  if (_jobs.empty() && !_running_job) { _idle.notify_all(); }
}

void spill_dictionary_registry::wait_idle()
{
  std::unique_lock lock(_mutex);
  _idle.wait(lock, [&] { return _jobs.empty() && !_running_job; });
}

std::size_t spill_dictionary_registry::size() const
{
  std::lock_guard lock(_mutex);
  return _entries.size();
}

void spill_dictionary_registry::run()
{
  std::unique_lock lock(_mutex);
  while (true) {
    _wake.wait(lock, [&] { return _stop || !_jobs.empty(); });
    if (_stop) { return; }
    auto j = std::move(_jobs.front());
    _jobs.pop_front();
    _running_job = true;
    lock.unlock();

    std::shared_ptr<simpatico::onpair_dictionary const> dict;
    try {
      nvtx_scoped_range nvtx_range{"sirius::compression::train_spill_dictionary"};
      dict = simpatico::train_onpair_dictionary(j.sample, j.config);
    } catch (std::exception const& e) {
      SIRIUS_LOG_DEBUG("[spill_compression] training a column dictionary failed: {}", e.what());
    }

    lock.lock();
    _running_job = false;
    // The column may have been erased, or erased and queued again, while this trained.
    if (auto const it = _entries.find(j.key);
        it != _entries.end() && it->second.generation == j.generation) {
      it->second.dict   = std::move(dict);
      it->second.failed = it->second.dict == nullptr;
    }
    if (_jobs.empty()) { _idle.notify_all(); }
  }
}

}  // namespace sirius::compression
