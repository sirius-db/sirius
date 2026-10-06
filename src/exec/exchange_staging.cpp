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

#include "exec/exchange_staging.hpp"

#include "data/data_repository_manager_registry.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace sirius::exec {

namespace {

constexpr std::string_view kPort = "exchange";

/// The registered view: lists and returns the batches of every live tracked repository, which is
/// all the downgrade executor's sweep reads (convertible_data_batch_provider). It owns no batch,
/// so the mutating verbs are refused rather than forwarded.
class forwarding_repository final : public cucascade::shared_data_repository {
 public:
  using targets_t = std::vector<std::weak_ptr<cucascade::shared_data_repository>>;

  forwarding_repository(std::shared_ptr<std::mutex> mutex, std::shared_ptr<targets_t> targets)
    : _mutex(std::move(mutex)), _targets(std::move(targets))
  {
  }

  void add_data_batch(std::shared_ptr<cucascade::data_batch>, std::size_t) override
  {
    throw std::logic_error("exchange_staging: the forwarding repository owns no batches");
  }

  std::shared_ptr<cucascade::data_batch> pop_next_data_batch(std::size_t) override
  {
    return nullptr;
  }

  std::shared_ptr<cucascade::data_batch> pop_data_batch_by_id(std::uint64_t, std::size_t) override
  {
    return nullptr;
  }

  std::shared_ptr<cucascade::data_batch> get_data_batch_by_id(std::uint64_t batch_id,
                                                              std::size_t partition) const override
  {
    for (auto const& repository : live()) {
      if (partition >= repository->num_partitions()) { continue; }
      if (auto batch = repository->get_data_batch_by_id(batch_id, partition)) { return batch; }
    }
    return nullptr;
  }

  std::vector<std::uint64_t> get_batch_ids(std::size_t partition) const override
  {
    std::vector<std::uint64_t> ids;
    for (auto const& repository : live()) {
      if (partition >= repository->num_partitions()) { continue; }
      auto const more = repository->get_batch_ids(partition);
      ids.insert(ids.end(), more.begin(), more.end());
    }
    return ids;
  }

 private:
  /// The tracked repositories still alive, pruning the rest.
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> live() const
  {
    std::vector<std::shared_ptr<cucascade::shared_data_repository>> alive;
    std::lock_guard lock(*_mutex);
    std::erase_if(*_targets, [&](auto const& weak) {
      auto repository = weak.lock();
      if (!repository) { return true; }
      alive.push_back(std::move(repository));
      return false;
    });
    return alive;
  }

  std::shared_ptr<std::mutex> _mutex;
  std::shared_ptr<targets_t> _targets;
};

}  // namespace

exchange_staging::exchange_staging(sirius::data::data_repository_manager_registry& registry)
  : _targets(std::make_shared<targets>())
{
  // Share the targets' mutex and vector with the registered view through aliasing pointers, so
  // the view stays valid after this object is gone (the registry may outlive it).
  auto mutex = std::shared_ptr<std::mutex>(_targets, &_targets->mutex);
  auto repositories =
    std::shared_ptr<forwarding_repository::targets_t>(_targets, &_targets->repositories);
  registry.create_for_query(query_id)->add_new_repository(
    0, kPort, std::make_unique<forwarding_repository>(std::move(mutex), std::move(repositories)));
}

void exchange_staging::track(std::shared_ptr<cucascade::shared_data_repository> const& repository)
{
  if (!repository) { return; }
  std::lock_guard lock(_targets->mutex);
  _targets->repositories.push_back(repository);
}

std::size_t exchange_staging::tracked() const
{
  std::lock_guard lock(_targets->mutex);
  return static_cast<std::size_t>(std::count_if(_targets->repositories.begin(),
                                                _targets->repositories.end(),
                                                [](auto const& w) { return !w.expired(); }));
}

}  // namespace sirius::exec
