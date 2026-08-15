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

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <data/data_repository_manager_registry.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using sirius::query_id_t;
using sirius::data::data_repository_manager_registry;

namespace {

const query_id_t kQueryA = sirius::make_query_id(1);
const query_id_t kQueryB = sirius::make_query_id(2);

}  // namespace

TEST_CASE("registry: create_for_query registers a retrievable manager", "[repository_registry]")
{
  data_repository_manager_registry registry;
  REQUIRE(registry.size() == 0);

  auto manager = registry.create_for_query(kQueryA);
  REQUIRE(manager != nullptr);
  CHECK(registry.size() == 1);
  // get() returns the same instance, not a copy.
  CHECK(registry.get(kQueryA).get() == manager.get());
}

TEST_CASE("registry: get of an unregistered query returns nullptr", "[repository_registry]")
{
  data_repository_manager_registry registry;
  CHECK(registry.get(kQueryA) == nullptr);

  registry.create_for_query(kQueryA);
  CHECK(registry.get(kQueryB) == nullptr);
}

TEST_CASE("registry: duplicate query id is rejected", "[repository_registry]")
{
  data_repository_manager_registry registry;
  registry.create_for_query(kQueryA);
  // Window ids are monotonic, so a duplicate signals a lifecycle bug rather than a
  // recoverable state — it must not silently replace the in-flight query's manager.
  CHECK_THROWS_AS(registry.create_for_query(kQueryA), std::runtime_error);
  CHECK(registry.size() == 1);
}

TEST_CASE("registry: queries own independent managers", "[repository_registry]")
{
  data_repository_manager_registry registry;
  auto manager_a = registry.create_for_query(kQueryA);
  auto manager_b = registry.create_for_query(kQueryB);
  REQUIRE(manager_a.get() != manager_b.get());

  // Operator ids restart at 0 for every query, so the same {operator_id, port_id} key is
  // registered in both managers. This is exactly what the single shared manager could not do.
  manager_a->add_new_repository(0, "default", std::make_unique<cucascade::data_repository>());
  CHECK_NOTHROW(
    manager_b->add_new_repository(0, "default", std::make_unique<cucascade::data_repository>()));

  CHECK(manager_a->get_repository_shared(0, "default").get() !=
        manager_b->get_repository_shared(0, "default").get());
}

TEST_CASE("registry: erase drops only the named query", "[repository_registry]")
{
  data_repository_manager_registry registry;
  auto manager_a = registry.create_for_query(kQueryA);
  auto manager_b = registry.create_for_query(kQueryB);
  manager_a->add_new_repository(0, "default", std::make_unique<cucascade::data_repository>());
  manager_b->add_new_repository(0, "default", std::make_unique<cucascade::data_repository>());

  registry.erase(kQueryA);

  CHECK(registry.get(kQueryA) == nullptr);
  REQUIRE(registry.get(kQueryB) != nullptr);
  CHECK(registry.size() == 1);
  // The surviving query's repositories are untouched — the property the wholesale
  // clear_all_repositories() teardown could not provide.
  CHECK_NOTHROW(manager_b->get_repository_shared(0, "default"));
}

TEST_CASE("registry: erasing an unknown query is a no-op", "[repository_registry]")
{
  data_repository_manager_registry registry;
  registry.create_for_query(kQueryA);

  REQUIRE_NOTHROW(registry.erase(kQueryB));

  CHECK(registry.size() == 1);
  CHECK(registry.get(kQueryA) != nullptr);
}

TEST_CASE("registry: erase destroys an unborrowed manager and its repositories",
          "[repository_registry]")
{
  data_repository_manager_registry registry;
  std::weak_ptr<cucascade::shared_data_repository_manager> manager_observer;
  std::weak_ptr<cucascade::data_repository> repo_observer;
  {
    auto manager = registry.create_for_query(kQueryA);
    manager->add_new_repository(7, "build", std::make_unique<cucascade::data_repository>());
    manager_observer = manager;
    repo_observer    = manager->get_repository_shared(7, "build");
  }

  // No borrower left: erase() drops the map entry, which is the last reference, so the
  // manager and repository die right here — the same promptness the old clearing erase had.
  registry.erase(kQueryA);

  CHECK(manager_observer.expired());
  CHECK(repo_observer.expired());
}

TEST_CASE("registry: get_all returns every manager in ascending query order",
          "[repository_registry]")
{
  data_repository_manager_registry registry;
  // Registered out of order to prove the ordering comes from the container, not insertion.
  auto manager_b = registry.create_for_query(kQueryB);
  auto manager_a = registry.create_for_query(kQueryA);

  auto all = registry.get_all();

  REQUIRE(all.size() == 2);
  // Ascending query id keeps downgrade spill-candidate selection reproducible across runs.
  CHECK(all[0].get() == manager_a.get());
  CHECK(all[1].get() == manager_b.get());
}

TEST_CASE("registry: a borrowed manager outlives erase", "[repository_registry]")
{
  data_repository_manager_registry registry;
  // Mirrors a downgrade worker holding a manager from get_all() while the query ends.
  auto borrowed = registry.create_for_query(kQueryA);
  std::weak_ptr<cucascade::shared_data_repository_manager> observer = borrowed;

  registry.erase(kQueryA);

  CHECK(registry.get(kQueryA) == nullptr);
  // Shared ownership means the manager object survives until the borrower releases it,
  // instead of leaving the worker with a dangling reference.
  CHECK_FALSE(observer.expired());

  borrowed.reset();
  CHECK(observer.expired());
}

TEST_CASE("registry: clear drops every manager", "[repository_registry]")
{
  data_repository_manager_registry registry;
  registry.create_for_query(kQueryA);
  registry.create_for_query(kQueryB);

  registry.clear();

  CHECK(registry.size() == 0);
  CHECK(registry.get(kQueryA) == nullptr);
  CHECK(registry.get(kQueryB) == nullptr);
  CHECK(registry.get_all().empty());
  // After clear the ids are free again, so teardown followed by reuse is legal.
  CHECK_NOTHROW(registry.create_for_query(kQueryA));
}

// B9 regression: cucascade's data_repository_manager::get_repository used to
// return a reference into the map WITHOUT taking _mutex, so a concurrent
// add_new_repository / clear_all_repositories raced both the lookup and the
// returned reference (a silent use-after-free in release builds). The locked
// shared_ptr accessor closes both: the lookup holds the mutex and the returned
// shared_ptr keeps the repository alive across a concurrent clear.
TEST_CASE("manager: get_repository_shared survives concurrent add/clear churn",
          "[repository_registry]")
{
  cucascade::shared_data_repository_manager manager;
  constexpr std::size_t kOperators = 8;
  constexpr int kIterations        = 2000;
  constexpr int kReaders           = 4;

  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> successful_gets{0};
  std::atomic<std::uint64_t> null_hits{0};

  // Writer: churn the map — register every operator, then clear them all.
  std::thread writer([&] {
    for (int iter = 0; iter < kIterations; ++iter) {
      for (std::size_t op = 0; op < kOperators; ++op) {
        manager.add_new_repository(op, "default", std::make_unique<cucascade::data_repository>());
      }
      manager.clear_all_repositories();
    }
    stop.store(true);
  });

  // Readers: race lookups against the churn and dereference every hit.
  // (Catch2 v2 assertion macros are not thread-safe — workers only count.)
  std::vector<std::thread> readers;
  for (int r = 0; r < kReaders; ++r) {
    readers.emplace_back([&] {
      while (!stop.load()) {
        for (std::size_t op = 0; op < kOperators; ++op) {
          try {
            auto repo = manager.get_repository_shared(op, "default");
            if (!repo) {
              null_hits.fetch_add(1);
              continue;
            }
            // Pre-fix this dereferenced a freed map node whenever the writer's
            // clear ran between lookup and use.
            (void)repo->total_size();
            successful_gets.fetch_add(1);
          } catch (const std::out_of_range&) {
            // The operator was between clear and re-add — expected.
          }
        }
      }
    });
  }

  writer.join();
  for (auto& t : readers) {
    t.join();
  }

  CHECK(null_hits.load() == 0);
  // The interleave actually exercised the racy window.
  REQUIRE(successful_gets.load() > 0);
}

// B9 lifetime guarantee, deterministically: a repository obtained just before
// clear_all_repositories remains usable through the returned shared_ptr.
TEST_CASE("manager: repository obtained before clear_all remains usable", "[repository_registry]")
{
  cucascade::shared_data_repository_manager manager;
  manager.add_new_repository(3, "build", std::make_unique<cucascade::data_repository>());

  auto repo = manager.get_repository_shared(3, "build");
  REQUIRE(repo != nullptr);

  manager.clear_all_repositories();
  CHECK_THROWS_AS(manager.get_repository_shared(3, "build"), std::out_of_range);

  // The detached repository is still fully usable.
  CHECK(repo->total_size() == 0);
  CHECK(repo->pop_next_data_batch() == nullptr);
}
