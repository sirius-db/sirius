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

#include <memory>
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

TEST_CASE("registry: exchange ownership survives query cleanup and fences a spill snapshot",
          "[repository_registry][exchange_memory]")
{
  data_repository_manager_registry registry;
  registry.create_for_query(kQueryA);
  auto repository = std::make_shared<cucascade::shared_data_repository>();
  std::weak_ptr<cucascade::shared_data_repository> lifetime = repository;
  registry.register_exchange(repository);
  registry.erase(kQueryA);
  auto snapshot = registry.get_exchange_repositories();
  REQUIRE(snapshot.size() == 1);
  CHECK(snapshot.front() == repository);
  repository.reset();
  CHECK_FALSE(lifetime.expired());
  snapshot.clear();
  CHECK(lifetime.expired());
  CHECK(registry.get_exchange_repositories().empty());
}

TEST_CASE("registry: retiring one exchange leaves other query ownership intact",
          "[repository_registry][exchange_memory]")
{
  data_repository_manager_registry registry;
  auto first  = std::make_shared<cucascade::shared_data_repository>();
  auto second = std::make_shared<cucascade::shared_data_repository>();
  registry.register_exchange(first);
  registry.register_exchange(second);
  first.reset();
  auto snapshot = registry.get_exchange_repositories();
  REQUIRE(snapshot.size() == 1);
  CHECK(snapshot.front() == second);
  registry.clear();
  CHECK(registry.get_exchange_repositories().empty());
  CHECK(snapshot.front() == second);
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

  CHECK(manager_a->get_repository(0, "default").get() !=
        manager_b->get_repository(0, "default").get());
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
  CHECK_NOTHROW(manager_b->get_repository(0, "default"));
}

TEST_CASE("registry: erasing an unknown query is a no-op", "[repository_registry]")
{
  data_repository_manager_registry registry;
  registry.create_for_query(kQueryA);

  auto leaked = registry.erase(kQueryB);

  CHECK(leaked.empty());
  CHECK(registry.size() == 1);
  CHECK(registry.get(kQueryA) != nullptr);
}

TEST_CASE("registry: erase reports repositories that still held batches", "[repository_registry]")
{
  data_repository_manager_registry registry;
  auto manager = registry.create_for_query(kQueryA);
  manager->add_new_repository(7, "build", std::make_unique<cucascade::data_repository>());

  // Nothing was pushed, so a clean query reports no leaks.
  auto leaked = registry.erase(kQueryA);
  CHECK(leaked.empty());
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
