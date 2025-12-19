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

#include "catch.hpp"
#include "data/common.hpp"
#include "data/data_batch.hpp"
#include "data/data_batch_view.hpp"
#include "data/data_repository.hpp"
#include "data/data_repository_manager.hpp"
#include "memory/null_device_memory_resource.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace cucascade;

// Mock memory_space for testing - provides a simple memory_space without real allocators
class mock_memory_space : public memory::memory_space {
 public:
  mock_memory_space(memory::Tier tier, size_t device_id = 0)
    : memory::memory_space(tier,
                           static_cast<int>(device_id),
                           1024 * 1024 * 1024,                      // memory_limit
                           (1024ULL * 1024ULL * 1024ULL) * 8 / 10,  // start_downgrading_threshold
                           (1024ULL * 1024ULL * 1024ULL) / 2,       // stop_downgrading_threshold
                           1024 * 1024 * 1024,                      // capacity
                           std::make_unique<memory::null_device_memory_resource>())
  {
  }
};

// Helper base class to hold memory_space - initialized before idata_representation
struct mock_memory_space_holder {
  std::shared_ptr<mock_memory_space> space;

  mock_memory_space_holder(memory::Tier tier, size_t device_id)
    : space(std::make_shared<mock_memory_space>(tier, device_id))
  {
  }
};

// Mock idata_representation for testing
// Inherits from mock_memory_space_holder first to ensure it's constructed before
// idata_representation
class mock_data_representation : private mock_memory_space_holder, public idata_representation {
 public:
  explicit mock_data_representation(memory::Tier tier, size_t size = 1024, size_t device_id = 0)
    : mock_memory_space_holder(tier, device_id)  // Construct holder first
      ,
      idata_representation(*space)  // Pass reference to base class
      ,
      _size(size)
  {
  }

  std::size_t get_size_in_bytes() const override { return _size; }

  std::unique_ptr<idata_representation> convert_to_memory_space(
    const memory::memory_space* target_memory_space,
    rmm::cuda_stream_view stream = rmm::cuda_stream_default) override
  {
    // Empty implementation for testing
    return nullptr;
  }

 private:
  size_t _size;
};

// =============================================================================
// Basic Construction and Initialization Tests
// =============================================================================

// Test basic construction
TEST_CASE("data_repository_manager Construction", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Manager should be empty initially
  // Accessing non-existent repository should throw
  REQUIRE_THROWS_AS(manager.get_repository(0, "default"), std::out_of_range);
}

// =============================================================================
// Repository Management Tests
// =============================================================================

// Test adding a single repository
TEST_CASE("data_repository_manager Add Single Repository", "[data_repository_manager]")
{
  data_repository_manager manager;

  size_t operator_id = 1;
  auto repository    = std::make_unique<idata_repository>();
  manager.add_new_repository(operator_id, "default", std::move(repository));

  // Repository should be accessible
  auto& repo = manager.get_repository(operator_id, "default");
  REQUIRE(repo != nullptr);
}

// Test adding multiple repositories
TEST_CASE("data_repository_manager Add Multiple Repositories", "[data_repository_manager]")
{
  data_repository_manager manager;

  constexpr int num_operators = 10;

  // Add repositories for multiple operators
  for (size_t i = 0; i < num_operators; ++i) {
    auto repository = std::make_unique<idata_repository>();
    manager.add_new_repository(i, "default", std::move(repository));
  }

  // All repositories should be accessible
  for (size_t i = 0; i < num_operators; ++i) {
    auto& repo = manager.get_repository(i, "default");
    REQUIRE(repo != nullptr);
  }
}

// Test replacing an existing repository
TEST_CASE("data_repository_manager Replace Repository", "[data_repository_manager]")
{
  data_repository_manager manager;

  size_t operator_id = 5;

  // Add first repository
  auto repository1 = std::make_unique<idata_repository>();
  auto* repo1_ptr  = repository1.get();
  manager.add_new_repository(operator_id, "default", std::move(repository1));

  REQUIRE(manager.get_repository(operator_id, "default").get() == repo1_ptr);

  // Replace with second repository
  auto repository2 = std::make_unique<idata_repository>();
  auto* repo2_ptr  = repository2.get();
  manager.add_new_repository(operator_id, "default", std::move(repository2));

  // Should now reference the new repository
  REQUIRE(manager.get_repository(operator_id, "default").get() == repo2_ptr);
  REQUIRE(manager.get_repository(operator_id, "default").get() != repo1_ptr);
}

// Test accessing non-existent repository
TEST_CASE("data_repository_manager Access Non-Existent Repository", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add some repositories
  manager.add_new_repository(1, "default", std::make_unique<idata_repository>());
  manager.add_new_repository(2, "default", std::make_unique<idata_repository>());

  // Accessing non-existent repositories should throw
  REQUIRE_THROWS_AS(manager.get_repository(0, "default"), std::out_of_range);
  REQUIRE_THROWS_AS(manager.get_repository(3, "default"), std::out_of_range);
  REQUIRE_THROWS_AS(manager.get_repository(999, "default"), std::out_of_range);
}

// =============================================================================
// Batch ID Generation Tests
// =============================================================================

// Test unique batch ID generation
TEST_CASE("data_repository_manager Unique Batch IDs", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Generate multiple IDs
  std::vector<uint64_t> ids;
  for (int i = 0; i < 100; ++i) {
    ids.push_back(manager.get_next_data_batch_id());
  }

  // All IDs should be unique
  std::sort(ids.begin(), ids.end());
  auto last = std::unique(ids.begin(), ids.end());
  REQUIRE(last == ids.end());
}

// Test batch ID monotonic increment
TEST_CASE("data_repository_manager Monotonic Batch IDs", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Generate IDs and verify they increment
  uint64_t prev_id = manager.get_next_data_batch_id();
  for (int i = 0; i < 100; ++i) {
    uint64_t next_id = manager.get_next_data_batch_id();
    REQUIRE(next_id > prev_id);
    prev_id = next_id;
  }
}

// Test batch ID starts at zero
TEST_CASE("data_repository_manager Batch ID Initial Value", "[data_repository_manager]")
{
  data_repository_manager manager;

  uint64_t first_id = manager.get_next_data_batch_id();
  REQUIRE(first_id == 0);
}

// =============================================================================
// Data Batch Management Tests
// =============================================================================

// Test adding data batch to single operator
TEST_CASE("data_repository_manager Add Data Batch Single Pipeline", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  // Create and add batch
  auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  uint64_t batch_id = manager.get_next_data_batch_id();
  auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));

  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};
  manager.add_new_data_batch(std::move(batch), operator_ports);

  // Repository should have the batch view
  auto& repo = manager.get_repository(operator_id, "default");
  auto view  = repo->pull_data_batch_view();
  REQUIRE(view != nullptr);
}

// Test adding data batch to multiple operators
TEST_CASE("data_repository_manager Add Data Batch Multiple Pipelines", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add multiple repositories
  std::vector<size_t> operator_ids = {1, 2, 3};
  for (size_t id : operator_ids) {
    manager.add_new_repository(id, "default", std::make_unique<idata_repository>());
  }

  // Create and add batch to all operators
  auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  uint64_t batch_id = manager.get_next_data_batch_id();
  auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));

  std::vector<std::pair<size_t, std::string_view>> operator_ports;
  for (size_t id : operator_ids) {
    operator_ports.push_back({id, "default"});
  }
  manager.add_new_data_batch(std::move(batch), operator_ports);

  // All repositories should have a view
  for (size_t id : operator_ids) {
    auto& repo = manager.get_repository(id, "default");
    auto view  = repo->pull_data_batch_view();
    REQUIRE(view != nullptr);
  }
}

// Test adding data batch with empty operator list
TEST_CASE("data_repository_manager Add Data Batch No Pipelines", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Create and add batch with empty operator list
  auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  uint64_t batch_id = manager.get_next_data_batch_id();
  auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));

  std::vector<std::pair<size_t, std::string_view>> empty_operator_ports;
  manager.add_new_data_batch(std::move(batch), empty_operator_ports);

  // Batch is stored but no views are created
  // This should not crash
}

// Test deleting data batch
TEST_CASE("data_repository_manager Delete Data Batch", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  // Create and add batch
  auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  uint64_t batch_id = manager.get_next_data_batch_id();
  auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));

  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};
  manager.add_new_data_batch(std::move(batch), operator_ports);

  // Pull the view from repository first (views hold pointers to the batch)
  auto& repo = manager.get_repository(operator_id, "default");
  auto view  = repo->pull_data_batch_view();
  REQUIRE(view != nullptr);

  // Now delete the batch - the view will be destroyed, triggering batch deletion
  view.reset();

  // Batch should be automatically deleted when last view is destroyed
}

// Note: Test for deleting non-existent batch removed because delete_data_batch
// is now private and can only be called by data_batch_view destructor

// Test adding multiple batches
TEST_CASE("data_repository_manager Add Multiple Batches", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  constexpr int num_batches                                       = 10;
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Add multiple batches
  for (int i = 0; i < num_batches; ++i) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    manager.add_new_data_batch(std::move(batch), operator_ports);
  }

  // Repository should have all batch views
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (auto view = repo->pull_data_batch_view()) {
    ++count;
  }
  REQUIRE(count == num_batches);
}

// =============================================================================
// Thread-Safety Tests
// =============================================================================

// Test concurrent batch ID generation
TEST_CASE("data_repository_manager Thread-Safe Batch ID Generation", "[data_repository_manager]")
{
  data_repository_manager manager;

  constexpr int num_threads    = 10;
  constexpr int ids_per_thread = 100;

  std::vector<std::thread> threads;
  std::vector<std::vector<uint64_t>> thread_ids(num_threads);

  // Launch threads to generate IDs
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < ids_per_thread; ++j) {
        thread_ids[i].push_back(manager.get_next_data_batch_id());
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Collect all IDs
  std::vector<uint64_t> all_ids;
  for (const auto& ids : thread_ids) {
    all_ids.insert(all_ids.end(), ids.begin(), ids.end());
  }

  // All IDs should be unique
  std::sort(all_ids.begin(), all_ids.end());
  auto last = std::unique(all_ids.begin(), all_ids.end());
  REQUIRE(last == all_ids.end());
  REQUIRE(all_ids.size() == num_threads * ids_per_thread);
}

// Test concurrent repository addition
TEST_CASE("data_repository_manager Thread-Safe Add Repository", "[data_repository_manager]")
{
  data_repository_manager manager;

  constexpr int num_threads = 10;
  std::vector<std::thread> threads;

  // Launch threads to add repositories
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      auto repository = std::make_unique<idata_repository>();
      manager.add_new_repository(i, "default", std::move(repository));
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // All repositories should be accessible
  for (int i = 0; i < num_threads; ++i) {
    auto& repo = manager.get_repository(i, "default");
    REQUIRE(repo != nullptr);
  }
}

// Test concurrent batch addition
TEST_CASE("data_repository_manager Thread-Safe Add Batch", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  constexpr int num_threads        = 10;
  constexpr int batches_per_thread = 50;

  std::vector<std::thread> threads;
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Launch threads to add batches
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < batches_per_thread; ++j) {
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), operator_ports);
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Repository should have all batch views
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (auto view = repo->pull_data_batch_view()) {
    ++count;
  }
  REQUIRE(count == num_threads * batches_per_thread);
}

// Test concurrent batch deletion
TEST_CASE("data_repository_manager Thread-Safe Delete Batch", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  constexpr int num_batches                                       = 100;
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};
  std::vector<uint64_t> batch_ids;

  // Add batches
  for (int i = 0; i < num_batches; ++i) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    batch_ids.push_back(batch_id);
    auto batch = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    manager.add_new_data_batch(std::move(batch), operator_ports);
  }

  // Pull all views from repository to allow safe deletion
  auto& repo = manager.get_repository(operator_id, "default");
  std::vector<std::unique_ptr<data_batch_view>> views;
  while (auto view = repo->pull_data_batch_view()) {
    views.push_back(std::move(view));
  }
  REQUIRE(views.size() == num_batches);

  constexpr int num_threads = 10;
  std::vector<std::thread> threads;

  // Launch threads to destroy views (which triggers batch cleanup)
  int views_per_thread = num_batches / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < views_per_thread; ++j) {
        int idx = i * views_per_thread + j;
        views[idx].reset();  // Destroy view, auto-deletes batch when ref count hits 0
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // All views destroyed - batches are automatically deleted
}

// Test concurrent mixed operations
TEST_CASE("data_repository_manager Thread-Safe Mixed Operations", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add initial repositories
  for (int i = 0; i < 5; ++i) {
    manager.add_new_repository(i, "default", std::make_unique<idata_repository>());
  }

  constexpr int num_threads           = 10;
  constexpr int operations_per_thread = 50;

  std::vector<std::thread> threads;
  std::atomic<int> batch_count{0};
  std::mutex view_mutex;
  std::vector<std::unique_ptr<data_batch_view>> all_views;

  // Launch threads doing mixed operations
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < operations_per_thread; ++j) {
        // Generate batch ID
        uint64_t batch_id = manager.get_next_data_batch_id();

        // Add batch to random operator
        size_t operator_id = (i + j) % 5;
        auto data          = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto batch         = std::make_unique<data_batch>(batch_id, manager, std::move(data));
        std::vector<std::pair<size_t, std::string_view>> operator_ports = {
          {operator_id, "default"}};
        manager.add_new_data_batch(std::move(batch), operator_ports);

        ++batch_count;

        // Occasionally pull and store a view (to test concurrent pull operations)
        if (j % 10 == 0) {
          auto& repo = manager.get_repository(operator_id, "default");
          if (auto view = repo->pull_data_batch_view()) {
            std::lock_guard<std::mutex> lock(view_mutex);
            all_views.push_back(std::move(view));
          }
        }
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Should complete without crashes
  REQUIRE(batch_count == num_threads * operations_per_thread);

  // Clean up views to trigger batch deletion
  all_views.clear();
}

// Test concurrent add and delete via data_batch_view destructor
TEST_CASE("data_repository_manager Concurrent Add and Delete via View Destructor",
          "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repositories for multiple operators
  constexpr int num_operators = 3;
  for (int i = 0; i < num_operators; ++i) {
    manager.add_new_repository(i, "default", std::make_unique<idata_repository>());
  }

  constexpr int num_adder_threads   = 5;
  constexpr int num_deleter_threads = 5;
  constexpr int batches_per_adder   = 100;

  std::vector<std::thread> threads;
  std::atomic<int> batches_added{0};
  std::atomic<int> batches_deleted{0};
  std::atomic<bool> keep_adding{true};

  // Launch adder threads - continuously add batches to repositories
  for (int i = 0; i < num_adder_threads; ++i) {
    threads.emplace_back([&, i]() {
      for (int j = 0; j < batches_per_adder; ++j) {
        // Generate batch ID
        uint64_t batch_id = manager.get_next_data_batch_id();

        // Add batch to one or more operators
        size_t operator_id = (i + j) % num_operators;
        auto data          = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto batch         = std::make_unique<data_batch>(batch_id, manager, std::move(data));
        std::vector<std::pair<size_t, std::string_view>> operator_ports = {
          {operator_id, "default"}};
        manager.add_new_data_batch(std::move(batch), operator_ports);

        ++batches_added;

        // Small delay to allow deleters to work
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });
  }

  // Launch deleter threads - pull views and destroy them (triggers batch deletion)
  for (int i = 0; i < num_deleter_threads; ++i) {
    threads.emplace_back([&, i]() {
      size_t operator_id = i % num_operators;
      auto& repo         = manager.get_repository(operator_id, "default");

      // Keep pulling and destroying views while adders are working
      while (keep_adding.load() || repo->pull_data_batch_view() != nullptr) {
        auto view = repo->pull_data_batch_view();
        if (view) {
          ++batches_deleted;
          // View destructor will be called here, triggering batch deletion
          // when it's the last view
          view.reset();
        } else {
          // Repository temporarily empty, yield to adders
          std::this_thread::yield();
        }
      }
    });
  }

  // Wait for adder threads to complete
  for (int i = 0; i < num_adder_threads; ++i) {
    threads[i].join();
  }

  // Signal deleters that adding is done
  keep_adding.store(false);

  // Wait for deleter threads to complete
  for (int i = num_adder_threads; i < threads.size(); ++i) {
    threads[i].join();
  }

  // Verify all batches were added
  REQUIRE(batches_added == num_adder_threads * batches_per_adder);

  // Verify all batches were deleted (all views destroyed)
  REQUIRE(batches_deleted == num_adder_threads * batches_per_adder);

  // All repositories should be empty
  for (int i = 0; i < num_operators; ++i) {
    auto& repo = manager.get_repository(i, "default");
    REQUIRE(repo->pull_data_batch_view() == nullptr);
  }
}

// Test high-contention concurrent add and delete via view destructor
TEST_CASE("data_repository_manager High Contention Add Delete via View Destructor",
          "[data_repository_manager]")
{
  data_repository_manager manager;

  // Single operator for maximum contention
  size_t operator_id = 0;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  constexpr int num_threads           = 20;
  constexpr int operations_per_thread = 50;

  std::vector<std::thread> threads;
  std::atomic<int> total_added{0};
  std::atomic<int> total_deleted{0};

  // Launch threads doing both add and delete operations
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      auto& repo = manager.get_repository(operator_id, "default");
      std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

      for (int j = 0; j < operations_per_thread; ++j) {
        // Add a batch
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 512);
        auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), operator_ports);
        ++total_added;

        // Immediately try to pull and delete a batch (might be ours or someone else's)
        auto view = repo->pull_data_batch_view();
        if (view) {
          ++total_deleted;
          view.reset();  // Destructor triggers deletion when last view
        }
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify counts
  REQUIRE(total_added == num_threads * operations_per_thread);

  // Clean up remaining batches
  auto& repo = manager.get_repository(operator_id, "default");
  while (auto view = repo->pull_data_batch_view()) {
    ++total_deleted;
  }

  // All batches should have been processed
  REQUIRE(total_deleted == total_added);
}

// Test concurrent add and delete with multiple views per batch
TEST_CASE("data_repository_manager Concurrent Add Delete Multiple Views per Batch",
          "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repositories
  constexpr int num_operators = 5;
  for (int i = 0; i < num_operators; ++i) {
    manager.add_new_repository(i, "default", std::make_unique<idata_repository>());
  }

  constexpr int num_batches = 50;
  std::atomic<int> views_deleted{0};

  // Add batches to ALL operators (each batch will have multiple views)
  std::vector<std::pair<size_t, std::string_view>> all_operator_ports;
  for (int i = 0; i < num_operators; ++i) {
    all_operator_ports.push_back({i, "default"});
  }

  for (int i = 0; i < num_batches; ++i) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    manager.add_new_data_batch(std::move(batch), all_operator_ports);
  }

  // Now concurrently delete views from different operators
  // The batch should only be deleted when ALL views are destroyed
  std::vector<std::thread> threads;

  for (int i = 0; i < num_operators; ++i) {
    threads.emplace_back([&, i]() {
      auto& repo = manager.get_repository(i, "default");

      // Pull and destroy all views from this operator
      while (auto view = repo->pull_data_batch_view()) {
        ++views_deleted;
        view.reset();  // Destructor called here
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Each batch was added to all operators, so we should have deleted
  // num_batches * num_operators views
  REQUIRE(views_deleted == num_batches * num_operators);

  // All repositories should be empty
  for (int i = 0; i < num_operators; ++i) {
    auto& repo = manager.get_repository(i, "default");
    REQUIRE(repo->pull_data_batch_view() == nullptr);
  }
}

// =============================================================================
// Integration Tests
// =============================================================================

// Test full workflow with multiple operators and batches
TEST_CASE("data_repository_manager Full Workflow", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Setup: Create 3 operators
  std::vector<size_t> operator_ids = {0, 1, 2};
  for (size_t id : operator_ids) {
    manager.add_new_repository(id, "default", std::make_unique<idata_repository>());
  }

  // Add batches to different operator combinations
  std::vector<uint64_t> batch_ids;

  // Batch 0: All operators
  {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    batch_ids.push_back(batch_id);
    auto batch = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    std::vector<std::pair<size_t, std::string_view>> all_ports;
    for (size_t id : operator_ids) {
      all_ports.push_back({id, "default"});
    }
    manager.add_new_data_batch(std::move(batch), all_ports);
  }

  // Batch 1: Operator 0 only
  {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
    uint64_t batch_id = manager.get_next_data_batch_id();
    batch_ids.push_back(batch_id);
    auto batch = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    std::vector<std::pair<size_t, std::string_view>> p0 = {{0, "default"}};
    manager.add_new_data_batch(std::move(batch), p0);
  }

  // Batch 2: Operators 1 and 2
  {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 4096);
    uint64_t batch_id = manager.get_next_data_batch_id();
    batch_ids.push_back(batch_id);
    auto batch = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    std::vector<std::pair<size_t, std::string_view>> p12 = {{1, "default"}, {2, "default"}};
    manager.add_new_data_batch(std::move(batch), p12);
  }

  // Verify: Operator 0 should have 2 batches (batch 0 and 1)
  {
    auto& repo = manager.get_repository(0, "default");
    int count  = 0;
    while (auto view = repo->pull_data_batch_view()) {
      ++count;
    }
    REQUIRE(count == 2);
  }

  // Verify: Operator 1 should have 2 batches (batch 0 and 2)
  {
    auto& repo = manager.get_repository(1, "default");
    int count  = 0;
    while (auto view = repo->pull_data_batch_view()) {
      ++count;
    }
    REQUIRE(count == 2);
  }

  // Verify: Operator 2 should have 2 batches (batch 0 and 2)
  {
    auto& repo = manager.get_repository(2, "default");
    int count  = 0;
    while (auto view = repo->pull_data_batch_view()) {
      ++count;
    }
    REQUIRE(count == 2);
  }

  // Batches are automatically deleted when all views go out of scope
}

// Test replacing repository with data
TEST_CASE("data_repository_manager Replace Repository With Data", "[data_repository_manager]")
{
  data_repository_manager manager;

  size_t operator_id = 1;

  // Add first repository
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  // Add batch to first repository
  auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
  uint64_t batch_id = manager.get_next_data_batch_id();
  auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};
  manager.add_new_data_batch(std::move(batch), operator_ports);

  // Replace repository
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  // New repository should be empty
  auto& new_repo = manager.get_repository(operator_id, "default");
  auto view      = new_repo->pull_data_batch_view();
  REQUIRE(view == nullptr);
}

// Test large number of operators
TEST_CASE("data_repository_manager Large Number of Pipelines", "[data_repository_manager]")
{
  data_repository_manager manager;

  constexpr int num_operators = 1000;

  // Add many operators
  for (int i = 0; i < num_operators; ++i) {
    manager.add_new_repository(i, "default", std::make_unique<idata_repository>());
  }

  // All operators should be accessible
  for (int i = 0; i < num_operators; ++i) {
    auto& repo = manager.get_repository(i, "default");
    REQUIRE(repo != nullptr);
  }
}

// Test large number of batches
TEST_CASE("data_repository_manager Large Number of Batches", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  constexpr int num_batches                                       = 1000;
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Add many batches
  for (int i = 0; i < num_batches; ++i) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    manager.add_new_data_batch(std::move(batch), operator_ports);
  }

  // Repository should have all batches
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (auto view = repo->pull_data_batch_view()) {
    ++count;
  }
  REQUIRE(count == num_batches);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

// Test with operator ID zero
TEST_CASE("data_repository_manager Pipeline ID Zero", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Operator ID 0 should work like any other ID
  manager.add_new_repository(0, "default", std::make_unique<idata_repository>());

  auto& repo = manager.get_repository(0, "default");
  REQUIRE(repo != nullptr);
}

// Test with large operator IDs
TEST_CASE("data_repository_manager Large Pipeline IDs", "[data_repository_manager]")
{
  data_repository_manager manager;

  std::vector<size_t> large_ids = {1000, 10000, 100000, SIZE_MAX - 1, SIZE_MAX};

  // Add repositories with large IDs
  for (size_t id : large_ids) {
    manager.add_new_repository(id, "default", std::make_unique<idata_repository>());
  }

  // All should be accessible
  for (size_t id : large_ids) {
    auto& repo = manager.get_repository(id, "default");
    REQUIRE(repo != nullptr);
  }
}

// Test batch with different data sizes
TEST_CASE("data_repository_manager Batches With Different Sizes", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  std::vector<size_t> sizes = {1, 1024, 1024 * 1024, 1024 * 1024 * 10};
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Add batches with different sizes
  for (size_t size : sizes) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, size);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    manager.add_new_data_batch(std::move(batch), operator_ports);
  }

  // All batches should be accessible
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (auto view = repo->pull_data_batch_view()) {
    ++count;
  }
  REQUIRE(count == sizes.size());
}

// Test batch with different memory tiers
TEST_CASE("data_repository_manager Batches With Different Tiers", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  std::vector<memory::Tier> tiers = {memory::Tier::GPU, memory::Tier::HOST, memory::Tier::DISK};
  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};

  // Add batches with different tiers
  for (memory::Tier tier : tiers) {
    auto data         = std::make_unique<mock_data_representation>(tier, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    manager.add_new_data_batch(std::move(batch), operator_ports);
  }

  // All batches should be accessible
  auto& repo = manager.get_repository(operator_id, "default");
  int count  = 0;
  while (auto view = repo->pull_data_batch_view()) {
    ++count;
  }
  REQUIRE(count == tiers.size());
}

// Test rapid add and delete cycles
TEST_CASE("data_repository_manager Rapid Add Delete Cycles", "[data_repository_manager]")
{
  data_repository_manager manager;

  // Add repository
  size_t operator_id = 1;
  manager.add_new_repository(operator_id, "default", std::make_unique<idata_repository>());

  std::vector<std::pair<size_t, std::string_view>> operator_ports = {{operator_id, "default"}};
  auto& repo = manager.get_repository(operator_id, "default");

  // Perform many cycles of add and delete
  for (int cycle = 0; cycle < 100; ++cycle) {
    auto data         = std::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch        = std::make_unique<data_batch>(batch_id, manager, std::move(data));
    manager.add_new_data_batch(std::move(batch), operator_ports);

    // Pull and destroy the view, which triggers batch deletion
    auto view = repo->pull_data_batch_view();
    REQUIRE(view != nullptr);
    view.reset();  // Explicitly destroy view, auto-deletes batch
  }

  // Repository should be empty
  REQUIRE(repo->pull_data_batch_view() == nullptr);
}
