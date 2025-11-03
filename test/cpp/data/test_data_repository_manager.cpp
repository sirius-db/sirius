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
#include <thread>
#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include "data/data_repository_manager.hpp"
#include "data/data_repository.hpp"
#include "data/data_batch.hpp"
#include "data/data_batch_view.hpp"
#include "data/common.hpp"
#include "memory/null_device_memory_resource.hpp"

using namespace sirius;

// Mock memory_space for testing - provides a simple memory_space without real allocators
class mock_memory_space : public memory::memory_space {
public:
    mock_memory_space(memory::Tier tier, size_t device_id = 0)
        : memory::memory_space(tier, device_id, 1024 * 1024 * 1024, create_null_allocators()) {}
    
private:
    static std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> create_null_allocators() {
        std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators;
        allocators.push_back(std::make_unique<memory::null_device_memory_resource>());
        return allocators;
    }
};

// Helper base class to hold memory_space - initialized before idata_representation
struct mock_memory_space_holder {
    std::shared_ptr<mock_memory_space> space;
    
    mock_memory_space_holder(memory::Tier tier, size_t device_id)
        : space(std::make_shared<mock_memory_space>(tier, device_id)) {}
};

// Mock idata_representation for testing
// Inherits from mock_memory_space_holder first to ensure it's constructed before idata_representation
class mock_data_representation : private mock_memory_space_holder, public idata_representation {
public:
    explicit mock_data_representation(memory::Tier tier, size_t size = 1024, size_t device_id = 0)
        : mock_memory_space_holder(tier, device_id)  // Construct holder first
        , idata_representation(*space)                // Pass reference to base class
        , _size(size) {}
    
    std::size_t get_size_in_bytes() const override {
        return _size;
    }
    
    sirius::unique_ptr<idata_representation> convert_to_memory_space(memory::memory_space& target_memory_space, rmm::cuda_stream_view stream = rmm::cuda_stream_default) override {
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
TEST_CASE("data_repository_manager Construction", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Manager should be empty initially
    // Accessing non-existent repository should throw
    REQUIRE_THROWS_AS(manager.get_repository(0), std::out_of_range);
}

// =============================================================================
// Repository Management Tests
// =============================================================================

// Test adding a single repository
TEST_CASE("data_repository_manager Add Single Repository", "[data_repository_manager]") {
    data_repository_manager manager;
    
    size_t pipeline_id = 1;
    auto repository = sirius::make_unique<idata_repository>();
    manager.add_new_repository(pipeline_id, std::move(repository));
    
    // Repository should be accessible
    auto& repo = manager.get_repository(pipeline_id);
    REQUIRE(repo != nullptr);
}

// Test adding multiple repositories
TEST_CASE("data_repository_manager Add Multiple Repositories", "[data_repository_manager]") {
    data_repository_manager manager;
    
    constexpr int num_pipelines = 10;
    
    // Add repositories for multiple pipelines
    for (size_t i = 0; i < num_pipelines; ++i) {
        auto repository = sirius::make_unique<idata_repository>();
        manager.add_new_repository(i, std::move(repository));
    }
    
    // All repositories should be accessible
    for (size_t i = 0; i < num_pipelines; ++i) {
        auto& repo = manager.get_repository(i);
        REQUIRE(repo != nullptr);
    }
}

// Test replacing an existing repository
TEST_CASE("data_repository_manager Replace Repository", "[data_repository_manager]") {
    data_repository_manager manager;
    
    size_t pipeline_id = 5;
    
    // Add first repository
    auto repository1 = sirius::make_unique<idata_repository>();
    auto* repo1_ptr = repository1.get();
    manager.add_new_repository(pipeline_id, std::move(repository1));
    
    REQUIRE(manager.get_repository(pipeline_id).get() == repo1_ptr);
    
    // Replace with second repository
    auto repository2 = sirius::make_unique<idata_repository>();
    auto* repo2_ptr = repository2.get();
    manager.add_new_repository(pipeline_id, std::move(repository2));
    
    // Should now reference the new repository
    REQUIRE(manager.get_repository(pipeline_id).get() == repo2_ptr);
    REQUIRE(manager.get_repository(pipeline_id).get() != repo1_ptr);
}

// Test accessing non-existent repository
TEST_CASE("data_repository_manager Access Non-Existent Repository", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add some repositories
    manager.add_new_repository(1, sirius::make_unique<idata_repository>());
    manager.add_new_repository(2, sirius::make_unique<idata_repository>());
    
    // Accessing non-existent repositories should throw
    REQUIRE_THROWS_AS(manager.get_repository(0), std::out_of_range);
    REQUIRE_THROWS_AS(manager.get_repository(3), std::out_of_range);
    REQUIRE_THROWS_AS(manager.get_repository(999), std::out_of_range);
}

// =============================================================================
// Batch ID Generation Tests
// =============================================================================

// Test unique batch ID generation
TEST_CASE("data_repository_manager Unique Batch IDs", "[data_repository_manager]") {
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
TEST_CASE("data_repository_manager Monotonic Batch IDs", "[data_repository_manager]") {
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
TEST_CASE("data_repository_manager Batch ID Initial Value", "[data_repository_manager]") {
    data_repository_manager manager;
    
    uint64_t first_id = manager.get_next_data_batch_id();
    REQUIRE(first_id == 0);
}

// =============================================================================
// Data Batch Management Tests
// =============================================================================

// Test adding data batch to single pipeline
TEST_CASE("data_repository_manager Add Data Batch Single Pipeline", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repository
    size_t pipeline_id = 1;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    // Create and add batch
    auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
    
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    manager.add_new_data_batch(std::move(batch), pipeline_ids);
    
    // Repository should have the batch view
    auto& repo = manager.get_repository(pipeline_id);
    auto view = repo->pull_data_batch_view();
    REQUIRE(view != nullptr);
}

// Test adding data batch to multiple pipelines
TEST_CASE("data_repository_manager Add Data Batch Multiple Pipelines", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add multiple repositories
    sirius::vector<size_t> pipeline_ids = {1, 2, 3};
    for (size_t id : pipeline_ids) {
        manager.add_new_repository(id, sirius::make_unique<idata_repository>());
    }
    
    // Create and add batch to all pipelines
    auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
    
    manager.add_new_data_batch(std::move(batch), pipeline_ids);
    
    // All repositories should have a view
    for (size_t id : pipeline_ids) {
        auto& repo = manager.get_repository(id);
        auto view = repo->pull_data_batch_view();
        REQUIRE(view != nullptr);
    }
}

// Test adding data batch with empty pipeline list
TEST_CASE("data_repository_manager Add Data Batch No Pipelines", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Create and add batch with empty pipeline list
    auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
    
    sirius::vector<size_t> empty_pipeline_ids;
    manager.add_new_data_batch(std::move(batch), empty_pipeline_ids);
    
    // Batch is stored but no views are created
    // This should not crash
}

// Test deleting data batch
TEST_CASE("data_repository_manager Delete Data Batch", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repository
    size_t pipeline_id = 1;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    // Create and add batch
    auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
    
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    manager.add_new_data_batch(std::move(batch), pipeline_ids);
    
    // Pull the view from repository first (views hold pointers to the batch)
    auto& repo = manager.get_repository(pipeline_id);
    auto view = repo->pull_data_batch_view();
    REQUIRE(view != nullptr);
    
    // Now delete the batch - the view will be destroyed, triggering batch deletion
    view.reset();
    
    // Batch should be automatically deleted when last view is destroyed
}

// Note: Test for deleting non-existent batch removed because delete_data_batch
// is now private and can only be called by data_batch_view destructor

// Test adding multiple batches
TEST_CASE("data_repository_manager Add Multiple Batches", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repository
    size_t pipeline_id = 1;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    constexpr int num_batches = 10;
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    
    // Add multiple batches
    for (int i = 0; i < num_batches; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), pipeline_ids);
    }
    
    // Repository should have all batch views
    auto& repo = manager.get_repository(pipeline_id);
    int count = 0;
    while (auto view = repo->pull_data_batch_view()) {
        ++count;
    }
    REQUIRE(count == num_batches);
}

// =============================================================================
// Thread-Safety Tests
// =============================================================================

// Test concurrent batch ID generation
TEST_CASE("data_repository_manager Thread-Safe Batch ID Generation", "[data_repository_manager]") {
    data_repository_manager manager;
    
    constexpr int num_threads = 10;
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
TEST_CASE("data_repository_manager Thread-Safe Add Repository", "[data_repository_manager]") {
    data_repository_manager manager;
    
    constexpr int num_threads = 10;
    std::vector<std::thread> threads;
    
    // Launch threads to add repositories
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            auto repository = sirius::make_unique<idata_repository>();
            manager.add_new_repository(i, std::move(repository));
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All repositories should be accessible
    for (int i = 0; i < num_threads; ++i) {
        auto& repo = manager.get_repository(i);
        REQUIRE(repo != nullptr);
    }
}

// Test concurrent batch addition
TEST_CASE("data_repository_manager Thread-Safe Add Batch", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repository
    size_t pipeline_id = 1;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    constexpr int num_threads = 10;
    constexpr int batches_per_thread = 50;
    
    std::vector<std::thread> threads;
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    
    // Launch threads to add batches
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < batches_per_thread; ++j) {
                auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
                uint64_t batch_id = manager.get_next_data_batch_id();
                auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
                manager.add_new_data_batch(std::move(batch), pipeline_ids);
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Repository should have all batch views
    auto& repo = manager.get_repository(pipeline_id);
    int count = 0;
    while (auto view = repo->pull_data_batch_view()) {
        ++count;
    }
    REQUIRE(count == num_threads * batches_per_thread);
}

// Test concurrent batch deletion
TEST_CASE("data_repository_manager Thread-Safe Delete Batch", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repository
    size_t pipeline_id = 1;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    constexpr int num_batches = 100;
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    std::vector<uint64_t> batch_ids;
    
    // Add batches
    for (int i = 0; i < num_batches; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        batch_ids.push_back(batch_id);
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), pipeline_ids);
    }
    
    // Pull all views from repository to allow safe deletion
    auto& repo = manager.get_repository(pipeline_id);
    std::vector<sirius::unique_ptr<data_batch_view>> views;
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
TEST_CASE("data_repository_manager Thread-Safe Mixed Operations", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add initial repositories
    for (int i = 0; i < 5; ++i) {
        manager.add_new_repository(i, sirius::make_unique<idata_repository>());
    }
    
    constexpr int num_threads = 10;
    constexpr int operations_per_thread = 50;
    
    std::vector<std::thread> threads;
    std::atomic<int> batch_count{0};
    std::mutex view_mutex;
    std::vector<sirius::unique_ptr<data_batch_view>> all_views;
    
    // Launch threads doing mixed operations
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                // Generate batch ID
                uint64_t batch_id = manager.get_next_data_batch_id();
                
                // Add batch to random pipeline
                size_t pipeline_id = (i + j) % 5;
                auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
                auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
                sirius::vector<size_t> pipeline_ids = {pipeline_id};
                manager.add_new_data_batch(std::move(batch), pipeline_ids);
                
                ++batch_count;
                
                // Occasionally pull and store a view (to test concurrent pull operations)
                if (j % 10 == 0) {
                    auto& repo = manager.get_repository(pipeline_id);
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
TEST_CASE("data_repository_manager Concurrent Add and Delete via View Destructor", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repositories for multiple pipelines
    constexpr int num_pipelines = 3;
    for (int i = 0; i < num_pipelines; ++i) {
        manager.add_new_repository(i, sirius::make_unique<idata_repository>());
    }
    
    constexpr int num_adder_threads = 5;
    constexpr int num_deleter_threads = 5;
    constexpr int batches_per_adder = 100;
    
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
                
                // Add batch to one or more pipelines
                size_t pipeline_id = (i + j) % num_pipelines;
                auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
                auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
                sirius::vector<size_t> pipeline_ids = {pipeline_id};
                manager.add_new_data_batch(std::move(batch), pipeline_ids);
                
                ++batches_added;
                
                // Small delay to allow deleters to work
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }
    
    // Launch deleter threads - pull views and destroy them (triggers batch deletion)
    for (int i = 0; i < num_deleter_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t pipeline_id = i % num_pipelines;
            auto& repo = manager.get_repository(pipeline_id);
            
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
    for (int i = 0; i < num_pipelines; ++i) {
        auto& repo = manager.get_repository(i);
        REQUIRE(repo->pull_data_batch_view() == nullptr);
    }
}

// Test high-contention concurrent add and delete via view destructor
TEST_CASE("data_repository_manager High Contention Add Delete via View Destructor", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Single pipeline for maximum contention
    size_t pipeline_id = 0;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    constexpr int num_threads = 20;
    constexpr int operations_per_thread = 50;
    
    std::vector<std::thread> threads;
    std::atomic<int> total_added{0};
    std::atomic<int> total_deleted{0};
    
    // Launch threads doing both add and delete operations
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            auto& repo = manager.get_repository(pipeline_id);
            sirius::vector<size_t> pipeline_ids = {pipeline_id};
            
            for (int j = 0; j < operations_per_thread; ++j) {
                // Add a batch
                uint64_t batch_id = manager.get_next_data_batch_id();
                auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 512);
                auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
                manager.add_new_data_batch(std::move(batch), pipeline_ids);
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
    auto& repo = manager.get_repository(pipeline_id);
    while (auto view = repo->pull_data_batch_view()) {
        ++total_deleted;
    }
    
    // All batches should have been processed
    REQUIRE(total_deleted == total_added);
}

// Test concurrent add and delete with multiple views per batch
TEST_CASE("data_repository_manager Concurrent Add Delete Multiple Views per Batch", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repositories
    constexpr int num_pipelines = 5;
    for (int i = 0; i < num_pipelines; ++i) {
        manager.add_new_repository(i, sirius::make_unique<idata_repository>());
    }
    
    constexpr int num_batches = 50;
    std::atomic<int> views_deleted{0};
    
    // Add batches to ALL pipelines (each batch will have multiple views)
    sirius::vector<size_t> all_pipeline_ids;
    for (int i = 0; i < num_pipelines; ++i) {
        all_pipeline_ids.push_back(i);
    }
    
    for (int i = 0; i < num_batches; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), all_pipeline_ids);
    }
    
    // Now concurrently delete views from different pipelines
    // The batch should only be deleted when ALL views are destroyed
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_pipelines; ++i) {
        threads.emplace_back([&, i]() {
            auto& repo = manager.get_repository(i);
            
            // Pull and destroy all views from this pipeline
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
    
    // Each batch was added to all pipelines, so we should have deleted
    // num_batches * num_pipelines views
    REQUIRE(views_deleted == num_batches * num_pipelines);
    
    // All repositories should be empty
    for (int i = 0; i < num_pipelines; ++i) {
        auto& repo = manager.get_repository(i);
        REQUIRE(repo->pull_data_batch_view() == nullptr);
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

// Test full workflow with multiple pipelines and batches
TEST_CASE("data_repository_manager Full Workflow", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Setup: Create 3 pipelines
    sirius::vector<size_t> pipeline_ids = {0, 1, 2};
    for (size_t id : pipeline_ids) {
        manager.add_new_repository(id, sirius::make_unique<idata_repository>());
    }
    
    // Add batches to different pipeline combinations
    std::vector<uint64_t> batch_ids;
    
    // Batch 0: All pipelines
    {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        batch_ids.push_back(batch_id);
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), pipeline_ids);
    }
    
    // Batch 1: Pipeline 0 only
    {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
        uint64_t batch_id = manager.get_next_data_batch_id();
        batch_ids.push_back(batch_id);
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        sirius::vector<size_t> p0 = {0};
        manager.add_new_data_batch(std::move(batch), p0);
    }
    
    // Batch 2: Pipelines 1 and 2
    {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 4096);
        uint64_t batch_id = manager.get_next_data_batch_id();
        batch_ids.push_back(batch_id);
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        sirius::vector<size_t> p12 = {1, 2};
        manager.add_new_data_batch(std::move(batch), p12);
    }
    
    // Verify: Pipeline 0 should have 2 batches (batch 0 and 1)
    {
        auto& repo = manager.get_repository(0);
        int count = 0;
        while (auto view = repo->pull_data_batch_view()) {
            ++count;
        }
        REQUIRE(count == 2);
    }
    
    // Verify: Pipeline 1 should have 2 batches (batch 0 and 2)
    {
        auto& repo = manager.get_repository(1);
        int count = 0;
        while (auto view = repo->pull_data_batch_view()) {
            ++count;
        }
        REQUIRE(count == 2);
    }
    
    // Verify: Pipeline 2 should have 2 batches (batch 0 and 2)
    {
        auto& repo = manager.get_repository(2);
        int count = 0;
        while (auto view = repo->pull_data_batch_view()) {
            ++count;
        }
        REQUIRE(count == 2);
    }
    
    // Batches are automatically deleted when all views go out of scope
}

// Test replacing repository with data
TEST_CASE("data_repository_manager Replace Repository With Data", "[data_repository_manager]") {
    data_repository_manager manager;
    
    size_t pipeline_id = 1;
    
    // Add first repository
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    // Add batch to first repository
    auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    manager.add_new_data_batch(std::move(batch), pipeline_ids);
    
    // Replace repository
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    // New repository should be empty
    auto& new_repo = manager.get_repository(pipeline_id);
    auto view = new_repo->pull_data_batch_view();
    REQUIRE(view == nullptr);
}

// Test large number of pipelines
TEST_CASE("data_repository_manager Large Number of Pipelines", "[data_repository_manager]") {
    data_repository_manager manager;
    
    constexpr int num_pipelines = 1000;
    
    // Add many pipelines
    for (int i = 0; i < num_pipelines; ++i) {
        manager.add_new_repository(i, sirius::make_unique<idata_repository>());
    }
    
    // All pipelines should be accessible
    for (int i = 0; i < num_pipelines; ++i) {
        auto& repo = manager.get_repository(i);
        REQUIRE(repo != nullptr);
    }
}

// Test large number of batches
TEST_CASE("data_repository_manager Large Number of Batches", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repository
    size_t pipeline_id = 1;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    constexpr int num_batches = 1000;
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    
    // Add many batches
    for (int i = 0; i < num_batches; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), pipeline_ids);
    }
    
    // Repository should have all batches
    auto& repo = manager.get_repository(pipeline_id);
    int count = 0;
    while (auto view = repo->pull_data_batch_view()) {
        ++count;
    }
    REQUIRE(count == num_batches);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

// Test with pipeline ID zero
TEST_CASE("data_repository_manager Pipeline ID Zero", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Pipeline ID 0 should work like any other ID
    manager.add_new_repository(0, sirius::make_unique<idata_repository>());
    
    auto& repo = manager.get_repository(0);
    REQUIRE(repo != nullptr);
}

// Test with large pipeline IDs
TEST_CASE("data_repository_manager Large Pipeline IDs", "[data_repository_manager]") {
    data_repository_manager manager;
    
    std::vector<size_t> large_ids = {1000, 10000, 100000, SIZE_MAX - 1, SIZE_MAX};
    
    // Add repositories with large IDs
    for (size_t id : large_ids) {
        manager.add_new_repository(id, sirius::make_unique<idata_repository>());
    }
    
    // All should be accessible
    for (size_t id : large_ids) {
        auto& repo = manager.get_repository(id);
        REQUIRE(repo != nullptr);
    }
}

// Test batch with different data sizes
TEST_CASE("data_repository_manager Batches With Different Sizes", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repository
    size_t pipeline_id = 1;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    std::vector<size_t> sizes = {1, 1024, 1024 * 1024, 1024 * 1024 * 10};
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    
    // Add batches with different sizes
    for (size_t size : sizes) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, size);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), pipeline_ids);
    }
    
    // All batches should be accessible
    auto& repo = manager.get_repository(pipeline_id);
    int count = 0;
    while (auto view = repo->pull_data_batch_view()) {
        ++count;
    }
    REQUIRE(count == sizes.size());
}

// Test batch with different memory tiers
TEST_CASE("data_repository_manager Batches With Different Tiers", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repository
    size_t pipeline_id = 1;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    std::vector<memory::Tier> tiers = {memory::Tier::GPU, memory::Tier::HOST, memory::Tier::DISK};
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    
    // Add batches with different tiers
    for (memory::Tier tier : tiers) {
        auto data = sirius::make_unique<mock_data_representation>(tier, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), pipeline_ids);
    }
    
    // All batches should be accessible
    auto& repo = manager.get_repository(pipeline_id);
    int count = 0;
    while (auto view = repo->pull_data_batch_view()) {
        ++count;
    }
    REQUIRE(count == tiers.size());
}

// Test rapid add and delete cycles
TEST_CASE("data_repository_manager Rapid Add Delete Cycles", "[data_repository_manager]") {
    data_repository_manager manager;
    
    // Add repository
    size_t pipeline_id = 1;
    manager.add_new_repository(pipeline_id, sirius::make_unique<idata_repository>());
    
    sirius::vector<size_t> pipeline_ids = {pipeline_id};
    auto& repo = manager.get_repository(pipeline_id);
    
    // Perform many cycles of add and delete
    for (int cycle = 0; cycle < 100; ++cycle) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        uint64_t batch_id = manager.get_next_data_batch_id();
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        manager.add_new_data_batch(std::move(batch), pipeline_ids);
        
        // Pull and destroy the view, which triggers batch deletion
        auto view = repo->pull_data_batch_view();
        REQUIRE(view != nullptr);
        view.reset();  // Explicitly destroy view, auto-deletes batch
    }
    
    // Repository should be empty
    REQUIRE(repo->pull_data_batch_view() == nullptr);
}
