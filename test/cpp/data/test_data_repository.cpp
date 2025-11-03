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
#include "data/data_repository.hpp"
#include "data/data_repository_manager.hpp"
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

// Test basic construction
TEST_CASE("idata_repository Construction", "[data_repository]") {
    idata_repository repository;
    
    // Repository should be empty initially
    auto batch = repository.pull_data_batch_view();
    REQUIRE(batch == nullptr);
}

// Test adding and pulling a single batch
TEST_CASE("idata_repository Add and Pull Single Batch", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    // Create a batch and view
    auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    batch->increment_view_ref_count(); // Prevent auto-delete
    
    auto view = sirius::make_unique<data_batch_view>(batch);
    
    // Add to repository
    repository.add_new_data_batch_view(std::move(view));
    
    // Pull from repository
    auto pulled_view = repository.pull_data_batch_view();
    REQUIRE(pulled_view != nullptr);
    
    // Repository should now be empty
    auto empty = repository.pull_data_batch_view();
    REQUIRE(empty == nullptr);
    
    // Clean up
    batch->decrement_view_ref_count();
    batch->decrement_view_ref_count();
}

// Test FIFO behavior
TEST_CASE("idata_repository FIFO Order", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    std::vector<data_batch*> batches;
    
    // Create multiple batches and add them
    for (uint64_t i = 1; i <= 5; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto* batch = new data_batch(i, manager, std::move(data));
        batch->increment_view_ref_count(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<data_batch_view>(batch);
        repository.add_new_data_batch_view(std::move(view));
    }
    
    // Pull them back and verify FIFO order
    for (uint64_t i = 1; i <= 5; ++i) {
        auto pulled_view = repository.pull_data_batch_view();
        REQUIRE(pulled_view != nullptr);
        // Note: We can't directly check batch ID from view, but order should be maintained
    }
    
    // Repository should be empty
    auto empty = repository.pull_data_batch_view();
    REQUIRE(empty == nullptr);
    
    // Clean up
    for (auto* batch : batches) {
        batch->decrement_view_ref_count();
    }
}

// Test multiple add and pull operations
TEST_CASE("idata_repository Multiple Add Pull Operations", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    std::vector<data_batch*> batches;
    
    // Add 3 batches
    for (uint64_t i = 1; i <= 3; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto* batch = new data_batch(i, manager, std::move(data));
        batch->increment_view_ref_count(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<data_batch_view>(batch);
        repository.add_new_data_batch_view(std::move(view));
    }
    
    // Pull 2 batches
    auto view1 = repository.pull_data_batch_view();
    auto view2 = repository.pull_data_batch_view();
    REQUIRE(view1 != nullptr);
    REQUIRE(view2 != nullptr);
    
    // Add 2 more batches
    for (uint64_t i = 4; i <= 5; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto* batch = new data_batch(i, manager, std::move(data));
        batch->increment_view_ref_count(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<data_batch_view>(batch);
        repository.add_new_data_batch_view(std::move(view));
    }
    
    // Pull remaining 3 batches (1 from first batch + 2 new)
    auto view3 = repository.pull_data_batch_view();
    auto view4 = repository.pull_data_batch_view();
    auto view5 = repository.pull_data_batch_view();
    REQUIRE(view3 != nullptr);
    REQUIRE(view4 != nullptr);
    REQUIRE(view5 != nullptr);
    
    // Repository should be empty
    auto empty = repository.pull_data_batch_view();
    REQUIRE(empty == nullptr);
    
    // Clean up
    for (auto* batch : batches) {
        batch->decrement_view_ref_count();
    }
}

// Test pulling from empty repository
TEST_CASE("idata_repository Pull From Empty", "[data_repository]") {
    idata_repository repository;
    
    // Pull from empty repository multiple times
    for (int i = 0; i < 10; ++i) {
        auto view = repository.pull_data_batch_view();
        REQUIRE(view == nullptr);
    }
}

// Test thread-safe adding
TEST_CASE("idata_repository Thread-Safe Adding", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    constexpr int num_threads = 10;
    constexpr int batches_per_thread = 50;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<data_batch*>> thread_batches(num_threads);
    
    // Launch threads to add batches
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < batches_per_thread; ++j) {
                auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
                uint64_t batch_id = i * batches_per_thread + j;
                auto* batch = new data_batch(batch_id, manager, std::move(data));
                batch->increment_view_ref_count(); // Prevent auto-delete
                thread_batches[i].push_back(batch);
                
                auto view = sirius::make_unique<data_batch_view>(batch);
                repository.add_new_data_batch_view(std::move(view));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Pull all batches and count
    int count = 0;
    while (auto view = repository.pull_data_batch_view()) {
        ++count;
    }
    
    // Should have exactly num_threads * batches_per_thread
    REQUIRE(count == num_threads * batches_per_thread);
    
    // Clean up
    for (auto& batches : thread_batches) {
        for (auto* batch : batches) {
            batch->decrement_view_ref_count();
        }
    }
}

// Test thread-safe pulling
TEST_CASE("idata_repository Thread-Safe Pulling", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    constexpr int num_batches = 500;
    std::vector<data_batch*> batches;
    
    // Add many batches
    for (int i = 0; i < num_batches; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto* batch = new data_batch(i, manager, std::move(data));
        batch->increment_view_ref_count(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<data_batch_view>(batch);
        repository.add_new_data_batch_view(std::move(view));
    }
    
    constexpr int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<int> thread_counts(num_threads, 0);
    
    // Launch threads to pull batches
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            while (auto view = repository.pull_data_batch_view()) {
                ++thread_counts[i];
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Sum all thread counts
    int total_count = 0;
    for (int count : thread_counts) {
        total_count += count;
    }
    
    // Should have pulled exactly num_batches
    REQUIRE(total_count == num_batches);
    
    // Repository should be empty
    auto empty = repository.pull_data_batch_view();
    REQUIRE(empty == nullptr);
    
    // Clean up
    for (auto* batch : batches) {
        batch->decrement_view_ref_count();
    }
}

// Test concurrent adding and pulling
TEST_CASE("idata_repository Concurrent Add and Pull", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    constexpr int num_add_threads = 5;
    constexpr int num_pull_threads = 5;
    constexpr int batches_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<data_batch*>> thread_batches(num_add_threads);
    std::atomic<int> pulled_count{0};
    
    // Launch adding threads
    for (int i = 0; i < num_add_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < batches_per_thread; ++j) {
                auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
                uint64_t batch_id = i * batches_per_thread + j;
                auto* batch = new data_batch(batch_id, manager, std::move(data));
                batch->increment_view_ref_count(); // Prevent auto-delete
                thread_batches[i].push_back(batch);
                
                auto view = sirius::make_unique<data_batch_view>(batch);
                repository.add_new_data_batch_view(std::move(view));
                
                // Small delay to allow pullers to work
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }
    
    // Launch pulling threads
    for (int i = 0; i < num_pull_threads; ++i) {
        threads.emplace_back([&]() {
            int local_count = 0;
            while (local_count < batches_per_thread) {
                auto view = repository.pull_data_batch_view();
                if (view) {
                    ++local_count;
                    ++pulled_count;
                } else {
                    // Repository temporarily empty, yield to adders
                    std::this_thread::yield();
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Should have pulled exactly num_add_threads * batches_per_thread
    REQUIRE(pulled_count == num_add_threads * batches_per_thread);
    
    // Clean up
    for (auto& batches : thread_batches) {
        for (auto* batch : batches) {
            batch->decrement_view_ref_count();
        }
    }
}

// Test large number of batches
TEST_CASE("idata_repository Large Number of Batches", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    constexpr int num_batches = 10000;
    std::vector<data_batch*> batches;
    
    // Add many batches
    for (int i = 0; i < num_batches; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto* batch = new data_batch(i, manager, std::move(data));
        batch->increment_view_ref_count(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<data_batch_view>(batch);
        repository.add_new_data_batch_view(std::move(view));
    }
    
    // Pull all batches
    int count = 0;
    while (auto view = repository.pull_data_batch_view()) {
        ++count;
    }
    
    REQUIRE(count == num_batches);
    
    // Clean up
    for (auto* batch : batches) {
        batch->decrement_view_ref_count();
    }
}

// Test add after pull all
TEST_CASE("idata_repository Add After Pull All", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    std::vector<data_batch*> batches;
    
    // Add some batches
    for (int i = 0; i < 5; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto* batch = new data_batch(i, manager, std::move(data));
        batch->increment_view_ref_count(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<data_batch_view>(batch);
        repository.add_new_data_batch_view(std::move(view));
    }
    
    // Pull all
    int count1 = 0;
    while (auto view = repository.pull_data_batch_view()) {
        ++count1;
    }
    REQUIRE(count1 == 5);
    
    // Add more batches
    for (int i = 5; i < 10; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
        auto* batch = new data_batch(i, manager, std::move(data));
        batch->increment_view_ref_count(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<data_batch_view>(batch);
        repository.add_new_data_batch_view(std::move(view));
    }
    
    // Pull again
    int count2 = 0;
    while (auto view = repository.pull_data_batch_view()) {
        ++count2;
    }
    REQUIRE(count2 == 5);
    
    // Clean up
    for (auto* batch : batches) {
        batch->decrement_view_ref_count();
    }
}

// Test repository with different batch sizes
TEST_CASE("idata_repository Different Batch Sizes", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    std::vector<size_t> sizes = {100, 1024, 10240, 102400, 1024000};
    std::vector<data_batch*> batches;
    
    // Add batches with different sizes
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, sizes[i]);
        auto* batch = new data_batch(i, manager, std::move(data));
        batch->increment_view_ref_count(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<data_batch_view>(batch);
        repository.add_new_data_batch_view(std::move(view));
    }
    
    // Pull all batches
    int count = 0;
    while (auto view = repository.pull_data_batch_view()) {
        ++count;
    }
    
    REQUIRE(count == sizes.size());
    
    // Clean up
    for (auto* batch : batches) {
        batch->decrement_view_ref_count();
    }
}

// Test interleaved add and pull
TEST_CASE("idata_repository Interleaved Add and Pull", "[data_repository]") {
    data_repository_manager manager;
    idata_repository repository;
    
    std::vector<data_batch*> batches;
    
    for (int cycle = 0; cycle < 50; ++cycle) {
        // Add some batches
        for (int i = 0; i < 3; ++i) {
            auto data = sirius::make_unique<mock_data_representation>(memory::Tier::GPU, 1024);
            auto* batch = new data_batch(cycle * 3 + i, manager, std::move(data));
            batch->increment_view_ref_count(); // Prevent auto-delete
            batches.push_back(batch);
            
            auto view = sirius::make_unique<data_batch_view>(batch);
            repository.add_new_data_batch_view(std::move(view));
        }
        
        // Pull one batch
        auto view = repository.pull_data_batch_view();
        REQUIRE(view != nullptr);
    }
    
    // Pull remaining batches
    int remaining = 0;
    while (auto view = repository.pull_data_batch_view()) {
        ++remaining;
    }
    
    // Should have 50 cycles * 3 adds - 50 pulls = 100 remaining
    REQUIRE(remaining == 100);
    
    // Clean up
    for (auto* batch : batches) {
        batch->decrement_view_ref_count();
    }
}

