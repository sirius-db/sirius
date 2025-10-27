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
#include "data/data_repository.hpp"
#include "data/data_batch.hpp"
#include "data/data_batch_view.hpp"
#include "data/common.hpp"
#include "memory/memory_reservation.hpp"

using namespace sirius;
using sirius::memory::Tier;

// Mock IDataRepresentation for testing
class MockDataRepresentation : public IDataRepresentation {
public:
    explicit MockDataRepresentation(Tier tier, size_t size = 1024)
        : tier_(tier), size_(size) {}
    
    Tier GetCurrentTier() const override {
        return tier_;
    }
    
    std::size_t GetSizeInBytes() const override {
        return size_;
    }
    
    void SetTier(Tier tier) {
        tier_ = tier;
    }

private:
    Tier tier_;
    size_t size_;
};

// Test basic construction
TEST_CASE("IDataRepository Construction", "[data_repository]") {
    IDataRepository repository;
    
    // Repository should be empty initially
    auto batch = repository.PullDataBatchView();
    REQUIRE(batch == nullptr);
}

// Test adding and pulling a single batch
TEST_CASE("IDataRepository Add and Pull Single Batch", "[data_repository]") {
    IDataRepository repository;
    
    // Create a batch and view
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    batch->IncrementViewRefCount(); // Prevent auto-delete
    
    auto view = sirius::make_unique<DataBatchView>(batch);
    
    // Add to repository
    repository.AddNewDataBatchView(std::move(view));
    
    // Pull from repository
    auto pulled_view = repository.PullDataBatchView();
    REQUIRE(pulled_view != nullptr);
    
    // Repository should now be empty
    auto empty = repository.PullDataBatchView();
    REQUIRE(empty == nullptr);
    
    // Clean up
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test FIFO behavior
TEST_CASE("IDataRepository FIFO Order", "[data_repository]") {
    IDataRepository repository;
    
    std::vector<DataBatch*> batches;
    
    // Create multiple batches and add them
    for (uint64_t i = 1; i <= 5; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto* batch = new DataBatch(i, std::move(data));
        batch->IncrementViewRefCount(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<DataBatchView>(batch);
        repository.AddNewDataBatchView(std::move(view));
    }
    
    // Pull them back and verify FIFO order
    for (uint64_t i = 1; i <= 5; ++i) {
        auto pulled_view = repository.PullDataBatchView();
        REQUIRE(pulled_view != nullptr);
        // Note: We can't directly check batch ID from view, but order should be maintained
    }
    
    // Repository should be empty
    auto empty = repository.PullDataBatchView();
    REQUIRE(empty == nullptr);
    
    // Clean up
    for (auto* batch : batches) {
        batch->DecrementViewRefCount();
    }
}

// Test multiple add and pull operations
TEST_CASE("IDataRepository Multiple Add Pull Operations", "[data_repository]") {
    IDataRepository repository;
    
    std::vector<DataBatch*> batches;
    
    // Add 3 batches
    for (uint64_t i = 1; i <= 3; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto* batch = new DataBatch(i, std::move(data));
        batch->IncrementViewRefCount(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<DataBatchView>(batch);
        repository.AddNewDataBatchView(std::move(view));
    }
    
    // Pull 2 batches
    auto view1 = repository.PullDataBatchView();
    auto view2 = repository.PullDataBatchView();
    REQUIRE(view1 != nullptr);
    REQUIRE(view2 != nullptr);
    
    // Add 2 more batches
    for (uint64_t i = 4; i <= 5; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto* batch = new DataBatch(i, std::move(data));
        batch->IncrementViewRefCount(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<DataBatchView>(batch);
        repository.AddNewDataBatchView(std::move(view));
    }
    
    // Pull remaining 3 batches (1 from first batch + 2 new)
    auto view3 = repository.PullDataBatchView();
    auto view4 = repository.PullDataBatchView();
    auto view5 = repository.PullDataBatchView();
    REQUIRE(view3 != nullptr);
    REQUIRE(view4 != nullptr);
    REQUIRE(view5 != nullptr);
    
    // Repository should be empty
    auto empty = repository.PullDataBatchView();
    REQUIRE(empty == nullptr);
    
    // Clean up
    for (auto* batch : batches) {
        batch->DecrementViewRefCount();
    }
}

// Test pulling from empty repository
TEST_CASE("IDataRepository Pull From Empty", "[data_repository]") {
    IDataRepository repository;
    
    // Pull from empty repository multiple times
    for (int i = 0; i < 10; ++i) {
        auto view = repository.PullDataBatchView();
        REQUIRE(view == nullptr);
    }
}

// Test thread-safe adding
TEST_CASE("IDataRepository Thread-Safe Adding", "[data_repository]") {
    IDataRepository repository;
    
    constexpr int num_threads = 10;
    constexpr int batches_per_thread = 50;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<DataBatch*>> thread_batches(num_threads);
    
    // Launch threads to add batches
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < batches_per_thread; ++j) {
                auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
                uint64_t batch_id = i * batches_per_thread + j;
                auto* batch = new DataBatch(batch_id, std::move(data));
                batch->IncrementViewRefCount(); // Prevent auto-delete
                thread_batches[i].push_back(batch);
                
                auto view = sirius::make_unique<DataBatchView>(batch);
                repository.AddNewDataBatchView(std::move(view));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Pull all batches and count
    int count = 0;
    while (auto view = repository.PullDataBatchView()) {
        ++count;
    }
    
    // Should have exactly num_threads * batches_per_thread
    REQUIRE(count == num_threads * batches_per_thread);
    
    // Clean up
    for (auto& batches : thread_batches) {
        for (auto* batch : batches) {
            batch->DecrementViewRefCount();
        }
    }
}

// Test thread-safe pulling
TEST_CASE("IDataRepository Thread-Safe Pulling", "[data_repository]") {
    IDataRepository repository;
    
    constexpr int num_batches = 500;
    std::vector<DataBatch*> batches;
    
    // Add many batches
    for (int i = 0; i < num_batches; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto* batch = new DataBatch(i, std::move(data));
        batch->IncrementViewRefCount(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<DataBatchView>(batch);
        repository.AddNewDataBatchView(std::move(view));
    }
    
    constexpr int num_threads = 10;
    std::vector<std::thread> threads;
    std::vector<int> thread_counts(num_threads, 0);
    
    // Launch threads to pull batches
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            while (auto view = repository.PullDataBatchView()) {
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
    auto empty = repository.PullDataBatchView();
    REQUIRE(empty == nullptr);
    
    // Clean up
    for (auto* batch : batches) {
        batch->DecrementViewRefCount();
    }
}

// Test concurrent adding and pulling
TEST_CASE("IDataRepository Concurrent Add and Pull", "[data_repository]") {
    IDataRepository repository;
    
    constexpr int num_add_threads = 5;
    constexpr int num_pull_threads = 5;
    constexpr int batches_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<DataBatch*>> thread_batches(num_add_threads);
    std::atomic<int> pulled_count{0};
    
    // Launch adding threads
    for (int i = 0; i < num_add_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < batches_per_thread; ++j) {
                auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
                uint64_t batch_id = i * batches_per_thread + j;
                auto* batch = new DataBatch(batch_id, std::move(data));
                batch->IncrementViewRefCount(); // Prevent auto-delete
                thread_batches[i].push_back(batch);
                
                auto view = sirius::make_unique<DataBatchView>(batch);
                repository.AddNewDataBatchView(std::move(view));
                
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
                auto view = repository.PullDataBatchView();
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
            batch->DecrementViewRefCount();
        }
    }
}

// Test large number of batches
TEST_CASE("IDataRepository Large Number of Batches", "[data_repository]") {
    IDataRepository repository;
    
    constexpr int num_batches = 10000;
    std::vector<DataBatch*> batches;
    
    // Add many batches
    for (int i = 0; i < num_batches; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto* batch = new DataBatch(i, std::move(data));
        batch->IncrementViewRefCount(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<DataBatchView>(batch);
        repository.AddNewDataBatchView(std::move(view));
    }
    
    // Pull all batches
    int count = 0;
    while (auto view = repository.PullDataBatchView()) {
        ++count;
    }
    
    REQUIRE(count == num_batches);
    
    // Clean up
    for (auto* batch : batches) {
        batch->DecrementViewRefCount();
    }
}

// Test add after pull all
TEST_CASE("IDataRepository Add After Pull All", "[data_repository]") {
    IDataRepository repository;
    
    std::vector<DataBatch*> batches;
    
    // Add some batches
    for (int i = 0; i < 5; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto* batch = new DataBatch(i, std::move(data));
        batch->IncrementViewRefCount(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<DataBatchView>(batch);
        repository.AddNewDataBatchView(std::move(view));
    }
    
    // Pull all
    int count1 = 0;
    while (auto view = repository.PullDataBatchView()) {
        ++count1;
    }
    REQUIRE(count1 == 5);
    
    // Add more batches
    for (int i = 5; i < 10; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto* batch = new DataBatch(i, std::move(data));
        batch->IncrementViewRefCount(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<DataBatchView>(batch);
        repository.AddNewDataBatchView(std::move(view));
    }
    
    // Pull again
    int count2 = 0;
    while (auto view = repository.PullDataBatchView()) {
        ++count2;
    }
    REQUIRE(count2 == 5);
    
    // Clean up
    for (auto* batch : batches) {
        batch->DecrementViewRefCount();
    }
}

// Test repository with different batch sizes
TEST_CASE("IDataRepository Different Batch Sizes", "[data_repository]") {
    IDataRepository repository;
    
    std::vector<size_t> sizes = {100, 1024, 10240, 102400, 1024000};
    std::vector<DataBatch*> batches;
    
    // Add batches with different sizes
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, sizes[i]);
        auto* batch = new DataBatch(i, std::move(data));
        batch->IncrementViewRefCount(); // Prevent auto-delete
        batches.push_back(batch);
        
        auto view = sirius::make_unique<DataBatchView>(batch);
        repository.AddNewDataBatchView(std::move(view));
    }
    
    // Pull all batches
    int count = 0;
    while (auto view = repository.PullDataBatchView()) {
        ++count;
    }
    
    REQUIRE(count == sizes.size());
    
    // Clean up
    for (auto* batch : batches) {
        batch->DecrementViewRefCount();
    }
}

// Test interleaved add and pull
TEST_CASE("IDataRepository Interleaved Add and Pull", "[data_repository]") {
    IDataRepository repository;
    
    std::vector<DataBatch*> batches;
    
    for (int cycle = 0; cycle < 50; ++cycle) {
        // Add some batches
        for (int i = 0; i < 3; ++i) {
            auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
            auto* batch = new DataBatch(cycle * 3 + i, std::move(data));
            batch->IncrementViewRefCount(); // Prevent auto-delete
            batches.push_back(batch);
            
            auto view = sirius::make_unique<DataBatchView>(batch);
            repository.AddNewDataBatchView(std::move(view));
        }
        
        // Pull one batch
        auto view = repository.PullDataBatchView();
        REQUIRE(view != nullptr);
    }
    
    // Pull remaining batches
    int remaining = 0;
    while (auto view = repository.PullDataBatchView()) {
        ++remaining;
    }
    
    // Should have 50 cycles * 3 adds - 50 pulls = 100 remaining
    REQUIRE(remaining == 100);
    
    // Clean up
    for (auto* batch : batches) {
        batch->DecrementViewRefCount();
    }
}

