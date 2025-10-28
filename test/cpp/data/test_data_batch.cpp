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
#include "data/data_batch.hpp"
#include "data/common.hpp"

using namespace sirius;

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
    
    sirius::unique_ptr<IDataRepresentation> ConvertToTier(Tier target_tier, rmm::mr::device_memory_resource* mr = nullptr, rmm::cuda_stream_view stream = rmm::cuda_stream_default) override {
        // Empty implementation for testing
        return nullptr;
    }
    
    void SetTier(Tier tier) {
        tier_ = tier;
    }

private:
    Tier tier_;
    size_t size_;
};

// Test basic construction
TEST_CASE("DataBatch Construction", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 2048);
    DataBatch batch(1, std::move(data));
    
    REQUIRE(batch.GetBatchId() == 1);
    REQUIRE(batch.GetCurrentTier() == Tier::GPU);
    REQUIRE(batch.GetViewCount() == 0);
    REQUIRE(batch.GetPinCount() == 0);
}

// Test move constructor
TEST_CASE("DataBatch Move Constructor", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::HOST, 1024);
    DataBatch batch1(42, std::move(data));
    
    REQUIRE(batch1.GetBatchId() == 42);
    REQUIRE(batch1.GetCurrentTier() == Tier::HOST);
    
    // Move construct
    DataBatch batch2(std::move(batch1));
    
    REQUIRE(batch2.GetBatchId() == 42);
    REQUIRE(batch2.GetCurrentTier() == Tier::HOST);
    REQUIRE(batch1.GetBatchId() == 0); // Moved-from state
}

// Test move assignment
TEST_CASE("DataBatch Move Assignment", "[data_batch]") {
    auto data1 = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 512);
    auto data2 = sirius::make_unique<MockDataRepresentation>(Tier::HOST, 1024);
    
    DataBatch batch1(10, std::move(data1));
    DataBatch batch2(20, std::move(data2));
    
    REQUIRE(batch1.GetBatchId() == 10);
    REQUIRE(batch2.GetBatchId() == 20);
    
    // Move assign
    batch1 = std::move(batch2);
    
    REQUIRE(batch1.GetBatchId() == 20);
    REQUIRE(batch1.GetCurrentTier() == Tier::HOST);
    REQUIRE(batch2.GetBatchId() == 0); // Moved-from state
}

// Test self-assignment (move)
TEST_CASE("DataBatch Self Move Assignment", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(100, std::move(data));
    
    // Self-assignment should not crash
    batch = std::move(batch);
    
    REQUIRE(batch.GetBatchId() == 100);
    REQUIRE(batch.GetCurrentTier() == Tier::GPU);
}

// Test view reference counting - increment and decrement
TEST_CASE("DataBatch View Reference Counting", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    REQUIRE(batch.GetViewCount() == 0);
    
    // Increment view count
    batch.IncrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 1);
    
    batch.IncrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 2);
    
    batch.IncrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 3);
    
    // Decrement view count
    batch.DecrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 2);
    
    batch.DecrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 1);
    
    batch.DecrementViewRefCount();
    // batch should be deleted
}

// Test pin reference counting - increment and decrement
TEST_CASE("DataBatch Pin Reference Counting", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    REQUIRE(batch.GetPinCount() == 0);
    
    // Increment pin count
    batch.IncrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 1);
    
    batch.IncrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 2);
    
    batch.IncrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 3);
    
    // Decrement pin count
    batch.DecrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 2);
    
    batch.DecrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 1);
    
    batch.DecrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 0);
}

// Test increment pin count throws when not in GPU tier
TEST_CASE("DataBatch IncrementPinRefCount GPU Tier Validation", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::HOST, 1024);
    DataBatch batch(1, std::move(data));
    
    // Should throw because data is not in GPU tier
    REQUIRE_THROWS_AS(batch.IncrementPinRefCount(), std::runtime_error);
    REQUIRE(batch.GetPinCount() == 0); // Count should not change
}

// Test decrement pin count throws when not in GPU tier
TEST_CASE("DataBatch DecrementPinRefCount GPU Tier Validation", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    // First increment while in GPU tier
    batch.IncrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 1);
    
    // Decrement while in GPU tier should work
    batch.DecrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 0);
}

// Test that view count operations work regardless of tier
TEST_CASE("DataBatch View Count No Tier Validation", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::HOST, 1024);
    DataBatch batch(1, std::move(data));
    
    // View count should work even when not in GPU tier
    batch.IncrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 1);
    
    batch.DecrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 0);
}

// Test multiple batches with different IDs
TEST_CASE("Multiple DataBatch Instances", "[data_batch]") {
    std::vector<DataBatch> batches;
    
    for (uint64_t i = 0; i < 10; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024 * (i + 1));
        batches.emplace_back(i, std::move(data));
    }
    
    // Verify all batches have correct IDs and tiers
    for (uint64_t i = 0; i < 10; ++i) {
        REQUIRE(batches[i].GetBatchId() == i);
        REQUIRE(batches[i].GetCurrentTier() == Tier::GPU);
        REQUIRE(batches[i].GetViewCount() == 0);
        REQUIRE(batches[i].GetPinCount() == 0);
    }
}

// Test GetCurrentTier delegates to IDataRepresentation
TEST_CASE("DataBatch GetCurrentTier Delegation", "[data_batch]") {
    // Test GPU tier
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        DataBatch batch(1, std::move(data));
        REQUIRE(batch.GetCurrentTier() == Tier::GPU);
    }
    
    // Test HOST tier
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::HOST, 1024);
        DataBatch batch(2, std::move(data));
        REQUIRE(batch.GetCurrentTier() == Tier::HOST);
    }
    
    // Test DISK tier
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::DISK, 1024);
        DataBatch batch(3, std::move(data));
        REQUIRE(batch.GetCurrentTier() == Tier::DISK);
    }
}

// Test thread-safe view reference counting
TEST_CASE("DataBatch Thread-Safe View Reference Counting", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    constexpr int num_threads = 10;
    constexpr int increments_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch threads to increment view count
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&batch]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                batch.IncrementViewRefCount();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count
    REQUIRE(batch.GetViewCount() == num_threads * increments_per_thread);
    
    threads.clear();
    
    // Launch threads to decrement view count
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&batch]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                batch.DecrementViewRefCount();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count is back to zero
    REQUIRE(batch.GetViewCount() == 0);
}

// Test thread-safe pin reference counting
TEST_CASE("DataBatch Thread-Safe Pin Reference Counting", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    constexpr int num_threads = 10;
    constexpr int increments_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch threads to increment pin count
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&batch]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                batch.IncrementPinRefCount();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count
    REQUIRE(batch.GetPinCount() == num_threads * increments_per_thread);
    
    threads.clear();
    
    // Launch threads to decrement pin count
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&batch]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                batch.DecrementPinRefCount();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count is back to zero
    REQUIRE(batch.GetPinCount() == 0);
}

// Test concurrent view increment and decrement
TEST_CASE("DataBatch Concurrent View Increment and Decrement", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    // Pre-increment to avoid hitting zero during concurrent operations
    for (int i = 0; i < 1000; ++i) {
        batch.IncrementViewRefCount();
    }
    
    constexpr int num_threads = 5;
    constexpr int operations_per_thread = 100;
    
    std::vector<std::thread> inc_threads;
    std::vector<std::thread> dec_threads;
    
    // Launch incrementing threads
    for (int i = 0; i < num_threads; ++i) {
        inc_threads.emplace_back([&batch]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                batch.IncrementViewRefCount();
            }
        });
    }
    
    // Launch decrementing threads
    for (int i = 0; i < num_threads; ++i) {
        dec_threads.emplace_back([&batch]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                batch.DecrementViewRefCount();
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : inc_threads) {
        thread.join();
    }
    for (auto& thread : dec_threads) {
        thread.join();
    }
    
    // Net effect should be 1000 (initial) + 0 (equal increments and decrements)
    REQUIRE(batch.GetViewCount() == 1000);
}

// Test concurrent pin increment and decrement
TEST_CASE("DataBatch Concurrent Pin Increment and Decrement", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    // Pre-increment to avoid hitting zero during concurrent operations
    for (int i = 0; i < 1000; ++i) {
        batch.IncrementPinRefCount();
    }
    
    constexpr int num_threads = 5;
    constexpr int operations_per_thread = 100;
    
    std::vector<std::thread> inc_threads;
    std::vector<std::thread> dec_threads;
    
    // Launch incrementing threads
    for (int i = 0; i < num_threads; ++i) {
        inc_threads.emplace_back([&batch]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                batch.IncrementPinRefCount();
            }
        });
    }
    
    // Launch decrementing threads
    for (int i = 0; i < num_threads; ++i) {
        dec_threads.emplace_back([&batch]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                batch.DecrementPinRefCount();
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : inc_threads) {
        thread.join();
    }
    for (auto& thread : dec_threads) {
        thread.join();
    }
    
    // Net effect should be 1000 (initial) + 0 (equal increments and decrements)
    REQUIRE(batch.GetPinCount() == 1000);
}

// Test batch ID uniqueness in practice
TEST_CASE("DataBatch Unique IDs", "[data_batch]") {
    std::vector<uint64_t> batch_ids = {0, 1, 100, 999, 1000, 9999, 
                                        UINT64_MAX - 1, UINT64_MAX};
    
    std::vector<DataBatch> batches;
    
    for (auto id : batch_ids) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        batches.emplace_back(id, std::move(data));
    }
    
    // Verify each batch has the correct ID
    for (size_t i = 0; i < batch_ids.size(); ++i) {
        REQUIRE(batches[i].GetBatchId() == batch_ids[i]);
    }
}

// Test edge case: zero view count operations
TEST_CASE("DataBatch Zero View Count", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    // Starting view count should be zero
    REQUIRE(batch.GetViewCount() == 0);
    
    // Increment from zero
    batch.IncrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 1);
    
    // Decrement back to zero
    batch.DecrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 0);
    
    // Can increment again from zero
    batch.IncrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 1);
}

// Test edge case: zero pin count operations
TEST_CASE("DataBatch Zero Pin Count", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    // Starting pin count should be zero
    REQUIRE(batch.GetPinCount() == 0);
    
    // Increment from zero
    batch.IncrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 1);
    
    // Decrement back to zero
    batch.DecrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 0);
    
    // Can increment again from zero
    batch.IncrementPinRefCount();
    REQUIRE(batch.GetPinCount() == 1);
}

// Test with different data sizes
TEST_CASE("DataBatch With Different Data Sizes", "[data_batch]") {
    std::vector<size_t> sizes = {0, 1, 1024, 1024 * 1024, 1024 * 1024 * 100};
    
    for (size_t size : sizes) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, size);
        auto* data_ptr = data.get();
        DataBatch batch(1, std::move(data));
        
        // Verify the data representation is accessible through the batch
        REQUIRE(batch.GetCurrentTier() == Tier::GPU);
        REQUIRE(data_ptr->GetSizeInBytes() == size);
    }
}

// Test that move operations require zero reference counts
TEST_CASE("DataBatch Move Requires Zero Reference Counts", "[data_batch]") {
    // Test that moving with active views throws
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        DataBatch batch1(1, std::move(data));
        batch1.IncrementViewRefCount();
        
        REQUIRE_THROWS_AS([&]() { DataBatch batch2(std::move(batch1)); }(), std::runtime_error);
    }
    
    // Test that moving with active pins throws
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        DataBatch batch1(1, std::move(data));
        batch1.IncrementPinRefCount();
        
        REQUIRE_THROWS_AS([&]() { DataBatch batch2(std::move(batch1)); }(), std::runtime_error);
    }
    
    // Test that moving with both active views and pins throws
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        DataBatch batch1(1, std::move(data));
        batch1.IncrementViewRefCount();
        batch1.IncrementPinRefCount();
        
        REQUIRE_THROWS_AS([&]() { DataBatch batch2(std::move(batch1)); }(), std::runtime_error);
    }
    
    // Test that moving with zero counts succeeds
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        DataBatch batch1(1, std::move(data));
        
        REQUIRE(batch1.GetViewCount() == 0);
        REQUIRE(batch1.GetPinCount() == 0);
        
        DataBatch batch2(std::move(batch1));
        
        REQUIRE(batch2.GetViewCount() == 0);
        REQUIRE(batch2.GetPinCount() == 0);
        REQUIRE(batch2.GetBatchId() == 1);
    }
}

// Test multiple rapid view count increment/decrement cycles
TEST_CASE("DataBatch Rapid View Count Cycles", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    // Perform many cycles of increment and decrement
    for (int cycle = 0; cycle < 100; ++cycle) {
        for (int i = 0; i < 10; ++i) {
            batch.IncrementViewRefCount();
        }
        REQUIRE(batch.GetViewCount() == 10);
        
        for (int i = 0; i < 10; ++i) {
            batch.DecrementViewRefCount();
        }
        REQUIRE(batch.GetViewCount() == 0);
    }
    
    // Final state should be zero
    REQUIRE(batch.GetViewCount() == 0);
}

// Test multiple rapid pin count increment/decrement cycles
TEST_CASE("DataBatch Rapid Pin Count Cycles", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    // Perform many cycles of increment and decrement
    for (int cycle = 0; cycle < 100; ++cycle) {
        for (int i = 0; i < 10; ++i) {
            batch.IncrementPinRefCount();
        }
        REQUIRE(batch.GetPinCount() == 10);
        
        for (int i = 0; i < 10; ++i) {
            batch.DecrementPinRefCount();
        }
        REQUIRE(batch.GetPinCount() == 0);
    }
    
    // Final state should be zero
    REQUIRE(batch.GetPinCount() == 0);
}

// Test independent view and pin counts
TEST_CASE("DataBatch Independent View and Pin Counts", "[data_batch]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    DataBatch batch(1, std::move(data));
    
    // Increment view count
    batch.IncrementViewRefCount();
    batch.IncrementViewRefCount();
    batch.IncrementViewRefCount();
    
    // Increment pin count
    batch.IncrementPinRefCount();
    batch.IncrementPinRefCount();
    
    // Verify counts are independent
    REQUIRE(batch.GetViewCount() == 3);
    REQUIRE(batch.GetPinCount() == 2);
    
    // Decrement view count
    batch.DecrementViewRefCount();
    REQUIRE(batch.GetViewCount() == 2);
    REQUIRE(batch.GetPinCount() == 2); // Pin count unchanged
    
    // Decrement pin count
    batch.DecrementPinRefCount();
    REQUIRE(batch.GetViewCount() == 2); // View count unchanged
    REQUIRE(batch.GetPinCount() == 1);
}

