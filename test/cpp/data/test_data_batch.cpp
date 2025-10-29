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

// Mock idata_representation for testing
class mock_data_representation : public idata_representation {
public:
    explicit mock_data_representation(Tier tier, size_t size = 1024)
        : _tier(tier), _size(size) {}
    
    Tier get_current_tier() const override {
        return _tier;
    }
    
    std::size_t get_size_in_bytes() const override {
        return _size;
    }
    
    sirius::unique_ptr<idata_representation> convert_to_tier(Tier target_tier, rmm::mr::device_memory_resource* mr = nullptr, rmm::cuda_stream_view stream = rmm::cuda_stream_default) override {
        // Empty implementation for testing
        return nullptr;
    }
    
    void set_tier(Tier tier) {
        _tier = tier;
    }

private:
    Tier _tier;
    size_t _size;
};

// Test basic construction
TEST_CASE("data_batch Construction", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 2048);
    data_batch batch(1, std::move(data));
    
    REQUIRE(batch.get_batch_id() == 1);
    REQUIRE(batch.get_current_tier() == Tier::GPU);
    REQUIRE(batch.get_view_count() == 0);
    REQUIRE(batch.get_pin_count() == 0);
}

// Test move constructor
TEST_CASE("data_batch Move Constructor", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::HOST, 1024);
    data_batch batch1(42, std::move(data));
    
    REQUIRE(batch1.get_batch_id() == 42);
    REQUIRE(batch1.get_current_tier() == Tier::HOST);
    
    // Move construct
    data_batch batch2(std::move(batch1));
    
    REQUIRE(batch2.get_batch_id() == 42);
    REQUIRE(batch2.get_current_tier() == Tier::HOST);
    REQUIRE(batch1.get_batch_id() == 0); // Moved-from state
}

// Test move assignment
TEST_CASE("data_batch Move Assignment", "[data_batch]") {
    auto data1 = sirius::make_unique<mock_data_representation>(Tier::GPU, 512);
    auto data2 = sirius::make_unique<mock_data_representation>(Tier::HOST, 1024);
    
    data_batch batch1(10, std::move(data1));
    data_batch batch2(20, std::move(data2));
    
    REQUIRE(batch1.get_batch_id() == 10);
    REQUIRE(batch2.get_batch_id() == 20);
    
    // Move assign
    batch1 = std::move(batch2);
    
    REQUIRE(batch1.get_batch_id() == 20);
    REQUIRE(batch1.get_current_tier() == Tier::HOST);
    REQUIRE(batch2.get_batch_id() == 0); // Moved-from state
}

// Test self-assignment (move)
TEST_CASE("data_batch Self Move Assignment", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(100, std::move(data));
    
    // Self-assignment should not crash
    batch = std::move(batch);
    
    REQUIRE(batch.get_batch_id() == 100);
    REQUIRE(batch.get_current_tier() == Tier::GPU);
}

// Test view reference counting - increment and decrement
TEST_CASE("data_batch View Reference Counting", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    REQUIRE(batch.get_view_count() == 0);
    
    // Increment view count
    batch.increment_view_ref_count();
    REQUIRE(batch.get_view_count() == 1);
    
    batch.increment_view_ref_count();
    REQUIRE(batch.get_view_count() == 2);
    
    batch.increment_view_ref_count();
    REQUIRE(batch.get_view_count() == 3);
    
    // Decrement view count
    batch.decrement_view_ref_count();
    REQUIRE(batch.get_view_count() == 2);
    
    batch.decrement_view_ref_count();
    REQUIRE(batch.get_view_count() == 1);
    
    batch.decrement_view_ref_count();
    // batch should be deleted
}

// Test pin reference counting - increment and decrement
TEST_CASE("data_batch Pin Reference Counting", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    REQUIRE(batch.get_pin_count() == 0);
    
    // Increment pin count
    batch.increment_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 1);
    
    batch.increment_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 2);
    
    batch.increment_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 3);
    
    // Decrement pin count
    batch.decrement_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 2);
    
    batch.decrement_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 1);
    
    batch.decrement_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 0);
}

// Test increment pin count throws when not in GPU tier
TEST_CASE("data_batch increment_pin_ref_count GPU Tier Validation", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::HOST, 1024);
    data_batch batch(1, std::move(data));
    
    // Should throw because data is not in GPU tier
    REQUIRE_THROWS_AS(batch.increment_pin_ref_count(), std::runtime_error);
    REQUIRE(batch.get_pin_count() == 0); // Count should not change
}

// Test decrement pin count throws when not in GPU tier
TEST_CASE("data_batch decrement_pin_ref_count GPU Tier Validation", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    // First increment while in GPU tier
    batch.increment_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 1);
    
    // Decrement while in GPU tier should work
    batch.decrement_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 0);
}

// Test that view count operations work regardless of tier
TEST_CASE("data_batch View Count No Tier Validation", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::HOST, 1024);
    data_batch batch(1, std::move(data));
    
    // View count should work even when not in GPU tier
    batch.increment_view_ref_count();
    REQUIRE(batch.get_view_count() == 1);
    
    batch.decrement_view_ref_count();
    REQUIRE(batch.get_view_count() == 0);
}

// Test multiple batches with different IDs
TEST_CASE("Multiple data_batch Instances", "[data_batch]") {
    std::vector<data_batch> batches;
    
    for (uint64_t i = 0; i < 10; ++i) {
        auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024 * (i + 1));
        batches.emplace_back(i, std::move(data));
    }
    
    // Verify all batches have correct IDs and tiers
    for (uint64_t i = 0; i < 10; ++i) {
        REQUIRE(batches[i].get_batch_id() == i);
        REQUIRE(batches[i].get_current_tier() == Tier::GPU);
        REQUIRE(batches[i].get_view_count() == 0);
        REQUIRE(batches[i].get_pin_count() == 0);
    }
}

// Test get_current_tier delegates to idata_representation
TEST_CASE("data_batch get_current_tier Delegation", "[data_batch]") {
    // Test GPU tier
    {
        auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
        data_batch batch(1, std::move(data));
        REQUIRE(batch.get_current_tier() == Tier::GPU);
    }
    
    // Test HOST tier
    {
        auto data = sirius::make_unique<mock_data_representation>(Tier::HOST, 1024);
        data_batch batch(2, std::move(data));
        REQUIRE(batch.get_current_tier() == Tier::HOST);
    }
    
    // Test DISK tier
    {
        auto data = sirius::make_unique<mock_data_representation>(Tier::DISK, 1024);
        data_batch batch(3, std::move(data));
        REQUIRE(batch.get_current_tier() == Tier::DISK);
    }
}

// Test thread-safe view reference counting
TEST_CASE("data_batch Thread-Safe View Reference Counting", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    constexpr int num_threads = 10;
    constexpr int increments_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch threads to increment view count
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&batch]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                batch.increment_view_ref_count();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count
    REQUIRE(batch.get_view_count() == num_threads * increments_per_thread);
    
    threads.clear();
    
    // Launch threads to decrement view count
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&batch]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                batch.decrement_view_ref_count();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count is back to zero
    REQUIRE(batch.get_view_count() == 0);
}

// Test thread-safe pin reference counting
TEST_CASE("data_batch Thread-Safe Pin Reference Counting", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    constexpr int num_threads = 10;
    constexpr int increments_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch threads to increment pin count
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&batch]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                batch.increment_pin_ref_count();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count
    REQUIRE(batch.get_pin_count() == num_threads * increments_per_thread);
    
    threads.clear();
    
    // Launch threads to decrement pin count
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&batch]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                batch.decrement_pin_ref_count();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count is back to zero
    REQUIRE(batch.get_pin_count() == 0);
}

// Test concurrent view increment and decrement
TEST_CASE("data_batch Concurrent View Increment and Decrement", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    // Pre-increment to avoid hitting zero during concurrent operations
    for (int i = 0; i < 1000; ++i) {
        batch.increment_view_ref_count();
    }
    
    constexpr int num_threads = 5;
    constexpr int operations_per_thread = 100;
    
    std::vector<std::thread> inc_threads;
    std::vector<std::thread> dec_threads;
    
    // Launch incrementing threads
    for (int i = 0; i < num_threads; ++i) {
        inc_threads.emplace_back([&batch]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                batch.increment_view_ref_count();
            }
        });
    }
    
    // Launch decrementing threads
    for (int i = 0; i < num_threads; ++i) {
        dec_threads.emplace_back([&batch]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                batch.decrement_view_ref_count();
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
    REQUIRE(batch.get_view_count() == 1000);
}

// Test concurrent pin increment and decrement
TEST_CASE("data_batch Concurrent Pin Increment and Decrement", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    // Pre-increment to avoid hitting zero during concurrent operations
    for (int i = 0; i < 1000; ++i) {
        batch.increment_pin_ref_count();
    }
    
    constexpr int num_threads = 5;
    constexpr int operations_per_thread = 100;
    
    std::vector<std::thread> inc_threads;
    std::vector<std::thread> dec_threads;
    
    // Launch incrementing threads
    for (int i = 0; i < num_threads; ++i) {
        inc_threads.emplace_back([&batch]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                batch.increment_pin_ref_count();
            }
        });
    }
    
    // Launch decrementing threads
    for (int i = 0; i < num_threads; ++i) {
        dec_threads.emplace_back([&batch]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                batch.decrement_pin_ref_count();
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
    REQUIRE(batch.get_pin_count() == 1000);
}

// Test batch ID uniqueness in practice
TEST_CASE("data_batch Unique IDs", "[data_batch]") {
    std::vector<uint64_t> batch_ids = {0, 1, 100, 999, 1000, 9999, 
                                        UINT64_MAX - 1, UINT64_MAX};
    
    std::vector<data_batch> batches;
    
    for (auto id : batch_ids) {
        auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
        batches.emplace_back(id, std::move(data));
    }
    
    // Verify each batch has the correct ID
    for (size_t i = 0; i < batch_ids.size(); ++i) {
        REQUIRE(batches[i].get_batch_id() == batch_ids[i]);
    }
}

// Test edge case: zero view count operations
TEST_CASE("data_batch Zero View Count", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    // Starting view count should be zero
    REQUIRE(batch.get_view_count() == 0);
    
    // Increment from zero
    batch.increment_view_ref_count();
    REQUIRE(batch.get_view_count() == 1);
    
    // Decrement back to zero
    batch.decrement_view_ref_count();
    REQUIRE(batch.get_view_count() == 0);
    
    // Can increment again from zero
    batch.increment_view_ref_count();
    REQUIRE(batch.get_view_count() == 1);
}

// Test edge case: zero pin count operations
TEST_CASE("data_batch Zero Pin Count", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    // Starting pin count should be zero
    REQUIRE(batch.get_pin_count() == 0);
    
    // Increment from zero
    batch.increment_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 1);
    
    // Decrement back to zero
    batch.decrement_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 0);
    
    // Can increment again from zero
    batch.increment_pin_ref_count();
    REQUIRE(batch.get_pin_count() == 1);
}

// Test with different data sizes
TEST_CASE("data_batch With Different Data Sizes", "[data_batch]") {
    std::vector<size_t> sizes = {0, 1, 1024, 1024 * 1024, 1024 * 1024 * 100};
    
    for (size_t size : sizes) {
        auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, size);
        auto* data_ptr = data.get();
        data_batch batch(1, std::move(data));
        
        // Verify the data representation is accessible through the batch
        REQUIRE(batch.get_current_tier() == Tier::GPU);
        REQUIRE(data_ptr->get_size_in_bytes() == size);
    }
}

// Test that move operations require zero reference counts
TEST_CASE("data_batch Move Requires Zero Reference Counts", "[data_batch]") {
    // Test that moving with active views throws
    {
        auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
        data_batch batch1(1, std::move(data));
        batch1.increment_view_ref_count();
        
        REQUIRE_THROWS_AS([&]() { data_batch batch2(std::move(batch1)); }(), std::runtime_error);
    }
    
    // Test that moving with active pins throws
    {
        auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
        data_batch batch1(1, std::move(data));
        batch1.increment_pin_ref_count();
        
        REQUIRE_THROWS_AS([&]() { data_batch batch2(std::move(batch1)); }(), std::runtime_error);
    }
    
    // Test that moving with both active views and pins throws
    {
        auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
        data_batch batch1(1, std::move(data));
        batch1.increment_view_ref_count();
        batch1.increment_pin_ref_count();
        
        REQUIRE_THROWS_AS([&]() { data_batch batch2(std::move(batch1)); }(), std::runtime_error);
    }
    
    // Test that moving with zero counts succeeds
    {
        auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
        data_batch batch1(1, std::move(data));
        
        REQUIRE(batch1.get_view_count() == 0);
        REQUIRE(batch1.get_pin_count() == 0);
        
        data_batch batch2(std::move(batch1));
        
        REQUIRE(batch2.get_view_count() == 0);
        REQUIRE(batch2.get_pin_count() == 0);
        REQUIRE(batch2.get_batch_id() == 1);
    }
}

// Test multiple rapid view count increment/decrement cycles
TEST_CASE("data_batch Rapid View Count Cycles", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    // Perform many cycles of increment and decrement
    for (int cycle = 0; cycle < 100; ++cycle) {
        for (int i = 0; i < 10; ++i) {
            batch.increment_view_ref_count();
        }
        REQUIRE(batch.get_view_count() == 10);
        
        for (int i = 0; i < 10; ++i) {
            batch.decrement_view_ref_count();
        }
        REQUIRE(batch.get_view_count() == 0);
    }
    
    // Final state should be zero
    REQUIRE(batch.get_view_count() == 0);
}

// Test multiple rapid pin count increment/decrement cycles
TEST_CASE("data_batch Rapid Pin Count Cycles", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    // Perform many cycles of increment and decrement
    for (int cycle = 0; cycle < 100; ++cycle) {
        for (int i = 0; i < 10; ++i) {
            batch.increment_pin_ref_count();
        }
        REQUIRE(batch.get_pin_count() == 10);
        
        for (int i = 0; i < 10; ++i) {
            batch.decrement_pin_ref_count();
        }
        REQUIRE(batch.get_pin_count() == 0);
    }
    
    // Final state should be zero
    REQUIRE(batch.get_pin_count() == 0);
}

// Test independent view and pin counts
TEST_CASE("data_batch Independent View and Pin Counts", "[data_batch]") {
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    data_batch batch(1, std::move(data));
    
    // Increment view count
    batch.increment_view_ref_count();
    batch.increment_view_ref_count();
    batch.increment_view_ref_count();
    
    // Increment pin count
    batch.increment_pin_ref_count();
    batch.increment_pin_ref_count();
    
    // Verify counts are independent
    REQUIRE(batch.get_view_count() == 3);
    REQUIRE(batch.get_pin_count() == 2);
    
    // Decrement view count
    batch.decrement_view_ref_count();
    REQUIRE(batch.get_view_count() == 2);
    REQUIRE(batch.get_pin_count() == 2); // Pin count unchanged
    
    // Decrement pin count
    batch.decrement_pin_ref_count();
    REQUIRE(batch.get_view_count() == 2); // View count unchanged
    REQUIRE(batch.get_pin_count() == 1);
}

