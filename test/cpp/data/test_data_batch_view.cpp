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
#include "data/data_batch_view.hpp"
#include "data/data_batch.hpp"
#include "data/data_repository_manager.hpp"
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

// Helper function to create a batch with proper manager setup
// Returns pair of (batch_id, batch_pointer)
// The batch is stored in the manager and will be auto-deleted when views reach 0
std::pair<uint64_t, data_batch*> create_managed_batch(
    data_repository_manager& manager,
    Tier tier = Tier::GPU,
    size_t size = 1024) {
    
    auto data = sirius::make_unique<mock_data_representation>(tier, size);
    uint64_t batch_id = manager.get_next_data_batch_id();
    auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
    auto* batch_ptr = batch.get();
    
    // Store in manager with empty pipelines (no views created)
    sirius::vector<size_t> empty_pipelines;
    manager.add_new_data_batch(std::move(batch), empty_pipelines);
    
    return {batch_id, batch_ptr};
}

// Helper for old-style tests that need a batch with manager but manual lifecycle
// Returns tuple of (manager, batch_id, batch_pointer)
// Note: The manager must outlive the batch and any views
struct managed_batch_holder {
    data_repository_manager manager;
    uint64_t batch_id;
    data_batch* batch_ptr;
    
    managed_batch_holder(Tier tier = Tier::GPU, size_t size = 1024) {
        auto data = sirius::make_unique<mock_data_representation>(tier, size);
        batch_id = manager.get_next_data_batch_id();
        auto batch = sirius::make_unique<data_batch>(batch_id, manager, std::move(data));
        batch_ptr = batch.get();
        
        sirius::vector<size_t> empty_pipelines;
        manager.add_new_data_batch(std::move(batch), empty_pipelines);
    }
};

// Test basic construction
TEST_CASE("data_batch_view Construction", "[data_batch_view]") {
    managed_batch_holder holder(Tier::GPU, 2048);
    auto* batch = holder.batch_ptr;
    
    REQUIRE(batch->get_view_count() == 0);
    
    // Prevent deletion by keeping an extra ref count
    batch->increment_view_ref_count();
    
    {
        // Create a view
        data_batch_view view(batch);
        
        // View count should be incremented
        REQUIRE(batch->get_view_count() == 2);
        REQUIRE(batch->get_pin_count() == 0);
    }
    
    // Clean up - decrement the extra ref count, batch will be deleted
    batch->decrement_view_ref_count();
}

// Test copy constructor
TEST_CASE("data_batch_view Copy Constructor", "[data_batch_view]") {
    managed_batch_holder holder(Tier::GPU, 2048);
    auto* batch = holder.batch_ptr;
    
    // Prevent deletion by keeping an extra ref count
    batch->increment_view_ref_count();
    
    {
        data_batch_view view1(batch);
        REQUIRE(batch->get_view_count() == 2);
        
        // Copy construct
        data_batch_view view2(view1);
        
        // View count should be incremented again
        REQUIRE(batch->get_view_count() == 3);
        REQUIRE(batch->get_pin_count() == 0);
    }
    
    // Clean up - decrement triggers deletion
    batch->decrement_view_ref_count();
}

// Test copy assignment
TEST_CASE("data_batch_view Copy Assignment", "[data_batch_view]") {
    managed_batch_holder holder1(Tier::GPU, 512);
    managed_batch_holder holder2(Tier::GPU, 1024);
    auto* batch1 = holder1.batch_ptr;
    auto* batch2 = holder2.batch_ptr;
    
    // Prevent deletion by keeping an extra ref count
    batch1->increment_view_ref_count();
    batch2->increment_view_ref_count();
    
    {
        data_batch_view view1(batch1);
        data_batch_view view2(batch2);
        
        REQUIRE(batch1->get_view_count() == 2);
        REQUIRE(batch2->get_view_count() == 2);
        
        // Copy assign - this decrements batch1's count and increments batch2's count
        view1 = view2;
        
        REQUIRE(batch1->get_view_count() == 1);
        REQUIRE(batch2->get_view_count() == 3);
    }
    
    // Clean up - both batches will be deleted
    batch1->decrement_view_ref_count();
    batch2->decrement_view_ref_count();
}

// Test self-assignment
TEST_CASE("data_batch_view Self Copy Assignment", "[data_batch_view]") {
    managed_batch_holder holder;
    auto* batch = holder.batch_ptr;
    
    // Prevent deletion by keeping an extra ref count
    batch->increment_view_ref_count();
    
    {
        data_batch_view view(batch);
        REQUIRE(batch->get_view_count() == 2);
        
        // Self-assignment should not change count
        view = view;
        
        REQUIRE(batch->get_view_count() == 2);
    }
    
    // Clean up
    batch->decrement_view_ref_count();
}

// Test destructor decrements view count
TEST_CASE("data_batch_view Destructor Decrements View Count", "[data_batch_view]") {
    managed_batch_holder holder;
    auto* batch = holder.batch_ptr;
    
    // Increment view count so batch doesn't self-delete
    batch->increment_view_ref_count();
    
    REQUIRE(batch->get_view_count() == 1);
    
    {
        data_batch_view view(batch);
        REQUIRE(batch->get_view_count() == 2);
    } // view destroyed here
    
    // View count should be decremented
    REQUIRE(batch->get_view_count() == 1);
    
    // Clean up
    batch->decrement_view_ref_count();
}

// Test Pin and Unpin operations
TEST_CASE("data_batch_view Pin and Unpin", "[data_batch_view]") {
    managed_batch_holder holder;
    auto* batch = holder.batch_ptr;
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    data_batch_view view(batch);
    
    REQUIRE(batch->get_pin_count() == 0);
    
    // Pin the view
    view.pin();
    REQUIRE(batch->get_pin_count() == 1);
    
    // Unpin the view
    view.unpin();
    REQUIRE(batch->get_pin_count() == 0);
    
    // Clean up
    batch->decrement_view_ref_count();
}

// Test Pin throws when already pinned
TEST_CASE("data_batch_view Pin Throws When Already Pinned", "[data_batch_view]") {
    managed_batch_holder holder;
    auto* batch = holder.batch_ptr;
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    data_batch_view view(batch);
    
    // Pin the view
    view.pin();
    REQUIRE(batch->get_pin_count() == 1);
    
    // Try to pin again - should throw
    REQUIRE_THROWS_AS(view.pin(), std::runtime_error);
    REQUIRE(batch->get_pin_count() == 1); // Count should not change
    
    // Unpin and clean up
    view.unpin();
    batch->decrement_view_ref_count();
}

// Test Unpin throws when not pinned
TEST_CASE("data_batch_view Unpin Throws When Not Pinned", "[data_batch_view]") {
    managed_batch_holder holder;
    auto* batch = holder.batch_ptr;
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    data_batch_view view(batch);
    
    REQUIRE(batch->get_pin_count() == 0);
    
    // Try to unpin when not pinned - should throw
    REQUIRE_THROWS_AS(view.unpin(), std::runtime_error);
    REQUIRE(batch->get_pin_count() == 0);
    
    // Clean up
    batch->decrement_view_ref_count();
}

// Test Pin throws when not in GPU tier
TEST_CASE("data_batch_view Pin Throws When Not In GPU Tier", "[data_batch_view]") {
    managed_batch_holder holder(Tier::HOST);
    auto* batch = holder.batch_ptr;
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    data_batch_view view(batch);
    
    // Should throw because data is not in GPU tier
    REQUIRE_THROWS_AS(view.pin(), std::runtime_error);
    REQUIRE(batch->get_pin_count() == 0);
    
    // Clean up
    batch->decrement_view_ref_count();
}

// Test destructor auto-unpins
TEST_CASE("data_batch_view Destructor Auto Unpins", "[data_batch_view]") {
    managed_batch_holder holder;
    auto* batch = holder.batch_ptr;
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    {
        data_batch_view view(batch);
        view.pin();
        REQUIRE(batch->get_pin_count() == 1);
        REQUIRE(batch->get_view_count() == 2);
    } // view destroyed here, should auto-unpin
    
    // Pin count should be back to zero
    REQUIRE(batch->get_pin_count() == 0);
    REQUIRE(batch->get_view_count() == 1);
    
    // Clean up
    batch->decrement_view_ref_count();
}

// Test multiple views on same batch
TEST_CASE("Multiple data_batch_views on Same Batch", "[data_batch_view]") {
    managed_batch_holder holder;
    auto* batch = holder.batch_ptr;
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    std::vector<std::unique_ptr<data_batch_view>> views;
    
    for (int i = 0; i < 10; ++i) {
        views.push_back(std::make_unique<data_batch_view>(batch));
    }
    
    // Should have 11 views total (1 from increment + 10 from views)
    REQUIRE(batch->get_view_count() == 11);
    
    // Pin half of them
    for (int i = 0; i < 5; ++i) {
        views[i]->pin();
    }
    REQUIRE(batch->get_pin_count() == 5);
    
    // Unpin them
    for (int i = 0; i < 5; ++i) {
        views[i]->unpin();
    }
    REQUIRE(batch->get_pin_count() == 0);
    
    // Clear all views
    views.clear();
    
    // Should be back to 1
    REQUIRE(batch->get_view_count() == 1);
    
    // Clean up
    batch->decrement_view_ref_count();
}

// Test view count is independent of pin count
TEST_CASE("data_batch_view Independent View and Pin Counts", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    {
        data_batch_view view(batch);
        
        REQUIRE(batch->get_view_count() == 2);
        REQUIRE(batch->get_pin_count() == 0);
        
        // Pin the view
        view.pin();
        
        // View count should be unchanged, pin count should increase
        REQUIRE(batch->get_view_count() == 2);
        REQUIRE(batch->get_pin_count() == 1);
        
        // Unpin
        view.unpin();
        REQUIRE(batch->get_view_count() == 2);
        REQUIRE(batch->get_pin_count() == 0);
    } // view destroyed here
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test multiple pin/unpin cycles
TEST_CASE("data_batch_view Multiple Pin Unpin Cycles", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    {
        data_batch_view view(batch);
        
        // Perform many cycles
        for (int i = 0; i < 100; ++i) {
            view.pin();
            REQUIRE(batch->get_pin_count() == 1);
            view.unpin();
            REQUIRE(batch->get_pin_count() == 0);
        }
    } // view destroyed here
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test thread-safe view count operations
TEST_CASE("data_batch_view Thread-Safe View Count", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    constexpr int num_threads = 10;
    constexpr int views_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<std::unique_ptr<data_batch_view>>> thread_views(num_threads);
    
    // Launch threads to create views
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < views_per_thread; ++j) {
                thread_views[i].push_back(std::make_unique<data_batch_view>(batch));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count (1 initial + num_threads * views_per_thread)
    REQUIRE(batch->get_view_count() == 1 + num_threads * views_per_thread);
    
    // Clear all views
    for (auto& views : thread_views) {
        views.clear();
    }
    
    // Verify count is back to 1
    REQUIRE(batch->get_view_count() == 1);
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test thread-safe pin operations
TEST_CASE("data_batch_view Thread-Safe Pin Operations", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    constexpr int num_views = 100;
    std::vector<std::unique_ptr<data_batch_view>> views;
    
    // Create many views
    for (int i = 0; i < num_views; ++i) {
        views.push_back(std::make_unique<data_batch_view>(batch));
    }
    
    constexpr int num_threads = 10;
    
    std::vector<std::thread> threads;
    
    // Launch threads to pin views
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            int start = i * (num_views / num_threads);
            int end = (i + 1) * (num_views / num_threads);
            for (int j = start; j < end; ++j) {
                views[j]->pin();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final pin count
    REQUIRE(batch->get_pin_count() == num_views);
    
    threads.clear();
    
    // Launch threads to unpin views
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            int start = i * (num_views / num_threads);
            int end = (i + 1) * (num_views / num_threads);
            for (int j = start; j < end; ++j) {
                views[j]->unpin();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final pin count is back to zero
    REQUIRE(batch->get_pin_count() == 0);
    
    // Clear all views
    views.clear();
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test concurrent view creation and destruction
TEST_CASE("data_batch_view Concurrent View Creation and Destruction", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    constexpr int num_threads = 5;
    constexpr int operations_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch threads that create and destroy views
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                // Create a view
                auto view = std::make_unique<data_batch_view>(batch);
                // Immediately destroy it
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All views should be destroyed, count should be back to 1
    REQUIRE(batch->get_view_count() == 1);
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test view copy preserves unpinned state
TEST_CASE("data_batch_view Copy Does Not Copy Pin State", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    {
        data_batch_view view1(batch);
        view1.pin();
        
        REQUIRE(batch->get_view_count() == 2);
        REQUIRE(batch->get_pin_count() == 1);
        
        // Copy construct - pin state is per-view
        data_batch_view view2(view1);
        
        // View count should increase, pin count unchanged (new view is not pinned)
        REQUIRE(batch->get_view_count() == 3);
        REQUIRE(batch->get_pin_count() == 1);
        
        // View2 should not be pinned
        REQUIRE_THROWS_AS(view2.unpin(), std::runtime_error);
        
        // Unpin view1
        view1.unpin();
        REQUIRE(batch->get_pin_count() == 0);
    } // views destroyed here
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test multiple views with independent pin states
TEST_CASE("data_batch_view Multiple Views Independent Pin States", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    {
        data_batch_view view1(batch);
        data_batch_view view2(batch);
        data_batch_view view3(batch);
        
        REQUIRE(batch->get_view_count() == 4);
        REQUIRE(batch->get_pin_count() == 0);
        
        // Pin from different views
        view1.pin();
        view2.pin();
        
        REQUIRE(batch->get_pin_count() == 2);
        
        // Unpin from view1
        view1.unpin();
        REQUIRE(batch->get_pin_count() == 1);
        
        // view2 should still be pinned
        REQUIRE_THROWS_AS(view2.pin(), std::runtime_error);
        
        // view3 should not be pinned
        REQUIRE_THROWS_AS(view3.unpin(), std::runtime_error);
        
        // Pin view3
        view3.pin();
        REQUIRE(batch->get_pin_count() == 2);
        
        // Unpin all
        view2.unpin();
        view3.unpin();
        REQUIRE(batch->get_pin_count() == 0);
    } // views destroyed here
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test rapid view creation and destruction cycles
TEST_CASE("data_batch_view Rapid Creation and Destruction Cycles", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    // Perform many cycles of view creation and destruction
    for (int cycle = 0; cycle < 100; ++cycle) {
        {
            data_batch_view view1(batch);
            data_batch_view view2(batch);
            data_batch_view view3(batch);
            REQUIRE(batch->get_view_count() == 4);
        }
        REQUIRE(batch->get_view_count() == 1);
    }
    
    // Final state should be 1
    REQUIRE(batch->get_view_count() == 1);
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test view with different batch IDs
TEST_CASE("data_batch_view Multiple Batches", "[data_batch_view]") {
    data_repository_manager manager;
    auto data1 = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto data2 = sirius::make_unique<mock_data_representation>(Tier::GPU, 2048);
    auto data3 = sirius::make_unique<mock_data_representation>(Tier::GPU, 4096);
    
    auto* batch1 = new data_batch(1, manager, std::move(data1));
    auto* batch2 = new data_batch(2, manager, std::move(data2));
    auto* batch3 = new data_batch(3, manager, std::move(data3));
    
    // Increment to prevent auto-delete
    batch1->increment_view_ref_count();
    batch2->increment_view_ref_count();
    batch3->increment_view_ref_count();
    
    {
        data_batch_view view1(batch1);
        data_batch_view view2(batch2);
        data_batch_view view3(batch3);
        
        // Each batch should have its own view count
        REQUIRE(batch1->get_view_count() == 2);
        REQUIRE(batch2->get_view_count() == 2);
        REQUIRE(batch3->get_view_count() == 2);
        
        REQUIRE(batch1->get_batch_id() == 1);
        REQUIRE(batch2->get_batch_id() == 2);
        REQUIRE(batch3->get_batch_id() == 3);
        
        // Pin each view independently
        view1.pin();
        view2.pin();
        view3.pin();
        
        REQUIRE(batch1->get_pin_count() == 1);
        REQUIRE(batch2->get_pin_count() == 1);
        REQUIRE(batch3->get_pin_count() == 1);
        
        // Unpin
        view1.unpin();
        view2.unpin();
        view3.unpin();
    } // views destroyed here
    
    // Clean up
    batch1->decrement_view_ref_count();
    delete batch1;
    batch2->decrement_view_ref_count();
    delete batch2;
    batch3->decrement_view_ref_count();
    delete batch3;
}

// Test stress test with many views and pins
TEST_CASE("data_batch_view Stress Test", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    constexpr int num_views = 1000;
    std::vector<std::unique_ptr<data_batch_view>> views;
    
    // Create many views
    for (int i = 0; i < num_views; ++i) {
        views.push_back(std::make_unique<data_batch_view>(batch));
    }
    
    REQUIRE(batch->get_view_count() == num_views + 1);
    
    // Pin all of them
    for (int i = 0; i < num_views; ++i) {
        views[i]->pin();
    }
    
    REQUIRE(batch->get_pin_count() == num_views);
    
    // Unpin all
    for (int i = 0; i < num_views; ++i) {
        views[i]->unpin();
    }
    
    REQUIRE(batch->get_pin_count() == 0);
    
    // Destroy all views
    views.clear();
    
    REQUIRE(batch->get_view_count() == 1);
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test view lifetime with mixed operations
TEST_CASE("data_batch_view Mixed Lifetime Operations", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    std::vector<std::unique_ptr<data_batch_view>> views;
    
    // Create some views
    for (int i = 0; i < 5; ++i) {
        views.push_back(std::make_unique<data_batch_view>(batch));
    }
    REQUIRE(batch->get_view_count() == 6);
    
    // Pin some
    views[0]->pin();
    views[2]->pin();
    views[4]->pin();
    REQUIRE(batch->get_pin_count() == 3);
    
    // Destroy some views (including pinned ones - should auto-unpin)
    views.erase(views.begin() + 2);
    REQUIRE(batch->get_view_count() == 5);
    REQUIRE(batch->get_pin_count() == 2); // view[2] was unpinned on destruction
    
    // Create more views
    for (int i = 0; i < 3; ++i) {
        views.push_back(std::make_unique<data_batch_view>(batch));
    }
    REQUIRE(batch->get_view_count() == 8);
    
    // Unpin remaining
    views[0]->unpin();
    views[3]->unpin(); // This was views[4] before erase
    REQUIRE(batch->get_pin_count() == 0);
    
    // Destroy all
    views.clear();
    REQUIRE(batch->get_view_count() == 1);
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test destructor with pinned view
TEST_CASE("data_batch_view Destructor With Pinned View", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    {
        data_batch_view view1(batch);
        data_batch_view view2(batch);
        
        view1.pin();
        view2.pin();
        
        REQUIRE(batch->get_pin_count() == 2);
        REQUIRE(batch->get_view_count() == 3);
        
        // Both views will be destroyed, should auto-unpin
    }
    
    // Both should be unpinned and view count decreased
    REQUIRE(batch->get_pin_count() == 0);
    REQUIRE(batch->get_view_count() == 1);
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test edge case: zero to non-zero pin transitions
TEST_CASE("data_batch_view Zero to Non-Zero Pin Transitions", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    {
        data_batch_view view(batch);
        
        // Start with zero pin count
        REQUIRE(batch->get_pin_count() == 0);
        
        // Transition from 0 to 1
        view.pin();
        REQUIRE(batch->get_pin_count() == 1);
        
        // Transition from 1 to 0
        view.unpin();
        REQUIRE(batch->get_pin_count() == 0);
        
        // Repeat multiple times
        for (int i = 0; i < 10; ++i) {
            view.pin();
            REQUIRE(batch->get_pin_count() == 1);
            view.unpin();
            REQUIRE(batch->get_pin_count() == 0);
        }
    } // view destroyed here
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// Test rapid pin/unpin with multiple views
TEST_CASE("data_batch_view Rapid Pin Unpin Multiple Views", "[data_batch_view]") {
    data_repository_manager manager;
    auto data = sirius::make_unique<mock_data_representation>(Tier::GPU, 1024);
    auto* batch = new data_batch(1, manager, std::move(data));
    
    // Increment to prevent auto-delete
    batch->increment_view_ref_count();
    
    constexpr int num_views = 10;
    std::vector<std::unique_ptr<data_batch_view>> views;
    
    for (int i = 0; i < num_views; ++i) {
        views.push_back(std::make_unique<data_batch_view>(batch));
    }
    
    // Rapid pin/unpin cycles
    for (int cycle = 0; cycle < 50; ++cycle) {
        // Pin all
        for (auto& view : views) {
            view->pin();
        }
        REQUIRE(batch->get_pin_count() == num_views);
        
        // Unpin all
        for (auto& view : views) {
            view->unpin();
        }
        REQUIRE(batch->get_pin_count() == 0);
    }
    
    // Clear views
    views.clear();
    
    // Clean up
    batch->decrement_view_ref_count();
    delete batch;
}

// =============================================================================
// Tests for automatic batch deletion through data_repository_manager
// =============================================================================

// Test that batch is auto-deleted when last view is destroyed
TEST_CASE("data_batch_view Auto Delete When Last View Destroyed", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    auto [batch_id, batch] = create_managed_batch(manager);
    
    REQUIRE(batch->get_view_count() == 0);
    
    {
        // Create a single view - this increments view count to 1
        data_batch_view view(batch);
        REQUIRE(batch->get_view_count() == 1);
    } 
    // View goes out of scope, view count goes 1 → 0
    // Batch should be auto-deleted by the manager via delete_data_batch()
    
    // NOTE: batch pointer is now dangling - we cannot safely access it
    // The test passes if no crash occurs (batch was properly cleaned up)
}

// Test that batch persists while multiple views exist
TEST_CASE("data_batch_view Batch Persists With Multiple Views", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    auto [batch_id, batch] = create_managed_batch(manager);
    
    {
        data_batch_view view1(batch);
        REQUIRE(batch->get_view_count() == 1);
        
        {
            data_batch_view view2(batch);
            REQUIRE(batch->get_view_count() == 2);
            
            data_batch_view view3(batch);
            REQUIRE(batch->get_view_count() == 3);
        } // view2 and view3 destroyed, count goes 3 → 2 → 1
        
        // Batch still exists because view1 is alive
        REQUIRE(batch->get_view_count() == 1);
    } 
    // Last view destroyed, batch should be deleted
}

// Test deletion with pinned views
TEST_CASE("data_batch_view Auto Delete With Pinned View", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    auto [batch_id, batch] = create_managed_batch(manager);
    
    {
        data_batch_view view(batch);
        view.pin();
        
        REQUIRE(batch->get_pin_count() == 1);
        REQUIRE(batch->get_view_count() == 1);
        
        // View destructor will auto-unpin then decrement view count
    }
    // Batch should be auto-deleted (view was unpinned then view count → 0)
}

// Test deletion with multiple pinned views
TEST_CASE("data_batch_view Auto Delete With Multiple Pinned Views", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    auto [batch_id, batch] = create_managed_batch(manager);
    
    {
        data_batch_view view1(batch);
        data_batch_view view2(batch);
        data_batch_view view3(batch);
        
        view1.pin();
        view2.pin();
        // view3 remains unpinned
        
        REQUIRE(batch->get_pin_count() == 2);
        REQUIRE(batch->get_view_count() == 3);
    }
    // All views destroyed, pinned views auto-unpinned, batch deleted
}

// Test that copy assignment doesn't cause premature deletion
TEST_CASE("data_batch_view Copy Assignment No Premature Deletion", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    auto [batch_id1, batch1] = create_managed_batch(manager);
    auto [batch_id2, batch2] = create_managed_batch(manager);
    
    {
        data_batch_view view1(batch1);
        REQUIRE(batch1->get_view_count() == 1);
        
        {
            data_batch_view view2(batch2);
            REQUIRE(batch2->get_view_count() == 1);
            
            // Assign view2 to view1 - this should decrement batch1's count
            // batch1 goes to 0 and is deleted
            view1 = view2;
            
            // batch2 now has 2 views
            REQUIRE(batch2->get_view_count() == 2);
        } // view2 destroyed, batch2 count goes 2 → 1
        
        // batch2 still exists
        REQUIRE(batch2->get_view_count() == 1);
    } // view1 destroyed, batch2 count goes 1 → 0, batch2 deleted
}

// Test concurrent view creation and deletion with manager
TEST_CASE("data_batch_view Concurrent Auto Deletion", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    auto [batch_id, batch] = create_managed_batch(manager);
    
    // Create an extra view to prevent deletion during concurrent access
    auto persistent_view = std::make_unique<data_batch_view>(batch);
    REQUIRE(batch->get_view_count() == 1);
    
    constexpr int num_threads = 5;
    constexpr int operations_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch threads that create and destroy views rapidly
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                auto view = std::make_unique<data_batch_view>(batch);
                // View immediately destroyed
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Only persistent_view remains
    REQUIRE(batch->get_view_count() == 1);
    
    // Destroy the persistent view - this triggers batch deletion
    persistent_view.reset();
}

// Test that exceptions during pin don't prevent deletion
TEST_CASE("data_batch_view Auto Delete After Pin Exception", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    // Create batch in HOST tier (can't be pinned)
    auto [batch_id, batch] = create_managed_batch(manager, Tier::HOST);
    
    {
        data_batch_view view(batch);
        REQUIRE(batch->get_view_count() == 1);
        
        // Try to pin - this will throw
        REQUIRE_THROWS_AS(view.pin(), std::runtime_error);
        
        // View should still be unpinned
        REQUIRE(batch->get_pin_count() == 0);
    }
    // View destroyed, batch should be auto-deleted despite the pin exception
}

// Test multiple batches deleted independently
TEST_CASE("data_batch_view Multiple Batches Auto Delete Independently", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    auto [batch_id1, batch1] = create_managed_batch(manager);
    auto [batch_id2, batch2] = create_managed_batch(manager);
    auto [batch_id3, batch3] = create_managed_batch(manager);
    
    data_batch_view view1(batch1);
    data_batch_view view2(batch2);
    data_batch_view view3(batch3);
    
    REQUIRE(batch1->get_view_count() == 1);
    REQUIRE(batch2->get_view_count() == 1);
    REQUIRE(batch3->get_view_count() == 1);
    
    // Create extra view for batch2 to keep it alive longer
    auto extra_view2 = std::make_unique<data_batch_view>(batch2);
    REQUIRE(batch2->get_view_count() == 2);
    
    // Destroy view1 - batch1 should be deleted
    view1.~data_batch_view();
    new (&view1) data_batch_view(batch3); // Reconstruct as view of batch3
    
    // batch2 still has 2 views
    REQUIRE(batch2->get_view_count() == 2);
    // batch3 now has 2 views
    REQUIRE(batch3->get_view_count() == 2);
    
    // Destroy extra_view2 - batch2 goes to 1 view
    extra_view2.reset();
    REQUIRE(batch2->get_view_count() == 1);
    
    // Let remaining views go out of scope
    // batch2 and batch3 will be auto-deleted
}

// Test rapid batch creation and deletion
TEST_CASE("data_batch_view Rapid Batch Creation And Deletion", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    // Create and delete many batches rapidly
    for (int i = 0; i < 100; ++i) {
        auto [batch_id, batch] = create_managed_batch(manager);
        
        {
            data_batch_view view(batch);
            REQUIRE(batch->get_view_count() == 1);
            
            if (i % 3 == 0) {
                view.pin();
                REQUIRE(batch->get_pin_count() == 1);
            }
        } // View destroyed, batch auto-deleted
    }
}

// Test that view count transitions correctly during deletion
TEST_CASE("data_batch_view View Count Transitions During Deletion", "[data_batch_view][deletion]") {
    data_repository_manager manager;
    
    auto [batch_id, batch] = create_managed_batch(manager);
    
    // Create multiple views
    std::vector<std::unique_ptr<data_batch_view>> views;
    for (int i = 0; i < 10; ++i) {
        views.push_back(std::make_unique<data_batch_view>(batch));
    }
    
    REQUIRE(batch->get_view_count() == 10);
    
    // Destroy views one by one
    for (int i = 9; i >= 1; --i) {
        views.pop_back();
        REQUIRE(batch->get_view_count() == i);
    }
    
    // Destroy last view - batch should be auto-deleted
    views.pop_back();
    REQUIRE(views.empty());
}

