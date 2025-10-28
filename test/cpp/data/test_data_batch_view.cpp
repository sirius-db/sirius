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
TEST_CASE("DataBatchView Construction", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 2048);
    auto* batch = new DataBatch(1, std::move(data));
    
    REQUIRE(batch->GetViewCount() == 0);
    
    // Create a view
    DataBatchView view(batch);
    
    // View count should be incremented
    REQUIRE(batch->GetViewCount() == 1);
    REQUIRE(batch->GetPinCount() == 0);
}

// Test copy constructor
TEST_CASE("DataBatchView Copy Constructor", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 2048);
    auto* batch = new DataBatch(42, std::move(data));
    
    DataBatchView view1(batch);
    REQUIRE(batch->GetViewCount() == 1);
    
    // Copy construct
    DataBatchView view2(view1);
    
    // View count should be incremented again
    REQUIRE(batch->GetViewCount() == 2);
    REQUIRE(batch->GetPinCount() == 0);
}

// Test copy assignment
TEST_CASE("DataBatchView Copy Assignment", "[data_batch_view]") {
    auto data1 = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 512);
    auto data2 = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    
    auto* batch1 = new DataBatch(10, std::move(data1));
    auto* batch2 = new DataBatch(20, std::move(data2));
    
    DataBatchView view1(batch1);
    DataBatchView view2(batch2);
    
    REQUIRE(batch1->GetViewCount() == 1);
    REQUIRE(batch2->GetViewCount() == 1);
    
    // Copy assign
    view1 = view2;
    
    REQUIRE(batch2->GetViewCount() == 2);
}

// Test self-assignment
TEST_CASE("DataBatchView Self Copy Assignment", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(100, std::move(data));
    
    DataBatchView view(batch);
    REQUIRE(batch->GetViewCount() == 1);
    
    // Self-assignment should not change count
    view = view;
    
    REQUIRE(batch->GetViewCount() == 1);
}

// Test destructor decrements view count
TEST_CASE("DataBatchView Destructor Decrements View Count", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment view count so batch doesn't self-delete
    batch->IncrementViewRefCount();
    
    REQUIRE(batch->GetViewCount() == 1);
    
    {
        DataBatchView view(batch);
        REQUIRE(batch->GetViewCount() == 2);
    } // view destroyed here
    
    // View count should be decremented
    REQUIRE(batch->GetViewCount() == 1);
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test Pin and Unpin operations
TEST_CASE("DataBatchView Pin and Unpin", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    DataBatchView view(batch);
    
    REQUIRE(batch->GetPinCount() == 0);
    
    // Pin the view
    view.Pin();
    REQUIRE(batch->GetPinCount() == 1);
    
    // Unpin the view
    view.Unpin();
    REQUIRE(batch->GetPinCount() == 0);
    
    // Clean up
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test Pin throws when already pinned
TEST_CASE("DataBatchView Pin Throws When Already Pinned", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    DataBatchView view(batch);
    
    // Pin the view
    view.Pin();
    REQUIRE(batch->GetPinCount() == 1);
    
    // Try to pin again - should throw
    REQUIRE_THROWS_AS(view.Pin(), std::runtime_error);
    REQUIRE(batch->GetPinCount() == 1); // Count should not change
    
    // Unpin and clean up
    view.Unpin();
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test Unpin throws when not pinned
TEST_CASE("DataBatchView Unpin Throws When Not Pinned", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    DataBatchView view(batch);
    
    REQUIRE(batch->GetPinCount() == 0);
    
    // Try to unpin when not pinned - should throw
    REQUIRE_THROWS_AS(view.Unpin(), std::runtime_error);
    REQUIRE(batch->GetPinCount() == 0);
    
    // Clean up
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test Pin throws when not in GPU tier
TEST_CASE("DataBatchView Pin Throws When Not In GPU Tier", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::HOST, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    DataBatchView view(batch);
    
    // Should throw because data is not in GPU tier
    REQUIRE_THROWS_AS(view.Pin(), std::runtime_error);
    REQUIRE(batch->GetPinCount() == 0);
    
    // Clean up
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test destructor auto-unpins
TEST_CASE("DataBatchView Destructor Auto Unpins", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    {
        DataBatchView view(batch);
        view.Pin();
        REQUIRE(batch->GetPinCount() == 1);
        REQUIRE(batch->GetViewCount() == 2);
    } // view destroyed here, should auto-unpin
    
    // Pin count should be back to zero
    REQUIRE(batch->GetPinCount() == 0);
    REQUIRE(batch->GetViewCount() == 1);
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test multiple views on same batch
TEST_CASE("Multiple DataBatchViews on Same Batch", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    std::vector<std::unique_ptr<DataBatchView>> views;
    
    for (int i = 0; i < 10; ++i) {
        views.push_back(std::make_unique<DataBatchView>(batch));
    }
    
    // Should have 11 views total (1 from increment + 10 from views)
    REQUIRE(batch->GetViewCount() == 11);
    
    // Pin half of them
    for (int i = 0; i < 5; ++i) {
        views[i]->Pin();
    }
    REQUIRE(batch->GetPinCount() == 5);
    
    // Unpin them
    for (int i = 0; i < 5; ++i) {
        views[i]->Unpin();
    }
    REQUIRE(batch->GetPinCount() == 0);
    
    // Clear all views
    views.clear();
    
    // Should be back to 1
    REQUIRE(batch->GetViewCount() == 1);
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test view count is independent of pin count
TEST_CASE("DataBatchView Independent View and Pin Counts", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    DataBatchView view(batch);
    
    REQUIRE(batch->GetViewCount() == 2);
    REQUIRE(batch->GetPinCount() == 0);
    
    // Pin the view
    view.Pin();
    
    // View count should be unchanged, pin count should increase
    REQUIRE(batch->GetViewCount() == 2);
    REQUIRE(batch->GetPinCount() == 1);
    
    // Unpin
    view.Unpin();
    REQUIRE(batch->GetViewCount() == 2);
    REQUIRE(batch->GetPinCount() == 0);
    
    // Clean up
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test multiple pin/unpin cycles
TEST_CASE("DataBatchView Multiple Pin Unpin Cycles", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    DataBatchView view(batch);
    
    // Perform many cycles
    for (int i = 0; i < 100; ++i) {
        view.Pin();
        REQUIRE(batch->GetPinCount() == 1);
        view.Unpin();
        REQUIRE(batch->GetPinCount() == 0);
    }
    
    // Clean up
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test thread-safe view count operations
TEST_CASE("DataBatchView Thread-Safe View Count", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    constexpr int num_threads = 10;
    constexpr int views_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<std::unique_ptr<DataBatchView>>> thread_views(num_threads);
    
    // Launch threads to create views
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < views_per_thread; ++j) {
                thread_views[i].push_back(std::make_unique<DataBatchView>(batch));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final count (1 initial + num_threads * views_per_thread)
    REQUIRE(batch->GetViewCount() == 1 + num_threads * views_per_thread);
    
    // Clear all views
    for (auto& views : thread_views) {
        views.clear();
    }
    
    // Verify count is back to 1
    REQUIRE(batch->GetViewCount() == 1);
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test thread-safe pin operations
TEST_CASE("DataBatchView Thread-Safe Pin Operations", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    constexpr int num_views = 100;
    std::vector<std::unique_ptr<DataBatchView>> views;
    
    // Create many views
    for (int i = 0; i < num_views; ++i) {
        views.push_back(std::make_unique<DataBatchView>(batch));
    }
    
    constexpr int num_threads = 10;
    
    std::vector<std::thread> threads;
    
    // Launch threads to pin views
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            int start = i * (num_views / num_threads);
            int end = (i + 1) * (num_views / num_threads);
            for (int j = start; j < end; ++j) {
                views[j]->Pin();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final pin count
    REQUIRE(batch->GetPinCount() == num_views);
    
    threads.clear();
    
    // Launch threads to unpin views
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            int start = i * (num_views / num_threads);
            int end = (i + 1) * (num_views / num_threads);
            for (int j = start; j < end; ++j) {
                views[j]->Unpin();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final pin count is back to zero
    REQUIRE(batch->GetPinCount() == 0);
    
    // Clear all views
    views.clear();
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test concurrent view creation and destruction
TEST_CASE("DataBatchView Concurrent View Creation and Destruction", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    constexpr int num_threads = 5;
    constexpr int operations_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch threads that create and destroy views
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                // Create a view
                auto view = std::make_unique<DataBatchView>(batch);
                // Immediately destroy it
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All views should be destroyed, count should be back to 1
    REQUIRE(batch->GetViewCount() == 1);
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test view copy preserves unpinned state
TEST_CASE("DataBatchView Copy Does Not Copy Pin State", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    DataBatchView view1(batch);
    view1.Pin();
    
    REQUIRE(batch->GetViewCount() == 2);
    REQUIRE(batch->GetPinCount() == 1);
    
    // Copy construct - pin state is per-view
    DataBatchView view2(view1);
    
    // View count should increase, pin count unchanged (new view is not pinned)
    REQUIRE(batch->GetViewCount() == 3);
    REQUIRE(batch->GetPinCount() == 1);
    
    // View2 should not be pinned
    REQUIRE_THROWS_AS(view2.Unpin(), std::runtime_error);
    
    // Unpin view1
    view1.Unpin();
    REQUIRE(batch->GetPinCount() == 0);
    
    // Clean up
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test multiple views with independent pin states
TEST_CASE("DataBatchView Multiple Views Independent Pin States", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    DataBatchView view1(batch);
    DataBatchView view2(batch);
    DataBatchView view3(batch);
    
    REQUIRE(batch->GetViewCount() == 4);
    REQUIRE(batch->GetPinCount() == 0);
    
    // Pin from different views
    view1.Pin();
    view2.Pin();
    
    REQUIRE(batch->GetPinCount() == 2);
    
    // Unpin from view1
    view1.Unpin();
    REQUIRE(batch->GetPinCount() == 1);
    
    // view2 should still be pinned
    REQUIRE_THROWS_AS(view2.Pin(), std::runtime_error);
    
    // view3 should not be pinned
    REQUIRE_THROWS_AS(view3.Unpin(), std::runtime_error);
    
    // Pin view3
    view3.Pin();
    REQUIRE(batch->GetPinCount() == 2);
    
    // Unpin all
    view2.Unpin();
    view3.Unpin();
    REQUIRE(batch->GetPinCount() == 0);
    
    // Clean up
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test rapid view creation and destruction cycles
TEST_CASE("DataBatchView Rapid Creation and Destruction Cycles", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    // Perform many cycles of view creation and destruction
    for (int cycle = 0; cycle < 100; ++cycle) {
        {
            DataBatchView view1(batch);
            DataBatchView view2(batch);
            DataBatchView view3(batch);
            REQUIRE(batch->GetViewCount() == 4);
        }
        REQUIRE(batch->GetViewCount() == 1);
    }
    
    // Final state should be 1
    REQUIRE(batch->GetViewCount() == 1);
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test view with different batch IDs
TEST_CASE("DataBatchView Multiple Batches", "[data_batch_view]") {
    auto data1 = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto data2 = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 2048);
    auto data3 = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 4096);
    
    auto* batch1 = new DataBatch(1, std::move(data1));
    auto* batch2 = new DataBatch(2, std::move(data2));
    auto* batch3 = new DataBatch(3, std::move(data3));
    
    // Increment to prevent auto-delete
    batch1->IncrementViewRefCount();
    batch2->IncrementViewRefCount();
    batch3->IncrementViewRefCount();
    
    DataBatchView view1(batch1);
    DataBatchView view2(batch2);
    DataBatchView view3(batch3);
    
    // Each batch should have its own view count
    REQUIRE(batch1->GetViewCount() == 2);
    REQUIRE(batch2->GetViewCount() == 2);
    REQUIRE(batch3->GetViewCount() == 2);
    
    REQUIRE(batch1->GetBatchId() == 1);
    REQUIRE(batch2->GetBatchId() == 2);
    REQUIRE(batch3->GetBatchId() == 3);
    
    // Pin each view independently
    view1.Pin();
    view2.Pin();
    view3.Pin();
    
    REQUIRE(batch1->GetPinCount() == 1);
    REQUIRE(batch2->GetPinCount() == 1);
    REQUIRE(batch3->GetPinCount() == 1);
    
    // Unpin
    view1.Unpin();
    view2.Unpin();
    view3.Unpin();
    
    // Clean up
    batch1->DecrementViewRefCount();
    batch1->DecrementViewRefCount();
    batch2->DecrementViewRefCount();
    batch2->DecrementViewRefCount();
    batch3->DecrementViewRefCount();
    batch3->DecrementViewRefCount();
}

// Test stress test with many views and pins
TEST_CASE("DataBatchView Stress Test", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    constexpr int num_views = 1000;
    std::vector<std::unique_ptr<DataBatchView>> views;
    
    // Create many views
    for (int i = 0; i < num_views; ++i) {
        views.push_back(std::make_unique<DataBatchView>(batch));
    }
    
    REQUIRE(batch->GetViewCount() == num_views + 1);
    
    // Pin all of them
    for (int i = 0; i < num_views; ++i) {
        views[i]->Pin();
    }
    
    REQUIRE(batch->GetPinCount() == num_views);
    
    // Unpin all
    for (int i = 0; i < num_views; ++i) {
        views[i]->Unpin();
    }
    
    REQUIRE(batch->GetPinCount() == 0);
    
    // Destroy all views
    views.clear();
    
    REQUIRE(batch->GetViewCount() == 1);
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test view lifetime with mixed operations
TEST_CASE("DataBatchView Mixed Lifetime Operations", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    std::vector<std::unique_ptr<DataBatchView>> views;
    
    // Create some views
    for (int i = 0; i < 5; ++i) {
        views.push_back(std::make_unique<DataBatchView>(batch));
    }
    REQUIRE(batch->GetViewCount() == 6);
    
    // Pin some
    views[0]->Pin();
    views[2]->Pin();
    views[4]->Pin();
    REQUIRE(batch->GetPinCount() == 3);
    
    // Destroy some views (including pinned ones - should auto-unpin)
    views.erase(views.begin() + 2);
    REQUIRE(batch->GetViewCount() == 5);
    REQUIRE(batch->GetPinCount() == 2); // view[2] was unpinned on destruction
    
    // Create more views
    for (int i = 0; i < 3; ++i) {
        views.push_back(std::make_unique<DataBatchView>(batch));
    }
    REQUIRE(batch->GetViewCount() == 8);
    
    // Unpin remaining
    views[0]->Unpin();
    views[3]->Unpin(); // This was views[4] before erase
    REQUIRE(batch->GetPinCount() == 0);
    
    // Destroy all
    views.clear();
    REQUIRE(batch->GetViewCount() == 1);
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test destructor with pinned view
TEST_CASE("DataBatchView Destructor With Pinned View", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    {
        DataBatchView view1(batch);
        DataBatchView view2(batch);
        
        view1.Pin();
        view2.Pin();
        
        REQUIRE(batch->GetPinCount() == 2);
        REQUIRE(batch->GetViewCount() == 3);
        
        // Both views will be destroyed, should auto-unpin
    }
    
    // Both should be unpinned and view count decreased
    REQUIRE(batch->GetPinCount() == 0);
    REQUIRE(batch->GetViewCount() == 1);
    
    // Clean up
    batch->DecrementViewRefCount();
}

// Test edge case: zero to non-zero pin transitions
TEST_CASE("DataBatchView Zero to Non-Zero Pin Transitions", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    DataBatchView view(batch);
    
    // Start with zero pin count
    REQUIRE(batch->GetPinCount() == 0);
    
    // Transition from 0 to 1
    view.Pin();
    REQUIRE(batch->GetPinCount() == 1);
    
    // Transition from 1 to 0
    view.Unpin();
    REQUIRE(batch->GetPinCount() == 0);
    
    // Repeat multiple times
    for (int i = 0; i < 10; ++i) {
        view.Pin();
        REQUIRE(batch->GetPinCount() == 1);
        view.Unpin();
        REQUIRE(batch->GetPinCount() == 0);
    }
    
    // Clean up
    batch->DecrementViewRefCount();
    batch->DecrementViewRefCount();
}

// Test rapid pin/unpin with multiple views
TEST_CASE("DataBatchView Rapid Pin Unpin Multiple Views", "[data_batch_view]") {
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto* batch = new DataBatch(1, std::move(data));
    
    // Increment to prevent auto-delete
    batch->IncrementViewRefCount();
    
    constexpr int num_views = 10;
    std::vector<std::unique_ptr<DataBatchView>> views;
    
    for (int i = 0; i < num_views; ++i) {
        views.push_back(std::make_unique<DataBatchView>(batch));
    }
    
    // Rapid pin/unpin cycles
    for (int cycle = 0; cycle < 50; ++cycle) {
        // Pin all
        for (auto& view : views) {
            view->Pin();
        }
        REQUIRE(batch->GetPinCount() == num_views);
        
        // Unpin all
        for (auto& view : views) {
            view->Unpin();
        }
        REQUIRE(batch->GetPinCount() == 0);
    }
    
    // Clear views
    views.clear();
    
    // Clean up
    batch->DecrementViewRefCount();
}

