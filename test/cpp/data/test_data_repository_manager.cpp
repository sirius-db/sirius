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
#include "data/data_repository_manager.hpp"
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
TEST_CASE("DataRepositoryManager Construction", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Manager should be initialized with batch ID starting at 0
    uint64_t id1 = manager.GetNextDataBatchId();
    REQUIRE(id1 == 0);
    
    uint64_t id2 = manager.GetNextDataBatchId();
    REQUIRE(id2 == 1);
}

// Test adding a repository
TEST_CASE("DataRepositoryManager Add Repository", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add a repository for pipeline 1
    auto repository = sirius::make_unique<IDataRepository>();
    manager.AddNewRepository(1, std::move(repository));
    
    // Get the repository back
    auto& retrieved = manager.GetRepository(1);
    REQUIRE(retrieved != nullptr);
}

// Test adding multiple repositories
TEST_CASE("DataRepositoryManager Multiple Repositories", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add repositories for multiple pipelines
    for (size_t i = 0; i < 10; ++i) {
        auto repository = sirius::make_unique<IDataRepository>();
        manager.AddNewRepository(i, std::move(repository));
    }
    
    // Verify all repositories can be retrieved
    for (size_t i = 0; i < 10; ++i) {
        auto& repository = manager.GetRepository(i);
        REQUIRE(repository != nullptr);
    }
}

// Test GetNextDataBatchId uniqueness
TEST_CASE("DataRepositoryManager Unique Batch IDs", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    constexpr int num_ids = 1000;
    std::vector<uint64_t> ids;
    
    // Generate many IDs
    for (int i = 0; i < num_ids; ++i) {
        ids.push_back(manager.GetNextDataBatchId());
    }
    
    // Verify all are unique and sequential
    for (int i = 0; i < num_ids; ++i) {
        REQUIRE(ids[i] == static_cast<uint64_t>(i));
    }
}

// Test AddNewDataBatch with single pipeline
TEST_CASE("DataRepositoryManager Add Batch Single Pipeline", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add a repository
    manager.AddNewRepository(1, sirius::make_unique<IDataRepository>());
    
    // Create and add a data batch
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
    
    sirius::vector<size_t> pipeline_ids = {1};
    manager.AddNewDataBatch(std::move(batch), pipeline_ids);
    
    // Pull from the repository
    auto& repository = manager.GetRepository(1);
    auto view = repository->PullDataBatchView();
    REQUIRE(view != nullptr);
}

// Test AddNewDataBatch with multiple pipelines
TEST_CASE("DataRepositoryManager Add Batch Multiple Pipelines", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add repositories for multiple pipelines
    manager.AddNewRepository(1, sirius::make_unique<IDataRepository>());
    manager.AddNewRepository(2, sirius::make_unique<IDataRepository>());
    manager.AddNewRepository(3, sirius::make_unique<IDataRepository>());
    
    // Create and add a data batch to all pipelines
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
    
    sirius::vector<size_t> pipeline_ids = {1, 2, 3};
    manager.AddNewDataBatch(std::move(batch), pipeline_ids);
    
    // Each pipeline should have a view
    auto& repo1 = manager.GetRepository(1);
    auto view1 = repo1->PullDataBatchView();
    REQUIRE(view1 != nullptr);
    
    auto& repo2 = manager.GetRepository(2);
    auto view2 = repo2->PullDataBatchView();
    REQUIRE(view2 != nullptr);
    
    auto& repo3 = manager.GetRepository(3);
    auto view3 = repo3->PullDataBatchView();
    REQUIRE(view3 != nullptr);
}

// Test multiple batches to same pipeline
TEST_CASE("DataRepositoryManager Multiple Batches Same Pipeline", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add a repository
    manager.AddNewRepository(1, sirius::make_unique<IDataRepository>());
    
    // Add multiple batches
    for (int i = 0; i < 5; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
        
        sirius::vector<size_t> pipeline_ids = {1};
        manager.AddNewDataBatch(std::move(batch), pipeline_ids);
    }
    
    // Pull all batches
    auto& repository = manager.GetRepository(1);
    int count = 0;
    while (auto view = repository->PullDataBatchView()) {
        ++count;
    }
    
    REQUIRE(count == 5);
}

// Test batches to different pipelines
TEST_CASE("DataRepositoryManager Batches To Different Pipelines", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add repositories
    manager.AddNewRepository(1, sirius::make_unique<IDataRepository>());
    manager.AddNewRepository(2, sirius::make_unique<IDataRepository>());
    
    // Add batches to pipeline 1
    for (int i = 0; i < 3; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
        
        sirius::vector<size_t> pipeline_ids = {1};
        manager.AddNewDataBatch(std::move(batch), pipeline_ids);
    }
    
    // Add batches to pipeline 2
    for (int i = 0; i < 5; ++i) {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
        
        sirius::vector<size_t> pipeline_ids = {2};
        manager.AddNewDataBatch(std::move(batch), pipeline_ids);
    }
    
    // Verify counts
    auto& repo1 = manager.GetRepository(1);
    int count1 = 0;
    while (auto view = repo1->PullDataBatchView()) {
        ++count1;
    }
    REQUIRE(count1 == 3);
    
    auto& repo2 = manager.GetRepository(2);
    int count2 = 0;
    while (auto view = repo2->PullDataBatchView()) {
        ++count2;
    }
    REQUIRE(count2 == 5);
}

// Test shared batch across pipelines
TEST_CASE("DataRepositoryManager Shared Batch Across Pipelines", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add repositories
    manager.AddNewRepository(1, sirius::make_unique<IDataRepository>());
    manager.AddNewRepository(2, sirius::make_unique<IDataRepository>());
    manager.AddNewRepository(3, sirius::make_unique<IDataRepository>());
    
    // Add a batch shared across all pipelines
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
    
    sirius::vector<size_t> pipeline_ids = {1, 2, 3};
    manager.AddNewDataBatch(std::move(batch), pipeline_ids);
    
    // Each pipeline should have exactly one view
    auto& repo1 = manager.GetRepository(1);
    auto view1 = repo1->PullDataBatchView();
    REQUIRE(view1 != nullptr);
    REQUIRE(repo1->PullDataBatchView() == nullptr);
    
    auto& repo2 = manager.GetRepository(2);
    auto view2 = repo2->PullDataBatchView();
    REQUIRE(view2 != nullptr);
    REQUIRE(repo2->PullDataBatchView() == nullptr);
    
    auto& repo3 = manager.GetRepository(3);
    auto view3 = repo3->PullDataBatchView();
    REQUIRE(view3 != nullptr);
    REQUIRE(repo3->PullDataBatchView() == nullptr);
}

// Test thread-safe batch ID generation
TEST_CASE("DataRepositoryManager Thread-Safe Batch ID Generation", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    constexpr int num_threads = 10;
    constexpr int ids_per_thread = 1000;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<uint64_t>> thread_ids(num_threads);
    
    // Launch threads to generate IDs
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < ids_per_thread; ++j) {
                thread_ids[i].push_back(manager.GetNextDataBatchId());
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Collect all IDs
    std::vector<uint64_t> all_ids;
    for (const auto& ids : thread_ids) {
        all_ids.insert(all_ids.end(), ids.begin(), ids.end());
    }
    
    // Sort and verify uniqueness
    std::sort(all_ids.begin(), all_ids.end());
    for (size_t i = 1; i < all_ids.size(); ++i) {
        REQUIRE(all_ids[i] != all_ids[i-1]); // All unique
    }
    
    // Should have exactly num_threads * ids_per_thread unique IDs
    REQUIRE(all_ids.size() == num_threads * ids_per_thread);
}

// Test thread-safe repository addition
TEST_CASE("DataRepositoryManager Thread-Safe Repository Addition", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    constexpr int num_threads = 10;
    constexpr int repos_per_thread = 10;
    
    std::vector<std::thread> threads;
    
    // Launch threads to add repositories
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < repos_per_thread; ++j) {
                size_t pipeline_id = i * repos_per_thread + j;
                auto repository = sirius::make_unique<IDataRepository>();
                manager.AddNewRepository(pipeline_id, std::move(repository));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all repositories can be retrieved
    for (int i = 0; i < num_threads * repos_per_thread; ++i) {
        auto& repository = manager.GetRepository(i);
        REQUIRE(repository != nullptr);
    }
}

// Test thread-safe batch addition
TEST_CASE("DataRepositoryManager Thread-Safe Batch Addition", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add repository
    manager.AddNewRepository(1, sirius::make_unique<IDataRepository>());
    
    constexpr int num_threads = 10;
    constexpr int batches_per_thread = 50;
    
    std::vector<std::thread> threads;
    
    // Launch threads to add batches
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < batches_per_thread; ++j) {
                auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
                auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
                
                sirius::vector<size_t> pipeline_ids = {1};
                manager.AddNewDataBatch(std::move(batch), pipeline_ids);
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Count batches in repository
    auto& repository = manager.GetRepository(1);
    int count = 0;
    while (auto view = repository->PullDataBatchView()) {
        ++count;
    }
    
    REQUIRE(count == num_threads * batches_per_thread);
}

// Test complex multi-pipeline scenario
TEST_CASE("DataRepositoryManager Complex Multi-Pipeline Scenario", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add 5 pipelines
    for (size_t i = 1; i <= 5; ++i) {
        manager.AddNewRepository(i, sirius::make_unique<IDataRepository>());
    }
    
    // Add batches with different pipeline combinations
    // Batch 1: pipelines 1, 2
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
        manager.AddNewDataBatch(std::move(batch), {1, 2});
    }
    
    // Batch 2: pipelines 2, 3, 4
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
        manager.AddNewDataBatch(std::move(batch), {2, 3, 4});
    }
    
    // Batch 3: pipeline 5 only
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
        manager.AddNewDataBatch(std::move(batch), {5});
    }
    
    // Batch 4: all pipelines
    {
        auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
        auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
        manager.AddNewDataBatch(std::move(batch), {1, 2, 3, 4, 5});
    }
    
    // Verify counts
    std::vector<int> expected_counts = {2, 3, 2, 2, 2}; // Pipeline 1-5
    for (size_t i = 1; i <= 5; ++i) {
        auto& repository = manager.GetRepository(i);
        int count = 0;
        while (auto view = repository->PullDataBatchView()) {
            ++count;
        }
        REQUIRE(count == expected_counts[i-1]);
    }
}

// Test replacing repository
TEST_CASE("DataRepositoryManager Replace Repository", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add initial repository
    auto repo1 = sirius::make_unique<IDataRepository>();
    manager.AddNewRepository(1, std::move(repo1));
    
    // Add a batch
    auto data1 = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto batch1 = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data1));
    manager.AddNewDataBatch(std::move(batch1), {1});
    
    // Replace with new repository
    auto repo2 = sirius::make_unique<IDataRepository>();
    manager.AddNewRepository(1, std::move(repo2));
    
    // New repository should be empty
    auto& repository = manager.GetRepository(1);
    auto view = repository->PullDataBatchView();
    REQUIRE(view == nullptr);
    
    // Add batch to new repository
    auto data2 = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto batch2 = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data2));
    manager.AddNewDataBatch(std::move(batch2), {1});
    
    // Should now have one batch
    auto view2 = repository->PullDataBatchView();
    REQUIRE(view2 != nullptr);
}

// Test batch ID monotonicity under concurrency
TEST_CASE("DataRepositoryManager Batch ID Monotonicity", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    constexpr int num_threads = 20;
    constexpr int ids_per_thread = 500;
    
    std::vector<std::thread> threads;
    std::atomic<uint64_t> max_seen{0};
    std::atomic<bool> monotonic{true};
    
    // Launch threads to generate IDs and check monotonicity
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            uint64_t last_id = 0;
            bool first = true;
            
            for (int j = 0; j < ids_per_thread; ++j) {
                uint64_t id = manager.GetNextDataBatchId();
                
                if (!first && id <= last_id) {
                    monotonic = false;
                }
                
                last_id = id;
                first = false;
                
                // Update max seen
                uint64_t current_max = max_seen.load();
                while (id > current_max && !max_seen.compare_exchange_weak(current_max, id)) {
                    // Retry
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Monotonicity within each thread should hold
    REQUIRE(monotonic.load());
}

// Test empty pipeline vector
TEST_CASE("DataRepositoryManager Empty Pipeline Vector", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    // Add repositories
    manager.AddNewRepository(1, sirius::make_unique<IDataRepository>());
    
    // Add batch with empty pipeline vector
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
    
    sirius::vector<size_t> empty_pipeline_ids;
    manager.AddNewDataBatch(std::move(batch), empty_pipeline_ids);
    
    // Repository should still be empty
    auto& repository = manager.GetRepository(1);
    auto view = repository->PullDataBatchView();
    REQUIRE(view == nullptr);
}

// Test large number of pipelines
TEST_CASE("DataRepositoryManager Large Number of Pipelines", "[data_repository_manager]") {
    DataRepositoryManager manager;
    
    constexpr int num_pipelines = 1000;
    
    // Add many repositories
    for (int i = 0; i < num_pipelines; ++i) {
        manager.AddNewRepository(i, sirius::make_unique<IDataRepository>());
    }
    
    // Add a batch to all pipelines
    auto data = sirius::make_unique<MockDataRepresentation>(Tier::GPU, 1024);
    auto batch = sirius::make_unique<DataBatch>(manager.GetNextDataBatchId(), std::move(data));
    
    sirius::vector<size_t> all_pipeline_ids;
    for (int i = 0; i < num_pipelines; ++i) {
        all_pipeline_ids.push_back(i);
    }
    
    manager.AddNewDataBatch(std::move(batch), all_pipeline_ids);
    
    // Each pipeline should have one view
    for (int i = 0; i < num_pipelines; ++i) {
        auto& repository = manager.GetRepository(i);
        auto view = repository->PullDataBatchView();
        REQUIRE(view != nullptr);
    }
}

