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
#include <chrono>
#include <memory>
#include <array>
#include "memory/memory_reservation.hpp"

#include "memory_test_common.hpp"

using namespace sirius::memory;

// create_test_allocators now provided by memory_test_common.hpp

// Helper function to initialize the manager for tests
void initializeTestManager() {
    memory_reservation_manager::reset_for_testing();
    // Initialize with test memory_spaces: GPU(id:0)=1000, HOST(id:0)=2000, DISK(id:0)=5000
    std::vector<memory_reservation_manager::memory_space_config> configs;
    configs.emplace_back(Tier::GPU, 0, 1000, create_test_allocators(Tier::GPU));
    configs.emplace_back(Tier::HOST, 0, 2000, create_test_allocators(Tier::HOST));
    configs.emplace_back(Tier::DISK, 0, 5000, create_test_allocators(Tier::DISK));
    memory_reservation_manager::initialize(std::move(configs));
}

// Helper function for multi-device initialization
void initializeMultiDeviceManager() {
    memory_reservation_manager::reset_for_testing();
    std::vector<memory_reservation_manager::memory_space_config> configs;
    configs.emplace_back(Tier::GPU, 0, 1000, create_test_allocators(Tier::GPU));    // GPU device 0
    configs.emplace_back(Tier::GPU, 1, 2000, create_test_allocators(Tier::GPU));    // GPU device 1
    configs.emplace_back(Tier::HOST, 0, 1500, create_test_allocators(Tier::HOST));  // Host
    configs.emplace_back(Tier::DISK, 0, 5000, create_test_allocators(Tier::DISK));  // Disk
    memory_reservation_manager::initialize(std::move(configs));
}

// Test basic initialization
TEST_CASE("MemoryReservationManager Initialization", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    // Get MemorySpaces
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    auto disk_space = manager.get_memory_space(Tier::DISK, 0);
    
    REQUIRE(gpu_space != nullptr);
    REQUIRE(host_space != nullptr);
    REQUIRE(disk_space != nullptr);
    
    REQUIRE(gpu_space->get_max_memory() == 1000);
    REQUIRE(host_space->get_max_memory() == 2000);
    REQUIRE(disk_space->get_max_memory() == 5000);
    
    REQUIRE(gpu_space->get_total_reserved_memory() == 0);
    REQUIRE(host_space->get_total_reserved_memory() == 0);
    REQUIRE(disk_space->get_total_reserved_memory() == 0);
    
    REQUIRE(gpu_space->get_active_reservation_count() == 0);
    REQUIRE(host_space->get_active_reservation_count() == 0);
    REQUIRE(disk_space->get_active_reservation_count() == 0);
    
    // Test tier-level helpers
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::GPU) == 0);
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::HOST) == 0);
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::DISK) == 0);
    
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::GPU) == 0);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::HOST) == 0);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::DISK) == 0);
    
    // Test memory spaces for tier
    auto gpu_spaces = manager.get_memory_spaces_for_tier(Tier::GPU);
    REQUIRE(gpu_spaces.size() == 1);
    REQUIRE(gpu_spaces[0] == gpu_space);
}

// Test basic reservation and release
TEST_CASE("Basic Reservation and Release", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    REQUIRE(gpu_space != nullptr);
    
    // Request a reservation using specific MemorySpace
    auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 500);
    REQUIRE(reservation != nullptr);
    REQUIRE(reservation->tier == Tier::GPU);
    REQUIRE(reservation->device_id == 0);
    REQUIRE(reservation->size == 500);
    
    // Check that memory is reserved
    REQUIRE(gpu_space->get_total_reserved_memory() == 500);
    REQUIRE(gpu_space->get_active_reservation_count() == 1);
    REQUIRE(gpu_space->get_available_memory() == 500);
    
    // Check tier-level stats
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::GPU) == 500);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::GPU) == 1);
    
    // Release the reservation
    manager.release_reservation(std::move(reservation));
    
    // Check that memory is released
    REQUIRE(gpu_space->get_total_reserved_memory() == 0);
    REQUIRE(gpu_space->get_active_reservation_count() == 0);
    REQUIRE(gpu_space->get_available_memory() == 1000);
    
    // Check tier-level stats
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::GPU) == 0);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::GPU) == 0);
}

// Test reservation using different strategies
TEST_CASE("Reservation Strategies", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    
    // Test specific MemorySpace strategy
    auto reservation1 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 300);
    REQUIRE(reservation1 != nullptr);
    REQUIRE(reservation1->tier == Tier::GPU);
    REQUIRE(reservation1->device_id == 0);
    
    // Test any in tier strategy
    auto reservation2 = manager.request_reservation(any_memory_space_in_tier(Tier::HOST), 400);
    REQUIRE(reservation2 != nullptr);
    REQUIRE(reservation2->tier == Tier::HOST);
    REQUIRE(reservation2->device_id == 0);
    
    // Test any available strategy (using all tiers)
    std::vector<Tier> all_tiers = {Tier::GPU, Tier::HOST, Tier::DISK};
    auto reservation3 = manager.request_reservation(any_memory_space_in_tiers(all_tiers), 100);
    REQUIRE(reservation3 != nullptr);
    
    // Test convenience methods
    auto reservation4 = manager.request_reservation(any_memory_space_in_tier(Tier::DISK), 1000);
    REQUIRE(reservation4 != nullptr);
    REQUIRE(reservation4->tier == Tier::DISK);
    REQUIRE(reservation4->device_id == 0);
    
    auto reservation5 = manager.request_reservation(any_memory_space_in_tiers(all_tiers), 50);
    REQUIRE(reservation5 != nullptr);
    
    // Clean up
    manager.release_reservation(std::move(reservation1));
    manager.release_reservation(std::move(reservation2));
    manager.release_reservation(std::move(reservation3));
    manager.release_reservation(std::move(reservation4));
    manager.release_reservation(std::move(reservation5));
}

// Test multiple reservations
TEST_CASE("Multiple Reservations", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    std::vector<std::unique_ptr<reservation>> reservations;
    
    // Create multiple reservations
    for (int i = 0; i < 5; ++i) {
        auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), 200);
        REQUIRE(reservation != nullptr);
        reservations.push_back(std::move(reservation));
    }
    
    // Check totals
    REQUIRE(host_space->get_total_reserved_memory() == 1000);
    REQUIRE(host_space->get_active_reservation_count() == 5);
    REQUIRE(host_space->get_available_memory() == 1000);
    
    // Release all reservations
    for (auto& reservation : reservations) {
        manager.release_reservation(std::move(reservation));
    }
    
    // Check that all memory is released
    REQUIRE(host_space->get_total_reserved_memory() == 0);
    REQUIRE(host_space->get_active_reservation_count() == 0);
    REQUIRE(host_space->get_available_memory() == 2000);
}

// Test memory limit enforcement
TEST_CASE("Memory Limit Enforcement", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    
    // Reserve most of the memory
    auto reservation1 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 800);
    REQUIRE(reservation1 != nullptr);
    
    // Check that we can't reserve more than available
    REQUIRE(gpu_space->get_available_memory() == 200);
    
    // Try to reserve exactly what's available
    auto reservation2 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 200);
    REQUIRE(reservation2 != nullptr);
    
    // Check totals
    REQUIRE(gpu_space->get_total_reserved_memory() == 1000);
    REQUIRE(gpu_space->get_active_reservation_count() == 2);
    REQUIRE(gpu_space->get_available_memory() == 0);
    
    // Release first reservation
    manager.release_reservation(std::move(reservation1));
    
    // Check available memory increased
    REQUIRE(gpu_space->get_total_reserved_memory() == 200);
    REQUIRE(gpu_space->get_active_reservation_count() == 1);
    REQUIRE(gpu_space->get_available_memory() == 800);
    
    // Release second reservation
    manager.release_reservation(std::move(reservation2));
    
    // Check all memory is released
    REQUIRE(gpu_space->get_total_reserved_memory() == 0);
    REQUIRE(gpu_space->get_active_reservation_count() == 0);
    REQUIRE(gpu_space->get_available_memory() == 1000);
}

// Test grow reservation
TEST_CASE("Grow Reservation", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    
    // Create initial reservation
    auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), 500);
    REQUIRE(reservation != nullptr);
    
    // Grow the reservation
    bool success = manager.grow_reservation(reservation.get(), 800);
    REQUIRE(success);
    REQUIRE(reservation->size == 800);
    REQUIRE(host_space->get_total_reserved_memory() == 800);
    
    // Try to grow beyond available memory (HOST limit is 2000)
    success = manager.grow_reservation(reservation.get(), 2500);
    REQUIRE_FALSE(success); // Should fail
    REQUIRE(reservation->size == 800); // Size should remain unchanged
    
    // Release reservation
    manager.release_reservation(std::move(reservation));
}

// Test shrink reservation
TEST_CASE("Shrink Reservation", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto disk_space = manager.get_memory_space(Tier::DISK, 0);
    
    // Create initial reservation
    auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::DISK, 0), 1000);
    REQUIRE(reservation != nullptr);
    
    // Shrink the reservation
    bool success = manager.shrink_reservation(reservation.get(), 600);
    REQUIRE(success);
    REQUIRE(reservation->size == 600);
    REQUIRE(disk_space->get_total_reserved_memory() == 600);
    
    // Try to shrink to same size (should fail)
    success = manager.shrink_reservation(reservation.get(), 600);
    REQUIRE_FALSE(success);
    
    // Try to shrink to larger size (should fail)
    success = manager.shrink_reservation(reservation.get(), 800);
    REQUIRE_FALSE(success);
    
    // Release reservation
    manager.release_reservation(std::move(reservation));
}

// Test edge cases
TEST_CASE("Edge Cases", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    
    // Test zero size reservation (should be allowed)
    auto zero_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 0);
    REQUIRE(zero_reservation != nullptr);
    REQUIRE(zero_reservation->size == 0);
    manager.release_reservation(std::move(zero_reservation));
    
    // Test invalid memory space (non-existent device)
    auto invalid_space = manager.get_memory_space(Tier::GPU, 999);
    REQUIRE(invalid_space == nullptr);
    
    // Test null reservation release
    manager.release_reservation(nullptr); // Should not crash
    
    // Test null reservation resize
    REQUIRE_FALSE(manager.grow_reservation(nullptr, 100));
    REQUIRE_FALSE(manager.shrink_reservation(nullptr, 50));
    
    // Test requesting more memory than available should succeed when no outstanding reservations
    auto oversize_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 2000);
    REQUIRE(oversize_reservation != nullptr);
    REQUIRE(oversize_reservation->size == 2000);
    // Clean up
    manager.release_reservation(std::move(oversize_reservation));
}

// Test different memory spaces
TEST_CASE("Different Memory Spaces", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    auto disk_space = manager.get_memory_space(Tier::DISK, 0);
    
    // Test all memory spaces
    auto gpu_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 300);
    auto host_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), 500);
    auto disk_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::DISK, 0), 1000);
    
    REQUIRE(gpu_reservation != nullptr);
    REQUIRE(host_reservation != nullptr);
    REQUIRE(disk_reservation != nullptr);
    
    // Check each memory space independently
    REQUIRE(gpu_space->get_total_reserved_memory() == 300);
    REQUIRE(host_space->get_total_reserved_memory() == 500);
    REQUIRE(disk_space->get_total_reserved_memory() == 1000);
    
    REQUIRE(gpu_space->get_active_reservation_count() == 1);
    REQUIRE(host_space->get_active_reservation_count() == 1);
    REQUIRE(disk_space->get_active_reservation_count() == 1);
    
    // Check tier-level stats
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::GPU) == 300);
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::HOST) == 500);
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::DISK) == 1000);
    
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::GPU) == 1);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::HOST) == 1);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::DISK) == 1);
    
    // Release all reservations
    manager.release_reservation(std::move(gpu_reservation));
    manager.release_reservation(std::move(host_reservation));
    manager.release_reservation(std::move(disk_reservation));
    
    // Check all are released
    REQUIRE(gpu_space->get_total_reserved_memory() == 0);
    REQUIRE(host_space->get_total_reserved_memory() == 0);
    REQUIRE(disk_space->get_total_reserved_memory() == 0);
    
    // Check tier-level stats
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::GPU) == 0);
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::HOST) == 0);
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::DISK) == 0);
}

// Test blocking behavior with proper thread coordination
TEST_CASE("Blocking Behavior", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    // Reserve most of the memory (900 out of 1000)
    auto reservation1 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 900);
    REQUIRE(reservation1 != nullptr);
    REQUIRE(gpu_space->get_available_memory() == 100);
    
    // Thread coordination variables
    std::atomic<bool> waiting_thread_started{false};
    std::atomic<bool> waiting_thread_completed{false};
    std::atomic<bool> release_thread_completed{false};
    std::unique_ptr<reservation> waiting_reservation{nullptr};
    // Thread 1: Try to reserve more than available (should block)
    std::thread waiting_thread([&]() {
        waiting_thread_started = true;
        
        // This should block because we're trying to reserve 200 but only 100 is available
        auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 200);
        
        waiting_reservation = std::move(reservation);
        waiting_thread_completed = true;
    });
    
    // Thread 2: Release memory after a short delay to unblock thread 1
    std::thread release_thread([&]() {
        // Wait for waiting thread to start
        while (!waiting_thread_started.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Give waiting thread time to block
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Release the reservation to unblock the waiting thread
        manager.release_reservation(std::move(reservation1));
        release_thread_completed = true;
    });
    
    // Wait for both threads to complete with timeout
    auto start_time = std::chrono::steady_clock::now();
    while (!waiting_thread_completed.load() || !release_thread_completed.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > std::chrono::seconds(5)) {
            // Cleanup and fail
            waiting_thread.detach();
            release_thread.detach();
            FAIL("Test timed out - blocking behavior may not be working correctly");
        }
    }
    // Join threads
    waiting_thread.join();
    release_thread.join();
    
    // Verify the waiting thread got its reservation
    REQUIRE(waiting_reservation != nullptr);
    REQUIRE(waiting_reservation->size == 200);
    
    // Verify final state
    REQUIRE(gpu_space->get_total_reserved_memory() == 200);
    REQUIRE(gpu_space->get_active_reservation_count() == 1);
    REQUIRE(gpu_space->get_available_memory() == 800);
    
    // Clean up
    manager.release_reservation(std::move(waiting_reservation));
    
    // Verify all memory is released
    REQUIRE(gpu_space->get_total_reserved_memory() == 0);
    REQUIRE(gpu_space->get_active_reservation_count() == 0);
    REQUIRE(gpu_space->get_available_memory() == 1000);
}

// Test reservation with maximum size
TEST_CASE("Maximum Size Reservation", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    
    // Reserve the maximum possible size
    auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 1000);
    REQUIRE(reservation != nullptr);
    
    REQUIRE(reservation->size == 1000);
    REQUIRE(gpu_space->get_total_reserved_memory() == 1000);
    REQUIRE(gpu_space->get_available_memory() == 0);
    
    // Release reservation
    manager.release_reservation(std::move(reservation));
    
    // Check memory is fully available again
    REQUIRE(gpu_space->get_total_reserved_memory() == 0);
    REQUIRE(gpu_space->get_available_memory() == 1000);
}

// Test mixed operations
TEST_CASE("Mixed Operations", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    
    // Create multiple reservations
    auto reservation1 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), 500);
    auto reservation2 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), 300);
    auto reservation3 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), 200);
    
    REQUIRE(reservation1 != nullptr);
    REQUIRE(reservation2 != nullptr);
    REQUIRE(reservation3 != nullptr);
    
    // Check initial state
    REQUIRE(host_space->get_total_reserved_memory() == 1000);
    REQUIRE(host_space->get_active_reservation_count() == 3);
    
    // Grow one reservation
    bool success = manager.grow_reservation(reservation1.get(), 700);
    REQUIRE(success);
    REQUIRE(host_space->get_total_reserved_memory() == 1200);
    
    // Shrink another reservation
    success = manager.shrink_reservation(reservation2.get(), 200);
    REQUIRE(success);
    REQUIRE(host_space->get_total_reserved_memory() == 1100);
    
    // Release one reservation
    manager.release_reservation(std::move(reservation3));
    REQUIRE(host_space->get_total_reserved_memory() == 900);
    REQUIRE(host_space->get_active_reservation_count() == 2);
    
    // Release remaining reservations
    manager.release_reservation(std::move(reservation1));
    manager.release_reservation(std::move(reservation2));
    
    // Check all memory is released
    REQUIRE(host_space->get_total_reserved_memory() == 0);
    REQUIRE(host_space->get_active_reservation_count() == 0);
}

// Test multiple MemorySpaces of same tier
TEST_CASE("Multiple MemorySpaces Same Tier", "[memory][.multi]") {
    initializeMultiDeviceManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu0_space = manager.get_memory_space(Tier::GPU, 0);
    auto gpu1_space = manager.get_memory_space(Tier::GPU, 1);
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    
    REQUIRE(gpu0_space != nullptr);
    REQUIRE(gpu1_space != nullptr);
    REQUIRE(host_space != nullptr);
    
    // Test reservations on different GPU devices
    auto reservation_gpu0 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 500);
    auto reservation_gpu1 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 1), 1000);
    
    REQUIRE(reservation_gpu0 != nullptr);
    REQUIRE(reservation_gpu1 != nullptr);
    
    // Check individual MemorySpace stats
    REQUIRE(gpu0_space->get_total_reserved_memory() == 500);
    REQUIRE(gpu1_space->get_total_reserved_memory() == 1000);
    REQUIRE(gpu0_space->get_active_reservation_count() == 1);
    REQUIRE(gpu1_space->get_active_reservation_count() == 1);
    
    // Check tier-level aggregated stats
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::GPU) == 1500);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::GPU) == 2);
    
    // Check available memory per device
    REQUIRE(gpu0_space->get_available_memory() == 500);
    REQUIRE(gpu1_space->get_available_memory() == 1000);
    REQUIRE(manager.get_available_memory_for_tier(Tier::GPU) == 1500);
    
    // Check MemorySpaces for tier
    auto gpu_spaces = manager.get_memory_spaces_for_tier(Tier::GPU);
    REQUIRE(gpu_spaces.size() == 2);
    REQUIRE(std::find(gpu_spaces.begin(), gpu_spaces.end(), gpu0_space) != gpu_spaces.end());
    REQUIRE(std::find(gpu_spaces.begin(), gpu_spaces.end(), gpu1_space) != gpu_spaces.end());
    
    // Test tier-based reservation (should pick any available GPU)
    auto tier_reservation = manager.request_reservation(any_memory_space_in_tier(Tier::GPU), 300);
    REQUIRE(tier_reservation != nullptr);
    REQUIRE(tier_reservation->tier == Tier::GPU);
    
    // Clean up
    manager.release_reservation(std::move(reservation_gpu0));
    manager.release_reservation(std::move(reservation_gpu1));
    manager.release_reservation(std::move(tier_reservation));
    
    // Check final state
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::GPU) == 0);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::GPU) == 0);
}

// Test reservation request strategies in detail
TEST_CASE("Advanced Reservation Strategies", "[memory][.multi]") {
    initializeMultiDeviceManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu0_space = manager.get_memory_space(Tier::GPU, 0);
    auto gpu1_space = manager.get_memory_space(Tier::GPU, 1);
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    
    // Test AnyMemorySpaceInTiers (ordered preference)
    std::vector<Tier> tier_preferences = {Tier::GPU, Tier::HOST, Tier::DISK};
    auto reservation1 = manager.request_reservation(any_memory_space_in_tiers(tier_preferences), 500);
    REQUIRE(reservation1 != nullptr);
    REQUIRE(reservation1->tier == Tier::GPU);  // Should pick GPU first
    
    // Test specific tier requests (simulating space-specific requests)
    auto reservation2 = manager.request_reservation(any_memory_space_in_tier(Tier::HOST), 800);
    REQUIRE(reservation2 != nullptr);
    REQUIRE(reservation2->tier == Tier::HOST);  // Should pick host
    REQUIRE(reservation2->device_id == 0);      // Should pick host device 0
    
    // Fill up host space and test GPU fallback using tier preferences
    auto reservation3 = manager.request_reservation(any_memory_space_in_tier(Tier::HOST), 700);  // Fill remaining host
    auto reservation4 = manager.request_reservation(any_memory_space_in_tiers(tier_preferences), 500);
    REQUIRE(reservation4 != nullptr);
    REQUIRE(reservation4->tier == Tier::GPU);  // Should fallback to GPU when host is full
    
    // Test system-wide queries
    REQUIRE(manager.get_total_reserved_memory() > 0);
    REQUIRE(manager.get_total_available_memory() > 0);
    REQUIRE(manager.get_active_reservation_count() == 4);
    
    // Clean up
    manager.release_reservation(std::move(reservation1));
    manager.release_reservation(std::move(reservation2));
    manager.release_reservation(std::move(reservation3));
    manager.release_reservation(std::move(reservation4));
    
    REQUIRE(manager.get_active_reservation_count() == 0);
}

// Test zero-size reservations functionality
TEST_CASE("Zero Size Reservations", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    
    // Test zero-size reservation creation
    auto zero_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 0);
    REQUIRE(zero_reservation != nullptr);
    REQUIRE(zero_reservation->size == 0);
    REQUIRE(zero_reservation->tier == Tier::GPU);
    REQUIRE(zero_reservation->device_id == 0);
    
    // Test that zero-size reservations don't affect memory accounting
    REQUIRE(gpu_space->get_total_reserved_memory() == 0);
    REQUIRE(gpu_space->get_active_reservation_count() == 1);  // Still counts as active reservation
    REQUIRE(gpu_space->get_available_memory() == 1000);     // No memory consumed
    
    // Test growing a zero-size reservation
    bool success = manager.grow_reservation(zero_reservation.get(), 100);
    REQUIRE(success);
    REQUIRE(zero_reservation->size == 100);
    REQUIRE(gpu_space->get_total_reserved_memory() == 100);
    
    // Test shrinking back to zero
    success = manager.shrink_reservation(zero_reservation.get(), 0);
    REQUIRE(success);
    REQUIRE(zero_reservation->size == 0);
    REQUIRE(gpu_space->get_total_reserved_memory() == 0);
    
    // Test multiple zero-size reservations
    auto zero_reservation2 = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), 0);
    REQUIRE(zero_reservation2 != nullptr);
    REQUIRE(host_space->get_active_reservation_count() == 1);
    REQUIRE(host_space->get_total_reserved_memory() == 0);
    
    // Test reservation strategies with zero size
    auto zero_reservation3 = manager.request_reservation(any_memory_space_in_tier(Tier::DISK), 0);
    REQUIRE(zero_reservation3 != nullptr);
    REQUIRE(zero_reservation3->size == 0);
    
    std::vector<Tier> all_tiers_for_anywhere = {Tier::GPU, Tier::HOST, Tier::DISK};
    auto zero_reservation4 = manager.request_reservation(any_memory_space_in_tiers(all_tiers_for_anywhere), 0);
    REQUIRE(zero_reservation4 != nullptr);
    REQUIRE(zero_reservation4->size == 0);
    
    // Clean up
    manager.release_reservation(std::move(zero_reservation));
    manager.release_reservation(std::move(zero_reservation2));
    manager.release_reservation(std::move(zero_reservation3));
    manager.release_reservation(std::move(zero_reservation4));
    
    // Verify all are cleaned up
    REQUIRE(gpu_space->get_active_reservation_count() == 0);
    REQUIRE(host_space->get_active_reservation_count() == 0);
    REQUIRE(manager.get_active_reservation_count() == 0);
}

// Test allocator functionality
TEST_CASE("Allocator Management", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    
    REQUIRE(gpu_space != nullptr);
    REQUIRE(host_space != nullptr);
    
    // Test that we can get the default allocator
    auto gpu_allocator = gpu_space->get_default_allocator();
    auto host_allocator = host_space->get_default_allocator();
    
    // Test that we have at least one allocator
    REQUIRE(gpu_space->get_allocator_count() >= 1);
    REQUIRE(host_space->get_allocator_count() >= 1);
    
    // Test getting allocator by index
    auto gpu_allocator_by_index = gpu_space->get_allocator(0);
    auto host_allocator_by_index = host_space->get_allocator(0);
    
    // Test out of bounds access
    REQUIRE_THROWS_AS(gpu_space->get_allocator(100), std::out_of_range);
    
    // The references should be valid (this mainly tests that they compile and don't crash)
    REQUIRE(&gpu_allocator != nullptr);
    REQUIRE(&host_allocator != nullptr);
    REQUIRE(&gpu_allocator_by_index != nullptr);
    REQUIRE(&host_allocator_by_index != nullptr);
}

// Test specific allocator implementations
TEST_CASE("Specific Allocator Types", "[memory]") {
    initializeTestManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    auto disk_space = manager.get_memory_space(Tier::DISK, 0);
    
    REQUIRE(gpu_space != nullptr);
    REQUIRE(host_space != nullptr);
    REQUIRE(disk_space != nullptr);
    
    // Test that GPU uses cuda_async_memory_resource
    auto gpu_allocator = gpu_space->get_default_allocator();
    
    // Test that HOST uses fixed_size_host_memory_resource
    auto host_allocator = host_space->get_default_allocator();
    
    // Test that DISK uses a valid allocator (null_device_memory_resource)
    auto disk_allocator = disk_space->get_default_allocator();
    
    // Verify allocators can be used (basic functionality test)
    // Note: We can't easily test the exact type without RTTI, but we can test functionality
    REQUIRE(&gpu_allocator != nullptr);
    REQUIRE(&host_allocator != nullptr);
    REQUIRE(&disk_allocator != nullptr);
    
    // Test that allocators work by creating reservations
    auto gpu_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), 1024);
    auto host_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), 2048);
    auto disk_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::DISK, 0), 4096);
    
    REQUIRE(gpu_reservation != nullptr);
    REQUIRE(host_reservation != nullptr);
    REQUIRE(disk_reservation != nullptr);
    
    REQUIRE(gpu_reservation->size == 1024);
    REQUIRE(host_reservation->size == 2048);
    REQUIRE(disk_reservation->size == 4096);
    
    // Clean up
    manager.release_reservation(std::move(gpu_reservation));
    manager.release_reservation(std::move(host_reservation));
    manager.release_reservation(std::move(disk_reservation));
}

// Test that allocators must be explicitly provided
TEST_CASE("Explicit Allocator Requirement", "[memory]") {
    // Test that memory_space_config requires at least one allocator
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> empty_allocators;
    
    REQUIRE_THROWS_AS(
        memory_reservation_manager::memory_space_config(Tier::GPU, 0, 1000, std::move(empty_allocators)),
        std::invalid_argument
    );
    
    // Test that valid allocators work
    auto valid_allocators = create_test_allocators(Tier::GPU);
    REQUIRE_NOTHROW(
        memory_reservation_manager::memory_space_config(Tier::GPU, 0, 1000, std::move(valid_allocators))
    );
}

// Test fixed_size_host_memory_resource functionality aligned with its API
TEST_CASE("Fixed Size Host Memory Resource", "[memory]") {
    // Configure small, predictable pool: block=256B, pool=4 blocks, 2 pools
    const std::size_t block_size = 256;
    const std::size_t pool_size = 4;
    const std::size_t initial_pools = 2;

    fixed_size_host_memory_resource resource(block_size, pool_size, initial_pools);

    // Initial state
    REQUIRE(resource.get_block_size() == block_size);
    REQUIRE(resource.get_total_blocks() == pool_size * initial_pools);
    REQUIRE(resource.get_free_blocks() == pool_size * initial_pools);

    // Allocate enough bytes to require multiple blocks (600B -> 3 blocks)
    {
        auto blocks = resource.allocate_multiple_blocks(600);
        REQUIRE(blocks.size() == 3);
        REQUIRE(resource.get_free_blocks() == pool_size * initial_pools - 3);
        REQUIRE(blocks[0] != nullptr);
        REQUIRE(blocks[1] != nullptr);
        REQUIRE(blocks[2] != nullptr);
        REQUIRE(blocks[0] != blocks[1]);
        REQUIRE(blocks[1] != blocks[2]);
    } // RAII release on scope exit restores free list

    REQUIRE(resource.get_free_blocks() == pool_size * initial_pools);

    // Zero-size multi-block allocation is a no-op
    {
        auto zero = resource.allocate_multiple_blocks(0);
        REQUIRE(zero.size() == 0);
        REQUIRE(resource.get_free_blocks() == pool_size * initial_pools);
    }

    // Request more than a single pool's capacity to force pool expansion
    {
        auto many = resource.allocate_multiple_blocks(block_size * pool_size + 1); // pool_size + 1 blocks
        REQUIRE(many.size() == pool_size + 1);
    }

    // After RAII release, total/free blocks should reflect any expansion (>= initial)
    REQUIRE(resource.get_total_blocks() >= pool_size * initial_pools);
    REQUIRE(resource.get_free_blocks() >= pool_size * initial_pools);
}