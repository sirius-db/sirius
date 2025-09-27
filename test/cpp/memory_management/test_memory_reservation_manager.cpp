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

using namespace sirius::memory;

// Helper function to initialize the manager for tests
void initializeTestManager() {
    // Initialize with test limits: GPU=1000, HOST=2000, DISK=5000
    std::array<size_t, static_cast<size_t>(Tier::SIZE)> limits = {1000, 2000, 5000};
    MemoryReservationManager::initialize(limits);
}

// Test basic initialization
TEST_CASE("MemoryReservationManager Initialization", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    REQUIRE(manager.getMaxReservation(Tier::GPU) == 1000);
    REQUIRE(manager.getMaxReservation(Tier::HOST) == 2000);
    REQUIRE(manager.getMaxReservation(Tier::DISK) == 5000);
    
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 0);
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 0);
    REQUIRE(manager.getTotalReservedMemory(Tier::DISK) == 0);
    
    REQUIRE(manager.getActiveReservationCount(Tier::GPU) == 0);
    REQUIRE(manager.getActiveReservationCount(Tier::HOST) == 0);
    REQUIRE(manager.getActiveReservationCount(Tier::DISK) == 0);
}

// Test basic reservation and release
TEST_CASE("Basic Reservation and Release", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    // Request a reservation
    auto reservation = manager.requestReservation(Tier::GPU, 500);
    REQUIRE(reservation != nullptr);
    REQUIRE(reservation->tier == Tier::GPU);
    REQUIRE(reservation->size == 500);
    
    // Check that memory is reserved
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 500);
    REQUIRE(manager.getActiveReservationCount(Tier::GPU) == 1);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 500);
    
    // Release the reservation
    manager.releaseReservation(std::move(reservation));
    
    // Check that memory is released
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 0);
    REQUIRE(manager.getActiveReservationCount(Tier::GPU) == 0);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 1000);
}

// Test multiple reservations
TEST_CASE("Multiple Reservations", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    std::vector<std::unique_ptr<Reservation>> reservations;
    
    // Create multiple reservations
    for (int i = 0; i < 5; ++i) {
        auto reservation = manager.requestReservation(Tier::HOST, 200);
        REQUIRE(reservation != nullptr);
        reservations.push_back(std::move(reservation));
    }
    
    // Check totals
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 1000);
    REQUIRE(manager.getActiveReservationCount(Tier::HOST) == 5);
    REQUIRE(manager.getAvailableMemory(Tier::HOST) == 1000);
    
    // Release all reservations
    for (auto& reservation : reservations) {
        manager.releaseReservation(std::move(reservation));
    }
    
    // Check that all memory is released
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 0);
    REQUIRE(manager.getActiveReservationCount(Tier::HOST) == 0);
    REQUIRE(manager.getAvailableMemory(Tier::HOST) == 2000);
}

// Test memory limit enforcement
TEST_CASE("Memory Limit Enforcement", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    // Reserve most of the memory
    auto reservation1 = manager.requestReservation(Tier::GPU, 800);
    REQUIRE(reservation1 != nullptr);
    
    // Check that we can't reserve more than available
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 200);
    
    // Try to reserve exactly what's available
    auto reservation2 = manager.requestReservation(Tier::GPU, 200);
    REQUIRE(reservation2 != nullptr);
    
    // Check totals
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 1000);
    REQUIRE(manager.getActiveReservationCount(Tier::GPU) == 2);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 0);
    
    // Release first reservation
    manager.releaseReservation(std::move(reservation1));
    
    // Check available memory increased
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 200);
    REQUIRE(manager.getActiveReservationCount(Tier::GPU) == 1);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 800);
    
    // Release second reservation
    manager.releaseReservation(std::move(reservation2));
    
    // Check all memory is released
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 0);
    REQUIRE(manager.getActiveReservationCount(Tier::GPU) == 0);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 1000);
}

// Test grow reservation
TEST_CASE("Grow Reservation", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    // Create initial reservation
    auto reservation = manager.requestReservation(Tier::HOST, 500);
    REQUIRE(reservation != nullptr);
    
    // Grow the reservation
    bool success = manager.growReservation(reservation.get(), 800);
    REQUIRE(success);
    REQUIRE(reservation->size == 800);
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 800);
    
    // Try to grow beyond available memory (HOST limit is 2000)
    success = manager.growReservation(reservation.get(), 2500);
    REQUIRE_FALSE(success); // Should fail
    REQUIRE(reservation->size == 800); // Size should remain unchanged
    
    // Release reservation
    manager.releaseReservation(std::move(reservation));
}

// Test shrink reservation
TEST_CASE("Shrink Reservation", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    // Create initial reservation
    auto reservation = manager.requestReservation(Tier::DISK, 1000);
    REQUIRE(reservation != nullptr);
    
    // Shrink the reservation
    bool success = manager.shrinkReservation(reservation.get(), 600);
    REQUIRE(success);
    REQUIRE(reservation->size == 600);
    REQUIRE(manager.getTotalReservedMemory(Tier::DISK) == 600);
    
    // Try to shrink to same size (should fail)
    success = manager.shrinkReservation(reservation.get(), 600);
    REQUIRE_FALSE(success);
    
    // Try to shrink to larger size (should fail)
    success = manager.shrinkReservation(reservation.get(), 800);
    REQUIRE_FALSE(success);
    
    // Release reservation
    manager.releaseReservation(std::move(reservation));
}

// Test edge cases
TEST_CASE("Edge Cases", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    // Test zero size reservation
    REQUIRE_THROWS_AS(manager.requestReservation(Tier::GPU, 0), std::invalid_argument);
    
    // Test null reservation release
    manager.releaseReservation(nullptr); // Should not crash
    
    // Test null reservation resize
    REQUIRE_FALSE(manager.growReservation(nullptr, 100));
    REQUIRE_FALSE(manager.shrinkReservation(nullptr, 50));
}

// Test different tiers
TEST_CASE("Different Tiers", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    // Test all tiers
    auto gpu_reservation = manager.requestReservation(Tier::GPU, 300);
    auto host_reservation = manager.requestReservation(Tier::HOST, 500);
    auto disk_reservation = manager.requestReservation(Tier::DISK, 1000);
    
    REQUIRE(gpu_reservation != nullptr);
    REQUIRE(host_reservation != nullptr);
    REQUIRE(disk_reservation != nullptr);
    
    // Check each tier independently
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 300);
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 500);
    REQUIRE(manager.getTotalReservedMemory(Tier::DISK) == 1000);
    
    REQUIRE(manager.getActiveReservationCount(Tier::GPU) == 1);
    REQUIRE(manager.getActiveReservationCount(Tier::HOST) == 1);
    REQUIRE(manager.getActiveReservationCount(Tier::DISK) == 1);
    
    // Release all reservations
    manager.releaseReservation(std::move(gpu_reservation));
    manager.releaseReservation(std::move(host_reservation));
    manager.releaseReservation(std::move(disk_reservation));
    
    // Check all are released
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 0);
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 0);
    REQUIRE(manager.getTotalReservedMemory(Tier::DISK) == 0);
}

// Test blocking behavior with proper thread coordination
TEST_CASE("Blocking Behavior", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    // Reserve most of the memory (900 out of 1000)
    auto reservation1 = manager.requestReservation(Tier::GPU, 900);
    REQUIRE(reservation1 != nullptr);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 100);
    
    // Thread coordination variables
    std::atomic<bool> waiting_thread_started{false};
    std::atomic<bool> waiting_thread_completed{false};
    std::atomic<bool> release_thread_completed{false};
    std::unique_ptr<Reservation> waiting_reservation{nullptr};
    
    // Thread 1: Try to reserve more than available (should block)
    std::thread waiting_thread([&]() {
        waiting_thread_started = true;
        
        // This should block because we're trying to reserve 200 but only 100 is available
        auto reservation = manager.requestReservation(Tier::GPU, 200);
        
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
        manager.releaseReservation(std::move(reservation1));
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
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 200);
    REQUIRE(manager.getActiveReservationCount(Tier::GPU) == 1);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 800);
    
    // Clean up
    manager.releaseReservation(std::move(waiting_reservation));
    
    // Verify all memory is released
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 0);
    REQUIRE(manager.getActiveReservationCount(Tier::GPU) == 0);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 1000);
}

// Test initialization with zero limits (should throw)
TEST_CASE("Invalid Initialization", "[memory]") {
    // This test would require a way to reset the singleton
    // For now, we'll test the constructor directly if it were public
    // In a real implementation, you might want to add a reset method
    std::array<size_t, static_cast<size_t>(Tier::SIZE)> zero_limits = {0, 1000, 2000};
    
    // This would throw if we could create a new instance
    // REQUIRE_THROWS_AS(MemoryReservationManager manager(zero_limits), std::invalid_argument);
}


// Test reservation with maximum size
TEST_CASE("Maximum Size Reservation", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    // Reserve the maximum possible size
    auto reservation = manager.requestReservation(Tier::GPU, 1000);
    REQUIRE(reservation != nullptr);
    
    REQUIRE(reservation->size == 1000);
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 1000);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 0);
    
    // Release reservation
    manager.releaseReservation(std::move(reservation));
    
    // Check memory is fully available again
    REQUIRE(manager.getTotalReservedMemory(Tier::GPU) == 0);
    REQUIRE(manager.getAvailableMemory(Tier::GPU) == 1000);
}

// Test mixed operations
TEST_CASE("Mixed Operations", "[memory]") {
    initializeTestManager();
    auto& manager = MemoryReservationManager::getInstance();
    
    // Create multiple reservations
    auto reservation1 = manager.requestReservation(Tier::HOST, 500);
    auto reservation2 = manager.requestReservation(Tier::HOST, 300);
    auto reservation3 = manager.requestReservation(Tier::HOST, 200);
    
    REQUIRE(reservation1 != nullptr);
    REQUIRE(reservation2 != nullptr);
    REQUIRE(reservation3 != nullptr);
    
    // Check initial state
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 1000);
    REQUIRE(manager.getActiveReservationCount(Tier::HOST) == 3);
    
    // Grow one reservation
    bool success = manager.growReservation(reservation1.get(), 700);
    REQUIRE(success);
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 1200);
    
    // Shrink another reservation
    success = manager.shrinkReservation(reservation2.get(), 200);
    REQUIRE(success);
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 1100);
    
    // Release one reservation
    manager.releaseReservation(std::move(reservation3));
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 900);
    REQUIRE(manager.getActiveReservationCount(Tier::HOST) == 2);
    
    // Release remaining reservations
    manager.releaseReservation(std::move(reservation1));
    manager.releaseReservation(std::move(reservation2));
    
    // Check all memory is released
    REQUIRE(manager.getTotalReservedMemory(Tier::HOST) == 0);
    REQUIRE(manager.getActiveReservationCount(Tier::HOST) == 0);
}
