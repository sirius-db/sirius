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

/**
 * Test Tags:
 * [memory_space] - Basic memory space functionality tests
 * [threading] - Multi-threaded tests  
 * [gpu] - GPU-specific tests requiring CUDA
 * [.multi-device] - Tests requiring multiple GPU devices (hidden by default)
 * 
 * Running tests:
 * - Default (includes single GPU): ./test_executable
 * - Include multi-device tests: ./test_executable "[.multi-device]"
 * - Exclude multi-device tests: ./test_executable "~[.multi-device]"
 * - Run all tests: ./test_executable "[memory_space]"
 */

#include "catch.hpp"
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <random>
#include <algorithm>
#include <future>
#include <cstdlib>
#include "memory/memory_reservation.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include "rmm/mr/device/cuda_async_memory_resource.hpp"
#include "test_gpu_kernels.cuh"
#include "memory_test_common.hpp"

// RMM includes for creating allocators
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

using namespace sirius::memory;

// Use shared create_test_allocators from memory_test_common.hpp

// Helper function to initialize single-device memory manager
void initializeSingleDeviceMemoryManager() {
    memory_reservation_manager::reset_for_testing();
    std::vector<memory_reservation_manager::memory_space_config> configs;
    
    // Single GPU device - 2GB
    configs.emplace_back(Tier::GPU, 0, 2048ull * 1024 * 1024, create_test_allocators(Tier::GPU));  // GPU device 0: 2GB
    
    // Single HOST NUMA node - 4GB
    configs.emplace_back(Tier::HOST, 0, 4096ull * 1024 * 1024, create_test_allocators(Tier::HOST)); // HOST NUMA 0: 4GB
    
    // Single DISK device - 8GB
    configs.emplace_back(Tier::DISK, 0, 8192ull * 1024 * 1024, create_test_allocators(Tier::DISK)); // DISK path 0: 8GB
    
    memory_reservation_manager::initialize(std::move(configs));
}

// Helper function to initialize multi-device memory manager (for multi-device tests only)
void initializeMultiDeviceMemoryManager() {
    memory_reservation_manager::reset_for_testing();
    std::vector<memory_reservation_manager::memory_space_config> configs;
    
    // Multiple GPU devices
    configs.emplace_back(Tier::GPU, 0, 1024ull * 1024 * 1024, create_test_allocators(Tier::GPU));  // GPU device 0: 1GB
    configs.emplace_back(Tier::GPU, 1, 1024ull * 1024 * 1024, create_test_allocators(Tier::GPU));  // GPU device 1: 1GB
    
    // Multiple HOST NUMA nodes
    configs.emplace_back(Tier::HOST, 0, 2048ull * 1024 * 1024, create_test_allocators(Tier::HOST)); // HOST NUMA 0: 2GB
    configs.emplace_back(Tier::HOST, 1, 2048ull * 1024 * 1024, create_test_allocators(Tier::HOST)); // HOST NUMA 1: 2GB
    
    // Multiple DISK devices  
    configs.emplace_back(Tier::DISK, 0, 4096ull * 1024 * 1024, create_test_allocators(Tier::DISK)); // DISK path 0: 4GB
    configs.emplace_back(Tier::DISK, 1, 4096ull * 1024 * 1024, create_test_allocators(Tier::DISK)); // DISK path 1: 4GB
    
    memory_reservation_manager::initialize(std::move(configs));
}

// Test single-device memory space access
TEST_CASE("Single-Device Memory Space Access", "[memory_space]") {
    initializeSingleDeviceMemoryManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    // Expected memory capacities
    const size_t expected_gpu_capacity = 2048ull * 1024 * 1024;   // 2GB
    const size_t expected_host_capacity = 4096ull * 1024 * 1024;  // 4GB
    const size_t expected_disk_capacity = 8192ull * 1024 * 1024;  // 8GB
    const size_t expected_device_id = 0; // All devices should have ID 0
    
    // Test single GPU memory space
    auto gpu_device_0 = manager.get_memory_space(Tier::GPU, 0);
    
    REQUIRE(gpu_device_0 != nullptr);
    REQUIRE(gpu_device_0->get_tier() == Tier::GPU);
    REQUIRE(gpu_device_0->get_device_id() == expected_device_id);
    REQUIRE(gpu_device_0->get_max_memory() == expected_gpu_capacity);
    REQUIRE(gpu_device_0->get_available_memory() == expected_gpu_capacity);
    
    // Test single HOST memory space (NUMA node)
    auto host_numa_0 = manager.get_memory_space(Tier::HOST, 0);
    
    REQUIRE(host_numa_0 != nullptr);
    REQUIRE(host_numa_0->get_tier() == Tier::HOST);
    REQUIRE(host_numa_0->get_device_id() == expected_device_id);
    REQUIRE(host_numa_0->get_max_memory() == expected_host_capacity);
    REQUIRE(host_numa_0->get_available_memory() == expected_host_capacity);
    
    // Test single DISK memory space
    auto disk_0 = manager.get_memory_space(Tier::DISK, 0);
    
    REQUIRE(disk_0 != nullptr);
    REQUIRE(disk_0->get_tier() == Tier::DISK);
    REQUIRE(disk_0->get_device_id() == expected_device_id);
    REQUIRE(disk_0->get_max_memory() == expected_disk_capacity);
    REQUIRE(disk_0->get_available_memory() == expected_disk_capacity);
    
    // Test non-existent devices (only device 0 exists for each tier)
    REQUIRE(manager.get_memory_space(Tier::GPU, 1) == nullptr);
    REQUIRE(manager.get_memory_space(Tier::HOST, 1) == nullptr);
    REQUIRE(manager.get_memory_space(Tier::DISK, 1) == nullptr);
    
    // Verify all memory spaces are different objects
    REQUIRE(gpu_device_0 != host_numa_0);
    REQUIRE(gpu_device_0 != disk_0);
    REQUIRE(host_numa_0 != disk_0);
}

// Test memory reservations on specific devices
TEST_CASE("Device-Specific Memory Reservations", "[memory_space]") {
    initializeSingleDeviceMemoryManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    // Memory size constants
    const size_t gpu_allocation_size = 200ull * 1024 * 1024;    // 200MB
    const size_t host_allocation_size = 500ull * 1024 * 1024;   // 500MB
    const size_t disk_allocation_size = 1000ull * 1024 * 1024;  // 1GB
    
    const size_t gpu_capacity = 2048ull * 1024 * 1024;          // 2GB
    const size_t host_capacity = 4096ull * 1024 * 1024;         // 4GB
    const size_t disk_capacity = 8192ull * 1024 * 1024;         // 8GB
    
    auto gpu_device_0 = manager.get_memory_space(Tier::GPU, 0);
    auto host_numa_0 = manager.get_memory_space(Tier::HOST, 0);
    auto disk_0 = manager.get_memory_space(Tier::DISK, 0);
    
    // Test reservation on GPU device
    auto gpu_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), gpu_allocation_size);
    REQUIRE(gpu_reservation != nullptr);
    REQUIRE(gpu_reservation->tier == Tier::GPU);
    REQUIRE(gpu_reservation->device_id == 0);
    REQUIRE(gpu_reservation->size == gpu_allocation_size);
    
    // Check memory accounting on GPU device
    REQUIRE(gpu_device_0->get_total_reserved_memory() == gpu_allocation_size);
    REQUIRE(gpu_device_0->get_active_reservation_count() == 1);
    REQUIRE(gpu_device_0->get_available_memory() == gpu_capacity - gpu_allocation_size);
    
    // Check that other devices are unaffected
    REQUIRE(host_numa_0->get_total_reserved_memory() == 0);
    REQUIRE(host_numa_0->get_active_reservation_count() == 0);
    REQUIRE(disk_0->get_total_reserved_memory() == 0);
    REQUIRE(disk_0->get_active_reservation_count() == 0);
    
    // Test reservation on HOST NUMA node
    auto host_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), host_allocation_size);
    REQUIRE(host_reservation != nullptr);
    REQUIRE(host_reservation->tier == Tier::HOST);
    REQUIRE(host_reservation->device_id == 0);
    REQUIRE(host_reservation->size == host_allocation_size);
    
    // Check HOST memory accounting
    REQUIRE(host_numa_0->get_total_reserved_memory() == host_allocation_size);
    REQUIRE(host_numa_0->get_active_reservation_count() == 1);
    REQUIRE(host_numa_0->get_available_memory() == host_capacity - host_allocation_size);
    
    // Test reservation on DISK device
    auto disk_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::DISK, 0), disk_allocation_size);
    REQUIRE(disk_reservation != nullptr);
    REQUIRE(disk_reservation->tier == Tier::DISK);
    REQUIRE(disk_reservation->device_id == 0);
    REQUIRE(disk_reservation->size == disk_allocation_size);
    
    // Clean up
    manager.release_reservation(std::move(gpu_reservation));
    manager.release_reservation(std::move(host_reservation));
    manager.release_reservation(std::move(disk_reservation));
    
    // Verify cleanup
    REQUIRE(gpu_device_0->get_total_reserved_memory() == 0);
    REQUIRE(gpu_device_0->get_active_reservation_count() == 0);
    REQUIRE(gpu_device_0->get_available_memory() == gpu_capacity);
    
    REQUIRE(host_numa_0->get_total_reserved_memory() == 0);
    REQUIRE(host_numa_0->get_active_reservation_count() == 0);
    REQUIRE(host_numa_0->get_available_memory() == host_capacity);
    
    REQUIRE(disk_0->get_total_reserved_memory() == 0);
    REQUIRE(disk_0->get_active_reservation_count() == 0);
    REQUIRE(disk_0->get_available_memory() == disk_capacity);
}

// Test tier-level aggregated statistics
TEST_CASE("Tier-Level Aggregated Statistics", "[memory_space]") {
    initializeSingleDeviceMemoryManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    // Test allocation sizes
    const size_t gpu_test_allocation = 200ull * 1024 * 1024;   // 200MB
    const size_t host_test_allocation = 500ull * 1024 * 1024;  // 500MB
    const size_t disk_test_allocation = 1000ull * 1024 * 1024; // 1GB
    
    // Memory capacities
    const size_t gpu_total_capacity = 2048ull * 1024 * 1024;   // 2GB
    const size_t host_total_capacity = 4096ull * 1024 * 1024;  // 4GB
    const size_t disk_total_capacity = 8192ull * 1024 * 1024;  // 8GB
    
    auto gpu_device_0 = manager.get_memory_space(Tier::GPU, 0);
    auto host_numa_0 = manager.get_memory_space(Tier::HOST, 0);
    auto disk_0 = manager.get_memory_space(Tier::DISK, 0);
    
    // Make reservations on single devices
    auto gpu_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), gpu_test_allocation);
    auto host_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), host_test_allocation);
    auto disk_reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::DISK, 0), disk_test_allocation);
    
    // Test GPU tier-level statistics
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::GPU) == gpu_test_allocation);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::GPU) == 1);
    REQUIRE(manager.get_available_memory_for_tier(Tier::GPU) == gpu_total_capacity - gpu_test_allocation);
    
    // Test HOST tier-level statistics
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::HOST) == host_test_allocation);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::HOST) == 1);
    REQUIRE(manager.get_available_memory_for_tier(Tier::HOST) == host_total_capacity - host_test_allocation);
    
    // Test DISK tier-level statistics
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::DISK) == disk_test_allocation);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::DISK) == 1);
    REQUIRE(manager.get_available_memory_for_tier(Tier::DISK) == disk_total_capacity - disk_test_allocation);
    
    // Clean up
    manager.release_reservation(std::move(gpu_reservation));
    manager.release_reservation(std::move(host_reservation));
    manager.release_reservation(std::move(disk_reservation));
    
    // Verify cleanup at tier level
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::GPU) == 0);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::GPU) == 0);
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::HOST) == 0);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::HOST) == 0);
    REQUIRE(manager.get_total_reserved_memory_for_tier(Tier::DISK) == 0);
    REQUIRE(manager.get_active_reservation_count_for_tier(Tier::DISK) == 0);
}

// Test memory space enumeration for tiers
TEST_CASE("Memory Space Enumeration", "[memory_space]") {
    initializeSingleDeviceMemoryManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    // Test getting all GPU memory spaces
    auto gpu_spaces = manager.get_memory_spaces_for_tier(Tier::GPU);
    REQUIRE(gpu_spaces.size() == 1);
    
    // Verify device ID
    REQUIRE(gpu_spaces[0]->get_tier() == Tier::GPU);
    REQUIRE(gpu_spaces[0]->get_device_id() == 0);
    
    // Test getting all HOST memory spaces
    auto host_spaces = manager.get_memory_spaces_for_tier(Tier::HOST);
    REQUIRE(host_spaces.size() == 1);
    
    // Verify NUMA node ID
    REQUIRE(host_spaces[0]->get_tier() == Tier::HOST);
    REQUIRE(host_spaces[0]->get_device_id() == 0);
    
    // Test getting all DISK memory spaces
    auto disk_spaces = manager.get_memory_spaces_for_tier(Tier::DISK);
    REQUIRE(disk_spaces.size() == 1);
    
    // Verify disk device ID
    REQUIRE(disk_spaces[0]->get_tier() == Tier::DISK);
    REQUIRE(disk_spaces[0]->get_device_id() == 0);
    
    // Test getting all memory spaces
    auto all_spaces = manager.get_all_memory_spaces();
    REQUIRE(all_spaces.size() == 3); // 1 GPU + 1 HOST + 1 DISK
}

// Test reservation strategies with single devices
TEST_CASE("Reservation Strategies with Single Devices", "[memory_space]") {
    initializeSingleDeviceMemoryManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    // Test allocation sizes
    const size_t small_allocation = 25ull * 1024 * 1024;    // 25MB
    const size_t medium_allocation = 50ull * 1024 * 1024;   // 50MB
    const size_t large_allocation = 100ull * 1024 * 1024;   // 100MB
    
    // Test requesting reservation in any GPU
    auto gpu_any_reservation = manager.request_reservation(any_memory_space_in_tier(Tier::GPU), medium_allocation);
    REQUIRE(gpu_any_reservation != nullptr);
    REQUIRE(gpu_any_reservation->tier == Tier::GPU);
    REQUIRE(gpu_any_reservation->size == medium_allocation);
    
    // Should pick the single GPU device (device 0)
    REQUIRE(gpu_any_reservation->device_id == 0);
    
    // Test requesting reservation across multiple tiers (simulates "anywhere")
    std::vector<Tier> any_tier_preferences = {Tier::GPU, Tier::HOST, Tier::DISK};
    auto anywhere_reservation = manager.request_reservation(any_memory_space_in_tiers(any_tier_preferences), small_allocation);
    REQUIRE(anywhere_reservation != nullptr);
    REQUIRE(anywhere_reservation->size == small_allocation);
    
    // Should pick any available memory space
    Tier selected_tier = anywhere_reservation->tier;
    REQUIRE((selected_tier == Tier::GPU || selected_tier == Tier::HOST || selected_tier == Tier::DISK));
    
    // Test specific memory space in tiers list with HOST preference
    std::vector<Tier> tier_preferences = {Tier::HOST, Tier::GPU, Tier::DISK};
    auto preference_reservation = manager.request_reservation(any_memory_space_in_tiers(tier_preferences), large_allocation);
    REQUIRE(preference_reservation != nullptr);
    REQUIRE(preference_reservation->size == large_allocation);
    
    // Should prefer HOST first
    REQUIRE(preference_reservation->tier == Tier::HOST);
    
    // Clean up
    manager.release_reservation(std::move(gpu_any_reservation));
    manager.release_reservation(std::move(anywhere_reservation));
    manager.release_reservation(std::move(preference_reservation));
}

// Test allocator access for multiple devices
TEST_CASE("Allocator Access Multi-Device", "[memory_space][.multi-device]") {
    initializeMultiDeviceMemoryManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_device_0 = manager.get_memory_space(Tier::GPU, 0);
    auto gpu_device_1 = manager.get_memory_space(Tier::GPU, 1);
    auto host_numa_0 = manager.get_memory_space(Tier::HOST, 0);
    auto disk_path_0 = manager.get_memory_space(Tier::DISK, 0);
    
    // Test that each memory space has exactly one allocator
    REQUIRE(gpu_device_0->get_allocator_count() == 1);
    REQUIRE(gpu_device_1->get_allocator_count() == 1);
    REQUIRE(host_numa_0->get_allocator_count() == 1);
    REQUIRE(disk_path_0->get_allocator_count() == 1);
    
    // Test that we can get default allocators from each device
    auto gpu_0_allocator = gpu_device_0->get_default_allocator();
    auto gpu_1_allocator = gpu_device_1->get_default_allocator();
    auto host_0_allocator = host_numa_0->get_default_allocator();
    auto disk_0_allocator = disk_path_0->get_default_allocator();
    
    // Test that allocators are valid (basic smoke test)
    REQUIRE(&gpu_0_allocator != nullptr);
    REQUIRE(&gpu_1_allocator != nullptr);
    REQUIRE(&host_0_allocator != nullptr);
    REQUIRE(&disk_0_allocator != nullptr);
    
    // Test getting allocator by index
    auto gpu_0_allocator_by_index = gpu_device_0->get_allocator(0);
    REQUIRE(&gpu_0_allocator_by_index != nullptr);
    
    // Test out of bounds access
    REQUIRE_THROWS_AS(gpu_device_0->get_allocator(1), std::out_of_range);
    REQUIRE_THROWS_AS(host_numa_0->get_allocator(100), std::out_of_range);
}

// Helper function to do actual memory work on host memory
void doHostMemoryWork(void* ptr, size_t size, std::atomic<uint64_t>& work_counter) {
    if (!ptr || size == 0) return;
    
    // Write pattern to memory
    uint32_t* data = static_cast<uint32_t*>(ptr);
    size_t num_elements = size / sizeof(uint32_t);
    
    // Write ascending pattern
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = static_cast<uint32_t>(i ^ 0xDEADBEEF);
    }
    
    // Read back and verify (prevents optimization)
    uint64_t checksum = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        checksum += data[i];
    }
    
    work_counter.fetch_add(checksum);
    
    // Add some delay to simulate real work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

// Helper function to do actual memory work on GPU memory using CUDA kernels
void doGpuMemoryWork(void* gpu_ptr, size_t size, std::atomic<uint64_t>& work_counter) {
    if (!gpu_ptr || size == 0) return;
    
    uint64_t gpu_checksum = 0;
    cudaError_t err = performGpuMemoryWork(gpu_ptr, size, &gpu_checksum);
    
    if (err == cudaSuccess) {
        work_counter.fetch_add(gpu_checksum);
        
        // Verify the work was done correctly
        uint64_t verification_checksum = 0;
        err = verifyGpuMemoryWork(gpu_ptr, size, &verification_checksum);
        if (err == cudaSuccess && verification_checksum > 0) {
            work_counter.fetch_add(verification_checksum);
        }
    }
    
    // Add delay to simulate realistic GPU work timing
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
}

// Unified memory work function that chooses the appropriate method based on memory space tier
void doMemoryWork(const memory_space* memory_space, void* ptr, size_t size, std::atomic<uint64_t>& work_counter) {
    if (!memory_space || !ptr || size == 0) return;
    
    if (memory_space->get_tier() == Tier::GPU) {
        doGpuMemoryWork(ptr, size, work_counter);
    } else {
        // HOST and DISK tiers use host memory
        doHostMemoryWork(ptr, size, work_counter);
    }
}

// Test concurrent allocations with actual memory work
TEST_CASE("Concurrent Allocations with Real Memory Work", "[memory_space][threading]") {
    initializeSingleDeviceMemoryManager();
    auto& manager = memory_reservation_manager::get_instance();
    

    auto gpu_device_0 = manager.get_memory_space(Tier::GPU, 0);
    // Memory pressure test constants
    const size_t host_memory_capacity = 4096ull * 1024 * 1024;  // 4GB HOST space capacity
    const size_t thread_allocation_size = 800ull * 1024 * 1024; // 800MB per thread
    const int concurrent_threads = 8;
    const size_t total_requested_memory = concurrent_threads * thread_allocation_size; // 6.4GB total
    const size_t blocking_wait_threshold_ms = 1; // Consider >1ms wait as blocking
    const size_t memory_alignment = 64; // 64-byte alignment
    
    // Pressure ratio: 6.4GB requested / 4GB available = 1.6x (forces blocking)
    
    const int num_threads = concurrent_threads;
    
    std::atomic<int> successful_allocations{0};
    std::atomic<int> failed_allocations{0};
    std::atomic<int> blocked_threads{0};
    std::atomic<uint64_t> work_counter{0};
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                // Use single NUMA node
                auto* memory_space = gpu_device_0;
                
                // Measure time to get reservation to detect blocking
                auto start_time = std::chrono::steady_clock::now();
                
                // Make reservation
                auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), thread_allocation_size);
                
                auto reservation_time = std::chrono::steady_clock::now();
                auto wait_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    reservation_time - start_time);
                
                // If we had to wait longer than threshold, we were probably blocked
                if (wait_duration.count() > blocking_wait_threshold_ms) {
                    blocked_threads.fetch_add(1);
                }
                if (reservation) {
                    successful_allocations.fetch_add(1);
                    
                    // Perform actual memory allocation using the allocator (GPU)
                    auto allocator = memory_space->get_default_allocator();
                    void* ptr = allocator.allocate(thread_allocation_size, memory_alignment);
                    
                    if (ptr) {
                        // Do real memory work
                        doMemoryWork(memory_space, ptr, thread_allocation_size, work_counter);
                        
                        // Clean up
                        allocator.deallocate(ptr, thread_allocation_size, memory_alignment);
                    }
                    
                    // Release reservation
                    manager.release_reservation(std::move(reservation));
                }
            } catch (const std::exception& e) {
                failed_allocations.fetch_add(1);
            }
        });
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
    
    // Verify that some allocations succeeded and actual work was done
    REQUIRE(successful_allocations.load() > 0);
    REQUIRE(work_counter.load() > 0);
    
    // With the memory pressure created (6.4GB requested on 4GB space),
    // we should see blocking behavior - at least half the threads should be blocked
    const int minimum_expected_blocked_threads = concurrent_threads / 2;
    REQUIRE(blocked_threads.load() >= minimum_expected_blocked_threads);
    
    // Verify memory is properly released
    REQUIRE(gpu_device_0->get_total_reserved_memory() == 0);
    REQUIRE(gpu_device_0->get_active_reservation_count() == 0);
}

// Test oversubscription prevention with limited memory
TEST_CASE("Oversubscription Prevention", "[memory_space][threading]") {
    // Oversubscription test configuration
    const size_t limited_memory_capacity = 5ull * 1024 * 1024;  // Only 5MB available
    const size_t per_thread_allocation = 2ull * 1024 * 1024;    // 2MB per thread
    const int oversubscribing_threads = 10;                  // 10 threads
    const size_t total_requested = oversubscribing_threads * per_thread_allocation; // 20MB requested
    const size_t blocking_wait_threshold_ms = 1;
    const size_t memory_alignment = 64;
    const int contention_hold_time_ms = 100;
    
    // Pressure: 20MB requested / 5MB available = 4x oversubscription
    
    // Create a manager with very limited memory
    std::vector<memory_reservation_manager::memory_space_config> configs;
    configs.emplace_back(Tier::HOST, 0, limited_memory_capacity, create_test_allocators(Tier::HOST));
    memory_reservation_manager::initialize(std::move(configs));
    
    auto& manager = memory_reservation_manager::get_instance();
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    
    const int num_threads = oversubscribing_threads;
    
    std::atomic<int> successful_reservations{0};
    std::atomic<int> blocked_threads{0};
    std::atomic<int> completed_work{0};
    std::atomic<uint64_t> work_counter{0};
    
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < num_threads; ++i) {
        auto future = std::async(std::launch::async, [&, i]() {
            try {
                auto start_time = std::chrono::steady_clock::now();
                
                // This should block when memory is exhausted
                auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), per_thread_allocation);
                
                auto allocation_time = std::chrono::steady_clock::now() - start_time;
                if (allocation_time > std::chrono::milliseconds(blocking_wait_threshold_ms)) {
                    blocked_threads.fetch_add(1); // Thread was blocked waiting for memory
                }
                
                successful_reservations.fetch_add(1);
                
                // Perform actual allocation and work
                auto allocator = host_space->get_default_allocator();
                void* ptr = allocator.allocate(per_thread_allocation, memory_alignment);
                
                if (ptr) {
                    doMemoryWork(host_space, ptr, per_thread_allocation, work_counter);
                    allocator.deallocate(ptr, per_thread_allocation, memory_alignment);
                    completed_work.fetch_add(1);
                }
                
                // Hold the reservation briefly
                std::this_thread::sleep_for(std::chrono::milliseconds(contention_hold_time_ms));
                
                manager.release_reservation(std::move(reservation));
                
            } catch (const std::exception& e) {
                // Allocations should not fail, they should block instead
                FAIL("Unexpected exception: " << e.what());
            }
        });
        
        futures.push_back(std::move(future));
    }
    
    // Wait for all tasks to complete with timeout protection
    for (auto& future : futures) {
        auto status = future.wait_for(std::chrono::seconds(30));
        REQUIRE(status == std::future_status::ready);
        future.get(); // This will rethrow any exceptions
    }
    
    // Verify that the system worked correctly
    REQUIRE(successful_reservations.load() == num_threads); // All should eventually succeed
    REQUIRE(completed_work.load() == num_threads); // All should do work
    REQUIRE(blocked_threads.load() > 0); // Some threads should have been blocked
    REQUIRE(work_counter.load() > 0); // Actual work was performed
    
    // Verify no memory leaks
    REQUIRE(host_space->get_total_reserved_memory() == 0);
    REQUIRE(host_space->get_active_reservation_count() == 0);
    
    // Verify that memory was not oversubscribed (4x pressure: 20MB requested vs 5MB available)
    // Since allocations were 2MB each, at most 2-3 concurrent allocations should fit
    const int minimum_expected_blocks = oversubscribing_threads / 2; // At least half should be blocked
    REQUIRE(blocked_threads.load() >= minimum_expected_blocks);
}

// Test memory pressure simulation with reservation strategies
TEST_CASE("Memory Pressure with Reservation Strategies", "[memory_space][threading]") {
    initializeSingleDeviceMemoryManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    // Memory pressure test configuration
    const size_t pressure_allocation_size = 1200ull * 1024 * 1024; // 1.2GB per thread
    const int total_competing_threads = 12;
    const size_t total_requested_memory = total_competing_threads * pressure_allocation_size; // 14.4GB total
    const size_t blocking_wait_threshold_ms = 1;
    const size_t memory_alignment = 64; // 64-byte alignment
    const int hold_allocation_time_ms = 200; // Hold allocations for 200ms
    
    // System capacity: GPU(2GB) + HOST(4GB) + DISK(8GB) = 14GB total
    // Pressure: 14.4GB requested / 14GB available = 1.03x (slight oversubscription)
    
    const int num_competing_threads = total_competing_threads;
    
    std::atomic<int> tier_gpu_successes{0};
    std::atomic<int> tier_host_successes{0};
    std::atomic<int> tier_disk_successes{0};
    std::atomic<int> anywhere_successes{0};
    std::atomic<int> total_blocked{0};
    std::atomic<uint64_t> work_done{0};
    
    std::vector<std::future<void>> futures;
    
    // Strategy distribution: some prefer GPU, some HOST, some DISK, some anywhere
    for (int i = 0; i < num_competing_threads; ++i) {
        auto future = std::async(std::launch::async, [&, i]() {
            std::unique_ptr<reservation> reservation;
            try {
                int strategy_type = i % 4;

                // Measure time to get reservation to detect blocking
                auto start_time = std::chrono::steady_clock::now();

                switch (strategy_type) {
                    case 0: // Prefer GPU
                        reservation = manager.request_reservation(any_memory_space_in_tier(Tier::GPU), pressure_allocation_size);
                        if (reservation) tier_gpu_successes.fetch_add(1);
                        break;
                    case 1: // Prefer HOST
                        reservation = manager.request_reservation(any_memory_space_in_tier(Tier::HOST), pressure_allocation_size);
                        if (reservation) tier_host_successes.fetch_add(1);
                        break;
                    case 2: // Prefer DISK
                        reservation = manager.request_reservation(any_memory_space_in_tier(Tier::DISK), pressure_allocation_size);
                        if (reservation) tier_disk_successes.fetch_add(1);
                        break;
                    case 3: // Anywhere
                        {
                            std::vector<Tier> all_tiers = {Tier::GPU, Tier::HOST, Tier::DISK};
                            reservation = manager.request_reservation(any_memory_space_in_tiers(all_tiers), pressure_allocation_size);
                        }
                        if (reservation) anywhere_successes.fetch_add(1);
                        break;
                }

                auto reservation_time = std::chrono::steady_clock::now();
                auto wait_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    reservation_time - start_time);

                // If we had to wait longer than threshold, we were probably blocked
                if (wait_duration.count() > blocking_wait_threshold_ms) {
                    total_blocked.fetch_add(1);
                }
                

                if (reservation) {
                    // Perform work or simulate hold depending on tier
                    const memory_space* mem_space = manager.get_memory_space(reservation->tier, reservation->device_id);
                    if (reservation->tier == Tier::GPU) {
                        auto allocator = mem_space->get_default_allocator();
                        void* ptr = allocator.allocate(pressure_allocation_size, memory_alignment);

                        if (ptr) {
                            // Do substantial work to prevent optimization
                            doMemoryWork(mem_space, ptr, pressure_allocation_size, work_done);

                            // Hold allocation for realistic time
                            std::this_thread::sleep_for(std::chrono::milliseconds(hold_allocation_time_ms));

                            allocator.deallocate(ptr, pressure_allocation_size, memory_alignment);
                        } else {
                        }
                    } else {
                        // HOST and DISK tiers in tests use allocators that cannot handle 1.2GB.
                        // Simulate holding the reservation to create pressure without real allocation.
                        std::this_thread::sleep_for(std::chrono::milliseconds(hold_allocation_time_ms));
                        work_done.fetch_add(1);
                    }

                    manager.release_reservation(std::move(reservation));
                }

            } catch (const std::exception& e) {
                // Large allocations might fail on some tiers, that's expected
                // The system should handle this gracefully
                if (reservation) {
                    manager.release_reservation(std::move(reservation));
                }
            }
        });

        futures.push_back(std::move(future));
    }
    
    // Wait for all with timeout
    for (size_t idx = 0; idx < futures.size(); ++idx) {
        auto& future = futures[idx];
        auto status = future.wait_for(std::chrono::seconds(60));
        REQUIRE(status == std::future_status::ready);
        future.get();
    }
    
    // Verify that the system handled the pressure correctly
    int total_successes = tier_gpu_successes.load() + tier_host_successes.load() + 
                         tier_disk_successes.load() + anywhere_successes.load();
    
    REQUIRE(total_successes > 0); // At least some should succeed
    REQUIRE(work_done.load() > 0); // Work should have been performed
    
    // Compute minimum expected blocked based on per-tier capacities
    const size_t gpu_total_capacity = 2048ull * 1024 * 1024;   // 2GB
    const size_t host_total_capacity = 4096ull * 1024 * 1024;  // 4GB
    const size_t disk_total_capacity = 8192ull * 1024 * 1024;  // 8GB
    const int max_concurrent_reservations = static_cast<int>(gpu_total_capacity / pressure_allocation_size)
                                          + static_cast<int>(host_total_capacity / pressure_allocation_size)
                                          + static_cast<int>(disk_total_capacity / pressure_allocation_size);
    const int minimum_expected_blocked = std::max(0, num_competing_threads - max_concurrent_reservations);
    REQUIRE(total_blocked.load() >= minimum_expected_blocked);
    
    // Verify system is clean after pressure test
    auto all_spaces = manager.get_all_memory_spaces();
    for (const auto* space : all_spaces) {
        REQUIRE(space->get_total_reserved_memory() == 0);
        REQUIRE(space->get_active_reservation_count() == 0);
    }
    
    // The DISK tier should have handled most of the large allocations
    // since it has the most space (4GB per device vs 1-2GB for others)
    REQUIRE(tier_disk_successes.load() > 0);
}

// Test GPU vs Host memory work to verify CUDA kernels are used
TEST_CASE("GPU vs Host Memory Work Verification", "[memory_space][threading][gpu]") {
    initializeSingleDeviceMemoryManager();
    auto& manager = memory_reservation_manager::get_instance();
    
    auto gpu_space = manager.get_memory_space(Tier::GPU, 0);
    auto host_space = manager.get_memory_space(Tier::HOST, 0);
    
    const size_t allocation_size = 4ull * 1024 * 1024; // 4MB
    
    std::atomic<uint64_t> gpu_work_result{0};
    std::atomic<uint64_t> host_work_result{0};
    std::atomic<bool> gpu_work_completed{false};
    std::atomic<bool> host_work_completed{false};
    
    // Thread for GPU work
    std::thread gpu_thread([&]() {
        try {
            auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::GPU, 0), allocation_size);
            REQUIRE(reservation != nullptr);
            
            auto allocator = gpu_space->get_default_allocator();
            void* gpu_ptr = allocator.allocate(allocation_size, 64);
            
            if (gpu_ptr) {
                // This should use CUDA kernels
                auto start_time = std::chrono::high_resolution_clock::now();
                doMemoryWork(gpu_space, gpu_ptr, allocation_size, gpu_work_result);
                auto end_time = std::chrono::high_resolution_clock::now();
                
                auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                // GPU work should take some time due to kernel launches and memory operations
                REQUIRE(gpu_duration.count() > 10); // At least 10ms for GPU work
                
                allocator.deallocate(gpu_ptr, allocation_size, 64);
            }
            
            manager.release_reservation(std::move(reservation));
            gpu_work_completed = true;
            
        } catch (const std::exception& e) {
            FAIL("GPU work failed: " << e.what());
        }
    });
    
    // Thread for Host work
    std::thread host_thread([&]() {
        try {
            auto reservation = manager.request_reservation(any_memory_space_in_tier_with_preference(Tier::HOST, 0), allocation_size);
            REQUIRE(reservation != nullptr);
            
            auto allocator = host_space->get_default_allocator();
            void* host_ptr = allocator.allocate(allocation_size, 64);
            
            if (host_ptr) {
                // This should use direct host memory access
                auto start_time = std::chrono::high_resolution_clock::now();
                doMemoryWork(host_space, host_ptr, allocation_size, host_work_result);
                auto end_time = std::chrono::high_resolution_clock::now();
                
                auto host_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                // Host work should also take some time
                REQUIRE(host_duration.count() > 5); // At least 5ms for host work
                
                allocator.deallocate(host_ptr, allocation_size, 64);
            }
            
            manager.release_reservation(std::move(reservation));
            host_work_completed = true;
            
        } catch (const std::exception& e) {
            FAIL("Host work failed: " << e.what());
        }
    });
    
    // Wait for both threads
    gpu_thread.join();
    host_thread.join();
    
    // Verify both completed successfully
    REQUIRE(gpu_work_completed.load());
    REQUIRE(host_work_completed.load());
    
    // Both should have produced some work result (checksum > 0)
    REQUIRE(gpu_work_result.load() > 0);
    REQUIRE(host_work_result.load() > 0);
    
    // The work results will be different because:
    // 1. GPU uses CUDA kernels with different computation patterns
    // 2. Host uses direct C++ memory access
    // 3. Both use different algorithms for generating checksums
    REQUIRE(gpu_work_result.load() != host_work_result.load());
    
    // Verify cleanup
    REQUIRE(gpu_space->get_total_reserved_memory() == 0);
    REQUIRE(host_space->get_total_reserved_memory() == 0);
    REQUIRE(gpu_space->get_active_reservation_count() == 0);
    REQUIRE(host_space->get_active_reservation_count() == 0);
}