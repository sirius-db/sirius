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
#include <memory>
#include <vector>
#include "data/cpu_data_representation.hpp"
#include "data/gpu_data_representation.hpp"
#include "data/common.hpp"
#include "memory/null_device_memory_resource.hpp"
#include "memory/host_table.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>

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

// Helper function to create a mock host_table_allocation for testing
sirius::unique_ptr<memory::host_table_allocation> create_mock_host_table_allocation(std::size_t data_size) {
    // Create empty allocation blocks (we're not testing actual allocation here)
    // Use an empty vector and nullptr since we're just mocking
    std::vector<void*> empty_blocks;
    memory::fixed_size_host_memory_resource::multiple_blocks_allocation empty_allocation(
        std::move(empty_blocks),
        nullptr,  // No actual memory resource in mock
        0         // Block size doesn't matter for empty allocation
    );
    
    // Create mock metadata
    auto metadata = sirius::make_unique<sirius::vector<uint8_t>>();
    metadata->push_back(0x01);
    metadata->push_back(0x02);
    metadata->push_back(0x03);
    
    return sirius::make_unique<memory::host_table_allocation>(
        std::move(empty_allocation),
        std::move(metadata),
        data_size
    );
}

// Helper function to create a simple cuDF table for testing
cudf::table create_simple_cudf_table(int num_rows = 100) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    
    // Create a simple INT32 column
    auto col1 = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        cudf::mask_state::UNALLOCATED
    );
    
    // Create another INT64 column
    auto col2 = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64},
        num_rows,
        cudf::mask_state::UNALLOCATED
    );
    
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    
    return cudf::table(std::move(columns));
}

// =============================================================================
// host_table_representation Tests
// =============================================================================

TEST_CASE("host_table_representation Construction", "[cpu_data_representation]") {
    mock_memory_space host_space(memory::Tier::HOST, 0);
    auto host_table = create_mock_host_table_allocation(2048);
    
    host_table_representation repr(std::move(host_table), host_space);
    
    REQUIRE(repr.get_current_tier() == memory::Tier::HOST);
    REQUIRE(repr.get_device_id() == 0);
    REQUIRE(repr.get_size_in_bytes() == 2048);
}

TEST_CASE("host_table_representation get_size_in_bytes", "[cpu_data_representation]") {
    mock_memory_space host_space(memory::Tier::HOST, 0);
    
    SECTION("Small data size") {
        auto host_table = create_mock_host_table_allocation(512);
        host_table_representation repr(std::move(host_table), host_space);
        
        REQUIRE(repr.get_size_in_bytes() == 512);
    }
    
    SECTION("Large data size") {
        auto host_table = create_mock_host_table_allocation(1024 * 1024);
        host_table_representation repr(std::move(host_table), host_space);
        
        REQUIRE(repr.get_size_in_bytes() == 1024 * 1024);
    }
    
    SECTION("Zero data size") {
        auto host_table = create_mock_host_table_allocation(0);
        host_table_representation repr(std::move(host_table), host_space);
        
        REQUIRE(repr.get_size_in_bytes() == 0);
    }
}

TEST_CASE("host_table_representation memory tier", "[cpu_data_representation]") {
    SECTION("HOST tier") {
        mock_memory_space host_space(memory::Tier::HOST, 0);
        auto host_table = create_mock_host_table_allocation(1024);
        host_table_representation repr(std::move(host_table), host_space);
        
        REQUIRE(repr.get_current_tier() == memory::Tier::HOST);
    }
}

TEST_CASE("host_table_representation device_id", "[cpu_data_representation]") {
    SECTION("Device 0") {
        mock_memory_space host_space(memory::Tier::HOST, 0);
        auto host_table = create_mock_host_table_allocation(1024);
        host_table_representation repr(std::move(host_table), host_space);
        
        REQUIRE(repr.get_device_id() == 0);
    }
    
    SECTION("Device 1") {
        mock_memory_space host_space(memory::Tier::HOST, 1);
        auto host_table = create_mock_host_table_allocation(1024);
        host_table_representation repr(std::move(host_table), host_space);
        
        REQUIRE(repr.get_device_id() == 1);
    }
}

TEST_CASE("host_table_representation convert_to_memory_space throws", "[cpu_data_representation]") {
    mock_memory_space host_space(memory::Tier::HOST, 0);
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    auto host_table = create_mock_host_table_allocation(1024);
    host_table_representation repr(std::move(host_table), host_space);
    
    // Currently not implemented, so should throw
    REQUIRE_THROWS_AS(
        repr.convert_to_memory_space(gpu_space),
        std::runtime_error
    );
}

// =============================================================================
// gpu_table_representation Tests
// =============================================================================

TEST_CASE("gpu_table_representation Construction", "[gpu_data_representation]") {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    auto table = create_simple_cudf_table(100);
    
    gpu_table_representation repr(std::move(table), gpu_space);
    
    REQUIRE(repr.get_current_tier() == memory::Tier::GPU);
    REQUIRE(repr.get_device_id() == 0);
    REQUIRE(repr.get_size_in_bytes() > 0);
}

TEST_CASE("gpu_table_representation get_size_in_bytes", "[gpu_data_representation]") {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    
    SECTION("100 rows") {
        auto table = create_simple_cudf_table(100);
        gpu_table_representation repr(std::move(table), gpu_space);
        
        // Size should be at least 100 rows * (4 bytes for INT32 + 8 bytes for INT64)
        std::size_t expected_min_size = 100 * (4 + 8);
        REQUIRE(repr.get_size_in_bytes() >= expected_min_size);
    }
    
    SECTION("1000 rows") {
        auto table = create_simple_cudf_table(1000);
        gpu_table_representation repr(std::move(table), gpu_space);
        
        // Size should be at least 1000 rows * (4 bytes for INT32 + 8 bytes for INT64)
        std::size_t expected_min_size = 1000 * (4 + 8);
        REQUIRE(repr.get_size_in_bytes() >= expected_min_size);
    }
    
    SECTION("Empty table") {
        auto table = create_simple_cudf_table(0);
        gpu_table_representation repr(std::move(table), gpu_space);
        
        REQUIRE(repr.get_size_in_bytes() == 0);
    }
}

TEST_CASE("gpu_table_representation get_table", "[gpu_data_representation]") {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    auto table = create_simple_cudf_table(100);
    
    // Store the number of columns before moving the table
    auto num_columns = table.num_columns();
    
    gpu_table_representation repr(std::move(table), gpu_space);
    
    const cudf::table& retrieved_table = repr.get_table();
    REQUIRE(retrieved_table.num_columns() == num_columns);
    REQUIRE(retrieved_table.num_rows() == 100);
}

TEST_CASE("gpu_table_representation memory tier", "[gpu_data_representation]") {
    SECTION("GPU tier") {
        mock_memory_space gpu_space(memory::Tier::GPU, 0);
        auto table = create_simple_cudf_table(100);
        gpu_table_representation repr(std::move(table), gpu_space);
        
        REQUIRE(repr.get_current_tier() == memory::Tier::GPU);
    }
}

TEST_CASE("gpu_table_representation device_id", "[gpu_data_representation]") {
    SECTION("Device 0") {
        mock_memory_space gpu_space(memory::Tier::GPU, 0);
        auto table = create_simple_cudf_table(100);
        gpu_table_representation repr(std::move(table), gpu_space);
        
        REQUIRE(repr.get_device_id() == 0);
    }
    
    SECTION("Device 1") {
        mock_memory_space gpu_space(memory::Tier::GPU, 1);
        auto table = create_simple_cudf_table(100);
        gpu_table_representation repr(std::move(table), gpu_space);
        
        REQUIRE(repr.get_device_id() == 1);
    }
}

TEST_CASE("gpu_table_representation convert_to_memory_space throws", "[gpu_data_representation]") {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    mock_memory_space host_space(memory::Tier::HOST, 0);
    auto table = create_simple_cudf_table(100);
    gpu_table_representation repr(std::move(table), gpu_space);
    
    // Currently not implemented, so should throw
    REQUIRE_THROWS_AS(
        repr.convert_to_memory_space(host_space),
        std::runtime_error
    );
}

// =============================================================================
// idata_representation Interface Tests
// =============================================================================

TEST_CASE("idata_representation cast functionality", "[cpu_data_representation][gpu_data_representation]") {
    SECTION("Cast host_table_representation") {
        mock_memory_space host_space(memory::Tier::HOST, 0);
        auto host_table = create_mock_host_table_allocation(1024);
        host_table_representation repr(std::move(host_table), host_space);
        
        idata_representation* base_ptr = &repr;
        
        // Cast to derived type
        host_table_representation& casted = base_ptr->cast<host_table_representation>();
        REQUIRE(&casted == &repr);
        REQUIRE(casted.get_size_in_bytes() == 1024);
    }
    
    SECTION("Cast gpu_table_representation") {
        mock_memory_space gpu_space(memory::Tier::GPU, 0);
        auto table = create_simple_cudf_table(100);
        gpu_table_representation repr(std::move(table), gpu_space);
        
        idata_representation* base_ptr = &repr;
        
        // Cast to derived type
        gpu_table_representation& casted = base_ptr->cast<gpu_table_representation>();
        REQUIRE(&casted == &repr);
        REQUIRE(casted.get_table().num_rows() == 100);
    }
}

TEST_CASE("idata_representation const cast functionality", "[cpu_data_representation][gpu_data_representation]") {
    SECTION("Const cast host_table_representation") {
        mock_memory_space host_space(memory::Tier::HOST, 0);
        auto host_table = create_mock_host_table_allocation(1024);
        host_table_representation repr(std::move(host_table), host_space);
        
        const idata_representation* base_ptr = &repr;
        
        // Const cast to derived type
        const host_table_representation& casted = base_ptr->cast<host_table_representation>();
        REQUIRE(&casted == &repr);
        REQUIRE(casted.get_size_in_bytes() == 1024);
    }
    
    SECTION("Const cast gpu_table_representation") {
        mock_memory_space gpu_space(memory::Tier::GPU, 0);
        auto table = create_simple_cudf_table(100);
        gpu_table_representation repr(std::move(table), gpu_space);
        
        const idata_representation* base_ptr = &repr;
        
        // Const cast to derived type
        const gpu_table_representation& casted = base_ptr->cast<gpu_table_representation>();
        REQUIRE(&casted == &repr);
        REQUIRE(casted.get_table().num_rows() == 100);
    }
}

// =============================================================================
// Cross-Tier Comparison Tests
// =============================================================================

TEST_CASE("Compare CPU and GPU representations", "[cpu_data_representation][gpu_data_representation]") {
    mock_memory_space host_space(memory::Tier::HOST, 0);
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    
    auto host_table = create_mock_host_table_allocation(1200);
    host_table_representation host_repr(std::move(host_table), host_space);
    
    auto gpu_table = create_simple_cudf_table(100);
    gpu_table_representation gpu_repr(std::move(gpu_table), gpu_space);
    
    // Verify they have different tiers
    REQUIRE(host_repr.get_current_tier() != gpu_repr.get_current_tier());
    REQUIRE(host_repr.get_current_tier() == memory::Tier::HOST);
    REQUIRE(gpu_repr.get_current_tier() == memory::Tier::GPU);
    
    // Both should have valid sizes
    REQUIRE(host_repr.get_size_in_bytes() > 0);
    REQUIRE(gpu_repr.get_size_in_bytes() > 0);
}

TEST_CASE("Multiple representations on same memory space", "[cpu_data_representation][gpu_data_representation]") {
    SECTION("Multiple host representations") {
        mock_memory_space host_space(memory::Tier::HOST, 0);
        
        auto host_table1 = create_mock_host_table_allocation(1024);
        host_table_representation repr1(std::move(host_table1), host_space);
        
        auto host_table2 = create_mock_host_table_allocation(2048);
        host_table_representation repr2(std::move(host_table2), host_space);
        
        REQUIRE(repr1.get_current_tier() == repr2.get_current_tier());
        REQUIRE(repr1.get_device_id() == repr2.get_device_id());
        REQUIRE(repr1.get_size_in_bytes() != repr2.get_size_in_bytes());
    }
    
    SECTION("Multiple GPU representations") {
        mock_memory_space gpu_space(memory::Tier::GPU, 0);
        
        auto table1 = create_simple_cudf_table(100);
        gpu_table_representation repr1(std::move(table1), gpu_space);
        
        auto table2 = create_simple_cudf_table(200);
        gpu_table_representation repr2(std::move(table2), gpu_space);
        
        REQUIRE(repr1.get_current_tier() == repr2.get_current_tier());
        REQUIRE(repr1.get_device_id() == repr2.get_device_id());
        // Different row counts should result in different sizes
        REQUIRE(repr1.get_size_in_bytes() != repr2.get_size_in_bytes());
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_CASE("gpu_table_representation with single column", "[gpu_data_representation]") {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    
    std::vector<std::unique_ptr<cudf::column>> columns;
    auto col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        100,
        cudf::mask_state::UNALLOCATED
    );
    columns.push_back(std::move(col));
    
    cudf::table table(std::move(columns));
    gpu_table_representation repr(std::move(table), gpu_space);
    
    REQUIRE(repr.get_table().num_columns() == 1);
    REQUIRE(repr.get_table().num_rows() == 100);
    REQUIRE(repr.get_size_in_bytes() >= 100 * 4); // At least 100 rows * 4 bytes
}

TEST_CASE("gpu_table_representation with multiple column types", "[gpu_data_representation]") {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    
    std::vector<std::unique_ptr<cudf::column>> columns;
    
    // INT8 column
    auto col1 = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT8},
        100,
        cudf::mask_state::UNALLOCATED
    );
    
    // INT16 column
    auto col2 = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT16},
        100,
        cudf::mask_state::UNALLOCATED
    );
    
    // INT32 column
    auto col3 = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        100,
        cudf::mask_state::UNALLOCATED
    );
    
    // INT64 column
    auto col4 = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64},
        100,
        cudf::mask_state::UNALLOCATED
    );
    
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    columns.push_back(std::move(col3));
    columns.push_back(std::move(col4));
    
    cudf::table table(std::move(columns));
    gpu_table_representation repr(std::move(table), gpu_space);
    
    REQUIRE(repr.get_table().num_columns() == 4);
    REQUIRE(repr.get_table().num_rows() == 100);
    // Size should be at least 100 * (1 + 2 + 4 + 8) = 1500 bytes
    REQUIRE(repr.get_size_in_bytes() >= 1500);
}

TEST_CASE("Representations polymorphism", "[cpu_data_representation][gpu_data_representation]") {
    mock_memory_space host_space(memory::Tier::HOST, 0);
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    
    // Create vector of base class pointers
    std::vector<std::unique_ptr<idata_representation>> representations;
    
    auto host_table = create_mock_host_table_allocation(1024);
    representations.push_back(
        std::make_unique<host_table_representation>(std::move(host_table), host_space)
    );
    
    auto gpu_table = create_simple_cudf_table(100);
    representations.push_back(
        std::make_unique<gpu_table_representation>(std::move(gpu_table), gpu_space)
    );
    
    // Access through base class interface
    REQUIRE(representations[0]->get_current_tier() == memory::Tier::HOST);
    REQUIRE(representations[1]->get_current_tier() == memory::Tier::GPU);
    
    REQUIRE(representations[0]->get_size_in_bytes() == 1024);
    REQUIRE(representations[1]->get_size_in_bytes() > 0);
}

