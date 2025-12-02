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
#include "data/data_repository_manager.hpp"
#include "data/gpu_data_representation.hpp"
#include "memory/memory_space.hpp"
#include "memory/memory_reservation.hpp"
#include "memory_management/memory_test_common.hpp"
#include "merge/gpu_merge_impl.hpp"
#include "aggregate/gpu_aggregate_impl.hpp"
#include "order/gpu_order_impl.hpp"
#include "utils/utils.hpp"

#include <cudf/utilities/bit.hpp>

using namespace sirius;
using namespace sirius::memory;
using namespace sirius::op;

namespace {

// Helper function to initialize single-device memory manager
void initialize_memory_manager() {
        memory_reservation_manager::reset_for_testing();
        std::vector<memory_reservation_manager::memory_space_config> configs;
        configs.emplace_back(Tier::GPU, 0, 1024 * 1024, create_test_allocators(Tier::GPU));  // GPU device 0: 1MB
    memory_reservation_manager::initialize(std::move(configs));
}

// Helper function to get the default GPU memory space
memory_space* get_default_memory_space() {
    initialize_memory_manager();
    auto& manager = memory_reservation_manager::get_instance();
    return const_cast<memory_space*>(manager.get_memory_space(Tier::GPU, 0));
}

sirius::vector<sirius::unique_ptr<data_batch_view>> create_batches_with_random_data(
        const int num_batches, const sirius::vector<int> num_rows,
        const sirius::vector<cudf::data_type>& column_types,
        const sirius::vector<std::optional<std::pair<int, int>>>& ranges,
        data_repository_manager& data_repo_manager, memory_space& mem_space) {
    sirius::vector<sirius::unique_ptr<data_batch_view>> batches;
    for (int i = 0; i < num_batches; ++i) {
        // Create a data batch
        auto table = create_cudf_table_with_random_data(
            num_rows[i],
            column_types,
            ranges,
            cudf::get_default_stream(),
            mem_space.get_default_allocator());
        auto gpu_repr = sirius::make_unique<gpu_table_representation>(*table, mem_space);
        auto batch = sirius::make_unique<data_batch>(data_repo_manager.get_next_data_batch_id(),
                                                    data_repo_manager,
                                                    std::move(gpu_repr));

        // Put batch into repository, create a view, and pin it
        auto* batch_ptr = batch.get();
        data_repo_manager.add_new_data_batch(std::move(batch), {});
        batches.push_back(sirius::make_unique<data_batch_view>(batch_ptr));
        batches.back()->pin();
    }
    return batches;
}

void validate_concat(const sirius::vector<sirius::unique_ptr<data_batch_view>>& input_views,
                    const sirius::data_batch& output) {
    sirius::vector<cudf::table_view> input_table_views;
    int expected_num_rows = 0;
    for (const auto& input_view: input_views) {
        input_table_views.push_back(input_view->get_cudf_table_view());
        expected_num_rows += input_table_views.back().num_rows();
    }
    cudf::table_view output_table_view = output.get_data()->cast<gpu_table_representation>().get_table().view();

    REQUIRE(expected_num_rows == output_table_view.num_rows());
    REQUIRE(input_table_views[0].num_columns() == output_table_view.num_columns());

    for (int c = 0; c < output_table_view.num_columns(); ++c) {
        REQUIRE(input_table_views[0].column(c).type().id() == output_table_view.column(c).type().id());
        if (expected_num_rows == 0) {
            continue;
        }

        switch (output_table_view.column(c).type().id()) {
            case cudf::type_id::INT32: {
                sirius::vector<int32_t> actual_data(expected_num_rows), expected_data(expected_num_rows);
                cudaMemcpy(actual_data.data(),
                        output_table_view.column(c).data<int32_t>(),
                        sizeof(int32_t) * expected_num_rows,
                        cudaMemcpyDeviceToHost);
                int num_input_copied = 0;
                for (const auto& input_table_view: input_table_views) {
                    cudaMemcpy(expected_data.data() + num_input_copied,
                            input_table_view.column(c).data<int32_t>(),
                            sizeof(int32_t) * input_table_view.num_rows(),
                            cudaMemcpyDeviceToHost);
                    num_input_copied += input_table_view.num_rows();
                }
                for (int r = 0; r < expected_num_rows; ++r) {
                    REQUIRE(expected_data[r] == actual_data[r]);
                }
                break;
            }
            case cudf::type_id::STRING: {
                sirius::vector<cudf::size_type> actual_offsets(expected_num_rows + 1);
                cudf::strings_column_view str_col(output_table_view.column(c));
                cudaMemcpy(actual_offsets.data(),
                        str_col.offsets().data<cudf::size_type>(),
                        (expected_num_rows + 1) * sizeof(cudf::size_type),
                        cudaMemcpyDeviceToHost);
                sirius::vector<char> actual_data(actual_offsets.back());
                cudaMemcpy(actual_data.data(),
                        str_col.chars_begin(cudf::get_default_stream()),
                        actual_offsets.back(),
                        cudaMemcpyDeviceToHost);
                
                sirius::vector<cudf::size_type> expected_offsets{0};
                sirius::vector<char> expected_data(actual_data.size());
                for (int i = 0; i < input_views.size(); ++i) {
                    if (input_table_views[i].num_rows() == 0) {
                        continue;
                    }
                    sirius::vector<cudf::size_type> input_offsets(input_table_views[i].num_rows() + 1);
                    str_col = cudf::strings_column_view(input_table_views[i].column(c));
                    cudaMemcpy(input_offsets.data(),
                            str_col.offsets().data<cudf::size_type>(),
                            (input_table_views[i].num_rows() + 1) * sizeof(cudf::size_type),
                            cudaMemcpyDeviceToHost);
                    int curr_last_offset = expected_offsets.back();
                    for (int r = 1; r <= input_table_views[i].num_rows(); ++r) {
                        expected_offsets.push_back(curr_last_offset + input_offsets[r]);
                    }
                    cudaMemcpy(expected_data.data() + curr_last_offset,
                        str_col.chars_begin(cudf::get_default_stream()),
                        input_offsets.back(),
                        cudaMemcpyDeviceToHost);
                }

                for (int r = 0; r <= expected_num_rows; ++r) {
                    REQUIRE(expected_offsets[r] == actual_offsets[r]);
                }
                for (int i = 0; i < expected_data.size(); ++i) {
                    REQUIRE(expected_data[i] == actual_data[i]);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported cudf::data_type in `validate_concat()`");
        }
    }
}

}

TEST_CASE("Concatenate multiple data batches", "[operator][merge_concat]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_rows_per_batch = 100;
    sirius::vector<int> num_input_rows(num_batches, num_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::STRING}};

    auto input_views = create_batches_with_random_data(
        num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt},
        data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::concat(input_views,
                                            cudf::get_default_stream(),
                                            *mem_space,
                                            data_repo_manager);
    validate_concat(input_views, *output_batch);
}

TEST_CASE("Concatenate multiple data batches with different size", "[operator][merge_concat]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    sirius::vector<int> num_input_rows;
    for (int i = 0; i < num_batches; ++i) {
        num_input_rows.push_back((i + 1) * 10);
    }
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::STRING}};

    auto input_views = create_batches_with_random_data(
        num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt},
        data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::concat(input_views,
                                            cudf::get_default_stream(),
                                            *mem_space,
                                            data_repo_manager);
    validate_concat(input_views, *output_batch);
}

TEST_CASE("Concatenate with invalid input", "[operator][merge_concat]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 1;
    constexpr size_t num_rows_per_batch = 100;
    sirius::vector<int> num_input_rows(num_batches, num_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::STRING}};

    // Invalid input: less than two input batches
    auto input_views = create_batches_with_random_data(
        num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt},
        data_repo_manager, *mem_space);
    REQUIRE_THROWS_AS(gpu_merge_impl::concat(input_views,
                                            cudf::get_default_stream(),
                                            *mem_space,
                                            data_repo_manager),
                    std::runtime_error);
}

TEST_CASE("Concatenate multiple data batches but no input rows", "[operator][merge_concat]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_rows_per_batch = 0;
    sirius::vector<int> num_input_rows(num_batches, num_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::STRING}};

    auto input_views = create_batches_with_random_data(
        num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt},
        data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::concat(input_views,
                                            cudf::get_default_stream(),
                                            *mem_space,
                                            data_repo_manager);
    validate_concat(input_views, *output_batch);
}

TEST_CASE("Concatenate mixed empty and non-empty data batches", "[operator][merge_concat]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    sirius::vector<int> num_input_rows;
    for (int i = 0; i < num_batches; ++i) {
        num_input_rows.push_back(i % 2 == 1 ? 0 : (i + 1) * 10);
    }
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::STRING}};

    auto input_views = create_batches_with_random_data(
        num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt},
        data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::concat(input_views,
                                            cudf::get_default_stream(),
                                            *mem_space,
                                            data_repo_manager);
    validate_concat(input_views, *output_batch);
}

namespace {

sirius::vector<sirius::unique_ptr<data_batch_view>> create_batches_with_local_ungrouped_agg_result(
        const int num_batches, const sirius::vector<int> num_base_input_rows,
        const sirius::vector<cudf::data_type>& column_types, const sirius::vector<cudf::aggregation::Kind>& aggregates,
        data_repository_manager& data_repo_manager, memory_space& mem_space) {
    // Base input batches
    auto base_input_batches = create_batches_with_random_data(
        num_batches, num_base_input_rows, column_types, {column_types.size(), std::nullopt},
        data_repo_manager, mem_space);
    
    // Compute local ungrouped aggregates
    sirius::vector<sirius::unique_ptr<data_batch_view>> local_aggregate_batches;
    sirius::vector<int> aggregate_idx(aggregates.size());
    for (int i = 0; i < aggregates.size(); ++i) {
        aggregate_idx[i] = i;
    }
    for (int i = 0; i < num_batches; ++i) {
        auto batch = gpu_aggregate_impl::local_ungrouped_aggregate(
            *base_input_batches[i], aggregates, aggregate_idx, cudf::get_default_stream(),
            mem_space, data_repo_manager);
        auto* batch_ptr = batch.get();
        data_repo_manager.add_new_data_batch(std::move(batch), {});
        local_aggregate_batches.push_back(sirius::make_unique<data_batch_view>(batch_ptr));
        local_aggregate_batches.back()->pin();
    }
    return local_aggregate_batches;
}

template <typename T>
void validate_ungrouped_aggregate_numeric(const sirius::vector<cudf::table_view>& input_table_views,
                                        cudf::table_view output_table_view,
                                        const sirius::vector<cudf::aggregation::Kind>& aggregates,
                                        int c) {
    // Handle the case where there is no input
    int num_valid_input_rows = 0;
    for (const auto& input_table_view: input_table_views) {
        const auto& col = input_table_view.column(c);
        num_valid_input_rows += input_table_view.num_rows() - col.null_count();
    }
    if (num_valid_input_rows == 0) {
        REQUIRE(output_table_view.column(c).null_count() == 1);
        return;
    }

    // Compare result
    T actual_result;
    cudaMemcpy(&actual_result,
            output_table_view.column(c).data<T>(),
            sizeof(T),
            cudaMemcpyDeviceToHost);
    sirius::vector<T> input_data_without_nulls;
    for (const auto& input_table_view: input_table_views) {
        sirius::vector<T> input_data(input_table_view.num_rows());
        cudaMemcpy(input_data.data(),
                input_table_view.column(c).data<T>(),
                sizeof(T) * input_table_view.num_rows(),
                cudaMemcpyDeviceToHost);
        auto* d_null_mask = input_table_view.column(c).null_mask();
        if (d_null_mask == nullptr) {
            input_data_without_nulls.insert(input_data_without_nulls.end(), input_data.begin(), input_data.end());
        } else {
            std::vector<cudf::bitmask_type> h_null_mask(
                cudf::bitmask_allocation_size_bytes(input_table_view.num_rows()) / sizeof(cudf::bitmask_type));
            cudaMemcpy(h_null_mask.data(),
                    d_null_mask,
                    h_null_mask.size() * sizeof(cudf::bitmask_type),
                    cudaMemcpyDeviceToHost);
            for (int r = 0; r < input_table_view.num_rows(); ++r) {
                if (cudf::bit_is_set(h_null_mask.data(), r)) {
                    input_data_without_nulls.push_back(input_data[r]);
                }
            }
        }
    }

    switch (aggregates[c]) {
        case cudf::aggregation::Kind::MIN: {
            T expected_result = *std::min_element(input_data_without_nulls.begin(), input_data_without_nulls.end());
            REQUIRE(expected_result == actual_result);
            break;
        }
        case cudf::aggregation::Kind::MAX: {
            T expected_result = *std::max_element(input_data_without_nulls.begin(), input_data_without_nulls.end());
            REQUIRE(expected_result == actual_result);
            break;
        }
        case cudf::aggregation::Kind::SUM:
        case cudf::aggregation::Kind::COUNT_ALL:
        case cudf::aggregation::Kind::COUNT_VALID: {
            int64_t expected_result = std::accumulate(
                input_data_without_nulls.begin(), input_data_without_nulls.end(), int64_t{0});
            REQUIRE(expected_result == actual_result);
            break;
        }
    }
}

void validate_ungrouped_aggregate(const sirius::vector<sirius::unique_ptr<data_batch_view>>& input_views,
                                const sirius::data_batch& output,
                                const sirius::vector<cudf::aggregation::Kind>& aggregates) {
    sirius::vector<cudf::table_view> input_table_views;
    for (const auto& input_view: input_views) {
        input_table_views.push_back(input_view->get_cudf_table_view());
    }
    cudf::table_view output_table_view = output.get_data()->cast<gpu_table_representation>().get_table().view();

    REQUIRE(output_table_view.num_rows() == 1);

    for (int c = 0; c < output_table_view.num_columns(); ++c) {
        // For ungrouped merge aggregate, type of output should be the same as input, since the overflow expansion
        // should be already performed in local aggregation.
        REQUIRE(output_table_view.column(c).type().id() == input_table_views[0].column(c).type().id());
        
        switch (output_table_view.column(c).type().id()) {
            case cudf::type_id::INT32: {
                validate_ungrouped_aggregate_numeric<int32_t>(input_table_views, output_table_view, aggregates, c);
                break;
            }
            case cudf::type_id::INT64: {
                validate_ungrouped_aggregate_numeric<int64_t>(input_table_views, output_table_view, aggregates, c);
                break;
            }
            default:
                throw std::runtime_error("Unsupported cudf::data_type in `validate_ungrouped_aggregate()`: "
                    + std::to_string(static_cast<int>(output_table_view.column(c).type().id())));
        }
    }
}

}

TEST_CASE("Ungrouped merge aggregate of min/max/count/sum", "[operator][merge_ungrouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::INT64},
                                                    cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN,
                                                        cudf::aggregation::Kind::MAX,
                                                        cudf::aggregation::Kind::COUNT_ALL,
                                                        cudf::aggregation::Kind::SUM};
    
    auto input_views = create_batches_with_local_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_ungrouped_aggregate(
        input_views, aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_ungrouped_aggregate(input_views, *output_batch, aggregates);
}

TEST_CASE("Ungrouped merge aggregate with invalid input", "[operator][merge_ungrouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    int num_batches = 1;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32}};
    sirius::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::SUM};

    // Invalid input: less than two input batches
    auto input_views = create_batches_with_local_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    REQUIRE_THROWS_AS(gpu_merge_impl::merge_ungrouped_aggregate(input_views,
                                                            aggregates,
                                                            cudf::get_default_stream(),
                                                            *mem_space,
                                                            data_repo_manager),
                    std::runtime_error);

    // Invalid input: mismatch between num columns and num aggregations
    num_batches = 10;
    num_base_input_rows = sirius::vector<int>(num_batches, num_base_input_rows_per_batch);
    input_views = create_batches_with_local_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    aggregates.push_back(cudf::aggregation::Kind::SUM);
    REQUIRE_THROWS_AS(gpu_merge_impl::merge_ungrouped_aggregate(input_views,
                                                            aggregates,
                                                            cudf::get_default_stream(),
                                                            *mem_space,
                                                            data_repo_manager),
                    std::runtime_error);
}

TEST_CASE("Ungrouped merge aggregate with empty local aggregate results", "[operator][merge_ungrouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 0;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::INT64},
                                                    cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN,
                                                        cudf::aggregation::Kind::MAX,
                                                        cudf::aggregation::Kind::COUNT_ALL,
                                                        cudf::aggregation::Kind::SUM};

    auto input_views = create_batches_with_local_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_ungrouped_aggregate(
        input_views, aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_ungrouped_aggregate(input_views, *output_batch, aggregates);
}

TEST_CASE("Ungrouped merge aggregate with mixed empty and non-empty local aggregate results",
        "[operator][merge_ungrouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    sirius::vector<int> num_base_input_rows;
    for (int i = 0; i < num_batches; ++i) {
        num_base_input_rows.push_back(i % 2 == 1 ? 0 : (i + 1) * 10);
    }
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::INT64},
                                                    cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN,
                                                        cudf::aggregation::Kind::MAX,
                                                        cudf::aggregation::Kind::COUNT_ALL,
                                                        cudf::aggregation::Kind::SUM};

    auto input_views = create_batches_with_local_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_ungrouped_aggregate(
        input_views, aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_ungrouped_aggregate(input_views, *output_batch, aggregates);
}

namespace {

sirius::vector<sirius::unique_ptr<data_batch_view>> create_batches_with_local_grouped_agg_result(
        const int num_batches, const sirius::vector<int> num_base_input_rows,
        const sirius::vector<cudf::data_type>& column_types, const sirius::vector<int>& group_idx,
        const sirius::vector<cudf::aggregation::Kind>& aggregates, const sirius::vector<int>& aggregate_idx,
        data_repository_manager& data_repo_manager, memory_space& mem_space) {
    // Base input batches, make group key value ranges small so that we have multiple values in a single group
    sirius::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);
    for (int group_col_id: group_idx) {
        ranges[group_col_id] = {0, 3};
    }
    auto base_input_batches = create_batches_with_random_data(
        num_batches, num_base_input_rows, column_types, ranges, data_repo_manager, mem_space);
    
    // Compute local grouped aggregates
    sirius::vector<sirius::unique_ptr<data_batch_view>> local_aggregate_batches;
    for (int i = 0; i < num_batches; ++i) {
        auto batch = gpu_aggregate_impl::local_grouped_aggregate(
            *base_input_batches[i], group_idx, aggregates, aggregate_idx, cudf::get_default_stream(),
            mem_space, data_repo_manager);
        auto* batch_ptr = batch.get();
        data_repo_manager.add_new_data_batch(std::move(batch), {});
        local_aggregate_batches.push_back(sirius::make_unique<data_batch_view>(batch_ptr));
        local_aggregate_batches.back()->pin();
    }

    return local_aggregate_batches;
}

void copy_data_to_host(cudf::table_view table, sirius::vector<sirius::vector<int64_t>>& h_data) {
    for (int c = 0; c < table.num_columns(); ++c) {
        const auto& col = table.column(c);
        switch (col.type().id()) {
            case cudf::type_id::INT32: {
                sirius::vector<int32_t> h_buf(table.num_rows());
                cudaMemcpy(h_buf.data(), col.data<int32_t>(), sizeof(int32_t) * table.num_rows(), 
                        cudaMemcpyDeviceToHost);
                for (auto val: h_buf) {
                    h_data[c].push_back(val);
                }
                break;
            }
            case cudf::type_id::INT64: {
                sirius::vector<int64_t> h_buf(table.num_rows());
                cudaMemcpy(h_buf.data(), col.data<int64_t>(), sizeof(int64_t) * table.num_rows(), 
                        cudaMemcpyDeviceToHost);
                for (auto val: h_buf) {
                    h_data[c].push_back(val);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported cudf::data_type in `pull_data_to_host()`: "
                    + std::to_string(static_cast<int>(col.type().id())));
        }
    }
}

void validate_grouped_aggregate(const sirius::vector<sirius::unique_ptr<data_batch_view>>& input_views,
                                const sirius::data_batch& output,
                                int num_group_cols,
                                const sirius::vector<cudf::aggregation::Kind>& aggregates) {
    sirius::vector<cudf::table_view> input_table_views;
    for (const auto& input_view: input_views) {
        input_table_views.push_back(input_view->get_cudf_table_view());
    }
    cudf::table_view output_table_view = output.get_data()->cast<gpu_table_representation>().get_table().view();

    // Compute expected results
    sirius::vector<sirius::vector<int64_t>> h_input_data(input_table_views[0].num_columns());
    for (const auto& table: input_table_views) {
        copy_data_to_host(table, h_input_data);
    }
    sirius::vector<std::map<sirius::vector<int64_t>, int64_t>> expected(aggregates.size());
    for (int r = 0; r < h_input_data[0].size(); ++r) {
        sirius::vector<int64_t> group_key;
        for (int c = 0; c < num_group_cols; ++c) {
            group_key.push_back(h_input_data[c][r]);
        }
        for (int i = 0; i < aggregates.size(); ++i) {
            int64_t val = h_input_data[num_group_cols + i][r];
            switch (aggregates[i]) {
                case cudf::aggregation::Kind::MIN: {
                    if (!expected[i].contains(group_key)) {
                        expected[i][group_key] = val;
                    } else {
                        expected[i][group_key] = min(expected[i][group_key], val);
                    }
                    break;
                }
                case cudf::aggregation::Kind::MAX: {
                    if (!expected[i].contains(group_key)) {
                        expected[i][group_key] = val;
                    } else {
                        expected[i][group_key] = max(expected[i][group_key], val);
                    }
                    break;
                }
                case cudf::aggregation::Kind::SUM:
                case cudf::aggregation::Kind::COUNT_ALL:
                case cudf::aggregation::Kind::COUNT_VALID: {
                    expected[i][group_key] += val;
                    break;
                }
            }
        }
    }

    // Get actual results
    sirius::vector<sirius::vector<int64_t>> actual(output_table_view.num_columns());
    copy_data_to_host(output_table_view, actual);

    // Check results
    REQUIRE(output_table_view.num_rows() == expected[0].size());
    REQUIRE(output_table_view.num_columns() == num_group_cols + aggregates.size());
    for (int r = 0; r < output_table_view.num_rows(); ++r) {
        sirius::vector<int64_t> group_key;
        for (int c = 0; c < num_group_cols; ++c) {
            group_key.push_back(actual[c][r]);
        }
        for (int i = 0; i < aggregates.size(); ++i) {
            int actual_val = actual[num_group_cols + i][r];
            int expected_val = expected[i][group_key];
            REQUIRE(actual_val == expected_val);
        }
    }
}

}

TEST_CASE("Grouped merge aggregate of min/max/count/sum", "[operator][merge_grouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<int> group_idx = {0, 1};
    sirius::vector<cudf::aggregation::Kind> aggregates = {
        cudf::aggregation::Kind::MIN, cudf::aggregation::Kind::MAX,
        cudf::aggregation::Kind::COUNT_ALL, cudf::aggregation::Kind::SUM};
    sirius::vector<int> aggregate_idx = {2, 3, 4, 5};

    auto input_views = create_batches_with_local_grouped_agg_result(
        num_batches, num_base_input_rows, column_types, group_idx, aggregates, aggregate_idx,
        data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_grouped_aggregate(
        input_views, group_idx.size(), aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_grouped_aggregate(input_views, *output_batch, group_idx.size(), aggregates);
}

TEST_CASE("Grouped merge aggregate with invalid input", "[operator][merge_grouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    int num_batches = 1;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<int> group_idx = {0};
    sirius::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN};
    sirius::vector<int> aggregate_idx = {1};

    // Invalid input: less than two input batches
    auto input_views = create_batches_with_local_grouped_agg_result(
        num_batches, num_base_input_rows, column_types, group_idx, aggregates, aggregate_idx,
        data_repo_manager, *mem_space);
    REQUIRE_THROWS_AS(gpu_merge_impl::merge_grouped_aggregate(
                        input_views, group_idx.size(), aggregates, cudf::get_default_stream(),
                        *mem_space, data_repo_manager),
                    std::runtime_error);

    // Invalid input: mismatch between num columns, num_groups, and num aggregations
    num_batches = 10;
    num_base_input_rows = sirius::vector<int>(num_batches, num_base_input_rows_per_batch);
    input_views = create_batches_with_local_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    group_idx.push_back(1);
    REQUIRE_THROWS_AS(gpu_merge_impl::merge_grouped_aggregate(
                        input_views, group_idx.size(), aggregates, cudf::get_default_stream(),
                        *mem_space, data_repo_manager),
                    std::runtime_error);
}

TEST_CASE("Grouped merge aggregate with empty local aggregate results", "[operator][merge_grouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 0;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<int> group_idx = {0, 1};
    sirius::vector<cudf::aggregation::Kind> aggregates = {
        cudf::aggregation::Kind::MIN, cudf::aggregation::Kind::MAX,
        cudf::aggregation::Kind::COUNT_ALL, cudf::aggregation::Kind::SUM};
    sirius::vector<int> aggregate_idx = {2, 3, 4, 5};

    auto input_views = create_batches_with_local_grouped_agg_result(
        num_batches, num_base_input_rows, column_types, group_idx, aggregates, aggregate_idx,
        data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_grouped_aggregate(
        input_views, group_idx.size(), aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_grouped_aggregate(input_views, *output_batch, group_idx.size(), aggregates);
}

TEST_CASE("Grouped merge aggregate with mixed empty and non-empty local aggregate results",
        "[operator][merge_grouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    sirius::vector<int> num_base_input_rows;
    for (int i = 0; i < num_batches; ++i) {
        num_base_input_rows.push_back(i % 2 == 1 ? 0 : (i + 1) * 10);
    }
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<int> group_idx = {0, 1};
    sirius::vector<cudf::aggregation::Kind> aggregates = {
        cudf::aggregation::Kind::MIN, cudf::aggregation::Kind::MAX,
        cudf::aggregation::Kind::COUNT_ALL, cudf::aggregation::Kind::SUM};
    sirius::vector<int> aggregate_idx = {2, 3, 4, 5};

    auto input_views = create_batches_with_local_grouped_agg_result(
        num_batches, num_base_input_rows, column_types, group_idx, aggregates, aggregate_idx,
        data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_grouped_aggregate(
        input_views, group_idx.size(), aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_grouped_aggregate(input_views, *output_batch, group_idx.size(), aggregates);
}

TEST_CASE("Grouped merge aggregate with multiple aggregations on the same column", "[operator][merge_grouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<int> group_idx = {0, 1};
    sirius::vector<cudf::aggregation::Kind> aggregates = {
        cudf::aggregation::Kind::MIN, cudf::aggregation::Kind::MAX,
        cudf::aggregation::Kind::COUNT_ALL, cudf::aggregation::Kind::SUM};
    sirius::vector<int> aggregate_idx = {2, 3, 2, 3};

    auto input_views = create_batches_with_local_grouped_agg_result(
        num_batches, num_base_input_rows, column_types, group_idx, aggregates, aggregate_idx,
        data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_grouped_aggregate(
        input_views, group_idx.size(), aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_grouped_aggregate(input_views, *output_batch, group_idx.size(), aggregates);
}

namespace {

sirius::vector<sirius::unique_ptr<data_batch_view>> create_batches_with_local_orderby_or_topn_result(
        const int num_batches, const sirius::vector<int> num_base_input_rows,
        const std::optional<std::pair<int, int>>& limit_offset, const sirius::vector<cudf::data_type>& column_types,
        const sirius::vector<int>& order_key_idx, const sirius::vector<cudf::order>& column_order,
        const sirius::vector<cudf::null_order>& null_precedence, data_repository_manager& data_repo_manager,
        memory_space& mem_space) {
    // Base input batches, make order key value ranges small so that some rows are compared by multiple columns
    sirius::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);
    for (int idx: order_key_idx) {
        ranges[idx] = {0, 4};
    }
    auto base_input_batches = create_batches_with_random_data(
        num_batches, num_base_input_rows, column_types, ranges, data_repo_manager, mem_space);

    // Compute local order_by
    sirius::vector<sirius::unique_ptr<data_batch_view>> local_order_by_batches;
    sirius::vector<int> projections(column_types.size());
    for (int i = 0; i < column_types.size(); ++i) {
        projections[i] = i;
    }
    for (int i = 0; i < num_batches; ++i) {
        auto batch = limit_offset.has_value()
            ? gpu_order_impl::local_top_n(
                *base_input_batches[i], limit_offset->first, limit_offset->second, order_key_idx, column_order,
                null_precedence, projections, cudf::get_default_stream(), mem_space, data_repo_manager)
            : gpu_order_impl::local_order_by(
                *base_input_batches[i], order_key_idx, column_order, null_precedence, projections,
                cudf::get_default_stream(), mem_space, data_repo_manager);
        auto* batch_ptr = batch.get();
        data_repo_manager.add_new_data_batch(std::move(batch), {});
        local_order_by_batches.push_back(sirius::make_unique<data_batch_view>(batch_ptr));
        local_order_by_batches.back()->pin();
    }
    return local_order_by_batches;
}

void validate_order_by(const sirius::vector<sirius::unique_ptr<data_batch_view>>& input_views,
                    const sirius::data_batch& output,
                    const sirius::vector<int>& order_key_idx,
                    const sirius::vector<cudf::order>& column_order) {
    sirius::vector<cudf::table_view> input_table_views;
    int expected_num_rows = 0;
    for (const auto& input_view: input_views) {
        input_table_views.push_back(input_view->get_cudf_table_view());
        expected_num_rows += input_table_views.back().num_rows();
    }
    cudf::table_view output_table_view = output.get_data()->cast<gpu_table_representation>().get_table().view();

    REQUIRE(output_table_view.num_rows() == expected_num_rows);
    REQUIRE(output_table_view.num_columns() == input_table_views[0].num_columns());
    for (int c = 0; c < output_table_view.num_columns(); ++c) {
        REQUIRE(output_table_view.column(c).type().id() == input_table_views[0].column(c).type().id());
    }

    sirius::vector<sirius::vector<int64_t>> actual(output_table_view.num_columns());
    copy_data_to_host(output_table_view, actual);
    auto comp = [&](int r) {
        for (int i = 0; i < order_key_idx.size(); ++i) {
            int col = order_key_idx[i];
            if (actual[col][r] == actual[col][r - 1]) {
                continue;
            }
            return (column_order[i] == cudf::order::ASCENDING && actual[col][r] > actual[col][r - 1])
                || (column_order[i] == cudf::order::DESCENDING && actual[col][r] < actual[col][r - 1]);
        }
        return true;
    };
    for (int r = 1; r < output_table_view.num_rows(); ++r) {
        REQUIRE(comp(r));
    }
}

}

TEST_CASE("Merge order-by basic", "[operator][merge_order_by]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<int> order_key_idx = {0, 1, 2};
    sirius::vector<cudf::order> column_order = {
        cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
    sirius::vector<cudf::null_order> null_precedence = {
        cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};
    
    auto input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, std::nullopt, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_order_by(
        input_views, order_key_idx, column_order, null_precedence, cudf::get_default_stream(),
        *mem_space, data_repo_manager);
    validate_order_by(input_views, *output_batch, order_key_idx, column_order);
}

TEST_CASE("Merge order-by with invalid input", "[operator][merge_order_by]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    int num_batches = 1;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<int> order_key_idx = {0, 1, 2};
    sirius::vector<cudf::order> column_order = {
        cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
    sirius::vector<cudf::null_order> null_precedence = {
        cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};

    // Invalid input: less than two input batches
    auto input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, std::nullopt, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    REQUIRE_THROWS_AS(gpu_merge_impl::merge_order_by(
                        input_views, order_key_idx, column_order, null_precedence, cudf::get_default_stream(),
                        *mem_space, data_repo_manager),
                    std::runtime_error);

    // Invalid input: mismatch between sizes of `order_key_idx`, `column_order`, and `null_precedence`
    num_batches = 10;
    num_base_input_rows = sirius::vector<int>(num_batches, num_base_input_rows_per_batch);
    input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, std::nullopt, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    order_key_idx.push_back(3);
    REQUIRE_THROWS_AS(gpu_merge_impl::merge_order_by(
                        input_views, order_key_idx, column_order, null_precedence, cudf::get_default_stream(),
                        *mem_space, data_repo_manager),
                    std::runtime_error);
}

TEST_CASE("Merge order-by with empty local order-by results", "[operator][merge_order_by]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 0;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<int> order_key_idx = {0, 1, 2};
    sirius::vector<cudf::order> column_order = {
        cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
    sirius::vector<cudf::null_order> null_precedence = {
        cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};
    
    auto input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, std::nullopt, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_order_by(
        input_views, order_key_idx, column_order, null_precedence, cudf::get_default_stream(),
        *mem_space, data_repo_manager);
    validate_order_by(input_views, *output_batch, order_key_idx, column_order);
}

TEST_CASE("Merge order-by with mixed empty and non-empty local order-by results", "[operator][merge_order_by]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    sirius::vector<int> num_base_input_rows;
    for (int i = 0; i < num_batches; ++i) {
        num_base_input_rows.push_back(i % 2 == 1 ? 0 : (i + 1) * 10);
    }
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    sirius::vector<int> order_key_idx = {0, 1, 2};
    sirius::vector<cudf::order> column_order = {
        cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
    sirius::vector<cudf::null_order> null_precedence = {
        cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};
    
    auto input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, std::nullopt, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_order_by(
        input_views, order_key_idx, column_order, null_precedence, cudf::get_default_stream(),
        *mem_space, data_repo_manager);
    validate_order_by(input_views, *output_batch, order_key_idx, column_order);
}

namespace {

void validate_top_n(const sirius::vector<sirius::unique_ptr<data_batch_view>>& input_views,
                    const sirius::data_batch& output,
                    const std::pair<int, int>& limit_offset,
                    const sirius::vector<int>& order_key_idx,
                    const sirius::vector<cudf::order>& column_order) {
    sirius::vector<cudf::table_view> input_table_views;
    int num_input_rows = 0;
    for (const auto& input_view: input_views) {
        input_table_views.push_back(input_view->get_cudf_table_view());
        num_input_rows += input_table_views.back().num_rows();
    }
    int limit = limit_offset.first, offset = limit_offset.second;
    cudf::table_view output_table_view = output.get_data()->cast<gpu_table_representation>().get_table().view();
    int expected_num_rows = (limit + offset <= num_input_rows) ? limit : std::max(0, num_input_rows - offset);

    // Compute sorted input
    sirius::vector<sirius::vector<int64_t>> h_input_data(input_table_views[0].num_columns());
    for (const auto& table: input_table_views) {
        copy_data_to_host(table, h_input_data);
    }
    sirius::vector<sirius::vector<int64_t>> h_input_data_rows(
        h_input_data[0].size(), sirius::vector<int64_t>(h_input_data.size()));
    for (int r = 0; r < h_input_data_rows.size(); ++r) {
        for (int c = 0; c < h_input_data_rows[0].size(); ++c) {
            h_input_data_rows[r][c] = h_input_data[c][r];
        }
    }
    sort(h_input_data_rows.begin(), h_input_data_rows.end(),
        [&](const sirius::vector<int64_t>& r1, const sirius::vector<int64_t>& r2) {
            for (int i = 0; i < order_key_idx.size(); ++i) {
                int col = order_key_idx[i];
                if (r1[col] == r2[col]) {
                    continue;
                }
                return (column_order[i] == cudf::order::ASCENDING && r1[col] < r2[col])
                    || (column_order[i] == cudf::order::DESCENDING && r1[col] > r2[col]);
            }
            return false;
        });

    // Check
    REQUIRE(output_table_view.num_rows() == expected_num_rows);
    REQUIRE(output_table_view.num_columns() == input_table_views[0].num_columns());
    for (int c = 0; c < output_table_view.num_columns(); ++c) {
        REQUIRE(output_table_view.column(c).type().id() == input_table_views[0].column(c).type().id());
    }

    sirius::vector<sirius::vector<int64_t>> actual(output_table_view.num_columns());
    copy_data_to_host(output_table_view, actual);
    auto comp_lower = [&](int r) {
        if (offset == 0) {
            return true;
        }
        const auto& lower = h_input_data_rows[offset - 1];
        for (int i = 0; i < order_key_idx.size(); ++i) {
            int col = order_key_idx[i];
            if (actual[col][r] == lower[col]) {
                continue;
            }
            return (column_order[i] == cudf::order::ASCENDING && actual[col][r] > lower[col])
                || (column_order[i] == cudf::order::DESCENDING && actual[col][r] < lower[col]);
        }
        return true;
    };
    auto comp_upper = [&](int r) {
        if (offset + limit >= num_input_rows) {
            return true;
        }
        const auto& upper = h_input_data_rows[offset + limit];
        for (int i = 0; i < order_key_idx.size(); ++i) {
            int col = order_key_idx[i];
            if (actual[col][r] == upper[col]) {
                continue;
            }
            return (column_order[i] == cudf::order::ASCENDING && actual[col][r] < upper[col])
                || (column_order[i] == cudf::order::DESCENDING && actual[col][r] > upper[col]);
        }
        return true;
    };
    for (int r = 0; r < output_table_view.num_rows(); ++r) {
        REQUIRE(comp_lower(r));
        REQUIRE(comp_upper(r));
    }
}

}

TEST_CASE("Merge top-n basic", "[operator][merge_top_n]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    std::pair<int, int> limit_offset = {10, 20};
    sirius::vector<int> order_key_idx = {0, 1, 2};
    sirius::vector<cudf::order> column_order = {
        cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
    sirius::vector<cudf::null_order> null_precedence = {
        cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};
    
    auto input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, limit_offset, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_top_n(
        input_views, limit_offset.first, limit_offset.second, order_key_idx, column_order, null_precedence,
        cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_top_n(input_views, *output_batch, limit_offset, order_key_idx, column_order);
}

TEST_CASE("Merge top-n with empty local top-n results", "[operator][merge_top_n]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 0;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    std::pair<int, int> limit_offset = {10, 20};
    sirius::vector<int> order_key_idx = {0, 1, 2};
    sirius::vector<cudf::order> column_order = {
        cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
    sirius::vector<cudf::null_order> null_precedence = {
        cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};
    
    auto input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, limit_offset, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_top_n(
        input_views, limit_offset.first, limit_offset.second, order_key_idx, column_order, null_precedence,
        cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_top_n(input_views, *output_batch, limit_offset, order_key_idx, column_order);
}

TEST_CASE("Merge top-n with mixed empty and non-empty local top-n results", "[operator][merge_top_n]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    sirius::vector<int> num_base_input_rows;
    for (int i = 0; i < num_batches; ++i) {
        num_base_input_rows.push_back(i % 2 == 1 ? 0 : (i + 1) * 10);
    }
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    std::pair<int, int> limit_offset = {10, 20};
    sirius::vector<int> order_key_idx = {0, 1, 2};
    sirius::vector<cudf::order> column_order = {
        cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
    sirius::vector<cudf::null_order> null_precedence = {
        cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};
    
    auto input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, limit_offset, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_top_n(
        input_views, limit_offset.first, limit_offset.second, order_key_idx, column_order, null_precedence,
        cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_top_n(input_views, *output_batch, limit_offset, order_key_idx, column_order);
}

TEST_CASE("Merge top-n with `limit = 0`", "[operator][merge_top_n]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    std::pair<int, int> limit_offset = {0, 20};
    sirius::vector<int> order_key_idx = {0, 1, 2};
    sirius::vector<cudf::order> column_order = {
        cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
    sirius::vector<cudf::null_order> null_precedence = {
        cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};
    
    auto input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, limit_offset, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_top_n(
        input_views, limit_offset.first, limit_offset.second, order_key_idx, column_order, null_precedence,
        cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_top_n(input_views, *output_batch, limit_offset, order_key_idx, column_order);
}

TEST_CASE("Merge top-n with `num_input_rows - limit <= offset < num_input-rows`", "[operator][merge_top_n]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64},
        cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}};
    std::pair<int, int> limit_offset = {200, 900};
    sirius::vector<int> order_key_idx = {0, 1, 2};
    sirius::vector<cudf::order> column_order = {
        cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
    sirius::vector<cudf::null_order> null_precedence = {
        cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};
    
    auto input_views = create_batches_with_local_orderby_or_topn_result(
        num_batches, num_base_input_rows, limit_offset, column_types, order_key_idx,
        column_order, null_precedence, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge_impl::merge_top_n(
        input_views, limit_offset.first, limit_offset.second, order_key_idx, column_order, null_precedence,
        cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_top_n(input_views, *output_batch, limit_offset, order_key_idx, column_order);
}
