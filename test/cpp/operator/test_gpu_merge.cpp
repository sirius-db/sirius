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
#include "merge/gpu_merge.hpp"
#include "utils/utils.hpp"

#include <cudf/utilities/bit.hpp>

using namespace sirius;
using namespace sirius::memory;
using namespace sirius::op;

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
    const int num_batches, const sirius::vector<int> num_rows, const sirius::vector<cudf::data_type>& column_types,
    data_repository_manager& data_repo_manager, memory_space& mem_space) {
    sirius::vector<sirius::unique_ptr<data_batch_view>> batches;
    for (int i = 0; i < num_batches; ++i) {
        // Create a data batch
        auto table = create_cudf_table_with_random_data(
            num_rows[i],
            column_types,
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

TEST_CASE("Concatenate multiple data batches", "[operator][merge_concat]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    constexpr size_t num_rows_per_batch = 100;
    sirius::vector<int> num_input_rows(num_batches, num_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::STRING}};

    auto input_views = create_batches_with_random_data(
        num_batches, num_input_rows, column_types, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge::concat(input_views,
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
        num_batches, num_input_rows, column_types, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge::concat(input_views,
                                        cudf::get_default_stream(),
                                        *mem_space,
                                        data_repo_manager);
    validate_concat(input_views, *output_batch);
}

TEST_CASE("Concatenate one data batch", "[operator][merge_concat]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 1;
    constexpr size_t num_rows_per_batch = 100;
    sirius::vector<int> num_input_rows(num_batches, num_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                                    cudf::data_type{cudf::type_id::STRING}};

    auto input_views = create_batches_with_random_data(
        num_batches, num_input_rows, column_types, data_repo_manager, *mem_space);
    REQUIRE_THROWS_AS(gpu_merge::concat(input_views,
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
        num_batches, num_input_rows, column_types, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge::concat(input_views,
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
        num_batches, num_input_rows, column_types, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge::concat(input_views,
                                        cudf::get_default_stream(),
                                        *mem_space,
                                        data_repo_manager);
    validate_concat(input_views, *output_batch);
}

sirius::unique_ptr<cudf::reduce_aggregation> get_local_reduce_aggregate(cudf::aggregation::Kind kind) {
    switch (kind) {
        case cudf::reduce_aggregation::MIN:
            return cudf::make_min_aggregation<cudf::reduce_aggregation>();
        case cudf::reduce_aggregation::MAX:
            return cudf::make_max_aggregation<cudf::reduce_aggregation>();
        case cudf::reduce_aggregation::COUNT_ALL:
            return cudf::make_count_aggregation<cudf::reduce_aggregation>(cudf::null_policy::INCLUDE);
        case cudf::reduce_aggregation::SUM:
            return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        default:
            throw std::runtime_error("Unsupported cudf aggregate kind in `get_local_aggregate()`: "
                + std::to_string(static_cast<int>(kind)));
  }
}

sirius::vector<sirius::unique_ptr<data_batch_view>> create_batches_with_partial_ungrouped_agg_result(
    const int num_batches, const sirius::vector<int> num_base_input_rows,
    const sirius::vector<cudf::data_type>& column_types, const sirius::vector<cudf::aggregation::Kind>& aggregates,
    data_repository_manager& data_repo_manager, memory_space& mem_space) {
    // Base input batches
    auto base_input_batches = create_batches_with_random_data(
        num_batches, num_base_input_rows, column_types, data_repo_manager, mem_space);
    
    // Compute partial ungrouped aggregates
    sirius::vector<sirius::unique_ptr<data_batch_view>> partial_aggregate_batches;
    sirius::vector<sirius::unique_ptr<cudf::reduce_aggregation>> reduce_aggregations;
    for (int c = 0; c < aggregates.size(); ++c) {
        reduce_aggregations.push_back(std::move(get_local_reduce_aggregate(aggregates[c])));
    }
    for (int i = 0; i < num_batches; ++i) {
        // Make partial aggregate table
        auto cudf_table = base_input_batches[i]->get_cudf_table_view();
        sirius::vector<sirius::unique_ptr<cudf::column>> partial_aggregate_cols;
        for (int c = 0; c < reduce_aggregations.size(); ++c) {
            auto output_scalar = cudf::reduce(cudf_table.column(c),
                                            *reduce_aggregations[c],
                                            cudf_table.column(c).type(),
                                            cudf::get_default_stream(),
                                            mem_space.get_default_allocator());
            partial_aggregate_cols.push_back(cudf::make_column_from_scalar(
                *output_scalar, 1, cudf::get_default_stream(), mem_space.get_default_allocator()));
        }
        auto partial_aggregate_cudf_table = sirius::make_unique<cudf::table>(std::move(partial_aggregate_cols));

        // Create a batch from the table
        auto gpu_repr = sirius::make_unique<gpu_table_representation>(*partial_aggregate_cudf_table, mem_space);
        auto batch = sirius::make_unique<data_batch>(data_repo_manager.get_next_data_batch_id(),
                                                    data_repo_manager,
                                                    std::move(gpu_repr));
        auto* batch_ptr = batch.get();
        data_repo_manager.add_new_data_batch(std::move(batch), {});
        partial_aggregate_batches.push_back(sirius::make_unique<data_batch_view>(batch_ptr));
        partial_aggregate_batches.back()->pin();
    }
    return partial_aggregate_batches;
}

template <typename TIn, typename TOut>
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
    TOut actual_result;
    cudaMemcpy(&actual_result,
            output_table_view.column(c).data<TOut>(),
            sizeof(TOut),
            cudaMemcpyDeviceToHost);
    sirius::vector<TIn> input_data_without_nulls;
    for (const auto& input_table_view: input_table_views) {
        sirius::vector<TIn> input_data(input_table_view.num_rows());
        cudaMemcpy(input_data.data(),
                input_table_view.column(c).data<TIn>(),
                sizeof(TIn) * input_table_view.num_rows(),
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
            TIn expected_result = *std::min_element(input_data_without_nulls.begin(), input_data_without_nulls.end());
            REQUIRE(expected_result == actual_result);
            break;
        }
        case cudf::aggregation::Kind::MAX: {
            TIn expected_result = *std::max_element(input_data_without_nulls.begin(), input_data_without_nulls.end());
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
        cudf::data_type expected_output_type = input_table_views[0].column(c).type();
        switch (aggregates[c]) {
            case cudf::aggregation::Kind::SUM: {
                if (expected_output_type.id() == cudf::type_id::INT8
                    || expected_output_type.id() == cudf::type_id::INT16
                    || expected_output_type.id() == cudf::type_id::INT32) {
                    expected_output_type = cudf::data_type(cudf::type_id::INT64);
                }
                break;
            }
            case cudf::aggregation::Kind::COUNT_ALL:
            case cudf::aggregation::Kind::COUNT_VALID: {
                expected_output_type = cudf::data_type(cudf::type_id::INT64);
                break;
            }
        }

        REQUIRE(output_table_view.column(c).type().id() == expected_output_type.id());

        if (expected_output_type.id() == cudf::type_id::INT64) {
            switch (output_table_view.column(c).type().id()) {
                case cudf::type_id::INT32: {
                    validate_ungrouped_aggregate_numeric<int32_t, int64_t>(
                        input_table_views, output_table_view, aggregates, c);
                    break;
                }
                case cudf::type_id::INT64: {
                    validate_ungrouped_aggregate_numeric<int64_t, int64_t>(
                        input_table_views, output_table_view, aggregates, c);
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported cudf::data_type in `validate_ungrouped_aggregate()`: "
                        + std::to_string(static_cast<int>(output_table_view.column(c).type().id())));
            }
        } else {
            switch (output_table_view.column(c).type().id()) {
                case cudf::type_id::INT32: {
                    validate_ungrouped_aggregate_numeric<int32_t, int32_t>(
                        input_table_views, output_table_view, aggregates, c);
                    break;
                }
                case cudf::type_id::INT64: {
                    validate_ungrouped_aggregate_numeric<int64_t, int64_t>(
                        input_table_views, output_table_view, aggregates, c);
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
    
    auto input_views = create_batches_with_partial_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge::merge_ungrouped_aggregate(
        input_views, aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_ungrouped_aggregate(input_views, *output_batch, aggregates);
}

TEST_CASE("Ungrouped merge aggregate on one data batch", "[operator][merge_ungrouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 1;
    constexpr size_t num_base_input_rows_per_batch = 100;
    sirius::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
    sirius::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32}};
    sirius::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::SUM};

    auto input_views = create_batches_with_partial_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    REQUIRE_THROWS_AS(gpu_merge::merge_ungrouped_aggregate(input_views,
                                                        aggregates,
                                                        cudf::get_default_stream(),
                                                        *mem_space,
                                                        data_repo_manager),
                    std::runtime_error);
}

TEST_CASE("Ungrouped merge aggregate of with empty partial aggregate results", "[operator][merge_ungrouped_agg]") {
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

    auto input_views = create_batches_with_partial_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge::merge_ungrouped_aggregate(
        input_views, aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_ungrouped_aggregate(input_views, *output_batch, aggregates);
}

TEST_CASE("Ungrouped merge aggregate of with mixed empty and non-empty partial aggregate results",
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

    auto input_views = create_batches_with_partial_ungrouped_agg_result(
        num_batches, num_base_input_rows, column_types, aggregates, data_repo_manager, *mem_space);
    auto output_batch = gpu_merge::merge_ungrouped_aggregate(
        input_views, aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_ungrouped_aggregate(input_views, *output_batch, aggregates);
}

TEST_CASE("Ungrouped merge aggregate with overflow", "[operator][merge_ungrouped_agg]") {
    data_repository_manager data_repo_manager;
    auto* mem_space = get_default_memory_space();
    constexpr int num_batches = 10;
    sirius::vector<cudf::aggregation::Kind> aggregates(3, cudf::aggregation::Kind::SUM);

    // Create partial aggregated batches that can trigger overflow
    sirius::vector<sirius::unique_ptr<cudf::scalar>> scalars;
    scalars.push_back(std::move(cudf::make_fixed_width_scalar<int8_t>(
        SCHAR_MAX, cudf::get_default_stream(), mem_space->get_default_allocator())));
    scalars.push_back(std::move(cudf::make_fixed_width_scalar<int16_t>(
        SHRT_MAX, cudf::get_default_stream(), mem_space->get_default_allocator())));
    scalars.push_back(std::move(cudf::make_fixed_width_scalar<int32_t>(
        INT_MAX, cudf::get_default_stream(), mem_space->get_default_allocator())));
    sirius::vector<sirius::unique_ptr<data_batch_view>> input_views;
    for (int i = 0; i < num_batches; ++i) {
        // Create cudf table
        sirius::vector<sirius::unique_ptr<cudf::column>> cudf_cols;
        for (const auto& scalar: scalars) {
            cudf_cols.push_back(std::move(
                cudf::make_column_from_scalar(
                    *scalar, 1, cudf::get_default_stream(), mem_space->get_default_allocator())));
        }
        auto cudf_table = sirius::make_unique<cudf::table>(std::move(cudf_cols));

        // Create a batch from the table
        auto gpu_repr = sirius::make_unique<gpu_table_representation>(*cudf_table, *mem_space);
        auto batch = sirius::make_unique<data_batch>(data_repo_manager.get_next_data_batch_id(),
                                                    data_repo_manager,
                                                    std::move(gpu_repr));
        auto* batch_ptr = batch.get();
        data_repo_manager.add_new_data_batch(std::move(batch), {});
        input_views.push_back(sirius::make_unique<data_batch_view>(batch_ptr));
        input_views.back()->pin();
    }

    // Call merge function
    auto output_batch = gpu_merge::merge_ungrouped_aggregate(
        input_views, aggregates, cudf::get_default_stream(), *mem_space, data_repo_manager);
    validate_ungrouped_aggregate(input_views, *output_batch, aggregates);
}
