#include "cudf/cudf_utils.hpp"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"
namespace duckdb {

void cudf_groupby(GPUColumn **keys, GPUColumn **aggregate_keys, uint64_t num_keys, uint64_t num_aggregates, AggregationType* agg_mode) 
{
  if (keys[0]->column_length == 0) {
    printf("N is 0\n");
    for (idx_t group = 0; group < num_keys; group++) {
      bool old_unique = keys[group]->is_unique;
      if (keys[group]->data_wrapper.type == ColumnType::VARCHAR) {
        keys[group] = new GPUColumn(0, keys[group]->data_wrapper.type, keys[group]->data_wrapper.data, keys[group]->data_wrapper.offset, 0, true);
      } else {
        keys[group] = new GPUColumn(0, keys[group]->data_wrapper.type, keys[group]->data_wrapper.data);
      }
      keys[group]->is_unique = old_unique;
    }

    for (int agg_idx = 0; agg_idx < num_aggregates; agg_idx++) {
      if (agg_mode[agg_idx] == AggregationType::COUNT_STAR || agg_mode[agg_idx] == AggregationType::COUNT) {
        aggregate_keys[agg_idx] = new GPUColumn(0, ColumnType::INT64, aggregate_keys[agg_idx]->data_wrapper.data);
      } else {
        aggregate_keys[agg_idx] = new GPUColumn(0, aggregate_keys[agg_idx]->data_wrapper.type, aggregate_keys[agg_idx]->data_wrapper.data);
      }
    }
    return;
  }

  GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
  cudf::set_current_device_resource(gpuBufferManager->mr);

  std::vector<cudf::column_view> keys_cudf;

  //TODO: This is a hack to get the size of the keys
  size_t size = keys[0]->column_length;

  for (int key = 0; key < num_keys; key++) {
    auto cudf_column = keys[key]->convertToCudfColumn();
    keys_cudf.push_back(cudf_column);
  }

  auto keys_table = cudf::table_view(keys_cudf);
  cudf::groupby::groupby grpby_obj(keys_table);

  std::vector<cudf::groupby::aggregation_request> requests;
  for (int agg = 0; agg < num_aggregates; agg++) {
    requests.emplace_back(cudf::groupby::aggregation_request());
    if (aggregate_keys[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::COUNT && aggregate_keys[agg]->column_length == 0) {
      auto aggregate = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
      requests[agg].aggregations.push_back(std::move(aggregate));
      auto const_scalar = cudf::make_fixed_width_scalar<uint64_t>(0);
      requests[agg].values = cudf::make_column_from_scalar(*const_scalar, size)->view();
    } else if (aggregate_keys[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::SUM && aggregate_keys[agg]->column_length == 0) {
      auto aggregate = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      auto const_scalar = cudf::make_fixed_width_scalar<uint64_t>(0);
      requests[agg].values = cudf::make_column_from_scalar(*const_scalar, size)->view();
    } else if (aggregate_keys[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::COUNT_STAR && aggregate_keys[agg]->column_length != 0) {
      auto aggregate = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
      requests[agg].aggregations.push_back(std::move(aggregate));
      auto const_scalar = cudf::make_fixed_width_scalar<uint64_t>(1);
      requests[agg].values = cudf::make_column_from_scalar(*const_scalar, size)->view();
    } else if (agg_mode[agg] == AggregationType::SUM) {
      auto aggregate = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::AVERAGE) {
      auto aggregate = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::MIN) {
      auto aggregate = cudf::make_min_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::MAX) {
      auto aggregate = cudf::make_max_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::COUNT) {
      auto aggregate = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else {
      throw NotImplementedException("Aggregate function not supported");
    }
  }

  auto result = grpby_obj.aggregate(requests);

  auto result_key = std::move(result.first);
  for (int key = 0; key < num_keys; key++) {
      cudf::column group_key = result_key->get_column(key);
      keys[key]->setFromCudfColumn(group_key, keys[key]->is_unique, nullptr, 0, gpuBufferManager);
      // keys[key] = gpuBufferManager->copyDataFromcuDFColumn(group_key, 0);
  }

  for (int agg = 0; agg < num_aggregates; agg++) {
      auto agg_val = std::move(result.second[agg].results[0]);
      if (agg_mode[agg] == AggregationType::COUNT || agg_mode[agg] == AggregationType::COUNT_STAR) {
        auto agg_val_view = agg_val->view();
        auto temp_data = convertInt32ToUInt64(const_cast<int32_t*>(agg_val_view.data<int32_t>()), agg_val_view.size());
        size_t size = agg_val_view.size();
        aggregate_keys[agg] = new GPUColumn(size, ColumnType::INT64, reinterpret_cast<uint8_t*>(temp_data));
      } else {
        aggregate_keys[agg]->setFromCudfColumn(*agg_val, false, nullptr, 0, gpuBufferManager);
        // aggregate_keys[agg] = gpuBufferManager->copyDataFromcuDFColumn(agg_val_view, 0);
      }
  }

}

} //namespace duckdb