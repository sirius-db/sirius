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

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <filesystem>

#include "sirius_extension.hpp"
#include "log/logging.hpp"

namespace duckdb {

// cudf::io::table_with_metadata read_csv(std::string const& file_path)
// {
//   auto source_info = cudf::io::source_info(file_path);
//   std::map<std::string, cudf::data_type> dtypes;
//   dtypes["Company"] = cudf::data_type(cudf::type_id::STRING);
//   dtypes["Date"] = cudf::data_type(cudf::type_id::STRING);
//   dtypes["Open"] = cudf::data_type(cudf::type_id::FLOAT64);
//   dtypes["High"] = cudf::data_type(cudf::type_id::FLOAT64);
//   dtypes["Low"] = cudf::data_type(cudf::type_id::FLOAT64);
//   dtypes["Close"] = cudf::data_type(cudf::type_id::FLOAT64);
//   dtypes["Volume"] = cudf::data_type(cudf::type_id::FLOAT64);
//   // auto builder     = cudf::io::csv_reader_options::builder(source_info);
//   auto options = cudf::io::csv_reader_options::builder(source_info);
//   options.header(0);
//   options.dtypes(dtypes);
//   options.delimiter(',');
//   // auto options     = builder.build();
//   // SIRIUS_LOG_DEBUG("{}", options)
//   SIRIUS_LOG_DEBUG("here");
//   return cudf::io::read_csv(options);
// }


cudf::io::table_with_metadata read_csv(std::string const& file_path)
{
  SIRIUS_LOG_DEBUG("Reading CSV file: {}", file_path);
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::csv_reader_options::builder(source_info);
  // auto options = cudf::io::csv_reader_options::builder(source_info);
  auto options     = builder.build();
  // SIRIUS_LOG_DEBUG("{}", options)
  return cudf::io::read_csv(options);
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}

std::vector<cudf::groupby::aggregation_request> make_single_aggregation_request(
  std::unique_ptr<cudf::groupby_aggregation>&& agg, cudf::column_view value)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].aggregations.push_back(std::move(agg));
  requests[0].values = value;
  return requests;
}

std::unique_ptr<cudf::table> average_closing_price(cudf::table_view stock_info_table)
{
  // Schema: | Company | Date | Open | High | Low | Close | Volume |
  auto keys = cudf::table_view{{stock_info_table.column(0)}};  // Company
  auto val  = stock_info_table.column(5);                      // Close

  // Compute the average of each company's closing price with entire column
  cudf::groupby::groupby grpby_obj(keys);
  auto requests =
    make_single_aggregation_request(cudf::make_mean_aggregation<cudf::groupby_aggregation>(), val);

  auto agg_results = grpby_obj.aggregate(requests);

  // Assemble the result
  auto result_key = std::move(agg_results.first);
  auto result_val = std::move(agg_results.second[0].results[0]);
  std::vector<cudf::column_view> columns{result_key->get_column(0), *result_val};
  return std::make_unique<cudf::table>(cudf::table_view(columns));
}

void test_cudf()
{
  // Construct a CUDA memory resource using RAPIDS Memory Manager (RMM)
  // This is the default memory resource for libcudf for allocating device memory.
  rmm::mr::cuda_memory_resource cuda_mr{};
  // Construct a memory pool using the CUDA memory resource
  // Using a memory pool for device memory allocations is important for good performance in libcudf.
  // The pool defaults to allocating half of the available GPU memory.
  rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(10)};

  // Set the pool resource to be used by default for all device memory allocations
  // Note: It is the user's responsibility to ensure the `mr` object stays alive for the duration of
  // it being set as the default
  // Also, call this before the first libcudf API call to ensure all data is allocated by the same
  // memory resource.
  cudf::set_current_device_resource(&mr);

  // Read the test data from the sirius directory
  const char* sirius_directory_path = getenv("SIRIUS_HOME_PATH");
  if (sirius_directory_path == nullptr) {
    throw std::runtime_error("Environment variable SIRIUS_HOME_PATH not set");
  }

  // Determine the paths
  std::filesystem::path sirius_dir(sirius_directory_path);
  std::filesystem::path input_file = sirius_dir / "4stock_5day.csv";
  std::filesystem::path output_file = sirius_dir / "4stock_5day_avg_close.csv";

  auto stock_table_with_metadata = read_csv(input_file.string());
  auto result = average_closing_price(*stock_table_with_metadata.tbl);

  // Write out result
  write_csv(*result, output_file.string());
}

} //namespace duckdb