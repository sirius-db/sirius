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

#pragma once

// sirius
#include <op/gpu_physical_operator.hpp>

// duckdb
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table_function.hpp>

namespace duckdb {
//===----------------------------------------------------------------------===//
// Physical Table Scan Adapter
//===----------------------------------------------------------------------===//

/**
 * @brief Adapter class that wraps a DuckDB PhysicalTableScan operator and inherits from
 * GPUPhysicalOperator.
 */
class physical_table_scan_adapter : GPUPhysicalOperator {
 public:
  static constexpr PhysicalOperatorType TYPE = PhysicalOperatorType::TABLE_SCAN;

  //===----------Constructor----------===//
  /**
   * @brief Construct a new physical_table_scan_adapter object (has the same signature as DuckDB's
   * PhysicalTableScan).
   *
   * @param[in] physical_table_scan The DuckDB PhysicalTableScan operator to be wrapped.
   */
  physical_table_scan_adapter(std::unique_ptr<PhysicalTableScan> physical_table_scan)
    : GPUPhysicalOperator(PhysicalOperatorType::TABLE_SCAN,
                          physical_table_scan->types,
                          physical_table_scan->estimated_cardinality),
      physical_table_scan(std::move(physical_table_scan))
  {
  }

  //===----------Methods----------===//
  /**
   * @brief Get the underlying PhysicalTableScan operator.
   *
   * @return PhysicalTableScan& Reference to the underlying PhysicalTableScan operator.
   */
  PhysicalTableScan const& get_physical_table_scan() const { return *physical_table_scan; }

  //===----------Fields----------===//
  std::unique_ptr<PhysicalTableScan> physical_table_scan;
};

}  // namespace duckdb
