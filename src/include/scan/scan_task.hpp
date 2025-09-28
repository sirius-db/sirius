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
#include "config.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "data/data_batch.hpp"

namespace duckdb {
namespace sirius {

// Scan Executor will leverage DuckDB task scheduler and thread pool, therefore the ScanTask has to be derived from duckdb::BaseExecutorTask.
// Header bellow following Yifei's Scan implementation on gpu_phyisical_table_scan.cpp
// ScanTaskQueue is not needed since we will use DuckDB task scheduler to manage tasks
class ScanGlobalSourceState : public GlobalSourceState {
public:
	ScanGlobalSourceState(ClientContext &context, const GPUPhysicalTableScan &op) {
	}

	idx_t max_threads = 0;
	unique_ptr<GlobalTableFunctionState> global_state;
	bool in_out_final = false;
	DataChunk input_chunk;
	unique_ptr<TableFilterSet> table_filters;

	optional_ptr<TableFilterSet> GetTableFilters(const GPUPhysicalTableScan &op) const {
		return table_filters ? table_filters.get() : op.fake_table_filters.get();
	}
	idx_t MaxThreads() override {
		return max_threads;
	}

    // The followings are used in `TableScanCoalesceTask`
    void InitForTableScanCoalesceTask(const GPUPhysicalTableScan& op, uint8_t** mask_ptr_p) {
    }

    void NextChunkOffsetsAligned(uint64_t chunk_rows, const vector<uint64_t>& chunk_column_sizes,
                                uint64_t* out_row_offset, vector<uint64_t>& out_column_data_offsets) {
    }

    inline void AssignBits(uint8_t from, int from_pos, uint8_t* to, int to_pos, int n) {
    }

    void NextChunkOffsetsUnaligned(uint64_t chunk_rows, const vector<uint64_t>& chunk_column_sizes,
                                    uint64_t* out_row_offset, vector<uint64_t>& out_column_data_offsets,
                                    const vector<uint8_t>& chunk_unaligned_mask_bytes) {
    }

    // For both rows which are null mask aligned and unaligned
    struct {
        std::mutex mutex;
        uint64_t row_offset;
        vector<uint64_t> column_data_offsets;
    } offset_info_aligned, offset_info_unaligned;

    // For compacting null mask bytes of unaligned portion per column. We write starting from last bit
    // since the unaligned portion is written from the end.
    uint8_t** mask_ptr;
    uint64_t unaligned_mask_byte_pos;
    int unaligned_mask_in_byte_pos;
};

class ScanLocalSourceState : public LocalSourceState {
public:
	ScanLocalSourceState(ExecutionContext &context, ScanGlobalSourceState &gstate,
	                     const GPUPhysicalTableScan &op) {
		if (op.function.init_local) {
			TableFunctionInitInput input(op.bind_data.get(), op.column_ids, op.scanned_ids,
			                             gstate.GetTableFilters(op), op.extra_info.sample_options);
			local_state = op.function.init_local(context, input, gstate.global_state.get());
		}
        num_rows = 0;
        column_size.resize(op.column_ids.size(), 0);
	}

	unique_ptr<LocalTableFunctionState> local_state;

    // Used in `TableScanGetSizeTask`
    uint64_t num_rows;
    vector<uint64_t> column_size;
};

class ScanGetSizeTask : public BaseExecutorTask {
public:
	ScanGetSizeTask(TaskExecutor &executor, int task_id_p, TableFunction& function_p, ExecutionContext& context_p,
                       GPUPhysicalTableScan& op_p, GlobalSourceState* g_state_p, LocalSourceState* l_state_p)
	    : BaseExecutorTask(executor), task_id(task_id_p), function(function_p), context(context_p),
        op(op_p), g_state(g_state_p), l_state(l_state_p) {}

	void ExecuteTask() override {
    }

private:
  int task_id;
  TableFunction& function;
  ExecutionContext& context;
  GPUPhysicalTableScan& op;
  GlobalSourceState* g_state;
  LocalSourceState* l_state;
};

class ScanCoalesceTask : public BaseExecutorTask {
public:
	ScanCoalesceTask(TaskExecutor &executor, int task_id_p, TableFunction& function_p, ExecutionContext& context_p,
                        GPUPhysicalTableScan& op_p, GlobalSourceState* g_state_p, LocalSourceState* l_state_p,
                        uint8_t** data_ptr_p, uint8_t** mask_ptr_p, uint64_t** offset_ptr_p,
                        int64_t* duckdb_storage_row_ids_ptr_p)
	    : BaseExecutorTask(executor), task_id(task_id_p), function(function_p), context(context_p),
        op(op_p), g_state(g_state_p), l_state(l_state_p), data_ptr(data_ptr_p), mask_ptr(mask_ptr_p),
        offset_ptr(offset_ptr_p), duckdb_storage_row_ids_ptr(duckdb_storage_row_ids_ptr_p) {}

	void ExecuteTask() override {
	}

private:
  int task_id;
  TableFunction& function;
  ExecutionContext& context;
  GPUPhysicalTableScan& op;
  GlobalSourceState* g_state;
  LocalSourceState* l_state;
  uint8_t** data_ptr;
  uint8_t** mask_ptr;
  uint64_t** offset_ptr;
  int64_t* duckdb_storage_row_ids_ptr;
};

} // namespace sirius
} // namespace duckdb