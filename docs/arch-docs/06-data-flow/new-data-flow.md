# New Mode Data Flow

Deep dive into data flow in Sirius New Mode (`gpu_execution`), focusing on the cucascade integration, port-based communication, and task-driven execution model.

---

## Overview

New Mode introduces a fundamentally different data flow model compared to Legacy Mode:

| Aspect | Legacy Mode | New Mode |
|--------|-------------|----------|
| **Data Unit** | `GPUIntermediateRelation` | `cucascade::data_batch` |
| **Pipeline Communication** | Direct push/pull | Port-based repositories |
| **Execution Model** | Pull-based (GetData) | Push-based (publish) |
| **Task Creation** | Static | Dynamic (hint-based) |
| **Memory Management** | GPUBufferManager | Multi-tier (GPU/HOST/DISK) |

**Key Innovation**: New Mode uses a **port-based publish-subscribe model** where pipelines communicate through shared data repositories, enabling better parallelism and memory management.

---

## Core Data Structures

### 1. cucascade::data_batch

The fundamental data unit in New Mode.

**Definition**: `cucascade/include/data_batch.hpp`

```cpp
class data_batch {
public:
    // Holds cuDF table with columnar data
    std::unique_ptr<cudf::table> table;

    // Memory tier: GPU, HOST, or DISK
    MemoryTier tier;

    // Metadata
    size_t num_rows;
    std::vector<cudf::data_type> schema;

    // Memory reservation ID
    ReservationID reservation_id;
};
```

**Properties**:
- **Immutable**: Once created, data_batch contents don't change
- **Movable**: Can be moved between memory tiers
- **Shared**: Multiple consumers can access via shared_ptr
- **Bounded**: Default size ~100K rows (configurable via `scan_batch_size`)

**Lifecycle**:
```
Create → Process → Publish → Store → Retrieve → Consume → Release
```

### 2. cucascade::shared_data_repository

Inter-pipeline data storage.

**Definition**: `cucascade/include/data_repository.hpp`

```cpp
class shared_data_repository {
private:
    // Multi-tier storage queues
    std::queue<data_batch> gpu_queue;
    std::queue<data_batch> host_queue;
    std::queue<data_batch> disk_queue;

    // Memory reservation manager
    memory_reservation_manager& mem_mgr;

    // Synchronization
    std::mutex mutex;
    std::condition_variable cv_data_ready;

public:
    // Producer API
    void push_data_batch(data_batch&& batch);
    void mark_complete();

    // Consumer API
    std::optional<data_batch> pull_batch();
    bool is_complete();
};
```

**Key Features**:
- **Multi-tier**: Automatically spills data (GPU → HOST → DISK)
- **Thread-safe**: Multiple producers/consumers
- **Blocking**: Consumers wait for data if queue empty
- **Completion signaling**: Producers signal end-of-stream

### 3. Port System

Connects pipelines to data repositories.

**Types**:
- **Input Port**: Reads from repository
- **Output Port**: Writes to repository

**Example** (`src/include/pipeline/sirius_pipeline.hpp:45-60`):

```cpp
class sirius_pipeline {
    // Input ports (read from repositories)
    std::vector<shared_ptr<shared_data_repository>> input_ports;

    // Output ports (write to repositories)
    std::vector<shared_ptr<shared_data_repository>> output_ports;

    // Port management
    void add_input_port(shared_ptr<shared_data_repository> repo);
    void add_output_port(shared_ptr<shared_data_repository> repo);
};
```

---

## Data Flow Patterns

### Pattern 1: Simple Pipeline (No Dependencies)

**Example Query**:
```sql
SELECT * FROM gpu_execution('SELECT * FROM data.parquet WHERE x > 100');
```

**Data Flow**:
```
SCAN (source)
    ↓
  Read Parquet
    ↓
For each batch (100K rows):
    ↓
  Create data_batch
    ↓
  FILTER (sink/source)
    ↓
  Apply predicate
    ↓
  RESULT_COLLECTOR (sink)
    ↓
  Materialize
    ↓
QueryResult
```

**No Repositories**: Single pipeline, no inter-pipeline communication needed.

**Code Path** (`src/op/sirius_physical_scan.cpp:120-145`):

```cpp
// SCAN operator creates batches
void sirius_physical_table_scan::execute(ExecutionContext& ctx) {
    // Read from Parquet file
    auto cudf_table = read_parquet_batch(file_path, batch_idx);

    // Create data_batch
    auto batch = data_batch{
        .table = std::move(cudf_table),
        .tier = MemoryTier::GPU,
        .num_rows = cudf_table->num_rows()
    };

    // Publish to next operator
    ctx.publish_output(std::move(batch));
}
```

### Pattern 2: Pipeline with Single Dependency

**Example Query**:
```sql
SELECT * FROM gpu_execution('
    SELECT category, SUM(price)
    FROM sales.parquet
    WHERE year = 2024
    GROUP BY category
    ORDER BY category
');
```

**Pipeline Break**: ORDER BY requires all GROUP BY results first.

**Data Flow**:

```
Pipeline 1: Scan + Filter + GroupBy
┌─────────────────────────────────┐
│ SCAN                            │
│   ↓                             │
│ FILTER (year = 2024)            │
│   ↓                             │
│ HASH_GROUP_BY (category)        │
│   ↓                             │
│ push_data_batch()               │
└─────────────────────────────────┘
          ↓
    Data Repository
    (shared_data_repository)
    - GPU queue: batches 0-5
    - HOST queue: batches 6-10 (spilled)
          ↓
Pipeline 2: Order By + Result Collection
┌─────────────────────────────────┐
│ pull_batch()                    │
│   ↓                             │
│ ORDER_BY (category)             │
│   ↓                             │
│ RESULT_COLLECTOR                │
└─────────────────────────────────┘
```

**Producer Code** (`src/op/sirius_physical_hash_group_by.cpp:200-230`):

```cpp
void sirius_physical_hash_group_by::sink(data_batch&& input_batch) {
    // Accumulate into hash table
    hash_table.aggregate(input_batch);

    // Check if memory pressure
    if (should_flush()) {
        // Finalize aggregations
        auto result_batch = hash_table.finalize();

        // Publish to output port (data repository)
        auto& output_repo = output_ports[0];
        output_repo->push_data_batch(std::move(result_batch));

        // Reset hash table
        hash_table.reset();
    }
}

void sirius_physical_hash_group_by::finalize() {
    // Flush remaining data
    auto final_batch = hash_table.finalize();
    output_ports[0]->push_data_batch(std::move(final_batch));

    // Signal completion
    output_ports[0]->mark_complete();
}
```

**Consumer Code** (`src/op/sirius_physical_order_by.cpp:150-180`):

```cpp
TaskCreationHint sirius_physical_order_by::get_next_task_hint() {
    auto& input_repo = input_ports[0];

    // Check if input available
    if (!input_repo->is_complete() && input_repo->empty()) {
        return TaskCreationHint::WAITING_FOR_INPUT_DATA;
    }

    if (input_repo->has_data()) {
        return TaskCreationHint::READY;
    }

    return TaskCreationHint::NO_MORE_TASKS;
}

data_batch sirius_physical_order_by::get_next_task_input_batch() {
    auto& input_repo = input_ports[0];

    // Pull batch from repository (may block)
    auto batch_opt = input_repo->pull_batch();

    if (batch_opt.has_value()) {
        return std::move(batch_opt.value());
    }

    throw InternalException("No input data available");
}
```

### Pattern 3: Multi-Pipeline with Complex Dependencies

**Example Query**:
```sql
SELECT * FROM gpu_execution('
    SELECT l.order_id, l.product, c.customer_name, l.quantity * l.price AS total
    FROM lineitem.parquet l
    JOIN customers.parquet c ON l.customer_id = c.id
    WHERE l.quantity > 10
    ORDER BY total DESC
    LIMIT 100
');
```

**Pipeline Structure**:

```
Pipeline 1: Lineitem Scan + Filter (Build Side)
┌──────────────────────────┐
│ SCAN lineitem.parquet    │
│   ↓                      │
│ FILTER (quantity > 10)   │
│   ↓                      │
│ HASH_JOIN (build)        │
│   → Build hash table     │
└──────────────────────────┘
         ↓
  Repository A (join probe input)

Pipeline 2: Customer Scan (Probe Side)
┌──────────────────────────┐
│ SCAN customers.parquet   │
│   ↓                      │
│ HASH_JOIN (probe)        │
│   ← Read hash table      │
│   → Probe and emit       │
└──────────────────────────┘
         ↓
  Repository B (join output)

Pipeline 3: Order + Limit + Result
┌──────────────────────────┐
│ ORDER_BY (total DESC)    │
│   ↓                      │
│ TOP_N (limit 100)        │
│   ↓                      │
│ RESULT_COLLECTOR         │
└──────────────────────────┘
```

**Execution Timeline**:

```
Time    Pipeline 1          Repository A         Pipeline 2          Repository B         Pipeline 3
─────   ──────────          ────────────         ──────────          ────────────         ──────────
0ms     Scan lineitem       [empty]              Scan customers      [empty]              [waiting]
        Filter batch 0

50ms    Build hash 0→5      [empty]              Waiting...          [empty]              [waiting]
        (GPU memory)

100ms   Build complete      [hash table ready]   Probe batch 0       [empty]              [waiting]
        Signal complete                          Emit joined 0

150ms   [complete]          [hash table ready]   Probe batch 1       [batch 0]            [waiting]
                                                  Emit joined 1       [batch 1]

200ms   [complete]          [hash table ready]   Probe batch 2       [batch 0-2]          Pull batch 0
                                                  Emit joined 2       [batch 1-2]          Sort local

250ms   [complete]          [hash table ready]   Probe complete      [batch 0-5]          Pull batch 1-5
                                                  Signal complete                          Sort all

300ms   [complete]          [hash table ready]   [complete]          [batch 0-5]          Top N (100)
                                                                      [complete]

350ms   [complete]          [hash table ready]   [complete]          [complete]           Result collect
```

---

## Task-Based Execution

### Task Creation Workflow

New Mode uses **dynamic task creation** based on runtime hints.

**Code** (`src/parallel/task_creator.cpp:80-120`):

```cpp
void task_creator::create_tasks_for_pipeline(sirius_pipeline& pipeline) {
    while (true) {
        // Ask operator for hint
        auto hint = pipeline.source->get_next_task_hint();

        switch (hint) {
            case TaskCreationHint::READY:
                // Input data available, create task
                auto input_batch = pipeline.source->get_next_task_input_batch();
                auto task = std::make_unique<sirius_pipeline_itask>(
                    pipeline, std::move(input_batch)
                );
                task_queue.enqueue(std::move(task));
                break;

            case TaskCreationHint::WAITING_FOR_INPUT_DATA:
                // No data yet, check again later
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                break;

            case TaskCreationHint::NO_MORE_TASKS:
                // Pipeline complete
                return;
        }
    }
}
```

### Task Execution Workflow

**Code** (`src/pipeline/sirius_pipeline_itask.cpp:50-90`):

```cpp
void sirius_pipeline_itask::compute_task() {
    // 1. Execute source operator
    if (pipeline.source->operator_type() != SiriusPhysicalOperatorType::DUMMY_SCAN) {
        // Source creates/transforms batch
        auto output = pipeline.source->execute(input_batch);
        current_batch = std::move(output);
    } else {
        // Dummy scan (data already in batch)
        current_batch = std::move(input_batch);
    }

    // 2. Execute intermediate operators in sequence
    for (auto& op : pipeline.intermediate_operators) {
        current_batch = op->execute(std::move(current_batch));

        // Check if batch was filtered out
        if (current_batch.num_rows == 0) {
            return; // Early exit
        }
    }

    // 3. Sink operator
    if (pipeline.sink) {
        pipeline.sink->sink(std::move(current_batch));
    }

    // 4. Publish output if needed
    publish_output();
}

void sirius_pipeline_itask::publish_output() {
    if (!pipeline.output_ports.empty() && current_batch.num_rows > 0) {
        // Push to all output ports
        for (auto& port : pipeline.output_ports) {
            port->push_data_batch(current_batch.clone());
        }
    }
}
```

---

## Memory Management During Data Flow

### Multi-Tier Storage

Data batches automatically move between memory tiers based on pressure.

**Memory Hierarchy**:
```
┌────────────────────────────────────────────────┐
│ GPU Memory (Fastest, Limited)                  │
│ - Active batches being processed               │
│ - Recently created batches                     │
│ - Hot data in repositories                     │
│ Capacity: ~16GB (typical A100)                 │
└────────────────────────────────────────────────┘
              ↓ Spill on pressure
┌────────────────────────────────────────────────┐
│ Host Memory (Fast, Larger)                     │
│ - Spilled batches awaiting processing          │
│ - Intermediate results                         │
│ Capacity: ~128GB (typical server)              │
└────────────────────────────────────────────────┘
              ↓ Spill on pressure
┌────────────────────────────────────────────────┐
│ Disk Storage (Slower, Unlimited)               │
│ - Cold batches rarely accessed                 │
│ - Large intermediate results                   │
│ Capacity: ~1TB+ (NVMe SSD)                     │
└────────────────────────────────────────────────┘
```

### Automatic Downgrade

**Code** (`src/memory/sirius_memory_reservation_manager.cpp:200-250`):

```cpp
void memory_reservation_manager::check_and_downgrade() {
    // Check GPU memory pressure
    if (gpu_usage > gpu_limit * 0.9) {
        // Find eviction candidates from repositories
        auto candidates = find_eviction_candidates(MemoryTier::GPU);

        for (auto& batch : candidates) {
            // Downgrade GPU → HOST
            downgrade_batch(batch, MemoryTier::HOST);

            if (gpu_usage < gpu_limit * 0.8) {
                break; // Sufficient space freed
            }
        }
    }

    // Check HOST memory pressure
    if (host_usage > host_limit * 0.9) {
        auto candidates = find_eviction_candidates(MemoryTier::HOST);

        for (auto& batch : candidates) {
            // Downgrade HOST → DISK
            downgrade_batch(batch, MemoryTier::DISK);

            if (host_usage < host_limit * 0.8) {
                break;
            }
        }
    }
}

void memory_reservation_manager::downgrade_batch(
    data_batch& batch, MemoryTier target_tier) {

    switch (target_tier) {
        case MemoryTier::HOST:
            // GPU → HOST: cudaMemcpy to pinned memory
            batch.table = copy_to_host(batch.table);
            break;

        case MemoryTier::DISK:
            // HOST → DISK: serialize to Parquet
            write_parquet(batch.table, temp_file_path());
            batch.table.reset(); // Free memory
            break;
    }

    batch.tier = target_tier;
}
```

### Automatic Upgrade

When a consumer pulls a batch from a repository, it's automatically upgraded if needed.

**Code** (`cucascade/src/data_repository.cpp:100-140`):

```cpp
std::optional<data_batch> shared_data_repository::pull_batch() {
    std::unique_lock<std::mutex> lock(mutex);

    // Wait for data if not available
    cv_data_ready.wait(lock, [this]() {
        return !is_empty() || completed;
    });

    if (is_empty()) {
        return std::nullopt; // No more data
    }

    // Try tiers in order: GPU → HOST → DISK
    data_batch batch;

    if (!gpu_queue.empty()) {
        batch = std::move(gpu_queue.front());
        gpu_queue.pop();
    } else if (!host_queue.empty()) {
        batch = std::move(host_queue.front());
        host_queue.pop();

        // Upgrade HOST → GPU
        upgrade_batch(batch, MemoryTier::GPU);
    } else if (!disk_queue.empty()) {
        batch = std::move(disk_queue.front());
        disk_queue.pop();

        // Upgrade DISK → GPU (via HOST)
        upgrade_batch(batch, MemoryTier::HOST);
        upgrade_batch(batch, MemoryTier::GPU);
    }

    return batch;
}

void shared_data_repository::upgrade_batch(
    data_batch& batch, MemoryTier target_tier) {

    switch (target_tier) {
        case MemoryTier::GPU:
            if (batch.tier == MemoryTier::HOST) {
                // HOST → GPU: cudaMemcpy from pinned memory
                batch.table = copy_to_gpu(batch.table);
            } else if (batch.tier == MemoryTier::DISK) {
                // DISK → HOST first
                batch.table = read_parquet(batch.disk_path);
            }
            break;

        case MemoryTier::HOST:
            if (batch.tier == MemoryTier::DISK) {
                // DISK → HOST: read Parquet
                batch.table = read_parquet(batch.disk_path);
            }
            break;
    }

    batch.tier = target_tier;
}
```

---

## Concrete Example: Join Query Data Flow

Let's trace a complete join query through the system.

**Query**:
```sql
SELECT * FROM gpu_execution('
    SELECT o.order_id, o.amount, c.name
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
');
```

**Setup**:
- `orders`: 1M rows, 50MB on GPU
- `customers`: 100K rows, 5MB on GPU

### Step-by-Step Data Flow

#### Phase 1: Build Hash Table (Pipeline 1)

```
Time: 0-100ms

SCAN customers
  ↓
Batch 0: 100K rows, 5MB GPU
  ↓
HASH_JOIN (build phase)
  ↓
Create hash table:
  - Keys: customer.id (100K unique)
  - Values: customer.name
  - Size: ~8MB GPU memory
  - Structure: cuDF hash map
  ↓
Store in shared state
Signal build complete
```

**Code** (`src/op/sirius_physical_hash_join.cpp:150-190`):

```cpp
void sirius_physical_hash_join::sink_build(data_batch&& batch) {
    // Build side of join (customers table)

    // Extract key columns
    auto key_col = batch.table->get_column(join_key_idx);

    // Extract payload columns
    auto payload_cols = batch.table->select(payload_indices);

    // Insert into hash table
    hash_table->insert(key_col, payload_cols);

    // Update statistics
    build_row_count += batch.num_rows;
}

void sirius_physical_hash_join::finalize_build() {
    // Build complete, signal probe can start
    hash_table->finalize();
    build_complete_flag.store(true);

    // Notify waiting probe tasks
    cv_build_complete.notify_all();
}
```

#### Phase 2: Probe Hash Table (Pipeline 2)

```
Time: 100-300ms (overlapped with build)

Wait for build complete (blocks until hash table ready)
  ↓
SCAN orders
  ↓
Batch 0: 100K rows, 5MB GPU
  ↓
HASH_JOIN (probe phase)
  ↓
For each row in batch:
  - Extract customer_id
  - Probe hash table
  - If match: emit (order_id, amount, customer.name)
  - If no match: skip
  ↓
Output Batch 0: 95K rows (95% match rate)
  ↓
push_data_batch() → Repository
  ↓
Repeat for batches 1-9...
  ↓
Final: 10 output batches, ~950K total rows
Signal probe complete
```

**Code** (`src/op/sirius_physical_hash_join.cpp:250-310`):

```cpp
data_batch sirius_physical_hash_join::execute_probe(data_batch&& batch) {
    // Probe side of join (orders table)

    // Wait for build to complete
    if (!build_complete_flag.load()) {
        std::unique_lock<std::mutex> lock(build_mutex);
        cv_build_complete.wait(lock, [this]() {
            return build_complete_flag.load();
        });
    }

    // Extract probe key column
    auto probe_key = batch.table->get_column(probe_key_idx);

    // Probe hash table
    auto matched_indices = hash_table->probe(probe_key);

    // Gather matched rows from both sides
    auto left_matches = cudf::gather(
        batch.table->view(),
        matched_indices.first  // Probe side indices
    );

    auto right_matches = cudf::gather(
        hash_table->payload_table->view(),
        matched_indices.second  // Build side indices
    );

    // Concatenate columns
    std::vector<std::unique_ptr<cudf::column>> output_cols;
    for (auto& col : left_matches->release()) {
        output_cols.push_back(std::move(col));
    }
    for (auto& col : right_matches->release()) {
        output_cols.push_back(std::move(col));
    }

    // Create output batch
    return data_batch{
        .table = std::make_unique<cudf::table>(std::move(output_cols)),
        .tier = MemoryTier::GPU,
        .num_rows = matched_indices.first.size()
    };
}
```

#### Phase 3: Collect Results (Pipeline 3)

```
Time: 300-350ms

pull_batch() from Repository
  ↓
Batch 0: 95K rows
  ↓
RESULT_COLLECTOR (accumulate)
  ↓
pull_batch() from Repository
  ↓
Batch 1: 95K rows
  ↓
Accumulate...
  ↓
Repository signals complete
  ↓
Finalize result collection
  ↓
Convert to DuckDB DataChunk
  ↓
Return QueryResult
```

**Memory Usage Timeline**:

```
Time    GPU Usage          HOST Usage         DISK Usage
────    ─────────          ──────────         ──────────
0ms     5MB (customers)    0MB                0MB
100ms   13MB (+hash)       0MB                0MB
150ms   18MB (+orders[0])  0MB                0MB
200ms   23MB (+orders[1])  0MB                0MB

        ... batches 0-5 processed in GPU ...

250ms   30MB (pressure!)   0MB                0MB
        Spill orders[0-2]  15MB (spilled)     0MB
300ms   20MB               15MB               0MB
350ms   5MB (cleanup)      0MB                0MB
```

---

## Performance Characteristics

### Data Transfer Overhead

**Typical Costs** (A100 GPU, PCIe Gen4):

| Transfer | Bandwidth | 100K Rows Cost | Notes |
|----------|-----------|----------------|-------|
| CPU → GPU | 32 GB/s | ~0.15ms (5MB) | Initial load |
| GPU → CPU | 32 GB/s | ~0.15ms (5MB) | Result collection |
| GPU → HOST (spill) | 32 GB/s | ~0.15ms (5MB) | Memory pressure |
| HOST → GPU (restore) | 32 GB/s | ~0.15ms (5MB) | Pull from spill |
| HOST → DISK (spill) | 5 GB/s | ~1ms (5MB) | NVMe write |
| DISK → HOST (restore) | 7 GB/s | ~0.7ms (5MB) | NVMe read |

**Optimization**: Keep hot data in GPU, minimize spilling.

### Repository Latency

**Pull Operation Latency**:

| Scenario | Latency | Notes |
|----------|---------|-------|
| Data in GPU queue | ~1μs | Lock + pop |
| Data in HOST queue | ~0.15ms | Lock + pop + upgrade |
| Data in DISK queue | ~0.8ms | Lock + pop + read + upgrade |
| No data (block) | Variable | Wait for producer |

### Task Creation Throughput

**Measured Performance** (8-core CPU, A100 GPU):

| Scenario | Tasks/sec | Notes |
|----------|-----------|-------|
| Simple scan | 20,000 | No dependencies |
| With dependencies | 5,000 | Hint checking overhead |
| Complex join | 2,000 | Synchronization overhead |

---

## Best Practices

### 1. Minimize Pipeline Breaks

**Bad** (4 pipelines):
```sql
-- Each operation forces pipeline break
SELECT * FROM gpu_execution('
    SELECT category, SUM(amount) as total
    FROM (
        SELECT category, amount
        FROM sales
        WHERE year = 2024
        ORDER BY category  -- Pipeline break 1
    )
    GROUP BY category      -- Pipeline break 2
    ORDER BY total DESC    -- Pipeline break 3
    LIMIT 10               -- Pipeline break 4
');
```

**Good** (2 pipelines):
```sql
-- Combine operations to reduce breaks
SELECT * FROM gpu_execution('
    SELECT category, SUM(amount) as total
    FROM sales
    WHERE year = 2024
    GROUP BY category      -- Pipeline break 1
    ORDER BY total DESC    -- Pipeline break 2 (unavoidable)
    LIMIT 10
');
```

### 2. Tune Batch Size

Configure based on memory and workload:

```ini
[execution]
# Small batches: less memory, more overhead
scan_batch_size = 50000    # For wide tables (many columns)

# Large batches: more memory, less overhead
scan_batch_size = 200000   # For narrow tables (few columns)

# Default
scan_batch_size = 100000   # Good for most cases
```

### 3. Monitor Memory Tiers

Check where data resides:

```sql
-- Enable monitoring
SET sirius_enable_monitoring = true;

-- Run query
SELECT * FROM gpu_execution('...');

-- Check tier distribution
SELECT * FROM sirius_memory_stats();
```

**Output**:
```
tier  | batches | total_size | hit_rate
------+---------+------------+---------
GPU   | 85      | 425MB      | 95%
HOST  | 12      | 60MB       | 4%
DISK  | 3       | 15MB       | 1%
```

**Interpretation**:
- 95% GPU hit rate: Excellent (minimal spilling)
- 4% HOST hit rate: Acceptable (some pressure)
- 1% DISK hit rate: Rare spilling

### 4. Size Memory Tiers Appropriately

**Recommended Configuration**:

```ini
[memory]
# GPU: 75-80% of available
gpu_memory_limit = 12288    # 12GB (for 16GB GPU)

# HOST: 75% of available RAM
host_memory_limit = 49152   # 48GB (for 64GB RAM)

# DISK: unlimited or large
disk_memory_limit = -1      # Unlimited

# Always enable spilling
enable_spilling = true
```

---

## Debugging Data Flow Issues

### Problem: Tasks Wait Forever (Deadlock)

**Symptoms**:
- Query hangs indefinitely
- `get_next_task_hint()` returns `WAITING_FOR_INPUT_DATA` forever

**Diagnosis**:
```bash
# Enable debug logging
export SIRIUS_LOG_LEVEL=DEBUG

# Check logs for repository state
grep "repository.*waiting" /tmp/sirius.log
```

**Common Causes**:
1. Producer never calls `mark_complete()`
2. Repository connection not established
3. Pipeline dependency cycle

**Solution**:
- Ensure all sink operators call `mark_complete()` in finalize
- Verify port connections in pipeline setup
- Check for circular dependencies in pipeline DAG

### Problem: Excessive Spilling

**Symptoms**:
- Query slow despite GPU
- High DISK tier usage
- Many "downgrade" log messages

**Diagnosis**:
```sql
SELECT * FROM sirius_memory_stats();
```

**Solutions**:
1. Increase GPU memory limit
2. Reduce batch size
3. Add more selective filters early
4. Split query into smaller parts

### Problem: Low GPU Utilization

**Symptoms**:
- GPU idle during query execution
- Long wait times in task creation

**Diagnosis**:
```sql
SELECT * FROM sirius_execution_stats()
WHERE query_id = last_query_id();
```

**Solutions**:
1. Increase `pipeline_executor_threads`
2. Increase `task_creator_threads`
3. Reduce data dependencies (minimize pipeline breaks)
4. Check for CPU bottlenecks in scan

---

## See Also

- [Query Lifecycle](query-lifecycle.md) - Complete query execution trace
- [New Mode Overview](../04-new-mode/overview.md) - New Mode introduction
- [Cucascade Integration](../04-new-mode/cucascade-integration.md) - Repository details
- [Pipeline Execution](../04-new-mode/pipeline-execution.md) - Task model
- [Memory Management](../05-core-components/memory-management.md) - Multi-tier memory
- [Threading Model](../05-core-components/threading-model.md) - Task executors
- [Performance Tips](../appendices/performance-tips.md) - Optimization strategies
