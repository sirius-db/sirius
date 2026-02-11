# Inter-Pipeline Communication

Detailed guide to how pipelines communicate with each other in Sirius, covering both Legacy Mode's direct model and New Mode's repository-based system.

---

## Overview

Pipeline communication is necessary when a query requires **pipeline breaks** - points where one pipeline must complete before another can proceed.

**Common Pipeline Breaks**:
- **Sorting**: ORDER BY requires all input data
- **Aggregation**: GROUP BY requires all input data (for finalization)
- **Joins**: Hash join build side must complete before probe
- **Window Functions**: Partitioning requires complete partitions

| Aspect | Legacy Mode | New Mode |
|--------|-------------|----------|
| **Communication Model** | Direct (operator state) | Repository-based (ports) |
| **Synchronization** | Implicit (call stack) | Explicit (repositories) |
| **Data Transfer** | In-memory pointers | Data batches |
| **Buffering** | In operator state | Multi-tier repositories |
| **Parallelism** | Sequential | Concurrent |

---

## Legacy Mode: Direct Communication

### Model

Pipelines communicate through **shared operator state**.

**Example**: Hash Join

```
Pipeline 1 (Build)         Shared State         Pipeline 2 (Probe)
──────────────────         ────────────         ──────────────────
Scan build side      →  Hash Table (GPU)  ←    Scan probe side
HASH_JOIN sink           - Keys: customer.id    HASH_JOIN source
Finalize build           - Values: customer.*   Probe hash table
                         - Size: 8MB             Emit joined rows
                         - Complete: true
```

### Implementation

**Hash Join Operator** (`src/operator/gpu_physical_hash_join.cpp:50-100`):

```cpp
class GPUPhysicalHashJoin : public GPUPhysicalOperator {
private:
    // Shared state between pipelines
    std::unique_ptr<cudf::hash_join> hash_table;
    std::unique_ptr<cudf::table> payload_table;

    // Synchronization
    bool build_complete = false;
    std::mutex build_mutex;
    std::condition_variable build_cv;

    // Results storage
    std::vector<GPUIntermediateRelation> probe_results;
    GPUIntermediateRelation final_result;
    bool has_emitted = false;

public:
    // Pipeline 1: Build side (sink)
    void SinkBuild(GPUIntermediateRelation& input) {
        // Insert into hash table
        auto keys = ExtractKeys(input);
        auto payload = ExtractPayload(input);

        if (!hash_table) {
            hash_table = std::make_unique<cudf::hash_join>(keys);
            payload_table = payload;
        } else {
            hash_table->append(keys);
            payload_table = Concatenate(payload_table, payload);
        }
    }

    void FinalizeBuild() {
        std::lock_guard<std::mutex> lock(build_mutex);

        hash_table->finalize();
        build_complete = true;

        // Wake up probe pipelines
        build_cv.notify_all();
    }

    // Pipeline 2: Probe side (sink)
    void SinkProbe(GPUIntermediateRelation& input) {
        // Wait for build to complete
        {
            std::unique_lock<std::mutex> lock(build_mutex);
            build_cv.wait(lock, [this]() {
                return build_complete;
            });
        }

        // Probe hash table
        auto keys = ExtractKeys(input);
        auto [left_idx, right_idx] = hash_table->probe(keys);

        // Gather and store results
        auto result = GatherMatches(input, payload_table, left_idx, right_idx);
        probe_results.push_back(std::move(result));
    }

    void FinalizeProbe() {
        // Concatenate all probe results
        final_result = ConcatenateAll(probe_results);
        probe_results.clear();
    }

    // Pipeline 3: Source (emit)
    GPUIntermediateRelation GetData() {
        if (has_emitted) {
            return GPUIntermediateRelation(); // Empty
        }
        has_emitted = true;
        return std::move(final_result);
    }
};
```

### Execution Flow

```
Step 1: Execute Pipeline 1 (Build)
───────────────────────────────────

GPUMetaPipeline::Execute()
  ↓
Pipeline[0].Execute()  // Build pipeline
  ↓
Loop:
  SCAN build → GetData() → batch
  HASH_JOIN → SinkBuild(batch)

  ... process all batches ...

HASH_JOIN → FinalizeBuild()
  - Set build_complete = true
  - Notify condition variable

Pipeline 1 Complete ✓


Step 2: Execute Pipeline 2 (Probe)
───────────────────────────────────

Pipeline[1].Execute()  // Probe pipeline
  ↓
Loop:
  SCAN probe → GetData() → batch
  HASH_JOIN → SinkProbe(batch)
    - Wait for build_complete (already true)
    - Probe hash table
    - Store results

  ... process all batches ...

HASH_JOIN → FinalizeProbe()
  - Concatenate probe results

Pipeline 2 Complete ✓


Step 3: Execute Pipeline 3 (Emit)
──────────────────────────────────

Pipeline[2].Execute()  // Result pipeline
  ↓
RESULT_COLLECTOR → GetData()
  ↓
HASH_JOIN → GetData()
  - Return final_result

Pipeline 3 Complete ✓
```

### Synchronization Mechanisms

#### Mutex + Condition Variable

**Pattern**: Producer signals consumers

```cpp
// Producer (build pipeline)
{
    std::lock_guard<std::mutex> lock(mutex);
    // Update shared state
    build_complete = true;
}
cv.notify_all(); // Wake all waiting threads

// Consumer (probe pipeline)
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() {
        return build_complete; // Predicate
    });
}
// Proceed with probe
```

#### Atomic Flags

**Pattern**: Simple boolean state

```cpp
// Producer
data_ready.store(true, std::memory_order_release);

// Consumer
while (!data_ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
}
```

### Limitations

1. **Sequential Execution**: Pipeline N+1 blocked until N completes
2. **Memory Growth**: Intermediate results stored in operator state
3. **No Spilling**: All data must fit in GPU memory
4. **Single Consumer**: Hard to parallelize probe phase

---

## New Mode: Repository-Based Communication

### Model

Pipelines communicate through **shared data repositories** connected via **ports**.

**Example**: Hash Join

```
Pipeline 1 (Build)         Repository A            Pipeline 2 (Probe)
──────────────────         ────────────            ──────────────────
Scan build side      →  Build Data Queue     ←    HASH_JOIN build
HASH_JOIN build sink    [batch0, batch1...]       Pull batches
Mark complete           Complete: true            Build hash table
                        ↓
                    Repository B            ←    Pipeline 3 (Result)
                    ────────────                  ─────────────────
                    Joined Results                Pull batches
                    [result0, result1...]         ORDER BY
                                                  RESULT_COLLECTOR
```

### Data Repository

**Definition**: `cucascade/include/data_repository.hpp`

```cpp
class shared_data_repository {
private:
    // Multi-tier storage
    std::deque<data_batch> gpu_queue;
    std::deque<data_batch> host_queue;
    std::deque<data_batch> disk_queue;

    // State
    bool completed = false;
    size_t total_batches_pushed = 0;
    size_t total_batches_pulled = 0;

    // Synchronization
    std::mutex mutex;
    std::condition_variable cv_data_ready;
    std::condition_variable cv_space_available;

    // Memory management
    memory_reservation_manager& mem_mgr;

public:
    // Producer API
    void push_data_batch(data_batch&& batch);
    void mark_complete();
    bool is_complete() const;

    // Consumer API
    std::optional<data_batch> pull_batch();
    bool has_data() const;
    bool is_empty() const;

    // Statistics
    size_t size() const;
    size_t gpu_size() const;
    size_t host_size() const;
    size_t disk_size() const;
};
```

### Push/Pull Operations

#### Push (Producer Side)

**Code** (`cucascade/src/data_repository.cpp:50-100`):

```cpp
void shared_data_repository::push_data_batch(data_batch&& batch) {
    std::unique_lock<std::mutex> lock(mutex);

    // Reserve memory for this batch
    auto reservation = mem_mgr.reserve(batch.size_bytes());

    // Determine target tier based on memory availability
    MemoryTier target_tier;
    if (mem_mgr.gpu_has_space(batch.size_bytes())) {
        target_tier = MemoryTier::GPU;
    } else if (mem_mgr.host_has_space(batch.size_bytes())) {
        target_tier = MemoryTier::HOST;
        // Downgrade batch
        batch = downgrade_to_host(std::move(batch));
    } else {
        target_tier = MemoryTier::DISK;
        // Downgrade batch
        batch = downgrade_to_host(std::move(batch));
        batch = downgrade_to_disk(std::move(batch));
    }

    // Add to appropriate queue
    switch (target_tier) {
        case MemoryTier::GPU:
            gpu_queue.push_back(std::move(batch));
            break;
        case MemoryTier::HOST:
            host_queue.push_back(std::move(batch));
            break;
        case MemoryTier::DISK:
            disk_queue.push_back(std::move(batch));
            break;
    }

    total_batches_pushed++;

    // Notify waiting consumers
    cv_data_ready.notify_one();

    LOG_DEBUG("Repository: pushed batch {} to {} tier",
              total_batches_pushed, to_string(target_tier));
}

void shared_data_repository::mark_complete() {
    std::lock_guard<std::mutex> lock(mutex);
    completed = true;

    // Wake all waiting consumers
    cv_data_ready.notify_all();

    LOG_DEBUG("Repository: marked complete ({} batches total)",
              total_batches_pushed);
}
```

#### Pull (Consumer Side)

**Code** (`cucascade/src/data_repository.cpp:150-220`):

```cpp
std::optional<data_batch> shared_data_repository::pull_batch() {
    std::unique_lock<std::mutex> lock(mutex);

    // Wait for data or completion
    cv_data_ready.wait(lock, [this]() {
        return has_data() || completed;
    });

    // Check if truly empty
    if (is_empty()) {
        if (completed) {
            LOG_DEBUG("Repository: pull found no data (complete)");
            return std::nullopt; // End of stream
        }
        // Should not happen (spurious wakeup)
        return std::nullopt;
    }

    // Pull from highest priority tier
    data_batch batch;

    if (!gpu_queue.empty()) {
        // GPU tier (fastest)
        batch = std::move(gpu_queue.front());
        gpu_queue.pop_front();

        LOG_DEBUG("Repository: pulled batch from GPU tier");

    } else if (!host_queue.empty()) {
        // HOST tier → upgrade to GPU
        batch = std::move(host_queue.front());
        host_queue.pop_front();

        // Upgrade to GPU if space available
        if (mem_mgr.gpu_has_space(batch.size_bytes())) {
            batch = upgrade_to_gpu(std::move(batch));
            LOG_DEBUG("Repository: pulled batch from HOST, upgraded to GPU");
        } else {
            LOG_DEBUG("Repository: pulled batch from HOST (no GPU space)");
        }

    } else if (!disk_queue.empty()) {
        // DISK tier → upgrade to GPU (via HOST)
        batch = std::move(disk_queue.front());
        disk_queue.pop_front();

        // Upgrade: DISK → HOST → GPU
        batch = upgrade_from_disk(std::move(batch));
        if (mem_mgr.gpu_has_space(batch.size_bytes())) {
            batch = upgrade_to_gpu(std::move(batch));
            LOG_DEBUG("Repository: pulled batch from DISK, upgraded to GPU");
        } else {
            LOG_DEBUG("Repository: pulled batch from DISK to HOST (no GPU space)");
        }
    }

    total_batches_pulled++;

    // Notify producers if waiting
    cv_space_available.notify_one();

    return batch;
}

bool shared_data_repository::has_data() const {
    return !gpu_queue.empty() ||
           !host_queue.empty() ||
           !disk_queue.empty();
}

bool shared_data_repository::is_empty() const {
    return !has_data();
}
```

### Port System

**Pipeline Connections** (`src/include/pipeline/sirius_pipeline.hpp:40-70`):

```cpp
class sirius_pipeline {
public:
    // Input ports: read from these repositories
    std::vector<std::shared_ptr<shared_data_repository>> input_ports;

    // Output ports: write to these repositories
    std::vector<std::shared_ptr<shared_data_repository>> output_ports;

    // Add ports
    void add_input_port(std::shared_ptr<shared_data_repository> repo) {
        input_ports.push_back(repo);
    }

    void add_output_port(std::shared_ptr<shared_data_repository> repo) {
        output_ports.push_back(repo);
    }

    // Access ports
    std::shared_ptr<shared_data_repository> get_input_port(size_t idx) {
        return input_ports[idx];
    }

    std::shared_ptr<shared_data_repository> get_output_port(size_t idx) {
        return output_ports[idx];
    }
};
```

### Connection Setup

**Pipeline Builder** (`src/planner/sirius_physical_plan_generator.cpp:800-860`):

```cpp
void BuildPipelineConnections(
    std::vector<sirius_pipeline>& pipelines,
    const PhysicalPlan& plan) {

    // Create repository for each pipeline break
    std::map<size_t, std::shared_ptr<shared_data_repository>> repositories;

    for (size_t i = 0; i < plan.pipeline_breaks.size(); i++) {
        auto repo = std::make_shared<shared_data_repository>(
            memory_manager,
            "pipeline_break_" + std::to_string(i)
        );
        repositories[i] = repo;
    }

    // Connect pipelines via ports
    for (size_t i = 0; i < pipelines.size(); i++) {
        auto& pipeline = pipelines[i];

        // Producer: add output port
        if (i < repositories.size()) {
            pipeline.add_output_port(repositories[i]);
        }

        // Consumer: add input port
        if (i > 0) {
            pipeline.add_input_port(repositories[i - 1]);
        }
    }
}
```

**Example Connection**:

```
Query: SELECT * FROM t1 JOIN t2 ON ... ORDER BY ...

Pipelines:
  [0] Scan t1 → Hash Join (build)
  [1] Scan t2 → Hash Join (probe)
  [2] Order By → Result

Repositories:
  Repo A: Pipeline 0 output → Pipeline 1 input (hash table)
  Repo B: Pipeline 1 output → Pipeline 2 input (join results)

Connections:
  Pipeline 0:
    - input_ports: [] (none, it's a scan)
    - output_ports: [Repo A]

  Pipeline 1:
    - input_ports: [Repo A] (hash table from pipeline 0)
    - output_ports: [Repo B] (join results)

  Pipeline 2:
    - input_ports: [Repo B] (join results from pipeline 1)
    - output_ports: [] (none, final result)
```

### Producer-Consumer Pattern

#### Producer (Sink Operator)

**Example**: Hash Group By

**Code** (`src/op/sirius_physical_hash_group_by.cpp:200-240`):

```cpp
void sirius_physical_hash_group_by::sink(data_batch&& input_batch) {
    // Accumulate into hash table
    hash_table->aggregate(input_batch);

    // Check if should flush
    if (should_flush()) {
        auto result_batch = hash_table->finalize();

        // Push to output port (repository)
        if (!output_ports.empty()) {
            auto& repo = output_ports[0];
            repo->push_data_batch(std::move(result_batch));
        }

        // Reset for next batch
        hash_table->reset();
    }
}

void sirius_physical_hash_group_by::finalize() {
    // Flush remaining data
    auto final_batch = hash_table->finalize();

    if (!output_ports.empty() && final_batch.num_rows > 0) {
        auto& repo = output_ports[0];
        repo->push_data_batch(std::move(final_batch));

        // Signal completion
        repo->mark_complete();
    }
}
```

#### Consumer (Source Operator)

**Example**: Order By

**Code** (`src/op/sirius_physical_order_by.cpp:150-200`):

```cpp
TaskCreationHint sirius_physical_order_by::get_next_task_hint() {
    if (input_ports.empty()) {
        return TaskCreationHint::NO_MORE_TASKS;
    }

    auto& repo = input_ports[0];

    // Check if data available
    if (repo->has_data()) {
        return TaskCreationHint::READY;
    }

    // Check if producer complete
    if (repo->is_complete()) {
        if (repo->is_empty()) {
            return TaskCreationHint::NO_MORE_TASKS;
        }
        return TaskCreationHint::READY; // Pull remaining data
    }

    // Waiting for producer
    return TaskCreationHint::WAITING_FOR_INPUT_DATA;
}

data_batch sirius_physical_order_by::get_next_task_input_batch() {
    if (input_ports.empty()) {
        throw InternalException("No input ports");
    }

    auto& repo = input_ports[0];

    // Pull batch (may block)
    auto batch_opt = repo->pull_batch();

    if (!batch_opt.has_value()) {
        throw InternalException("No data available");
    }

    return std::move(batch_opt.value());
}

data_batch sirius_physical_order_by::execute(data_batch&& input_batch) {
    // Sort this batch locally
    auto sorted = cudf::sort(input_batch.table);

    return data_batch{
        .table = std::move(sorted),
        .tier = MemoryTier::GPU,
        .num_rows = input_batch.num_rows
    };
}
```

### Execution Flow

```
Time: 0ms - All Pipelines Start
────────────────────────────────

Task Creator Thread 1 (Pipeline 0):
  Check hint: READY
  Create task: SCAN → HASH_JOIN (build)
  Execute task:
    - Scan batch 0
    - Insert into hash table
    - Push to Repo A

Task Creator Thread 2 (Pipeline 1):
  Check hint: WAITING_FOR_INPUT_DATA (Repo A empty)
  Sleep 100μs

  ... wait for Repo A to have data ...


Time: 10ms - Repo A Has Data
─────────────────────────────

Task Creator Thread 2:
  Check hint: READY (Repo A has data)
  Create task: HASH_JOIN (probe)
  Execute task:
    - Pull batch from Repo A (hash table)
    - Scan probe data
    - Probe hash table
    - Push results to Repo B

Task Creator Thread 3 (Pipeline 2):
  Check hint: WAITING_FOR_INPUT_DATA (Repo B empty)
  Sleep 100μs


Time: 20ms - Repo B Has Data
─────────────────────────────

Task Creator Thread 3:
  Check hint: READY (Repo B has data)
  Create task: ORDER_BY
  Execute task:
    - Pull batch from Repo B
    - Sort batch
    - Continue...


Time: 100ms - Pipeline 0 Complete
──────────────────────────────────

Pipeline 0 finalize:
  - Repo A.mark_complete()
  - Notify Pipeline 1


Time: 150ms - Pipeline 1 Complete
──────────────────────────────────

Pipeline 1 finalize:
  - Repo B.mark_complete()
  - Notify Pipeline 2


Time: 200ms - Pipeline 2 Complete
──────────────────────────────────

Pipeline 2 finalize:
  - Collect results
  - Query complete
```

### Advantages Over Legacy Mode

#### 1. Concurrent Execution

**Legacy Mode** (Sequential):
```
Pipeline 0: ████████████░░░░░░░░░░
Pipeline 1:             ████████░░░░
Pipeline 2:                     ████

Total: 24 units sequential
```

**New Mode** (Overlapped):
```
Pipeline 0: ████████████
Pipeline 1:       ████████
Pipeline 2:             ████

Total: 16 units (33% faster)
```

#### 2. Memory Spilling

**Legacy Mode**:
```
GPU Full → OutOfMemoryException
```

**New Mode**:
```
GPU Full → Automatic Spill
  GPU Queue:  [batch0, batch1]
  HOST Queue: [batch2, batch3, batch4]  ← Spilled
  DISK Queue: [batch5, batch6]          ← Spilled
```

#### 3. Backpressure

**Legacy Mode**:
- Producer creates all results
- Stores in memory
- No flow control

**New Mode**:
- Repository has max capacity
- Producer blocks if full
- Natural flow control

**Code** (`cucascade/src/data_repository.cpp:70-85`):

```cpp
void shared_data_repository::push_data_batch(data_batch&& batch) {
    std::unique_lock<std::mutex> lock(mutex);

    // Wait if repository full
    cv_space_available.wait(lock, [this]() {
        return size() < max_capacity || completed;
    });

    // ... push batch ...
}
```

---

## Comparison: Join Query

### Legacy Mode Communication

**Query**:
```sql
SELECT * FROM t1 JOIN t2 ON t1.id = t2.id;
```

**Pipeline Structure**:
```
Pipeline 0: SCAN t1 → HASH_JOIN (build sink)
Pipeline 1: SCAN t2 → HASH_JOIN (probe sink)
Pipeline 2: HASH_JOIN (source) → RESULT_COLLECTOR
```

**Communication**:
```cpp
class GPUPhysicalHashJoin {
    // Shared state (direct communication)
    std::unique_ptr<cudf::hash_join> hash_table;  // In operator memory
    std::vector<GPUIntermediateRelation> results; // In operator memory

    // No repositories, no ports
    // Pipelines access operator state directly
};
```

**Execution Timeline**:
```
0-100ms:  Pipeline 0 (build)
          ├─ Scan t1 batches
          ├─ Build hash table
          └─ Set build_complete = true

100-200ms: Pipeline 1 (probe)
           ├─ Wait for build_complete
           ├─ Scan t2 batches
           ├─ Probe hash table
           └─ Store results in vector

200-210ms: Pipeline 2 (emit)
           ├─ Return results vector
           └─ Collect

Total: 210ms (sequential)
```

**Memory Usage**:
```
Peak: hash_table (50MB) + results vector (100MB) = 150MB
All in GPU memory (no spilling)
```

### New Mode Communication

**Query**:
```sql
SELECT * FROM gpu_execution('SELECT * FROM t1 JOIN t2 ON t1.id = t2.id');
```

**Pipeline Structure**:
```
Pipeline 0: SCAN t1 → HASH_JOIN (build)
            └─ Output: Repo A

Pipeline 1: SCAN t2 → HASH_JOIN (probe)
            ├─ Input: Repo A
            └─ Output: Repo B

Pipeline 2: ORDER_BY → RESULT_COLLECTOR
            └─ Input: Repo B
```

**Communication**:
```cpp
class sirius_physical_hash_join {
    // No shared state
    // Communication through ports
    std::vector<shared_ptr<shared_data_repository>> input_ports;
    std::vector<shared_ptr<shared_data_repository>> output_ports;
};

// Repositories (external)
Repo A: Build hash table (multi-tier storage)
Repo B: Join results (multi-tier storage)
```

**Execution Timeline**:
```
0-100ms:  Pipeline 0 (build)
          ├─ Scan t1 batch 0
          ├─ Build hash table partial
          ├─ Push to Repo A (batch 0)
          ├─ Pipeline 1 starts (concurrent!)
          │   ├─ Pull batch 0 from Repo A
          │   └─ Begin probe
          ├─ Scan t1 batch 1
          ├─ Build hash table partial
          └─ Push to Repo A (batch 1)

50-150ms: Pipeline 1 (probe) - Overlapped!
          ├─ Pull batches from Repo A
          ├─ Probe hash table
          ├─ Push results to Repo B
          └─ Pipeline 2 starts (concurrent!)

100-160ms: Pipeline 2 (order) - Overlapped!
           ├─ Pull batches from Repo B
           ├─ Sort batches
           └─ Collect results

Total: 160ms (24% faster due to overlap)
```

**Memory Usage**:
```
Peak GPU: 50MB (hash table + active batches)
Spillover:
  - Repo A HOST: 20MB
  - Repo B HOST: 30MB
  - Repo B DISK: 50MB

Total capacity: 150MB (same as legacy)
But distributed across tiers with automatic management
```

---

## Advanced Patterns

### Multiple Consumers

**Scenario**: One producer, multiple consumers

```
Pipeline 0: SCAN → FILTER
            └─ Output: Repo A

Pipeline 1: Aggregation
            └─ Input: Repo A

Pipeline 2: Top-K Selection
            └─ Input: Repo A
```

**Problem**: Repository consumed by first consumer

**Solution**: Clone batches or use multiple repositories

```cpp
void pipeline_0_sink(data_batch&& batch) {
    // Push to both repositories
    output_ports[0]->push_data_batch(batch.clone());
    output_ports[1]->push_data_batch(std::move(batch));
}
```

### Multiple Producers

**Scenario**: Multiple producers, one consumer

```
Pipeline 0: SCAN t1
            └─ Output: Repo A

Pipeline 1: SCAN t2
            └─ Output: Repo A

Pipeline 2: UNION
            └─ Input: Repo A
```

**Implementation**: Repository handles multiple producers naturally

```cpp
// Pipeline 0
repo_A->push_data_batch(batch_from_t1);
repo_A->mark_complete_producer(0);

// Pipeline 1
repo_A->push_data_batch(batch_from_t2);
repo_A->mark_complete_producer(1);

// Pipeline 2 consumer
// Pulls batches from both producers intermixed
while (auto batch = repo_A->pull_batch()) {
    process(batch);
}
```

**Extended Repository**:

```cpp
class shared_data_repository {
private:
    size_t num_producers;
    std::atomic<size_t> completed_producers{0};

public:
    void mark_complete_producer(size_t producer_id) {
        completed_producers++;
        if (completed_producers == num_producers) {
            mark_complete(); // All producers done
        }
    }
};
```

---

## Debugging Communication

### Problem: Deadlock

**Symptoms**:
- Query hangs indefinitely
- Threads waiting on condition variables

**Diagnosis**:

```bash
# Enable debug logging
export SIRIUS_LOG_LEVEL=DEBUG

# Check repository state
grep "repository.*waiting" /tmp/sirius.log
```

**Log Output**:
```
[DEBUG] Repository A: pushed batch 5
[DEBUG] Pipeline 1: waiting for data from Repository A
[DEBUG] Repository A: pulled batch 5
[DEBUG] Pipeline 1: waiting for data from Repository A
[DEBUG] Pipeline 1: waiting for data from Repository A
... [infinite loop] ...
```

**Root Cause**: Producer never called `mark_complete()`

**Fix**:
```cpp
void producer_operator::finalize() {
    // ... flush data ...

    // CRITICAL: Mark complete
    if (!output_ports.empty()) {
        output_ports[0]->mark_complete();
    }
}
```

### Problem: Memory Leak

**Symptoms**:
- GPU memory usage grows over time
- Eventually OOM

**Diagnosis**:

```cpp
// Add monitoring to repository
size_t shared_data_repository::memory_usage() {
    size_t total = 0;
    for (const auto& batch : gpu_queue) {
        total += batch.size_bytes();
    }
    return total;
}
```

**Log Output**:
```
[DEBUG] Repo A memory: 50MB (5 batches)
[DEBUG] Repo A memory: 100MB (10 batches)
[DEBUG] Repo A memory: 200MB (20 batches)  ← Growing!
[DEBUG] Repo A memory: 400MB (40 batches)  ← Leak!
```

**Root Cause**: Consumer not pulling batches

**Fix**: Ensure consumer running and pulling

### Problem: Slow Queries

**Symptoms**:
- Query slower than expected
- Frequent spilling

**Diagnosis**:

```sql
SELECT * FROM sirius_repository_stats()
WHERE query_id = last_query_id();
```

**Output**:
```
repo_name | pushed | pulled | gpu_hits | host_hits | disk_hits
──────────┼────────┼────────┼──────────┼───────────┼──────────
Repo A    | 100    | 100    | 20       | 60        | 20
Repo B    | 100    | 100    | 10       | 40        | 50
```

**Analysis**:
- Repo B: 50% disk hits → frequent spilling
- Slow query due to disk I/O

**Solutions**:
1. Increase GPU memory limit
2. Reduce batch size
3. Add more selective filters early
4. Increase task parallelism to process faster

---

## Performance Tuning

### Repository Capacity

**Configuration**:

```cpp
auto repo = std::make_shared<shared_data_repository>(
    memory_manager,
    "pipeline_break_0",
    max_batches = 100  // Max batches in repository
);
```

**Trade-offs**:
- **Small capacity (10-20 batches)**: Low memory, more backpressure
- **Large capacity (100+ batches)**: More memory, less blocking

**Recommendation**: 50 batches (5GB at 100MB/batch)

### Tier Thresholds

**Configuration**:

```ini
[memory]
gpu_memory_limit = 12288     # 12GB
host_memory_limit = 49152    # 48GB
disk_memory_limit = -1       # Unlimited

# Spill thresholds
gpu_spill_threshold = 0.9    # Spill at 90% GPU
host_spill_threshold = 0.9   # Spill at 90% HOST
```

**Impact on Repository**:
- Low thresholds → aggressive spilling → more HOST/DISK batches
- High thresholds → less spilling → more GPU batches

### Upgrade Strategy

**Options**:

1. **Eager Upgrade**: Upgrade immediately on pull
2. **Lazy Upgrade**: Upgrade only when needed
3. **Batch Upgrade**: Upgrade multiple batches at once

**Current**: Eager upgrade (implemented above)

**Future Optimization**: Lazy upgrade for better memory utilization

---

## See Also

- [New Data Flow](new-data-flow.md) - New Mode data flow details
- [Legacy Data Flow](legacy-data-flow.md) - Legacy Mode data flow
- [Query Lifecycle](query-lifecycle.md) - Complete execution trace
- [Memory Management](../05-core-components/memory-management.md) - Multi-tier memory
- [Threading Model](../05-core-components/threading-model.md) - Task executors
- [New Mode Overview](../04-new-mode/overview.md) - New Mode architecture
- [Performance Tips](../appendices/performance-tips.md) - Optimization guide
