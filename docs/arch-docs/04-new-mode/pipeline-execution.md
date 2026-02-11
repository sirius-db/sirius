# Pipeline Execution (New Mode)

Comprehensive guide to pipeline structure and execution in Sirius New Mode, covering pipeline organization, task-based execution, and the coordination between pipelines.

---

## Overview

New Mode uses a **dynamic, task-based execution model** where pipelines create tasks on-demand based on runtime conditions.

**Key Differences from Legacy Mode**:

| Aspect | Legacy Mode | New Mode |
|--------|-------------|----------|
| **Task Model** | Static | Dynamic (hint-based) |
| **Pipeline Structure** | Sequential execution | Parallel execution |
| **Data Flow** | Pull-based (GetData) | Push-based (repositories) |
| **Task Creation** | Upfront (all tasks) | On-demand (as data available) |
| **Scheduling** | Pipeline-level | Task-level |

---

## Pipeline Structure

### sirius_pipeline

**Definition**: `src/include/pipeline/sirius_pipeline.hpp`

```cpp
class sirius_pipeline {
public:
    // Pipeline metadata
    size_t pipeline_id;
    std::string name;

    // Operator structure
    sirius_physical_operator* source;  // Entry point
    std::vector<sirius_physical_operator*> intermediate_operators;
    sirius_physical_operator* sink;  // Exit point (optional)

    // Port connections
    std::vector<std::shared_ptr<shared_data_repository>> input_ports;
    std::vector<std::shared_ptr<shared_data_repository>> output_ports;

    // Execution state
    std::atomic<size_t> tasks_created{0};
    std::atomic<size_t> tasks_completed{0};
    std::atomic<bool> finalized{false};

    // CUDA streams
    std::vector<cudaStream_t> streams;

public:
    // Constructor
    sirius_pipeline(size_t id, std::string name);

    // Operator management
    void set_source(sirius_physical_operator* op);
    void add_intermediate(sirius_physical_operator* op);
    void set_sink(sirius_physical_operator* op);

    // Port management
    void add_input_port(std::shared_ptr<shared_data_repository> repo);
    void add_output_port(std::shared_ptr<shared_data_repository> repo);

    // Task creation
    TaskCreationHint get_next_task_hint();
    std::unique_ptr<sirius_pipeline_itask> create_next_task();

    // Finalization
    void finalize();

    // Utilities
    bool is_complete() const;
    std::string to_string() const;
    void print() const;
};
```

### Pipeline Types

#### Source Pipeline

**Characteristics**:
- Starts with source operator (SCAN)
- No input ports
- Has output ports (if pipeline break)

**Example**:
```
SCAN → FILTER → HASH_GROUP_BY (sink)
                └─ Output: Repository A
```

**Structure**:
```cpp
pipeline.source = table_scan;
pipeline.intermediate_operators = {filter};
pipeline.sink = hash_group_by;
pipeline.input_ports = {};  // Empty
pipeline.output_ports = {repo_A};
```

#### Intermediate Pipeline

**Characteristics**:
- Reads from input repository
- Produces to output repository
- Transform or reduce data

**Example**:
```
Input: Repository A
    ↓
HASH_JOIN (probe) → PROJECTION
                    └─ Output: Repository B
```

**Structure**:
```cpp
pipeline.source = hash_join_probe;
pipeline.intermediate_operators = {projection};
pipeline.sink = nullptr;
pipeline.input_ports = {repo_A};
pipeline.output_ports = {repo_B};
```

#### Sink Pipeline

**Characteristics**:
- Reads from input repository
- No output ports
- Final result collection

**Example**:
```
Input: Repository B
    ↓
ORDER_BY → RESULT_COLLECTOR
```

**Structure**:
```cpp
pipeline.source = order_by;
pipeline.intermediate_operators = {};
pipeline.sink = result_collector;
pipeline.input_ports = {repo_B};
pipeline.output_ports = {};  // Empty
```

---

## Task Structure

### sirius_pipeline_itask

**Definition**: `src/include/pipeline/sirius_pipeline_itask.hpp`

```cpp
class sirius_pipeline_itask {
private:
    // Associated pipeline
    sirius_pipeline& pipeline;

    // Input batch for this task
    data_batch input_batch;

    // CUDA stream for execution
    cudaStream_t stream;

    // Execution state
    TaskState state;

    // Output batch (if any)
    data_batch output_batch;

public:
    // Constructor
    sirius_pipeline_itask(
        sirius_pipeline& pipe,
        data_batch&& input,
        cudaStream_t stream
    );

    // Main execution
    void compute_task();

    // Output publishing
    void publish_output();

    // Utilities
    TaskState get_state() const { return state; }
    size_t get_pipeline_id() const { return pipeline.pipeline_id; }
};

enum class TaskState {
    CREATED,      // Task created, not started
    RUNNING,      // Currently executing
    COMPLETED,    // Execution complete
    FAILED        // Execution failed
};
```

### Task Execution Flow

**Code** (`src/pipeline/sirius_pipeline_itask.cpp:60-150`):

```cpp
void sirius_pipeline_itask::compute_task() {
    state = TaskState::RUNNING;

    try {
        // 1. Execute source operator
        if (pipeline.source) {
            if (pipeline.source->get_type() == SiriusPhysicalOperatorType::DUMMY_SCAN) {
                // Dummy scan: input IS the data
                current_batch = std::move(input_batch);
            } else {
                // Real source: execute to get data
                current_batch = pipeline.source->execute(std::move(input_batch));
            }

            if (current_batch.num_rows == 0) {
                // Empty result, early exit
                state = TaskState::COMPLETED;
                return;
            }
        }

        // 2. Execute intermediate operators (in sequence)
        for (auto& op : pipeline.intermediate_operators) {
            current_batch = op->execute(std::move(current_batch));

            // Check if filtered out
            if (current_batch.num_rows == 0) {
                LOG_TRACE("Pipeline {}: batch filtered out by {}",
                          pipeline.pipeline_id, op->get_name());
                state = TaskState::COMPLETED;
                return;
            }
        }

        // 3. Execute sink operator (if any)
        if (pipeline.sink) {
            pipeline.sink->sink(std::move(current_batch));
            // Sink consumes data, no output batch
        } else {
            // No sink, save output for publishing
            output_batch = std::move(current_batch);
        }

        // 4. Publish output
        publish_output();

        state = TaskState::COMPLETED;
        pipeline.tasks_completed++;

        LOG_TRACE("Pipeline {}: task completed (total={})",
                  pipeline.pipeline_id, pipeline.tasks_completed.load());

    } catch (const std::exception& e) {
        LOG_ERROR("Pipeline {}: task failed: {}",
                  pipeline.pipeline_id, e.what());
        state = TaskState::FAILED;
        throw;
    }
}

void sirius_pipeline_itask::publish_output() {
    if (output_batch.num_rows == 0) {
        return; // Nothing to publish
    }

    if (pipeline.output_ports.empty()) {
        return; // No output ports
    }

    // Push to all output ports
    for (size_t i = 0; i < pipeline.output_ports.size(); i++) {
        auto& port = pipeline.output_ports[i];

        if (i < pipeline.output_ports.size() - 1) {
            // Clone for all except last
            port->push_data_batch(output_batch.clone());
        } else {
            // Move to last
            port->push_data_batch(std::move(output_batch));
        }
    }

    LOG_TRACE("Pipeline {}: published output to {} ports",
              pipeline.pipeline_id, pipeline.output_ports.size());
}
```

---

## Task Creation

### Dynamic Task Creation

**Task Creator Thread** (`src/parallel/task_creator.cpp:100-180`):

```cpp
void task_creator::create_tasks_for_pipeline(sirius_pipeline& pipeline) {
    LOG_INFO("Task creator: starting for pipeline {}", pipeline.pipeline_id);

    while (!should_stop) {
        // 1. Check if pipeline can create task
        TaskCreationHint hint = pipeline.get_next_task_hint();

        switch (hint) {
            case TaskCreationHint::READY: {
                // Create and enqueue task
                auto task = pipeline.create_next_task();

                if (task) {
                    task_queue.enqueue(std::move(task));
                    pipeline.tasks_created++;

                    LOG_TRACE("Pipeline {}: created task {}",
                              pipeline.pipeline_id,
                              pipeline.tasks_created.load());
                }
                break;
            }

            case TaskCreationHint::WAITING_FOR_INPUT_DATA: {
                // No data yet, wait briefly
                LOG_TRACE("Pipeline {}: waiting for input data",
                          pipeline.pipeline_id);

                std::this_thread::sleep_for(std::chrono::microseconds(100));
                break;
            }

            case TaskCreationHint::NO_MORE_TASKS: {
                // Pipeline complete
                LOG_INFO("Pipeline {}: no more tasks, finalizing",
                         pipeline.pipeline_id);

                pipeline.finalize();
                return;  // Exit task creator for this pipeline
            }
        }

        // Brief yield to prevent tight spinning
        std::this_thread::yield();
    }

    LOG_INFO("Task creator: stopped for pipeline {}", pipeline.pipeline_id);
}
```

### get_next_task_hint() Implementation

**Pipeline Method** (`src/pipeline/sirius_pipeline.cpp:80-120`):

```cpp
TaskCreationHint sirius_pipeline::get_next_task_hint() {
    if (finalized) {
        return TaskCreationHint::NO_MORE_TASKS;
    }

    // Delegate to source operator
    if (source) {
        return source->get_next_task_hint();
    }

    // No source, pipeline is done
    return TaskCreationHint::NO_MORE_TASKS;
}
```

**Operator Examples**:

**1. TABLE_SCAN** (`src/op/sirius_physical_table_scan.cpp:100-120`):

```cpp
TaskCreationHint sirius_physical_table_scan::get_next_task_hint() {
    // Check if more batches to scan
    if (current_batch_idx >= total_batches) {
        return TaskCreationHint::NO_MORE_TASKS;
    }

    // Always ready (no dependencies)
    return TaskCreationHint::READY;
}
```

**2. HASH_JOIN (probe)** (`src/op/sirius_physical_hash_join.cpp:200-230`):

```cpp
TaskCreationHint sirius_physical_hash_join::get_next_task_hint() {
    // Wait for build phase to complete
    if (!build_complete) {
        LOG_TRACE("Hash join: waiting for build to complete");
        return TaskCreationHint::WAITING_FOR_INPUT_DATA;
    }

    // Check input repository
    if (input_ports.empty()) {
        return TaskCreationHint::NO_MORE_TASKS;
    }

    auto& input_repo = input_ports[0];

    if (input_repo->has_data()) {
        return TaskCreationHint::READY;
    }

    if (input_repo->is_complete()) {
        if (input_repo->is_empty()) {
            return TaskCreationHint::NO_MORE_TASKS;
        }
        return TaskCreationHint::READY;  // Pull remaining
    }

    return TaskCreationHint::WAITING_FOR_INPUT_DATA;
}
```

### create_next_task() Implementation

**Pipeline Method** (`src/pipeline/sirius_pipeline.cpp:150-180`):

```cpp
std::unique_ptr<sirius_pipeline_itask> sirius_pipeline::create_next_task() {
    // Get input batch (if needed)
    data_batch input;

    if (source && source->get_type() != SiriusPhysicalOperatorType::DUMMY_SCAN) {
        // Source generates input internally
        input = data_batch{};  // Empty placeholder
    } else if (!input_ports.empty()) {
        // Pull from input repository
        auto& input_repo = input_ports[0];
        auto input_opt = input_repo->pull_batch();

        if (!input_opt.has_value()) {
            return nullptr;  // No input available
        }

        input = std::move(input_opt.value());
    }

    // Select CUDA stream (round-robin)
    cudaStream_t stream = streams[tasks_created % streams.size()];

    // Create task
    return std::make_unique<sirius_pipeline_itask>(
        *this,
        std::move(input),
        stream
    );
}
```

---

## Pipeline Coordination

### sirius_meta_pipeline

**Definition**: `src/include/pipeline/sirius_meta_pipeline.hpp`

```cpp
class sirius_meta_pipeline {
private:
    // All pipelines
    std::vector<std::unique_ptr<sirius_pipeline>> pipelines;

    // Pipeline dependencies (DAG)
    std::vector<std::vector<size_t>> dependencies;

    // Shared repositories
    std::vector<std::shared_ptr<shared_data_repository>> repositories;

    // Execution context
    SiriusContext& context;

public:
    // Constructor
    sirius_meta_pipeline(SiriusContext& ctx);

    // Pipeline management
    void add_pipeline(std::unique_ptr<sirius_pipeline> pipeline);
    void add_dependency(size_t from, size_t to);

    // Repository management
    std::shared_ptr<shared_data_repository> create_repository(std::string name);
    void connect_pipelines(size_t producer_id, size_t consumer_id,
                            std::shared_ptr<shared_data_repository> repo);

    // Execution
    void initialize();
    void execute();
    void finalize();

    // Utilities
    std::vector<size_t> topological_sort();
    void print() const;
};
```

### Pipeline Dependency Graph

**Example Query**:
```sql
SELECT * FROM gpu_execution('
    SELECT o.order_id, c.name, o.amount
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
    WHERE o.amount > 100
    ORDER BY o.amount DESC
    LIMIT 10
');
```

**Pipeline Structure**:

```
Pipeline 0: SCAN customers → HASH_JOIN (build sink)
            └─ Output: Repo A (hash table)

Pipeline 1: SCAN orders → FILTER → HASH_JOIN (probe)
            ├─ Input: Repo A (hash table)
            └─ Output: Repo B (joined results)

Pipeline 2: ORDER_BY → LIMIT → RESULT_COLLECTOR
            └─ Input: Repo B (joined results)
```

**Dependency Graph**:

```
Pipeline 0 ──→ Pipeline 1 ──→ Pipeline 2
   (build)      (probe)         (sort+limit)
```

**Code** (`src/planner/sirius_physical_plan_generator.cpp:600-650`):

```cpp
void build_meta_pipeline(PhysicalPlan& plan, SiriusContext& ctx) {
    auto meta = std::make_unique<sirius_meta_pipeline>(ctx);

    // Create pipelines
    for (size_t i = 0; i < plan.pipelines.size(); i++) {
        auto pipeline = create_pipeline(plan.pipelines[i], i, ctx);
        meta->add_pipeline(std::move(pipeline));
    }

    // Create repositories for pipeline breaks
    for (size_t i = 0; i < plan.pipeline_breaks.size(); i++) {
        auto repo = meta->create_repository("pipeline_break_" + std::to_string(i));

        // Connect producer and consumer
        size_t producer_id = plan.pipeline_breaks[i].producer_pipeline;
        size_t consumer_id = plan.pipeline_breaks[i].consumer_pipeline;

        meta->connect_pipelines(producer_id, consumer_id, repo);

        // Add dependency
        meta->add_dependency(producer_id, consumer_id);
    }

    return meta;
}
```

### Topological Sort

**Implementation** (`src/pipeline/sirius_meta_pipeline.cpp:100-160`):

```cpp
std::vector<size_t> sirius_meta_pipeline::topological_sort() {
    std::vector<size_t> result;
    std::vector<size_t> in_degree(pipelines.size(), 0);

    // Calculate in-degrees
    for (const auto& deps : dependencies) {
        for (size_t dep : deps) {
            in_degree[dep]++;
        }
    }

    // Find pipelines with no dependencies (in-degree 0)
    std::queue<size_t> ready;
    for (size_t i = 0; i < pipelines.size(); i++) {
        if (in_degree[i] == 0) {
            ready.push(i);
        }
    }

    // Process in topological order
    while (!ready.empty()) {
        size_t current = ready.front();
        ready.pop();
        result.push_back(current);

        // Decrement in-degrees of dependent pipelines
        if (current < dependencies.size()) {
            for (size_t dep : dependencies[current]) {
                in_degree[dep]--;
                if (in_degree[dep] == 0) {
                    ready.push(dep);
                }
            }
        }
    }

    // Check for cycles
    if (result.size() != pipelines.size()) {
        throw InternalException("Cycle detected in pipeline dependency graph");
    }

    return result;
}
```

---

## Execution Model

### Initialization

**Meta Pipeline** (`src/pipeline/sirius_meta_pipeline.cpp:200-240`):

```cpp
void sirius_meta_pipeline::initialize() {
    LOG_INFO("Meta pipeline: initializing {} pipelines", pipelines.size());

    // Initialize CUDA streams for each pipeline
    for (auto& pipeline : pipelines) {
        size_t num_streams = context.config.cuda_streams_per_pipeline;

        for (size_t i = 0; i < num_streams; i++) {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            pipeline->streams.push_back(stream);
        }

        LOG_DEBUG("Pipeline {}: created {} CUDA streams",
                  pipeline->pipeline_id, num_streams);
    }

    // Topological sort for initialization order
    auto order = topological_sort();

    // Initialize pipelines in dependency order
    for (size_t pipeline_id : order) {
        auto& pipeline = pipelines[pipeline_id];

        // Initialize operators
        if (pipeline->source) {
            pipeline->source->initialize();
        }
        for (auto& op : pipeline->intermediate_operators) {
            op->initialize();
        }
        if (pipeline->sink) {
            pipeline->sink->initialize();
        }

        LOG_INFO("Pipeline {}: initialized", pipeline_id);
    }
}
```

### Execution

**Meta Pipeline** (`src/pipeline/sirius_meta_pipeline.cpp:280-350`):

```cpp
void sirius_meta_pipeline::execute() {
    LOG_INFO("Meta pipeline: executing {} pipelines", pipelines.size());

    // Get execution context
    auto& task_creator_pool = context.get_task_creator_pool();
    auto& pipeline_executor_pool = context.get_pipeline_executor_pool();

    // Launch task creators for ALL pipelines (in parallel)
    std::vector<std::future<void>> creator_futures;

    for (auto& pipeline : pipelines) {
        auto future = task_creator_pool.submit([&pipeline]() {
            task_creator creator;
            creator.create_tasks_for_pipeline(*pipeline);
        });
        creator_futures.push_back(std::move(future));
    }

    LOG_INFO("Meta pipeline: launched {} task creators", creator_futures.size());

    // Task creators run concurrently, creating tasks as data becomes available
    // Pipeline executors consume tasks from shared queue

    // Wait for all task creators to complete
    for (auto& future : creator_futures) {
        future.get();
    }

    LOG_INFO("Meta pipeline: all task creators completed");
}
```

**Key Insight**: All task creators start immediately, but they self-regulate based on data availability:
- Pipeline 0 (no dependencies): Creates tasks immediately
- Pipeline 1 (depends on Pipeline 0): Waits for data in repository
- Pipeline 2 (depends on Pipeline 1): Waits for data in repository

This enables **concurrent pipeline execution** where later pipelines start as soon as data becomes available, without waiting for earlier pipelines to fully complete.

### Finalization

**Meta Pipeline** (`src/pipeline/sirius_meta_pipeline.cpp:390-430`):

```cpp
void sirius_meta_pipeline::finalize() {
    LOG_INFO("Meta pipeline: finalizing {} pipelines", pipelines.size());

    // Finalize pipelines in dependency order
    auto order = topological_sort();

    for (size_t pipeline_id : order) {
        auto& pipeline = pipelines[pipeline_id];

        // Finalize operators
        if (pipeline->sink) {
            pipeline->sink->finalize();
        }
        for (auto& op : pipeline->intermediate_operators) {
            op->finalize();
        }
        if (pipeline->source) {
            pipeline->source->finalize();
        }

        LOG_INFO("Pipeline {}: finalized", pipeline_id);
    }

    // Mark all repositories complete
    for (auto& repo : repositories) {
        if (!repo->is_complete()) {
            LOG_WARN("Repository '{}' not marked complete, marking now",
                     repo->get_name());
            repo->mark_complete();
        }
    }

    // Destroy CUDA streams
    for (auto& pipeline : pipelines) {
        for (auto stream : pipeline->streams) {
            cudaStreamDestroy(stream);
        }
        pipeline->streams.clear();
    }

    LOG_INFO("Meta pipeline: finalization complete");
}
```

---

## Execution Timeline Example

### Query

```sql
SELECT * FROM gpu_execution('
    SELECT category, AVG(price)
    FROM products
    WHERE price > 50
    GROUP BY category
    ORDER BY avg_price DESC
');
```

### Pipeline Structure

```
Pipeline 0: SCAN → FILTER → HASH_GROUP_BY (sink)
            └─ Output: Repo A (1K groups)

Pipeline 1: ORDER_BY (source) → RESULT_COLLECTOR
            └─ Input: Repo A
```

### Timeline

```
Time: 0ms - Initialization
──────────────────────────

- Create 2 pipelines
- Create Repository A
- Create CUDA streams (4 per pipeline)
- Initialize operators


Time: 1ms - Execution Start
───────────────────────────

[Task Creator Thread 1 - Pipeline 0]
  hint = READY (SCAN has data)
  Create task 0: SCAN batch 0
  Enqueue task 0

[Task Creator Thread 2 - Pipeline 1]
  hint = WAITING_FOR_INPUT_DATA (Repo A empty)
  Sleep 100μs


Time: 5ms - Task 0 Executes
────────────────────────────

[Pipeline Executor Thread A]
  Dequeue task 0
  compute_task():
    - SCAN: read batch 0 (100K rows)
    - FILTER: apply price > 50 (80K rows)
    - HASH_GROUP_BY (sink): accumulate
  Task 0 complete


Time: 10ms - More Tasks Created
────────────────────────────────

[Task Creator Thread 1 - Pipeline 0]
  hint = READY
  Create task 1: SCAN batch 1
  Enqueue task 1

  hint = READY
  Create task 2: SCAN batch 2
  Enqueue task 2

  ... (tasks 3-9 created)


Time: 15ms - Parallel Execution
────────────────────────────────

[Pipeline Executor Thread A]
  Execute task 1

[Pipeline Executor Thread B]
  Execute task 2

[Pipeline Executor Thread C]
  Execute task 3

[Pipeline Executor Thread D]
  Execute task 4

Note: 4 executor threads, so 4 tasks run concurrently


Time: 50ms - Pipeline 0 Complete
─────────────────────────────────

[Task Creator Thread 1 - Pipeline 0]
  hint = NO_MORE_TASKS (all batches scanned)
  Finalize pipeline:
    - HASH_GROUP_BY::finalize()
    - Flush remaining data to Repo A
    - Repo A->mark_complete()
  Task creator exits


Time: 51ms - Pipeline 1 Starts
───────────────────────────────

[Task Creator Thread 2 - Pipeline 1]
  hint = READY (Repo A has data)
  Create task 10: ORDER_BY batch 0
  Enqueue task 10


Time: 55ms - ORDER_BY Buffering
────────────────────────────────

[Pipeline Executor Thread A]
  Execute task 10:
    - Pull batch 0 from Repo A
    - Buffer locally (ORDER_BY waits for all input)
  Task 10 complete

[Task Creator Thread 2]
  hint = READY (Repo A has more data)
  Create task 11: ORDER_BY batch 1


Time: 60ms - All Input Buffered
────────────────────────────────

[Task Creator Thread 2 - Pipeline 1]
  hint = READY (Repo A complete, all data buffered)
  Create task 12: ORDER_BY sort task


Time: 65ms - Sorting
────────────────────

[Pipeline Executor Thread A]
  Execute task 12:
    - ORDER_BY: concatenate all buffered batches
    - Sort combined table (1K rows)
    - Output sorted batch
  Task 12 complete


Time: 70ms - Result Collection
───────────────────────────────

[Task Creator Thread 2]
  hint = NO_MORE_TASKS
  Finalize pipeline:
    - RESULT_COLLECTOR::finalize()
  Task creator exits


Time: 75ms - Finalization
─────────────────────────

- Wait for all task creators
- Wait for all pipeline executors
- Finalize meta pipeline
- Destroy CUDA streams
- Return result


Total Time: 75ms
  Pipeline 0: 0-50ms (66%)
  Pipeline 1: 50-70ms (27%)
  Overhead: 5ms (7%)
```

---

## Performance Characteristics

### Concurrency

**Task-level Parallelism**:
```
4 executor threads × 4 CUDA streams = 16 concurrent tasks
```

**Pipeline-level Parallelism**:
```
Pipeline 0: ████████████
Pipeline 1:       ████████ (starts before Pipeline 0 completes)
```

**Speedup**:
- Sequential (Legacy Mode): 50ms + 20ms = 70ms
- Overlapped (New Mode): max(50ms, 20ms with offset) = 55ms
- **Speedup: 1.27x**

### Memory Efficiency

**Repository Buffering**:
```
Max buffered data = max_batches × batch_size
                  = 50 × 100MB = 5GB
```

**vs. Legacy Mode**:
- Legacy: All intermediate results in operator state
- New: Spill to HOST/DISK if needed

### Scalability

**More Executor Threads**:
```
2 threads: 2 concurrent tasks
4 threads: 4 concurrent tasks (default)
8 threads: 8 concurrent tasks (+2x throughput)
16 threads: Limited by GPU parallelism
```

**Recommendation**: 4-8 threads for most workloads

---

## Configuration

### Thread Pools

**INI Format** (`sirius.cfg`):

```ini
[threading]
pipeline_executor_threads = 4     # Task execution
task_creator_threads = 2          # Task creation
```

**Impact**:
- More executors → higher throughput
- More creators → faster task creation (diminishing returns)

### CUDA Streams

```ini
[cuda]
cuda_streams_per_pipeline = 1     # CUDA concurrency
```

**Trade-offs**:
- 1 stream: Simplest, less overhead
- 2-4 streams: More GPU concurrency
- >4 streams: Diminishing returns

---

## Debugging

### Enable Logging

```bash
export SIRIUS_LOG_LEVEL=DEBUG
export SIRIUS_LOG_FILE=/tmp/sirius_pipeline.log
```

### Trace Pipeline Execution

**Log Output**:
```
[INFO] Meta pipeline: initializing 2 pipelines
[INFO] Pipeline 0: initialized
[INFO] Pipeline 1: initialized
[INFO] Meta pipeline: executing 2 pipelines
[DEBUG] Pipeline 0: created 4 CUDA streams
[TRACE] Pipeline 0: created task 1
[TRACE] Pipeline 0: task completed (total=1)
[TRACE] Pipeline 0: waiting for input data
[TRACE] Repository A: pushed batch to GPU queue
[TRACE] Pipeline 1: created task 1
[INFO] Pipeline 0: no more tasks, finalizing
[INFO] Pipeline 0: finalized
[INFO] Pipeline 1: finalized
[INFO] Meta pipeline: all task creators completed
[INFO] Meta pipeline: finalization complete
```

### Monitor Task Queue

```cpp
// Add to task queue
void print_queue_stats() {
    printf("Task Queue Stats:\n");
    printf("  Size: %zu\n", task_queue.size());
    printf("  Enqueued: %zu\n", total_enqueued);
    printf("  Dequeued: %zu\n", total_dequeued);
}
```

---

## See Also

- [New Mode Overview](overview.md) - Introduction to New Mode
- [Operators](operators.md) - Operator implementations
- [Task Creation](task-creation.md) - Task creation details
- [Cucascade Integration](cucascade-integration.md) - Data repository
- [New Data Flow](../06-data-flow/new-data-flow.md) - Complete data flow
- [Threading Model](../05-core-components/threading-model.md) - Thread pools
- [Performance Tips](../appendices/performance-tips.md) - Optimization guide
