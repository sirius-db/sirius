# Task Creation (New Mode)

Deep dive into Sirius New Mode's dynamic task creation system, focusing on task creation hints, operator readiness, and the coordination between task creators and executors.

---

## Overview

**Dynamic Task Creation** is a key innovation in New Mode, allowing tasks to be created on-demand based on runtime conditions rather than upfront.

**Benefits**:
- **Adaptive**: Creates tasks only when data is available
- **Memory-efficient**: Doesn't pre-allocate all tasks
- **Responsive**: Pipelines start as soon as dependencies are satisfied
- **Scalable**: Handles variable input sizes gracefully

| Aspect | Static (Legacy) | Dynamic (New Mode) |
|--------|-----------------|---------------------|
| **When Created** | Upfront (all tasks) | On-demand (as needed) |
| **Memory** | O(total_batches) | O(active_batches) |
| **Startup Latency** | High (create all) | Low (create first) |
| **Adaptability** | Fixed | Runtime-adaptive |

---

## Task Creation Hints

### TaskCreationHint Enum

**Definition**: `src/include/pipeline/task_creation_hint.hpp`

```cpp
enum class TaskCreationHint {
    // Task can be created immediately
    READY,

    // Waiting for input data from repository
    WAITING_FOR_INPUT_DATA,

    // No more tasks to create (operator complete)
    NO_MORE_TASKS
};
```

### Hint Semantics

#### READY

**Meaning**: Task can be created and executed immediately.

**When to Return**:
- Source operator: Has data to scan
- Processing operator: Input repository has data
- Sink operator: Ready to emit final result

**Task Creator Action**:
```cpp
case TaskCreationHint::READY:
    auto task = pipeline.create_next_task();
    task_queue.enqueue(std::move(task));
    break;
```

**Example**: TABLE_SCAN with more batches to read

```cpp
TaskCreationHint sirius_physical_table_scan::get_next_task_hint() {
    if (current_batch_idx < total_batches) {
        return TaskCreationHint::READY;  // More data to scan
    }
    return TaskCreationHint::NO_MORE_TASKS;
}
```

#### WAITING_FOR_INPUT_DATA

**Meaning**: Operator is blocked waiting for input from repository.

**When to Return**:
- Input repository is empty
- Input repository not marked complete
- Build phase not finished (for joins)
- Dependent operator still running

**Task Creator Action**:
```cpp
case TaskCreationHint::WAITING_FOR_INPUT_DATA:
    // Brief sleep to avoid tight spinning
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    break;
```

**Example**: ORDER_BY waiting for all input

```cpp
TaskCreationHint sirius_physical_order_by::get_next_task_hint() {
    if (!all_input_received) {
        auto& input_repo = input_ports[0];

        if (!input_repo->has_data() && !input_repo->is_complete()) {
            return TaskCreationHint::WAITING_FOR_INPUT_DATA;  // Still waiting
        }
    }
    return TaskCreationHint::READY;
}
```

#### NO_MORE_TASKS

**Meaning**: Operator has completed all work.

**When to Return**:
- All input consumed
- All output produced
- Operator finalized

**Task Creator Action**:
```cpp
case TaskCreationHint::NO_MORE_TASKS:
    pipeline.finalize();  // Finalize operators
    return;  // Exit task creator thread
```

**Example**: SCAN with all batches read

```cpp
TaskCreationHint sirius_physical_table_scan::get_next_task_hint() {
    if (current_batch_idx >= total_batches) {
        return TaskCreationHint::NO_MORE_TASKS;  // All batches scanned
    }
    return TaskCreationHint::READY;
}
```

---

## Operator Task Creation Patterns

### Pattern 1: Simple Source (No Dependencies)

**Example**: TABLE_SCAN

**Implementation** (`src/op/sirius_physical_table_scan.cpp:100-140`):

```cpp
class sirius_physical_table_scan : public sirius_physical_operator {
private:
    std::string file_path;
    size_t current_batch_idx = 0;
    size_t total_batches;

public:
    TaskCreationHint get_next_task_hint() override {
        // Check if more batches to scan
        if (current_batch_idx >= total_batches) {
            return TaskCreationHint::NO_MORE_TASKS;
        }

        // Always ready (no external dependencies)
        return TaskCreationHint::READY;
    }

    data_batch get_next_task_input_batch() override {
        // No input needed (source operator)
        return data_batch{};
    }

    data_batch execute(data_batch&&) override {
        // Read next batch from file
        auto batch = read_parquet_batch(
            file_path,
            current_batch_idx,
            context.config.scan_batch_size
        );

        current_batch_idx++;

        return batch;
    }
};
```

**Task Creation Flow**:

```
Iteration 1:
  get_next_task_hint() → READY (batch 0 available)
  create_next_task() → Task 0
  enqueue(Task 0)

Iteration 2:
  get_next_task_hint() → READY (batch 1 available)
  create_next_task() → Task 1
  enqueue(Task 1)

  ... (batches 2-9) ...

Iteration 11:
  get_next_task_hint() → NO_MORE_TASKS (all 10 batches scanned)
  pipeline.finalize()
  exit
```

### Pattern 2: Input from Repository

**Example**: FILTER reading from repository

**Implementation** (`src/op/sirius_physical_filter.cpp:80-130`):

```cpp
class sirius_physical_filter : public sirius_physical_operator {
private:
    std::unique_ptr<Expression> filter_expr;

public:
    TaskCreationHint get_next_task_hint() override {
        // Use default implementation (check input port)
        return sirius_physical_operator::get_next_task_hint();
    }

    data_batch get_next_task_input_batch() override {
        // Use default implementation (pull from input port)
        return sirius_physical_operator::get_next_task_input_batch();
    }

    data_batch execute(data_batch&& input) override {
        // Apply filter
        auto mask = evaluate_filter(input.table, filter_expr.get());
        auto filtered = cudf::apply_boolean_mask(input.table, mask);

        return data_batch{
            .table = std::move(filtered),
            .tier = MemoryTier::GPU,
            .num_rows = filtered->num_rows()
        };
    }
};
```

**Default Implementation** (`src/op/sirius_physical_operator.cpp:50-90`):

```cpp
TaskCreationHint sirius_physical_operator::get_next_task_hint() {
    if (input_ports.empty()) {
        // No inputs: check internal state
        return has_more_work() ? TaskCreationHint::READY
                                : TaskCreationHint::NO_MORE_TASKS;
    }

    // Check first input port
    auto& input_repo = input_ports[0];

    if (input_repo->has_data()) {
        return TaskCreationHint::READY;  // Data available
    }

    if (input_repo->is_complete()) {
        if (input_repo->is_empty()) {
            return TaskCreationHint::NO_MORE_TASKS;  // Done
        }
        return TaskCreationHint::READY;  // Pull remaining
    }

    // Waiting for producer
    return TaskCreationHint::WAITING_FOR_INPUT_DATA;
}

data_batch sirius_physical_operator::get_next_task_input_batch() {
    if (input_ports.empty()) {
        throw InternalException("No input ports");
    }

    auto& input_repo = input_ports[0];

    // Pull batch (may block)
    auto batch_opt = input_repo->pull_batch();

    if (!batch_opt.has_value()) {
        throw InternalException("No input batch available");
    }

    return std::move(batch_opt.value());
}
```

**Task Creation Flow**:

```
Iteration 1:
  get_next_task_hint()
    → Check input_repo->has_data()
    → TRUE (producer pushed batch 0)
    → Return READY
  create_next_task()
    → Pull batch 0 from repository
    → Create Task 0
  enqueue(Task 0)

Iteration 2:
  get_next_task_hint()
    → Check input_repo->has_data()
    → FALSE (producer still processing)
    → Check input_repo->is_complete()
    → FALSE (producer not done)
    → Return WAITING_FOR_INPUT_DATA
  sleep(100μs)

Iteration 3:
  get_next_task_hint()
    → Check input_repo->has_data()
    → TRUE (producer pushed batch 1)
    → Return READY
  create_next_task()
    → Pull batch 1
    → Create Task 1

  ... (repeat for remaining batches) ...

Final Iteration:
  get_next_task_hint()
    → Check input_repo->has_data()
    → FALSE
    → Check input_repo->is_complete()
    → TRUE (producer marked complete)
    → Check input_repo->is_empty()
    → TRUE (all data pulled)
    → Return NO_MORE_TASKS
  pipeline.finalize()
```

### Pattern 3: Two-Phase Operator (Join)

**Example**: HASH_JOIN with build and probe phases

**Implementation** (`src/op/sirius_physical_hash_join.cpp:200-280`):

```cpp
class sirius_physical_hash_join : public sirius_physical_operator {
private:
    // Build phase state
    std::unique_ptr<cudf::hash_join> hash_table;
    std::atomic<bool> build_complete{false};
    std::mutex build_mutex;
    std::condition_variable build_cv;

public:
    // Build pipeline calls this (separate pipeline)
    void sink_build(data_batch&& batch) {
        // Insert into hash table
        auto keys = extract_keys(batch);
        hash_table->insert(keys);
    }

    void finalize_build() {
        std::lock_guard<std::mutex> lock(build_mutex);
        hash_table->finalize();
        build_complete.store(true);
        build_cv.notify_all();  // Wake probe tasks
    }

    // Probe pipeline calls this
    TaskCreationHint get_next_task_hint() override {
        // Wait for build to complete
        if (!build_complete.load(std::memory_order_acquire)) {
            LOG_TRACE("Hash join: waiting for build to complete");
            return TaskCreationHint::WAITING_FOR_INPUT_DATA;
        }

        // Build complete, check input repository
        if (input_ports.empty()) {
            return TaskCreationHint::NO_MORE_TASKS;
        }

        auto& input_repo = input_ports[0];

        if (input_repo->has_data()) {
            return TaskCreationHint::READY;
        }

        if (input_repo->is_complete() && input_repo->is_empty()) {
            return TaskCreationHint::NO_MORE_TASKS;
        }

        return TaskCreationHint::WAITING_FOR_INPUT_DATA;
    }

    data_batch execute(data_batch&& probe_batch) override {
        // Wait for build (should already be complete due to hint)
        {
            std::unique_lock<std::mutex> lock(build_mutex);
            build_cv.wait(lock, [this]() {
                return build_complete.load();
            });
        }

        // Probe hash table
        auto keys = extract_keys(probe_batch);
        auto [left_indices, right_indices] = hash_table->probe(keys);

        // Gather and return joined data
        return gather_joined_rows(probe_batch, left_indices, right_indices);
    }
};
```

**Task Creation Flow**:

```
Build Pipeline (runs first):
  Task 0: scan build side batch 0 → sink_build()
  Task 1: scan build side batch 1 → sink_build()
  ...
  finalize_build() → build_complete = true

Probe Pipeline (waits for build):
  Iteration 1 (t=0ms):
    get_next_task_hint()
      → build_complete == false
      → Return WAITING_FOR_INPUT_DATA
    sleep(100μs)

  Iteration 2 (t=0.1ms):
    get_next_task_hint()
      → build_complete == false
      → Return WAITING_FOR_INPUT_DATA
    sleep(100μs)

  ... (repeat until build complete) ...

  Iteration N (t=50ms, build completes):
    get_next_task_hint()
      → build_complete == true
      → Check input_repo->has_data()
      → TRUE (probe data available)
      → Return READY
    create_next_task()
      → Pull probe batch 0
      → Create Task 0

  Iteration N+1:
    get_next_task_hint()
      → build_complete == true
      → Check input_repo->has_data()
      → TRUE
      → Return READY
    create_next_task()
      → Pull probe batch 1
      → Create Task 1

  ... (probe all batches) ...

  Final Iteration:
    get_next_task_hint()
      → build_complete == true
      → input_repo->is_complete() && is_empty()
      → Return NO_MORE_TASKS
    pipeline.finalize()
```

### Pattern 4: Buffering Operator (Requires All Input)

**Example**: ORDER_BY must buffer all input before sorting

**Implementation** (`src/op/sirius_physical_order_by.cpp:150-240`):

```cpp
class sirius_physical_order_by : public sirius_physical_operator {
private:
    std::vector<SortColumn> sort_columns;

    // Buffering state
    std::vector<data_batch> buffered_batches;
    bool all_input_received = false;
    bool has_emitted = false;

public:
    TaskCreationHint get_next_task_hint() override {
        if (has_emitted) {
            return TaskCreationHint::NO_MORE_TASKS;
        }

        if (!all_input_received) {
            // Still collecting input
            auto& input_repo = input_ports[0];

            if (input_repo->has_data()) {
                return TaskCreationHint::READY;  // Pull and buffer
            }

            if (input_repo->is_complete()) {
                all_input_received = true;
                return TaskCreationHint::READY;  // Ready to sort
            }

            return TaskCreationHint::WAITING_FOR_INPUT_DATA;
        }

        // All input received, ready to sort and emit
        return TaskCreationHint::READY;
    }

    data_batch execute(data_batch&& input) override {
        if (!all_input_received) {
            // Buffering phase
            if (input.num_rows > 0) {
                buffered_batches.push_back(std::move(input));
                LOG_TRACE("ORDER_BY: buffered batch {} (total={})",
                          buffered_batches.size() - 1, buffered_batches.size());
            }
            return data_batch{};  // No output yet
        } else {
            // Sorting phase
            if (has_emitted) {
                return data_batch{};  // Already emitted
            }

            LOG_INFO("ORDER_BY: sorting {} buffered batches",
                     buffered_batches.size());

            // Concatenate all batches
            auto combined = concatenate_batches(buffered_batches);
            buffered_batches.clear();  // Free memory

            // Sort
            auto sorted = cudf::sort(combined.table, sort_columns);

            has_emitted = true;

            return data_batch{
                .table = std::move(sorted),
                .tier = MemoryTier::GPU,
                .num_rows = sorted->num_rows()
            };
        }
    }
};
```

**Task Creation Flow**:

```
Phase 1: Buffering (all_input_received = false)
────────────────────────────────────────────────

Iteration 1:
  get_next_task_hint()
    → all_input_received == false
    → input_repo->has_data() == TRUE
    → Return READY
  create_next_task()
    → Pull batch 0
    → Create Task 0
  execute(Task 0)
    → Buffer batch 0
    → Return empty batch

Iteration 2:
  get_next_task_hint()
    → input_repo->has_data() == TRUE
    → Return READY
  create_next_task()
    → Pull batch 1
    → Create Task 1
  execute(Task 1)
    → Buffer batch 1

  ... (buffer batches 2-9) ...

Iteration 11:
  get_next_task_hint()
    → input_repo->has_data() == FALSE
    → input_repo->is_complete() == TRUE
    → Set all_input_received = true
    → Return READY
  create_next_task()
    → Create Task 10 (sorting task)


Phase 2: Sorting (all_input_received = true)
─────────────────────────────────────────────

Iteration 12:
  execute(Task 10)
    → Concatenate 10 buffered batches
    → Sort combined table
    → Set has_emitted = true
    → Return sorted batch

Iteration 13:
  get_next_task_hint()
    → has_emitted == true
    → Return NO_MORE_TASKS
  pipeline.finalize()
```

---

## Task Creator Thread

### Main Loop

**Implementation** (`src/parallel/task_creator.cpp:100-180`):

```cpp
void task_creator::create_tasks_for_pipeline(sirius_pipeline& pipeline) {
    LOG_INFO("Task creator: starting for pipeline {}", pipeline.pipeline_id);

    size_t wait_iterations = 0;
    const size_t max_wait_iterations = 10000;  // ~1 second at 100μs/iter

    while (!should_stop) {
        // 1. Get hint from pipeline
        TaskCreationHint hint = pipeline.get_next_task_hint();

        switch (hint) {
            case TaskCreationHint::READY: {
                // Reset wait counter
                wait_iterations = 0;

                // Create task
                auto task = pipeline.create_next_task();

                if (task) {
                    // Enqueue for execution
                    task_queue.enqueue(std::move(task));
                    pipeline.tasks_created++;

                    LOG_TRACE("Pipeline {}: created task {}",
                              pipeline.pipeline_id,
                              pipeline.tasks_created.load());
                }
                break;
            }

            case TaskCreationHint::WAITING_FOR_INPUT_DATA: {
                // Increment wait counter
                wait_iterations++;

                if (wait_iterations % 1000 == 0) {
                    LOG_DEBUG("Pipeline {}: waiting for input data ({}s)",
                              pipeline.pipeline_id,
                              wait_iterations * 100.0 / 1000000);
                }

                // Brief sleep to avoid busy-waiting
                std::this_thread::sleep_for(std::chrono::microseconds(100));

                // Check for timeout (deadlock detection)
                if (wait_iterations >= max_wait_iterations) {
                    LOG_ERROR("Pipeline {}: timeout waiting for input data",
                              pipeline.pipeline_id);
                    throw TimeoutException("Task creation timeout");
                }
                break;
            }

            case TaskCreationHint::NO_MORE_TASKS: {
                LOG_INFO("Pipeline {}: no more tasks, finalizing",
                         pipeline.pipeline_id);

                // Finalize pipeline
                pipeline.finalize();

                LOG_INFO("Pipeline {}: task creator exiting (created {} tasks)",
                         pipeline.pipeline_id, pipeline.tasks_created.load());

                return;  // Exit task creator thread
            }
        }

        // Brief yield to prevent monopolizing CPU
        std::this_thread::yield();
    }

    LOG_WARN("Pipeline {}: task creator stopped prematurely", pipeline.pipeline_id);
}
```

### Coordination with Executors

**Task Queue** (lock-free MPMC queue):

```cpp
// Task creator (producer)
auto task = pipeline.create_next_task();
task_queue.enqueue(std::move(task));  // Non-blocking push

// Pipeline executor (consumer)
auto task_opt = task_queue.try_dequeue();  // Non-blocking pop
if (task_opt.has_value()) {
    auto task = std::move(task_opt.value());
    task->compute_task();
}
```

**Flow**:

```
Task Creator Thread 1          Task Queue          Pipeline Executor Thread A
───────────────────           ──────────          ──────────────────────────
Create Task 0        ──────→  [Task 0]
Create Task 1        ──────→  [Task 0, Task 1]
Create Task 2        ──────→  [Task 0, Task 1, Task 2]
                                   ↓
                              Dequeue Task 0  ←── Execute Task 0
                                   ↓
Create Task 3        ──────→  [Task 1, Task 2, Task 3]
                                   ↓
                              Dequeue Task 1  ←── Execute Task 1
                                                  (concurrent with Task 0)
```

---

## Performance Tuning

### Task Creator Threads

**Configuration**:

```ini
[threading]
task_creator_threads = 2  # Number of task creator threads
```

**Trade-offs**:
- **1 thread**: Simplest, may bottleneck with many pipelines
- **2 threads** (default): Good for 2-4 pipelines
- **4+ threads**: For many pipelines (8+), diminishing returns

**Recommendation**: 2 threads for most workloads

### Wait Sleep Duration

**Current**: 100μs

**Trade-offs**:
- **Shorter (10μs)**: More responsive, higher CPU usage
- **Longer (1ms)**: Lower CPU usage, higher latency

**Tuning**:

```cpp
// Adaptive backoff
size_t sleep_us = 100;
if (consecutive_waits > 10) {
    sleep_us = 500;  // Back off if waiting long
}
std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
```

### Task Queue Size

**Configuration**:

```cpp
// Maximum tasks in queue
const size_t MAX_QUEUE_SIZE = 1000;

// Block task creators if queue full
while (task_queue.size() >= MAX_QUEUE_SIZE) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}
```

**Benefits**:
- Prevents unbounded memory growth
- Natural backpressure on task creators

---

## Debugging

### Enable Task Creation Logging

```bash
export SIRIUS_LOG_LEVEL=TRACE
```

**Log Output**:

```
[INFO] Task creator: starting for pipeline 0
[TRACE] Pipeline 0: created task 1
[TRACE] Pipeline 0: created task 2
[TRACE] Pipeline 0: created task 3
[TRACE] Pipeline 0: waiting for input data
[TRACE] Pipeline 0: waiting for input data
[DEBUG] Pipeline 0: waiting for input data (0.1s)
[TRACE] Pipeline 0: created task 4
[INFO] Pipeline 0: no more tasks, finalizing
[INFO] Pipeline 0: task creator exiting (created 4 tasks)
```

### Trace Task Hints

Add instrumentation:

```cpp
TaskCreationHint get_next_task_hint() override {
    auto hint = compute_hint();

    LOG_TRACE("Operator {}: hint = {}",
              get_name(), to_string(hint));

    return hint;
}
```

**Output**:

```
[TRACE] Operator TABLE_SCAN: hint = READY
[TRACE] Operator FILTER: hint = READY
[TRACE] Operator HASH_JOIN: hint = WAITING_FOR_INPUT_DATA
[TRACE] Operator HASH_JOIN: hint = WAITING_FOR_INPUT_DATA
[TRACE] Operator HASH_JOIN: hint = READY
```

### Monitor Task Statistics

```cpp
void print_pipeline_stats(const sirius_pipeline& pipeline) {
    printf("Pipeline %zu Statistics:\n", pipeline.pipeline_id);
    printf("  Tasks created: %zu\n", pipeline.tasks_created.load());
    printf("  Tasks completed: %zu\n", pipeline.tasks_completed.load());
    printf("  In flight: %zu\n",
           pipeline.tasks_created - pipeline.tasks_completed);
}
```

---

## Common Issues

### Issue 1: Deadlock (Infinite WAITING_FOR_INPUT_DATA)

**Symptoms**:
- Task creator stuck in WAITING_FOR_INPUT_DATA
- Query hangs indefinitely

**Root Causes**:
1. Producer never calls `mark_complete()`
2. Circular pipeline dependency
3. Repository connection not established

**Solution**:
- Ensure all sink operators call `mark_complete()` in finalize
- Verify repository connections in pipeline setup
- Check for dependency cycles

### Issue 2: Premature NO_MORE_TASKS

**Symptoms**:
- Query returns incomplete results
- Pipeline exits early

**Root Causes**:
1. Incorrect hint logic (returns NO_MORE_TASKS too early)
2. Repository empty check before producer complete

**Solution**:
```cpp
// Wrong: May return NO_MORE_TASKS prematurely
if (!input_repo->has_data()) {
    return TaskCreationHint::NO_MORE_TASKS;  // ❌ Wrong!
}

// Correct: Check completion first
if (input_repo->is_complete() && input_repo->is_empty()) {
    return TaskCreationHint::NO_MORE_TASKS;  // ✓ Correct
}
```

### Issue 3: Excessive Task Creation

**Symptoms**:
- High memory usage
- Task queue grows unbounded

**Root Causes**:
- No backpressure on task creators
- Task execution slower than creation

**Solution**:
- Limit task queue size
- Add backpressure when queue full

---

## See Also

- [New Mode Overview](overview.md) - Introduction to New Mode
- [Pipeline Execution](pipeline-execution.md) - Pipeline structure
- [Operators](operators.md) - Operator implementations
- [New Data Flow](../06-data-flow/new-data-flow.md) - Complete data flow
- [Threading Model](../05-core-components/threading-model.md) - Thread pools
- [Performance Tips](../appendices/performance-tips.md) - Optimization guide
