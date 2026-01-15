# Pipeline Executor Tests

This directory contains unit tests for the pipeline executor and GPU pipeline executor components.

## Test Coverage

### `test_pipeline_executor.cpp`

This test file validates the functionality of the pipeline execution framework without executing actual GPU operations. The tests focus on:

#### Thread Pool Functionality
- **Start/Stop Lifecycle**: Verifies that executors can be started and stopped gracefully without hanging or crashing
- **Multiple Cycles**: Ensures executors can be restarted after stopping and continue to work correctly
- **Worker Thread Management**: Tests that worker threads are properly created and terminated

#### Queue Functionality
- **Task Queue Operations**: Tests task submission and retrieval from the pipeline queue
- **Request Queue Operations**: Tests task request submission and retrieval
- **Empty Queue Handling**: Verifies graceful shutdown when no tasks are in the queue
- **Rapid Submission**: Tests queue behavior under heavy load with many tasks submitted quickly
- **Synchronization**: Validates proper coordination between task queue and request queue

#### Pipeline Executor & GPU Pipeline Executor Interaction
- **Task Dispatching**: Tests that the pipeline_executor correctly dispatches tasks to GPU executors
- **Multi-GPU Support**: Verifies tasks can be routed to different GPU executors based on device_id
- **Task-Request Pairing**: Ensures tasks and their corresponding requests are properly synchronized
- **Task Execution**: Validates that scheduled tasks are actually executed by the GPU executors

## Mock Components

The tests use mock implementations to avoid dependencies on actual GPU hardware:

- **`mock_gpu_pipeline_task`**: Simulates a GPU pipeline task by incrementing counters and recording execution
- **`mock_gpu_pipeline_task_global_state`**: Tracks execution metrics across all tasks
- **`mock_gpu_pipeline_task_local_state`**: Stores per-task information like task ID and expected GPU ID

## Running the Tests

These tests are integrated into the main test suite and will run as part of the standard test execution:

```bash
# Build and run all tests
make test

# Or run just the pipeline executor tests using catch2 tags
./build/release/test/unittest "[pipeline_executor]"
./build/release/test/unittest "[gpu_pipeline_executor]"
./build/release/test/unittest "[pipeline_queue]"
```

## Test Timeouts

All tests include timeout protection (5-15 seconds depending on test complexity) to prevent infinite hangs if something goes wrong with the threading or queuing logic.
