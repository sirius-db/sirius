# Cucascade Integration

Comprehensive guide to how Sirius New Mode integrates with Cucascade for data management, including data_batch, repositories, and multi-tier memory.

---

## Overview

**Cucascade** is the data management layer for Sirius New Mode, providing:
- **data_batch**: Immutable data unit
- **shared_data_repository**: Inter-pipeline storage
- **memory_reservation_manager**: Multi-tier memory management
- **data_repository_manager**: Centralized repository coordination

| Component | Purpose | Key Feature |
|-----------|---------|-------------|
| **data_batch** | Data container | Immutable, tier-aware |
| **shared_data_repository** | Pipeline communication | Multi-tier buffering |
| **memory_reservation_manager** | Memory allocation | GPU/HOST/DISK spilling |
| **data_repository_manager** | Repository lifecycle | Centralized management |

**Location**: `cucascade/` subdirectory (external library integrated into Sirius)

---

## Core Concepts

### Data Batch

The fundamental unit of data in New Mode.

**Definition**: `cucascade/include/data_batch.hpp`

```cpp
class data_batch {
public:
    // Data storage
    std::unique_ptr<cudf::table> table;

    // Memory tier
    MemoryTier tier;

    // Metadata
    size_t num_rows;
    size_t size_bytes;
    std::vector<cudf::data_type> schema;

    // Memory reservation
    ReservationID reservation_id;

    // Constructors
    data_batch();
    data_batch(std::unique_ptr<cudf::table> tbl, MemoryTier t);

    // Move-only (no copying)
    data_batch(data_batch&&) = default;
    data_batch& operator=(data_batch&&) = default;
    data_batch(const data_batch&) = delete;
    data_batch& operator=(const data_batch&) = delete;

    // Clone (explicit copy)
    data_batch clone() const;

    // Utilities
    bool empty() const { return num_rows == 0; }
    void clear();
};
```

**Memory Tier Enumeration**:

```cpp
enum class MemoryTier {
    GPU,   // Fastest: GPU VRAM (cudaMalloc)
    HOST,  // Fast: Pinned host memory (cudaMallocHost)
    DISK   // Slow: Disk storage (Parquet files)
};
```

### Properties

#### Immutability

Once created, a data_batch's contents don't change:

```cpp
// Create batch
data_batch batch = create_batch(data);

// Operations return new batches
data_batch filtered = filter(std::move(batch));

// Original batch is moved (no longer valid)
// filtered is a NEW batch
```

**Benefits**:
- Thread-safe: Multiple operators can reference same batch
- Predictable: No hidden side effects
- Cacheable: Safe to store in repositories

#### Tier Awareness

Each batch knows its memory tier:

```cpp
data_batch gpu_batch = read_parquet("data.parquet");
// gpu_batch.tier == MemoryTier::GPU

// Automatic downgrade on memory pressure
if (memory_full()) {
    gpu_batch = downgrade_to_host(std::move(gpu_batch));
    // gpu_batch.tier == MemoryTier::HOST
}
```

#### Size Tracking

Batches track their memory footprint:

```cpp
data_batch batch;
batch.num_rows = 100000;
batch.size_bytes = calculate_size(batch.table);

// Used for memory reservation
reservation_mgr.reserve(batch.size_bytes);
```

**Size Calculation**:

```cpp
size_t calculate_batch_size(const cudf::table& table) {
    size_t total = 0;

    for (size_t i = 0; i < table.num_columns(); i++) {
        auto& col = table.column(i);

        // Data buffer
        total += col.size() * cudf::size_of(col.type());

        // Validity mask (1 bit per row)
        if (col.nullable()) {
            total += (col.size() + 7) / 8;
        }

        // String data (if string column)
        if (col.type().id() == cudf::type_id::STRING) {
            total += get_strings_size(col);
        }
    }

    return total;
}
```

---

## Shared Data Repository

Inter-pipeline data storage with multi-tier buffering.

**Definition**: `cucascade/include/data_repository.hpp`

```cpp
class shared_data_repository {
private:
    // Repository metadata
    std::string name;
    size_t repository_id;

    // Multi-tier queues
    std::deque<data_batch> gpu_queue;
    std::deque<data_batch> host_queue;
    std::deque<data_batch> disk_queue;

    // State tracking
    bool completed = false;
    size_t total_batches_pushed = 0;
    size_t total_batches_pulled = 0;

    // Synchronization
    std::mutex mutex;
    std::condition_variable cv_data_ready;
    std::condition_variable cv_space_available;

    // Memory management
    memory_reservation_manager& mem_mgr;

    // Capacity limits
    size_t max_batches;
    size_t max_bytes;

public:
    // Constructor
    shared_data_repository(
        memory_reservation_manager& mgr,
        std::string name,
        size_t max_batches = 100,
        size_t max_bytes = 10ULL * 1024 * 1024 * 1024 // 10GB
    );

    // Producer API
    void push_data_batch(data_batch&& batch);
    void mark_complete();

    // Consumer API
    std::optional<data_batch> pull_batch();
    std::optional<data_batch> try_pull_batch();  // Non-blocking

    // Status queries
    bool is_complete() const;
    bool has_data() const;
    bool is_empty() const;
    bool is_full() const;

    // Statistics
    size_t size() const;
    size_t gpu_size() const;
    size_t host_size() const;
    size_t disk_size() const;
    size_t total_size_bytes() const;

    // Debugging
    void print_stats() const;
};
```

### Push Operation

**Producer Side** (`cucascade/src/data_repository.cpp:50-120`):

```cpp
void shared_data_repository::push_data_batch(data_batch&& batch) {
    if (batch.empty()) {
        return; // Ignore empty batches
    }

    std::unique_lock<std::mutex> lock(mutex);

    // Wait if repository full (backpressure)
    cv_space_available.wait(lock, [this]() {
        return !is_full() || completed;
    });

    if (completed) {
        throw InternalException("Cannot push to completed repository");
    }

    // Reserve memory for this batch
    ReservationID reservation = mem_mgr.reserve(
        batch.size_bytes,
        batch.tier
    );
    batch.reservation_id = reservation;

    // Determine target tier based on memory availability
    MemoryTier target_tier = determine_target_tier(batch.size_bytes);

    // Downgrade if needed
    if (target_tier != batch.tier) {
        batch = downgrade_batch(std::move(batch), target_tier);
    }

    // Add to appropriate queue
    switch (target_tier) {
        case MemoryTier::GPU:
            gpu_queue.push_back(std::move(batch));
            LOG_TRACE("Repository {}: pushed batch to GPU queue (size={})",
                      name, gpu_queue.size());
            break;

        case MemoryTier::HOST:
            host_queue.push_back(std::move(batch));
            LOG_TRACE("Repository {}: pushed batch to HOST queue (size={})",
                      name, host_queue.size());
            break;

        case MemoryTier::DISK:
            disk_queue.push_back(std::move(batch));
            LOG_TRACE("Repository {}: pushed batch to DISK queue (size={})",
                      name, disk_queue.size());
            break;
    }

    total_batches_pushed++;

    // Notify waiting consumers
    cv_data_ready.notify_one();
}

MemoryTier shared_data_repository::determine_target_tier(size_t size_bytes) {
    // Try GPU first
    if (mem_mgr.gpu_has_space(size_bytes)) {
        return MemoryTier::GPU;
    }

    // Try HOST next
    if (mem_mgr.host_has_space(size_bytes)) {
        return MemoryTier::HOST;
    }

    // Fall back to DISK
    return MemoryTier::DISK;
}
```

**Downgrade Process**:

```cpp
data_batch shared_data_repository::downgrade_batch(
    data_batch&& batch,
    MemoryTier target_tier
) {
    if (batch.tier == target_tier) {
        return std::move(batch); // Already at target tier
    }

    switch (target_tier) {
        case MemoryTier::HOST:
            // GPU → HOST
            return downgrade_gpu_to_host(std::move(batch));

        case MemoryTier::DISK:
            // GPU/HOST → DISK
            if (batch.tier == MemoryTier::GPU) {
                batch = downgrade_gpu_to_host(std::move(batch));
            }
            return downgrade_host_to_disk(std::move(batch));

        default:
            throw InternalException("Invalid target tier for downgrade");
    }
}

data_batch shared_data_repository::downgrade_gpu_to_host(data_batch&& batch) {
    LOG_DEBUG("Repository {}: downgrading batch GPU → HOST", name);

    // Allocate pinned host memory
    void* host_buffer;
    cudaMallocHost(&host_buffer, batch.size_bytes);

    // Copy data from GPU to HOST
    cudaMemcpy(
        host_buffer,
        batch.table->data(),
        batch.size_bytes,
        cudaMemcpyDeviceToHost
    );

    // Free GPU memory
    batch.table.reset();

    // Update reservation
    mem_mgr.move_reservation(
        batch.reservation_id,
        MemoryTier::GPU,
        MemoryTier::HOST
    );

    // Create new batch at HOST tier
    return data_batch{
        .table = reconstruct_from_buffer(host_buffer, batch.schema),
        .tier = MemoryTier::HOST,
        .num_rows = batch.num_rows,
        .size_bytes = batch.size_bytes,
        .schema = batch.schema,
        .reservation_id = batch.reservation_id
    };
}

data_batch shared_data_repository::downgrade_host_to_disk(data_batch&& batch) {
    LOG_DEBUG("Repository {}: downgrading batch HOST → DISK", name);

    // Generate temp file path
    std::string temp_path = generate_temp_parquet_path();

    // Write batch to Parquet
    write_parquet(batch.table, temp_path);

    // Free host memory
    batch.table.reset();

    // Update reservation
    mem_mgr.move_reservation(
        batch.reservation_id,
        MemoryTier::HOST,
        MemoryTier::DISK
    );

    // Create new batch at DISK tier
    return data_batch{
        .table = nullptr,  // No in-memory table
        .tier = MemoryTier::DISK,
        .num_rows = batch.num_rows,
        .size_bytes = batch.size_bytes,
        .schema = batch.schema,
        .disk_path = temp_path,
        .reservation_id = batch.reservation_id
    };
}
```

### Pull Operation

**Consumer Side** (`cucascade/src/data_repository.cpp:180-270`):

```cpp
std::optional<data_batch> shared_data_repository::pull_batch() {
    std::unique_lock<std::mutex> lock(mutex);

    // Wait for data or completion
    cv_data_ready.wait(lock, [this]() {
        return has_data() || completed;
    });

    // Check if truly done
    if (is_empty()) {
        if (completed) {
            LOG_TRACE("Repository {}: pull found no data (completed)", name);
            return std::nullopt;
        }
        // Spurious wakeup
        return std::nullopt;
    }

    // Pull from highest priority tier: GPU → HOST → DISK
    data_batch batch;
    MemoryTier source_tier;

    if (!gpu_queue.empty()) {
        batch = std::move(gpu_queue.front());
        gpu_queue.pop_front();
        source_tier = MemoryTier::GPU;

        LOG_TRACE("Repository {}: pulled batch from GPU queue (remaining={})",
                  name, gpu_queue.size());

    } else if (!host_queue.empty()) {
        batch = std::move(host_queue.front());
        host_queue.pop_front();
        source_tier = MemoryTier::HOST;

        LOG_TRACE("Repository {}: pulled batch from HOST queue (remaining={})",
                  name, host_queue.size());

    } else if (!disk_queue.empty()) {
        batch = std::move(disk_queue.front());
        disk_queue.pop_front();
        source_tier = MemoryTier::DISK;

        LOG_TRACE("Repository {}: pulled batch from DISK queue (remaining={})",
                  name, disk_queue.size());
    }

    total_batches_pulled++;

    // Notify producers if waiting (backpressure released)
    cv_space_available.notify_one();

    // Upgrade to GPU if possible and needed
    if (source_tier != MemoryTier::GPU) {
        batch = try_upgrade_to_gpu(std::move(batch));
    }

    return batch;
}

data_batch shared_data_repository::try_upgrade_to_gpu(data_batch&& batch) {
    // Check if GPU has space
    if (!mem_mgr.gpu_has_space(batch.size_bytes)) {
        LOG_TRACE("Repository {}: cannot upgrade batch (no GPU space)", name);
        return std::move(batch); // Keep at current tier
    }

    // Upgrade based on current tier
    switch (batch.tier) {
        case MemoryTier::HOST:
            return upgrade_host_to_gpu(std::move(batch));

        case MemoryTier::DISK:
            // DISK → HOST → GPU (two-step)
            batch = upgrade_disk_to_host(std::move(batch));
            if (mem_mgr.gpu_has_space(batch.size_bytes)) {
                return upgrade_host_to_gpu(std::move(batch));
            }
            return std::move(batch);

        case MemoryTier::GPU:
            return std::move(batch); // Already at GPU
    }
}

data_batch shared_data_repository::upgrade_host_to_gpu(data_batch&& batch) {
    LOG_DEBUG("Repository {}: upgrading batch HOST → GPU", name);

    // Allocate GPU memory
    void* gpu_buffer;
    cudaMalloc(&gpu_buffer, batch.size_bytes);

    // Copy data from HOST to GPU
    cudaMemcpy(
        gpu_buffer,
        batch.table->data(),
        batch.size_bytes,
        cudaMemcpyHostToDevice
    );

    // Free host memory
    cudaFreeHost(batch.table->data());
    batch.table.reset();

    // Update reservation
    mem_mgr.move_reservation(
        batch.reservation_id,
        MemoryTier::HOST,
        MemoryTier::GPU
    );

    // Reconstruct table on GPU
    return data_batch{
        .table = reconstruct_from_buffer(gpu_buffer, batch.schema),
        .tier = MemoryTier::GPU,
        .num_rows = batch.num_rows,
        .size_bytes = batch.size_bytes,
        .schema = batch.schema,
        .reservation_id = batch.reservation_id
    };
}

data_batch shared_data_repository::upgrade_disk_to_host(data_batch&& batch) {
    LOG_DEBUG("Repository {}: upgrading batch DISK → HOST", name);

    // Read from Parquet file
    auto table = read_parquet(batch.disk_path);

    // Delete temp file
    std::filesystem::remove(batch.disk_path);

    // Update reservation
    mem_mgr.move_reservation(
        batch.reservation_id,
        MemoryTier::DISK,
        MemoryTier::HOST
    );

    return data_batch{
        .table = std::move(table),
        .tier = MemoryTier::HOST,
        .num_rows = batch.num_rows,
        .size_bytes = batch.size_bytes,
        .schema = batch.schema,
        .reservation_id = batch.reservation_id
    };
}
```

### Completion Signaling

**Mark Complete** (`cucascade/src/data_repository.cpp:300-320`):

```cpp
void shared_data_repository::mark_complete() {
    std::lock_guard<std::mutex> lock(mutex);

    if (completed) {
        LOG_WARN("Repository {}: already marked complete", name);
        return;
    }

    completed = true;

    LOG_INFO("Repository {}: marked complete ({} batches pushed, {} pulled)",
             name, total_batches_pushed, total_batches_pulled);

    // Wake ALL waiting consumers
    cv_data_ready.notify_all();
}

bool shared_data_repository::is_complete() const {
    std::lock_guard<std::mutex> lock(mutex);
    return completed;
}
```

---

## Memory Reservation Manager

Manages multi-tier memory allocation and spilling.

**Definition**: `cucascade/include/memory_reservation_manager.hpp`

```cpp
class memory_reservation_manager {
private:
    // Memory limits (bytes)
    size_t gpu_limit;
    size_t host_limit;
    size_t disk_limit;  // -1 for unlimited

    // Current usage (bytes)
    std::atomic<size_t> gpu_usage{0};
    std::atomic<size_t> host_usage{0};
    std::atomic<size_t> disk_usage{0};

    // Reservation tracking
    std::unordered_map<ReservationID, Reservation> reservations;
    std::atomic<ReservationID> next_reservation_id{0};

    // Synchronization
    std::mutex mutex;

public:
    // Constructor
    memory_reservation_manager(
        size_t gpu_limit,
        size_t host_limit,
        size_t disk_limit = -1
    );

    // Reservation API
    ReservationID reserve(size_t size_bytes, MemoryTier tier);
    void release(ReservationID id);
    void move_reservation(ReservationID id, MemoryTier from, MemoryTier to);

    // Capacity checks
    bool gpu_has_space(size_t size_bytes) const;
    bool host_has_space(size_t size_bytes) const;
    bool disk_has_space(size_t size_bytes) const;

    // Statistics
    size_t get_gpu_usage() const { return gpu_usage.load(); }
    size_t get_host_usage() const { return host_usage.load(); }
    size_t get_disk_usage() const { return disk_usage.load(); }

    double get_gpu_utilization() const {
        return static_cast<double>(gpu_usage) / gpu_limit;
    }

    void print_stats() const;
};

struct Reservation {
    ReservationID id;
    size_t size_bytes;
    MemoryTier tier;
    std::chrono::time_point<std::chrono::steady_clock> created_at;
};
```

### Reserve Memory

**Implementation** (`cucascade/src/memory_reservation_manager.cpp:50-100`):

```cpp
ReservationID memory_reservation_manager::reserve(
    size_t size_bytes,
    MemoryTier tier
) {
    std::lock_guard<std::mutex> lock(mutex);

    // Check capacity
    switch (tier) {
        case MemoryTier::GPU:
            if (gpu_usage + size_bytes > gpu_limit) {
                throw OutOfMemoryException("GPU memory limit exceeded");
            }
            gpu_usage += size_bytes;
            break;

        case MemoryTier::HOST:
            if (host_usage + size_bytes > host_limit) {
                throw OutOfMemoryException("HOST memory limit exceeded");
            }
            host_usage += size_bytes;
            break;

        case MemoryTier::DISK:
            if (disk_limit != -1 && disk_usage + size_bytes > disk_limit) {
                throw OutOfMemoryException("DISK space limit exceeded");
            }
            disk_usage += size_bytes;
            break;
    }

    // Create reservation
    ReservationID id = next_reservation_id++;
    reservations[id] = Reservation{
        .id = id,
        .size_bytes = size_bytes,
        .tier = tier,
        .created_at = std::chrono::steady_clock::now()
    };

    LOG_TRACE("Reserved {} bytes in {} tier (ID={})",
              size_bytes, to_string(tier), id);

    return id;
}
```

### Move Reservation

**Implementation** (`cucascade/src/memory_reservation_manager.cpp:150-200`):

```cpp
void memory_reservation_manager::move_reservation(
    ReservationID id,
    MemoryTier from_tier,
    MemoryTier to_tier
) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = reservations.find(id);
    if (it == reservations.end()) {
        throw InternalException("Reservation not found");
    }

    Reservation& res = it->second;

    if (res.tier != from_tier) {
        throw InternalException("Reservation tier mismatch");
    }

    // Update usage counters
    switch (from_tier) {
        case MemoryTier::GPU:
            gpu_usage -= res.size_bytes;
            break;
        case MemoryTier::HOST:
            host_usage -= res.size_bytes;
            break;
        case MemoryTier::DISK:
            disk_usage -= res.size_bytes;
            break;
    }

    switch (to_tier) {
        case MemoryTier::GPU:
            if (gpu_usage + res.size_bytes > gpu_limit) {
                throw OutOfMemoryException("GPU limit exceeded during move");
            }
            gpu_usage += res.size_bytes;
            break;
        case MemoryTier::HOST:
            if (host_usage + res.size_bytes > host_limit) {
                throw OutOfMemoryException("HOST limit exceeded during move");
            }
            host_usage += res.size_bytes;
            break;
        case MemoryTier::DISK:
            if (disk_limit != -1 && disk_usage + res.size_bytes > disk_limit) {
                throw OutOfMemoryException("DISK limit exceeded during move");
            }
            disk_usage += res.size_bytes;
            break;
    }

    // Update reservation
    res.tier = to_tier;

    LOG_TRACE("Moved reservation {} from {} to {} ({} bytes)",
              id, to_string(from_tier), to_string(to_tier), res.size_bytes);
}
```

### Automatic Spilling

**Triggered by Downgrade Executor** (`src/parallel/downgrade_executor.cpp:80-150`):

```cpp
void downgrade_executor::check_and_spill() {
    // Check GPU pressure
    double gpu_util = mem_mgr.get_gpu_utilization();

    if (gpu_util > gpu_spill_threshold) {
        LOG_INFO("GPU memory pressure: {:.1f}% (threshold={:.1f}%)",
                 gpu_util * 100, gpu_spill_threshold * 100);

        // Find candidates for spilling from repositories
        auto candidates = find_spill_candidates(MemoryTier::GPU);

        for (auto& [repo, batch_idx] : candidates) {
            // Downgrade GPU → HOST
            repo->downgrade_batch_at_index(batch_idx, MemoryTier::HOST);

            gpu_util = mem_mgr.get_gpu_utilization();
            if (gpu_util < gpu_restore_threshold) {
                break; // Sufficient space freed
            }
        }
    }

    // Check HOST pressure
    double host_util = mem_mgr.get_host_utilization();

    if (host_util > host_spill_threshold) {
        LOG_INFO("HOST memory pressure: {:.1f}% (threshold={:.1f}%)",
                 host_util * 100, host_spill_threshold * 100);

        auto candidates = find_spill_candidates(MemoryTier::HOST);

        for (auto& [repo, batch_idx] : candidates) {
            // Downgrade HOST → DISK
            repo->downgrade_batch_at_index(batch_idx, MemoryTier::DISK);

            host_util = mem_mgr.get_host_utilization();
            if (host_util < host_restore_threshold) {
                break;
            }
        }
    }
}

std::vector<std::pair<shared_data_repository*, size_t>>
downgrade_executor::find_spill_candidates(MemoryTier tier) {
    std::vector<std::pair<shared_data_repository*, size_t>> candidates;

    // Iterate all repositories
    for (auto& repo : repository_manager.get_all_repositories()) {
        // Find batches at given tier
        size_t idx = 0;
        for (const auto& batch : repo->get_tier_queue(tier)) {
            candidates.push_back({repo.get(), idx});
            idx++;
        }
    }

    // Sort by LRU (oldest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return a.second < b.second; // Earlier index = older
              });

    return candidates;
}
```

---

## Data Repository Manager

Centralized management of all repositories.

**Definition**: `cucascade/include/data_repository_manager.hpp`

```cpp
class data_repository_manager {
private:
    // All repositories
    std::vector<std::shared_ptr<shared_data_repository>> repositories;

    // Name → repository mapping
    std::unordered_map<std::string, std::shared_ptr<shared_data_repository>> by_name;

    // Memory manager reference
    memory_reservation_manager& mem_mgr;

    // Synchronization
    std::mutex mutex;

public:
    // Constructor
    data_repository_manager(memory_reservation_manager& mgr);

    // Repository lifecycle
    std::shared_ptr<shared_data_repository> create_repository(
        std::string name,
        size_t max_batches = 100,
        size_t max_bytes = 10ULL * 1024 * 1024 * 1024
    );

    std::shared_ptr<shared_data_repository> get_repository(const std::string& name);

    void destroy_repository(const std::string& name);
    void destroy_all();

    // Statistics
    size_t count() const;
    std::vector<std::shared_ptr<shared_data_repository>> get_all_repositories();

    void print_all_stats() const;
};
```

### Create Repository

**Implementation** (`cucascade/src/data_repository_manager.cpp:40-70`):

```cpp
std::shared_ptr<shared_data_repository>
data_repository_manager::create_repository(
    std::string name,
    size_t max_batches,
    size_t max_bytes
) {
    std::lock_guard<std::mutex> lock(mutex);

    // Check if already exists
    if (by_name.count(name) > 0) {
        LOG_WARN("Repository '{}' already exists, returning existing", name);
        return by_name[name];
    }

    // Create new repository
    auto repo = std::make_shared<shared_data_repository>(
        mem_mgr,
        name,
        max_batches,
        max_bytes
    );

    // Register
    repositories.push_back(repo);
    by_name[name] = repo;

    LOG_INFO("Created repository '{}' (max_batches={}, max_bytes={})",
             name, max_batches, max_bytes);

    return repo;
}
```

---

## Example: Complete Data Flow

### Query

```sql
SELECT * FROM gpu_execution('
    SELECT category, SUM(amount) as total
    FROM sales
    GROUP BY category
    ORDER BY total DESC
');
```

### Pipeline Structure

```
Pipeline 1: SCAN → HASH_GROUP_BY (sink)
            └─ Output: Repo A

Pipeline 2: ORDER_BY (source) → RESULT_COLLECTOR
            └─ Input: Repo A
```

### Data Flow with Cucascade

```
Time: 0ms - Pipeline 1 Start
──────────────────────────────

1. SCAN operator:
   - Read Parquet batch 0 (100K rows, 5MB)
   - Create data_batch:
       .table = cudf::table from Parquet
       .tier = MemoryTier::GPU
       .num_rows = 100000
       .size_bytes = 5 * 1024 * 1024
   - Reserve memory:
       mem_mgr.reserve(5MB, GPU)
       → reservation_id = 1

2. HASH_GROUP_BY operator (sink):
   - Receive data_batch (batch 0)
   - Aggregate into hash table
   - Check if should flush: NO
   - Continue...

Time: 50ms - Scan batch 5
─────────────────────────

3. SCAN operator:
   - Read batch 5 (100K rows, 5MB)
   - Create data_batch (reservation_id = 6)

4. HASH_GROUP_BY operator:
   - Aggregate batch 5
   - Check if should flush: YES (500K rows accumulated)
   - Finalize partial result:
       result_batch: 1K unique categories
   - Push to Repository A:
       repo_A->push_data_batch(result_batch)

5. Repository A (push):
   - Reserve memory: mem_mgr.reserve(100KB, GPU)
   - Determine target tier: GPU (space available)
   - Add to gpu_queue: [result_batch_0]
   - Notify consumers: cv_data_ready.notify_one()

Time: 55ms - Pipeline 2 Start
──────────────────────────────

6. Task Creator (Pipeline 2):
   - Check hint: ORDER_BY->get_next_task_hint()
       → Check repo_A->has_data()
       → YES (1 batch in gpu_queue)
       → Return: TaskCreationHint::READY

7. Create Task:
   - Get input: ORDER_BY->get_next_task_input_batch()
       → repo_A->pull_batch()

8. Repository A (pull):
   - Lock mutex
   - Check gpu_queue: NOT EMPTY
   - Pop batch: result_batch_0
   - Batch already at GPU tier
   - Return batch

9. ORDER_BY operator (execute):
   - Receive result_batch_0
   - Buffer locally (wait for all input)

Time: 100ms - Pipeline 1 Complete
──────────────────────────────────

10. HASH_GROUP_BY finalize():
    - Flush remaining data
    - Push final_batch to repo_A
    - Mark complete: repo_A->mark_complete()

11. Repository A:
    - Set completed = true
    - Notify all: cv_data_ready.notify_all()

Time: 120ms - Pipeline 2 Continues
───────────────────────────────────

12. Task Creator (Pipeline 2):
    - Check hint: ORDER_BY->get_next_task_hint()
        → repo_A->is_complete() && repo_A->is_empty()
        → All input received
        → Return: TaskCreationHint::READY (for sorting)

13. ORDER_BY operator (execute with dummy input):
    - Concatenate all buffered batches
    - Sort combined table
    - Emit sorted result

14. RESULT_COLLECTOR:
    - Receive sorted batch
    - Convert to DuckDB format
    - Return to user

Time: 130ms - Query Complete
─────────────────────────────

Memory Statistics:
  GPU Usage: Peak 30MB (hash table + batches)
  HOST Usage: 0MB (no spilling)
  DISK Usage: 0MB (no spilling)

Repository A Statistics:
  Batches pushed: 10
  Batches pulled: 10
  GPU hits: 10 (100%)
  HOST hits: 0
  DISK hits: 0
```

---

## Performance Considerations

### Memory Overhead

**Per data_batch**:
```
Overhead = sizeof(data_batch) + metadata
         = ~200 bytes (pointers, counters, etc.)
```

**Per repository**:
```
Overhead = queues + mutexes + counters
         = ~1 KB per repository
```

**Total for typical query**:
```
10 repositories * 50 batches/repo * 200 bytes/batch
= 100 KB overhead

Negligible compared to data size (GB scale)
```

### Tier Transfer Costs

**Measured on A100 + PCIe Gen4**:

| Transfer | Bandwidth | 5MB Batch Cost |
|----------|-----------|----------------|
| GPU → HOST | 32 GB/s | ~0.15ms |
| HOST → GPU | 32 GB/s | ~0.15ms |
| HOST → DISK | 5 GB/s (NVMe write) | ~1ms |
| DISK → HOST | 7 GB/s (NVMe read) | ~0.7ms |

**Recommendation**: Minimize spilling for best performance

---

## Configuration

### Memory Limits

**INI Format** (`sirius.cfg`):

```ini
[memory]
gpu_memory_limit = 12288     # 12GB
host_memory_limit = 49152    # 48GB
disk_memory_limit = -1       # Unlimited

# Spill thresholds (0.0-1.0)
gpu_spill_threshold = 0.9    # Spill at 90%
host_spill_threshold = 0.9
gpu_restore_threshold = 0.8  # Stop spilling at 80%
host_restore_threshold = 0.8
```

### Repository Limits

**Programmatic**:

```cpp
auto repo = repo_mgr.create_repository(
    "pipeline_break_0",
    max_batches = 50,          // Max 50 batches
    max_bytes = 5ULL << 30     // Max 5GB
);
```

---

## See Also

- [New Mode Overview](overview.md) - Introduction to New Mode
- [Operators](operators.md) - Operator implementations
- [New Data Flow](../06-data-flow/new-data-flow.md) - Complete data flow
- [Inter-Pipeline Communication](../06-data-flow/inter-pipeline-communication.md) - Repository details
- [Memory Management](../05-core-components/memory-management.md) - Memory system
- [Performance Tips](../appendices/performance-tips.md) - Optimization guide
