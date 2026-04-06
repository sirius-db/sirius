# Codebase Structure

**Analysis Date:** 2026-04-06

## Directory Layout

```
/home/bwyogatama/doris/
├── src/                           # All Sirius source code
│   ├── sirius_extension.cpp       # DuckDB extension entry point
│   ├── sirius_interface.cpp       # Query lifecycle management
│   ├── sirius_engine.cpp          # Pipeline orchestration
│   ├── gpu_buffer_manager.cpp     # GPU memory allocation
│   ├── sirius_context.cpp         # Per-connection subsystem ownership
│   ├── sirius_config.cpp          # Configuration parsing
│   ├── fallback.cpp               # CPU fallback detection
│   ├── cpu_cache.cpp              # CPU caching for warm runs
│   ├── extension_lock.cpp         # Thread-safe extension loading
│   │
│   ├── include/                   # Headers mirroring src/ structure
│   │   ├── sirius_*.hpp           # Core interfaces (engine, context, interface)
│   │   ├── config*.hpp            # Configuration structures
│   │   ├── op/                    # Sirius physical operators (NEW)
│   │   │   ├── sirius_physical_operator.hpp              # Base operator class
│   │   │   ├── sirius_physical_*.hpp                    # 40+ operators
│   │   │   ├── scan/              # Scan operators (DuckDB, Parquet, Iceberg)
│   │   │   ├── aggregate/         # Aggregate operators
│   │   │   ├── order/             # Sort/merge operators
│   │   │   ├── partition/         # Partition/concat operators
│   │   │   ├── merge/             # Merge operators
│   │   │   └── result/            # Result collection
│   │   │
│   │   ├── pipeline/              # Pipeline execution framework
│   │   │   ├── sirius_pipeline.hpp                # Pipeline container
│   │   │   ├── sirius_meta_pipeline.hpp           # Pipeline graph
│   │   │   ├── pipeline_executor.hpp              # Top-level executor
│   │   │   ├── gpu_pipeline_executor.hpp          # Per-GPU task execution
│   │   │   ├── gpu_pipeline_task.hpp              # Task state
│   │   │   └── completion_handler.hpp             # Task completion
│   │   │
│   │   ├── planner/               # Physical plan generation
│   │   │   ├── sirius_physical_plan_generator.hpp # Main planner
│   │   │   └── query.hpp                          # Query context
│   │   │
│   │   ├── creator/               # Task creation
│   │   │   └── task_creator.hpp   # Schedule scan/GPU tasks
│   │   │
│   │   ├── memory/                # GPU memory management
│   │   │   └── sirius_memory_reservation_manager.hpp
│   │   │
│   │   ├── downgrade/             # GPU→Host spilling
│   │   │   └── downgrade_executor.hpp
│   │   │
│   │   ├── data/                  # Data format conversion
│   │   │   └── host_parquet_representation.hpp
│   │   │
│   │   ├── expression_executor/   # GPU expression evaluation
│   │   │   └── gpu_expression_executor.hpp
│   │   │
│   │   ├── helper/                # Utility types
│   │   │   ├── types.hpp
│   │   │   └── utils.hpp
│   │   │
│   │   ├── log/                   # Logging
│   │   │   └── logging.hpp
│   │   │
│   │   └── util/                  # Utilities
│   │       └── segfault_backtrace.hpp
│   │
│   ├── legacy/                    # Legacy Sirius (gpu_processing) - DEPRECATED
│   │   ├── gpu_executor.cpp
│   │   ├── gpu_physical_plan_generator.cpp
│   │   ├── operator/              # Legacy operators
│   │   └── plan/                  # Legacy plan builders
│   │
│   ├── op/                        # Physical operator implementations (NEW)
│   │   ├── sirius_physical_*.cpp  # 40+ operator implementations
│   │   ├── scan/
│   │   │   ├── duckdb_scan_executor.cpp      # DuckDB table scan
│   │   │   ├── duckdb_scan_task.cpp
│   │   │   ├── parquet_scan_executor.cpp     # Parquet file scan
│   │   │   ├── parquet_scan_task.cpp
│   │   │   ├── iceberg_scan_task.cpp         # Iceberg table scan
│   │   │   └── iceberg_metadata_reader.cpp   # Iceberg delete file cache
│   │   ├── aggregate/
│   │   │   └── (aggregate-specific implementations)
│   │   ├── order/
│   │   │   └── (sort/merge implementations)
│   │   ├── partition/
│   │   │   └── (partition/concat implementations)
│   │   ├── merge/
│   │   │   └── (merge operator implementations)
│   │   └── result/
│   │       └── result_collector implementations
│   │
│   ├── planner/                   # Physical plan builders (NEW)
│   │   ├── sirius_physical_plan_generator.cpp  # Main dispatcher
│   │   ├── sirius_plan_*.cpp                   # 15+ operator-specific builders
│   │   └── query.cpp              # Query metadata
│   │
│   ├── pipeline/                  # Pipeline execution (NEW)
│   │   ├── pipeline_executor.cpp
│   │   ├── gpu_pipeline_executor.cpp
│   │   ├── gpu_pipeline_task.cpp
│   │   ├── sirius_pipeline.cpp
│   │   ├── sirius_meta_pipeline.cpp
│   │   └── completion_handler.cpp
│   │
│   ├── creator/                   # Task creation (NEW)
│   │   └── task_creator.cpp
│   │
│   ├── downgrade/                 # GPU→Host spilling (NEW)
│   │   ├── downgrade_executor.cpp
│   │   └── downgrade_task.cpp
│   │
│   ├── memory/                    # Memory management (NEW)
│   │   └── (memory manager implementations)
│   │
│   ├── data/                      # Data format conversion (NEW)
│   │   ├── host_parquet_representation.cpp
│   │   └── host_parquet_representation_converters.cpp
│   │
│   ├── expression_executor/       # Expression evaluation (NEW)
│   │   ├── gpu_expression_executor.cpp
│   │   ├── gpu_expression_translator.cpp
│   │   ├── gpu_expression_executor_state.cpp
│   │   ├── specializations/       # Expression type specializations
│   │   │   ├── gpu_execute_cast.cpp
│   │   │   ├── gpu_execute_comparison.cpp
│   │   │   ├── gpu_execute_conjunction.cpp
│   │   │   ├── gpu_execute_function.cpp
│   │   │   ├── gpu_execute_case.cpp
│   │   │   └── ... (8 more)
│   │   └── regex/
│   │       └── regex_playground.cpp
│   │
│   ├── parallel/                  # Thread pools (NEW)
│   │   └── task_executor.cpp
│   │
│   ├── cuda/                      # CUDA kernel implementations
│   │   ├── expression_executor/   # GPU expression dispatch
│   │   ├── operator/              # Operator-specific kernels
│   │   ├── cudf/                  # cuDF wrapper utilities
│   │   └── iceberg/               # Iceberg-specific kernels
│   │
│   └── util/                      # Utilities
│       ├── segfault_backtrace_handler.cpp
│       └── stream_check_wrapper.cpp
│
├── duckdb/                        # DuckDB submodule (git submodule)
│
├── CMakeLists.txt                 # Main build configuration
├── extension_config.cmake         # Extension-specific config
├── Makefile                       # Build wrapper
│
├── docs/                          # Documentation
│   └── super-sirius/              # Architecture and design docs
│       ├── README.md
│       ├── architecture-overview.md
│       ├── physical-plan-generation.md
│       ├── pipeline-execution.md
│       ├── execution-flow.md
│       ├── operators.md
│       ├── memory-management.md
│       ├── data-management.md
│       ├── task-creator.md
│       ├── scan.md
│       ├── expression-executor.md
│       ├── optimizations.md
│       └── configuration.md
│
├── test/                          # Tests
│   ├── cpp/                       # C++ unit tests
│   │   ├── sirius_unittest        # Catch2 test executable
│   │   └── log/                   # Test output logs
│   ├── sql/                       # SQLLogicTests
│   │   ├── tpch-sirius.test
│   │   └── ... (other .test files)
│   └── tpch_performance/          # Performance benchmarks
│
├── CLAUDE.md                      # This file (project guidelines)
└── .planning/codebase/            # GSD analysis documents
    ├── ARCHITECTURE.md
    ├── STRUCTURE.md
    ├── CONVENTIONS.md
    ├── TESTING.md
    ├── STACK.md
    ├── INTEGRATIONS.md
    └── CONCERNS.md
```

## Directory Purposes

**src/:**
- Contains all Sirius C++ source code
- Mirrors header structure for maintainability
- Build outputs (object files) go to `build/` during compilation

**src/include/:**
- All header files (.hpp)
- Organized by module (op, pipeline, planner, memory, etc.)
- Consumers include this directory with `-I src/include`

**src/op/:**
- Physical operator implementations (NEW Sirius, sirius namespace)
- Organized by operator type (scan, aggregate, order, partition, merge, result)
- Each operator has .hpp in `src/include/op/` and .cpp in `src/op/`

**src/planner/:**
- Physical plan generation from DuckDB logical operators
- `sirius_physical_plan_generator.cpp` is dispatcher
- `sirius_plan_*.cpp` contains operator-specific builders
- Each logical operator type has a corresponding plan builder

**src/pipeline/:**
- Task scheduling and execution framework
- `pipeline_executor` is the top-level orchestrator
- `gpu_pipeline_executor` manages GPU streams per device
- `sirius_pipeline` is the data structure
- `sirius_meta_pipeline` is the dependency graph builder

**src/creator/:**
- `task_creator` thread that watches data repositories
- Follows hint chain to find ready operators
- Creates scan tasks and GPU pipeline tasks

**src/downgrade/:**
- GPU memory pressure relief
- `downgrade_executor` monitors memory, moves data GPU→Host via cuCascade

**src/memory/:**
- GPU/Host/Disk memory tier management
- Wrapper around cuCascade's reservation system

**src/data/:**
- Data format conversions (DuckDB ↔ GPU ↔ Parquet)
- Parquet representation for caching and file I/O

**src/expression_executor/:**
- GPU scalar expression evaluation
- `gpu_expression_executor` translates DuckDB expressions to cuDF
- `specializations/` folder has optimized paths for each expression type

**src/cuda/:**
- CUDA kernel code (.cu files)
- Expression dispatch to cuDF/RMM
- Operator-specific GPU kernels

**src/legacy/:**
- Legacy Sirius (gpu_processing path)
- Deprecated but still present for backward compatibility
- Uses namespace `duckdb` not `sirius`

**docs/super-sirius/:**
- Architecture and design documentation
- **READ BEFORE MODIFYING SUPER SIRIUS**
- Covers execution flow, operators, pipeline splitting, memory management, task creation

**test/cpp/:**
- C++ unit tests using Catch2
- Organized by component (test_cpu_cache, test_exchange, etc.)
- Run with: `build/release/extension/sirius/test/cpp/sirius_unittest`

**test/sql/:**
- SQLLogicTests for end-to-end validation
- Run with: `build/release/test/unittest --test-dir . test/sql/tpch-sirius.test`

## Key File Locations

**Entry Points:**
- `src/sirius_extension.cpp` — DuckDB extension loader, registers `gpu_execution` table function
- `src/sirius_interface.cpp` — Query lifecycle (begin, execute, fetch, end)
- `src/sirius_engine.cpp` — Pipeline construction and orchestration

**Configuration:**
- `src/sirius_config.cpp` — Runtime config from file or environment
- `src/sirius_context.cpp` — Subsystem initialization and lifetime management
- `src/include/config.hpp` — Config macros and defaults

**Core Logic:**
- `src/planner/sirius_physical_plan_generator.cpp` — Logical→Physical operator conversion
- `src/pipeline/pipeline_executor.cpp` — Task scheduling and completion
- `src/creator/task_creator.cpp` — Scan/GPU task creation

**Memory Management:**
- `src/gpu_buffer_manager.cpp` — GPU buffer initialization via `gpu_buffer_init()`
- `src/include/memory/sirius_memory_reservation_manager.hpp` — Reservation system
- `src/downgrade/downgrade_executor.cpp` — GPU→Host spilling

**Testing:**
- `src/sirius_extension.cpp` contains CPU cache test hooks
- Test harness: `test/cpp/` with Catch2 framework
- SQL tests: `test/sql/tpch-sirius.test` (SQLLogicTest format)

## Naming Conventions

**Files:**
- `sirius_*.cpp` / `sirius_*.hpp` — Sirius-specific implementations
- `gpu_*.cpp` / `gpu_*.hpp` — GPU-related (legacy or dual-path code)
- `sirius_physical_*.hpp` — Physical operator headers in `src/include/op/`
- `sirius_plan_*.cpp` — Physical plan builder in `src/planner/`
- Test files: `*_test.cpp` (unit tests), `*.test` (SQLLogicTests)

**Directories:**
- `src/op/` — Physical operator implementations
- `src/planner/` — Plan generation
- `src/pipeline/` — Pipeline execution
- `src/creator/` — Task creation
- `src/downgrade/` — Memory spilling
- `src/memory/` — Memory management
- `src/data/` — Data format conversion
- `src/expression_executor/` — Expression evaluation
- `src/legacy/` — Deprecated code (gpu_processing)
- `src/include/op/scan/` — Scan operators (DuckDB, Parquet, Iceberg)

**Symbols:**
- Classes: `sirius_*` (e.g., `sirius_physical_hash_join`, `sirius_pipeline`, `sirius_engine`)
- Enums: `SiriusPhysicalOperatorType`, `MemoryBarrierType`, `TaskCreationHint`
- Namespaces: `sirius`, `sirius::op`, `sirius::pipeline`, `sirius::planner`, `sirius::creator`, `sirius::memory`
- Legacy namespace: `duckdb` (for gpu_processing)

## Where to Add New Code

**New Operator:**
1. Header: `src/include/op/sirius_physical_YOUR_OP.hpp` (inherit from `sirius_physical_operator`)
2. Implementation: `src/op/sirius_physical_YOUR_OP.cpp` (implement `execute()`, optionally `sink()`)
3. Planner: `src/planner/sirius_plan_YOUR_OP.cpp` (create operator from logical plan)
4. Dispatch: Add case to `sirius_physical_plan_generator::create_plan()` switch statement
5. Type enum: Add to `SiriusPhysicalOperatorType` in `src/include/op/sirius_physical_operator_type.hpp`
6. Metadata: Add to `get_name()`, `to_string()` in `src/op/sirius_physical_operator_type.cpp`
7. Tests: Unit tests in `test/cpp/operator/`, SQL tests in `test/sql/`

**New Expression Specialization:**
1. Header (if needed): `src/include/expression_executor/gpu_expression_executor.hpp`
2. Implementation: `src/expression_executor/specializations/gpu_execute_YOUR_EXPR.cpp`
3. Dispatcher: Add branch to `gpu_expression_executor::execute()` for new expression type
4. CUDA kernel (if needed): `src/cuda/expression_executor/YOUR_EXPR.cu`
5. Tests: Unit tests in `test/cpp/expression_executor/`

**New Scan Type:**
1. Operator header: `src/include/op/sirius_physical_YOUR_SCAN.hpp`
2. Operator impl: `src/op/sirius_physical_YOUR_SCAN.cpp`
3. Scan executor: `src/include/op/scan/YOUR_scan_executor.hpp` + `src/op/scan/YOUR_scan_executor.cpp`
4. Scan task: `src/include/op/scan/YOUR_scan_task.hpp` + `src/op/scan/YOUR_scan_task.cpp`
5. Task creator binding: Add to `task_creator::prepare_for_query()` switch
6. Planner: Create plan builder in `src/planner/sirius_plan_get.cpp` or dedicated file
7. Tests: In `test/sql/` with appropriate test data

**New Configuration Option:**
1. Header: Add to `src/include/sirius_config.hpp` (in `sirius_config` class)
2. Parser: Add parsing logic to `src/sirius_config.cpp` (in `parse_config_file()`)
3. Getter: Add accessor method to `sirius_config`
4. Usage: Pass to subsystems during initialization in `SiriusContext::initialize()`
5. Docs: Document in `docs/super-sirius/configuration.md`

**New Test:**
- Unit test: `test/cpp/YOUR_TEST.cpp` (Catch2 framework, use TEST_CASE macro)
- SQL test: `test/sql/YOUR_QUERIES.test` (SQLLogicTest format)
- Performance test: `test/tpch_performance/YOUR_BENCH.py` (Python script)
- Register in CMakeLists.txt or run harness

## Special Directories

**docs/super-sirius/:**
- Purpose: Authoritative design documentation
- Generated: No (hand-written)
- Committed: Yes (essential for contributors)
- Read: **Before modifying core architecture**

**test/cpp/log/:**
- Purpose: Test execution logs and crash dumps
- Generated: Yes (during test runs)
- Committed: No (`.gitignore`d)
- Cleanup: `rm -rf test/cpp/log` before committing

**build/:**
- Purpose: Compilation outputs
- Generated: Yes (by CMake)
- Committed: No (`.gitignore`d)
- Location: `build/release/extension/sirius/` for extension, `build/release/test/` for tests

**duckdb/:**
- Purpose: DuckDB source (git submodule)
- Generated: No
- Committed: Yes (via submodule reference)
- Frozen: Changes here affect Sirius compatibility

**.planning/codebase/:**
- Purpose: GSD analysis documents (this collection)
- Generated: Yes (by `/gsd-map-codebase`)
- Committed: Yes (shared with future Claude instances)
- Readonly: Documents guide implementation, not updated by PR

---

*Structure analysis: 2026-04-06*
