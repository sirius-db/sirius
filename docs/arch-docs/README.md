# Sirius Onboarding Documentation

Welcome to the Sirius onboarding documentation! This guide will help you understand Sirius, a GPU-native SQL execution engine that plugs into DuckDB.

## What is Sirius?

Sirius is a high-performance SQL query execution engine that leverages GPU acceleration to process analytical queries. It integrates with DuckDB as an extension, allowing queries to be offloaded to the GPU for faster execution on large datasets.

**Key Features:**
- GPU-accelerated query execution using CUDA and cuDF
- Seamless integration with DuckDB
- Two execution modes: Legacy (`gpu_processing`) and Modern (`gpu_execution`)
- Pipeline-based execution model
- Advanced memory management across GPU/HOST/DISK tiers

## Quick Navigation

### Getting Started
- [What is Sirius?](01-overview/what-is-sirius.md) - High-level introduction
- [Quick Start Guide](01-overview/quick-start.md) - Get up and running
- [Key Concepts](01-overview/key-concepts.md) - Core terminology and concepts

### Architecture
- [System Overview](02-architecture/system-overview.md) - Overall architecture
- [DuckDB Integration](02-architecture/duckdb-integration.md) - How Sirius plugs into DuckDB
- [Execution Modes](02-architecture/execution-modes.md) - Legacy vs New mode comparison

### Legacy Mode (gpu_processing)
- [Overview](03-legacy-mode/overview.md) - Introduction to legacy execution
- [Entry Points](03-legacy-mode/entry-points.md) - `gpu_processing` table function
- [Operators](03-legacy-mode/operators.md) - GPUPhysicalOperator implementations
- [Pipeline Execution](03-legacy-mode/pipeline-execution.md) - Pipeline structure and DAG
- [Memory Management](03-legacy-mode/memory-management.md) - GPUBufferManager and allocation
- [Data Structures](03-legacy-mode/data-structures.md) - GPUColumn, GPUIntermediateRelation
- [Architecture Diagram](03-legacy-mode/architecture-diagram.md) - Visual reference

### New Mode (gpu_execution)
- [Overview](04-new-mode/overview.md) - Introduction to modern execution
- [Entry Points](04-new-mode/entry-points.md) - `gpu_execution` and sirius_interface
- [Operators](04-new-mode/operators.md) - sirius_physical_operator base class
- [Cucascade Integration](04-new-mode/cucascade-integration.md) - Data repository and memory
- [Pipeline Execution](04-new-mode/pipeline-execution.md) - Pipeline and task model
- [Task Creation](04-new-mode/task-creation.md) - Hint-based task generation
- [Operator Guide](04-new-mode/operator-guide.md) - Detailed operator reference
- [Architecture Diagram](04-new-mode/architecture-diagram.md) - Visual reference

### Core Components
- [Planner](05-core-components/planner.md) - Physical plan generation
- [Configuration](05-core-components/configuration.md) - Config system
- [Memory Management](05-core-components/memory-management.md) - Multi-tier memory hierarchy
- [Threading Model](05-core-components/threading-model.md) - Task executors and thread pools
- [Expression Executor](05-core-components/expression-executor.md) - Expression evaluation
- [Result Collection](05-core-components/result-collection.md) - Result materialization

### Data Flow
- [Query Lifecycle](06-data-flow/query-lifecycle.md) - End-to-end query execution
- [Legacy Data Flow](06-data-flow/legacy-data-flow.md) - Data flow in legacy mode
- [New Data Flow](06-data-flow/new-data-flow.md) - Data flow in new mode
- [Inter-Pipeline Communication](06-data-flow/inter-pipeline-communication.md) - Data repositories

### Development Guide
- [Building and Testing](07-development/building-and-testing.md) - Build system and tests
- [Debugging](07-development/debugging.md) - Logging and debugging tips
- [Adding Operators](07-development/adding-operators.md) - How to add new operators
- [Testing Guide](07-development/testing-guide.md) - Unit and integration tests
- [Code Organization](07-development/code-organization.md) - Directory structure

### Reference
- [Glossary](08-reference/glossary.md) - Terms and definitions
- [API Reference](08-reference/api-reference.md) - Key classes and methods
- [File Index](08-reference/file-index.md) - Important files by category
- [Config Options](08-reference/config-options.md) - Configuration parameters

### Appendices
- [Performance Tips](appendices/performance-tips.md) - Performance tuning
- [Limitations](appendices/limitations.md) - Current limitations
- [Roadmap](appendices/roadmap.md) - Future plans
- [Comparison to Other Systems](appendices/comparison-to-other-systems.md) - How Sirius differs

## Documentation Reading Paths

### For New Team Members (All Backgrounds)
1. Start with [What is Sirius?](01-overview/what-is-sirius.md)
2. Read [System Overview](02-architecture/system-overview.md)
3. Understand [Execution Modes](02-architecture/execution-modes.md)
4. Choose your focus:
   - Working with legacy code? → Legacy Mode section
   - Working with new features? → New Mode section
5. Review [Core Components](05-core-components/planner.md)
6. Explore [Development Guide](07-development/building-and-testing.md)

### For Seasoned DB/GPU Engineers
**Fast Track** (2-3 hours):
1. [System Overview](02-architecture/system-overview.md) - 15 min
2. [Execution Modes](02-architecture/execution-modes.md) - 20 min
3. [New Mode Architecture Diagram](04-new-mode/architecture-diagram.md) - 10 min
4. [Cucascade Integration](04-new-mode/cucascade-integration.md) - 30 min
5. [Pipeline Execution](04-new-mode/pipeline-execution.md) - 30 min
6. [Planner](05-core-components/planner.md) - 20 min
7. [Memory Management](05-core-components/memory-management.md) - 20 min
8. [Query Lifecycle](06-data-flow/query-lifecycle.md) - 30 min

### For Grad Students / Junior Engineers
**Thorough Path** (1-2 days):
1. [What is Sirius?](01-overview/what-is-sirius.md) - understand the big picture
2. [Key Concepts](01-overview/key-concepts.md) - learn terminology
3. [Quick Start Guide](01-overview/quick-start.md) - hands-on experience
4. [System Overview](02-architecture/system-overview.md) - architecture basics
5. [DuckDB Integration](02-architecture/duckdb-integration.md) - understand integration
6. [New Mode Overview](04-new-mode/overview.md) - modern execution path
7. Work through each New Mode document in order
8. Explore [Development Guide](07-development/building-and-testing.md)
9. Reference [Glossary](08-reference/glossary.md) as needed

### For Debugging Specific Issues
1. Check [File Index](08-reference/file-index.md) to locate relevant code
2. Review specific operator documentation in [Operator Guide](04-new-mode/operator-guide.md)
3. Consult [Debugging](07-development/debugging.md) for logging and tools
4. Review [Data Flow](06-data-flow/query-lifecycle.md) to understand execution path

## Repository Structure

The Sirius codebase is located at `/home/roaramburu/coding/sirius/`. Key directories:

```
sirius/
├── src/
│   ├── include/           # Header files
│   │   ├── op/           # New mode operators
│   │   ├── pipeline/     # Pipeline infrastructure
│   │   ├── memory/       # Memory management
│   │   └── parallel/     # Threading and task execution
│   ├── op/               # New mode operator implementations
│   ├── operator/         # Legacy mode operator implementations
│   ├── planner/          # Physical plan generation
│   ├── sirius_extension.cpp  # DuckDB extension entry point
│   ├── sirius_engine.cpp     # New mode execution engine
│   └── gpu_executor.cpp      # Legacy mode executor
├── test/
│   ├── cpp/              # C++ unit tests
│   └── sql/              # SQL integration tests
└── cucascade/            # Submodule for data management
```

## Contributing to Documentation

This documentation is maintained separately from the main Sirius repository to allow easy editing and potential migration to Google Docs. When making changes:

1. **Keep it modular**: Each document should be self-contained but link to related topics
2. **Include code references**: Always reference specific files with line numbers
3. **Add examples**: Use concrete code snippets and SQL queries
4. **Target both audiences**: Include high-level summaries for experienced engineers and detailed explanations for newcomers
5. **Update diagrams**: Keep Mermaid diagrams in sync with code changes
6. **Cross-reference**: Link liberally between related documents

## Getting Help

- Check the [Glossary](08-reference/glossary.md) for term definitions
- Review [API Reference](08-reference/api-reference.md) for class/method documentation
- Consult [Debugging Guide](07-development/debugging.md) for troubleshooting
- Ask the team on Slack (if applicable)

## Version Information

- **Sirius Version**: Development (as of 2026-02)
- **DuckDB Version**: Compatible with DuckDB extension API
- **CUDA Version**: 11.x or higher
- **cuDF Version**: Compatible with RMM memory management

---

**Next Steps**: Start with [What is Sirius?](01-overview/what-is-sirius.md) or jump to your area of interest using the navigation above.
