# Roadmap

Future development plans and priorities for Sirius. This roadmap is subject to change based on user needs and technical constraints.

## Current Status

**Version**: Development (Pre-1.0)
**Active Development**: New Mode (`gpu_execution`)
**Maintenance**: Legacy Mode (`gpu_processing`)

---

## Short Term (Next 3-6 Months)

### 1. Complete New Mode Feature Parity

**Goal**: New Mode supports all Legacy Mode operators

**Tasks**:
- [x] Basic operators (SCAN, FILTER, PROJECT)
- [x] Aggregates (HASH_GROUP_BY, UNGROUPED_AGGREGATE)
- [x] Joins (HASH_JOIN)
- [x] Sorts (ORDER_BY, TOP_N)
- [ ] Window functions (comprehensive support)
- [ ] Advanced aggregates (PERCENTILE, MODE, MEDIAN)
- [ ] Set operations (UNION, INTERSECT, EXCEPT)

**Status**: 80% complete

### 2. Improve Error Messages and Debugging

**Goal**: Make Sirius easier to debug and understand

**Tasks**:
- [ ] Better error messages with context
- [ ] Query plan visualization tool
- [ ] Enhanced logging with structured output
- [ ] Built-in profiling dashboard
- [ ] Memory usage tracking and reporting

**Status**: 20% complete

### 3. Performance Optimizations

**Goal**: 2x speedup over current New Mode

**Tasks**:
- [ ] Operator fusion (combine adjacent operators into single kernel)
- [ ] Adaptive batch sizing
- [ ] Better memory prefetching
- [ ] CUDA graph optimization
- [ ] Operator-specific GPU kernel tuning

**Status**: 30% complete

### 4. Testing and Validation

**Goal**: Production-ready quality

**Tasks**:
- [ ] Expand SQL test suite (>5000 tests)
- [ ] TPC-H all 22 queries passing
- [ ] TPC-DS support
- [ ] Fuzzing infrastructure
- [ ] Correctness validation framework

**Status**: 60% complete

---

## Medium Term (6-12 Months)

### 1. Multi-GPU Support

**Goal**: Scale beyond single GPU

**Architecture**:
```
Query
  ↓
Query Partitioning
  ↓
┌────────┬────────┬────────┐
│ GPU 0  │ GPU 1  │ GPU 2  │
└────────┴────────┴────────┘
  ↓        ↓        ↓
Results Aggregation
  ↓
Final Result
```

**Tasks**:
- [ ] Multi-GPU query partitioning
- [ ] Data distribution strategies
- [ ] Inter-GPU communication (NVLink)
- [ ] Load balancing across GPUs
- [ ] Fault tolerance

**Benefits**:
- Process larger datasets
- Higher throughput
- Better resource utilization

**Status**: 0% complete (design phase)

### 2. Advanced SQL Features

**Goal**: Support complex analytical queries

**Tasks**:
- [ ] Recursive CTEs
- [ ] Full window function support
- [ ] Complex subqueries (correlated, EXISTS)
- [ ] Lateral joins
- [ ] PIVOT/UNPIVOT operations

**Status**: 10% complete

### 3. Data Type Expansion

**Goal**: Support more DuckDB data types

**Tasks**:
- [ ] Complex types (LIST, STRUCT, MAP)
- [ ] JSON support
- [ ] Full DECIMAL precision
- [ ] TIME type
- [ ] INTERVAL type

**Status**: 5% complete

### 4. Improved Memory Management

**Goal**: More efficient memory usage

**Tasks**:
- [ ] Smarter eviction policies (LRU, LFU)
- [ ] Compression in host memory
- [ ] Direct GPU ↔ Disk transfers (GPUDirect Storage)
- [ ] Memory pooling optimizations
- [ ] Per-query memory budgets

**Status**: 20% complete

---

## Long Term (1-2 Years)

### 1. Distributed Execution

**Goal**: Scale to clusters of GPUs

**Architecture**:
```
Coordinator Node
       ↓
┌──────────────────────────────┐
│  Worker Node 1   Worker Node 2│
│  ┌─────┬─────┐  ┌─────┬─────┐│
│  │GPU 0│GPU 1│  │GPU 0│GPU 1││
│  └─────┴─────┘  └─────┴─────┘│
└──────────────────────────────┘
```

**Tasks**:
- [ ] Distributed query planning
- [ ] Network-efficient data shuffle
- [ ] Fault tolerance and recovery
- [ ] Dynamic resource allocation
- [ ] Query scheduling across cluster

**Benefits**:
- Petabyte-scale datasets
- High availability
- Elastic scaling

**Status**: 0% complete (research phase)

### 2. Query Optimizer

**Goal**: Cost-based optimization for GPU execution

**Tasks**:
- [ ] GPU-specific cost model
- [ ] Statistics collection
- [ ] Cardinality estimation
- [ ] Join order optimization
- [ ] Operator selection (hash vs nested loop)

**Status**: 5% complete

### 3. Integration with Data Formats

**Goal**: Native support for popular formats

**Tasks**:
- [ ] Native Parquet reader (GPU-direct)
- [ ] Apache Arrow integration
- [ ] ORC format support
- [ ] Delta Lake support
- [ ] Iceberg support

**Status**: 10% complete (basic Parquet)

### 4. Machine Learning Integration

**Goal**: Seamless ML on GPU data

**Tasks**:
- [ ] Integration with cuML (GPU ML library)
- [ ] In-database feature engineering
- [ ] Model inference within queries
- [ ] AutoML capabilities
- [ ] Vector similarity search

**Status**: 0% complete

---

## Experimental Features

Features under active research:

### 1. Adaptive Query Execution

Dynamically adjust execution based on runtime statistics:
- Switch join algorithms mid-execution
- Adjust parallelism based on data skew
- Re-partition data based on observed patterns

**Status**: Prototype

### 2. Heterogeneous Execution

**Goal**: Intelligently use CPU + GPU

- Route operators to CPU or GPU based on characteristics
- Overlap CPU and GPU execution
- Dynamic workload balancing

**Status**: Early research

### 3. Approximate Query Processing

**Goal**: Fast approximate answers

- Sampling-based aggregates
- Approximate join results
- Confidence intervals
- Progressive refinement

**Status**: Design phase

### 4. Persistent GPU Data

**Goal**: Keep hot data on GPU across queries

- GPU-resident tables
- Cached query results
- Pre-computed aggregates
- Incremental updates

**Status**: Concept

---

## Deprecation Plans

### Legacy Mode (`gpu_processing`)

**Timeline**:
- **Now - 6 months**: Maintenance mode (bug fixes only)
- **6-12 months**: Deprecation warnings
- **12-18 months**: Removal from main branch
- **18+ months**: Legacy branch for compatibility

**Migration Path**:
- All queries should use `gpu_execution()`
- Documentation updated to show New Mode only
- Migration guide for legacy users

---

## Community Requests

Features requested by users (prioritized by demand):

### High Priority
1. **Windows Support** - Many users on Windows
2. **Multi-GPU** - Scaling beyond single GPU
3. **Complex Types** - LIST, STRUCT, JSON support
4. **Better Error Messages** - Easier debugging

### Medium Priority
1. **ARM Support** - Grace Hopper, Jetson platforms
2. **AMD GPU Support** - Via ROCm (significant effort)
3. **Python UDFs** - Custom functions in Python
4. **Query Caching** - Reuse compiled plans

### Lower Priority
1. **GUI Tool** - Visual query builder
2. **Cloud Integration** - AWS, Azure, GCP optimizations
3. **Monitoring Dashboard** - Real-time metrics
4. **Auto-tuning** - Automatic configuration

---

## Performance Targets

### Current Performance (TPC-H SF10 on A100)

| Query | GPU Time | CPU Time | Speedup |
|-------|----------|----------|---------|
| Q1 | 0.62s | 1.2s | 1.9x |
| Q3 | 0.78s | 2.1s | 2.7x |
| Q6 | 0.28s | 0.8s | 2.9x |
| Q9 | 1.35s | 4.2s | 3.1x |

### 6-Month Targets

| Query | Target GPU Time | Target Speedup vs CPU |
|-------|-----------------|----------------------|
| Q1 | 0.4s | 3.0x |
| Q3 | 0.5s | 4.2x |
| Q6 | 0.15s | 5.3x |
| Q9 | 0.8s | 5.3x |

### 12-Month Targets

| Query | Target GPU Time | Target Speedup vs CPU |
|-------|-----------------|----------------------|
| Q1 | 0.3s | 4.0x |
| Q3 | 0.35s | 6.0x |
| Q6 | 0.1s | 8.0x |
| Q9 | 0.5s | 8.4x |

---

## Technical Debt

Items to address:

### High Priority
1. **Code duplication** - Between legacy and new modes
2. **Test coverage** - Expand to 90%+
3. **Documentation** - Complete all sections
4. **Error handling** - More robust error recovery

### Medium Priority
1. **Refactoring** - Simplify pipeline builder
2. **Performance** - Profile and optimize hot paths
3. **Memory leaks** - Audit with valgrind/cuda-memcheck
4. **Type system** - Unify type handling

### Lower Priority
1. **Build system** - Simplify dependencies
2. **CI/CD** - Faster builds
3. **Code style** - Enforce formatting
4. **Naming** - Consistent conventions

---

## Contributing

Want to contribute? Here are priority areas:

### For Database Engineers
- Implement missing operators
- Optimize query planner
- Add SQL feature support
- Performance tuning

### For GPU Engineers
- Write CUDA kernels
- Optimize memory transfers
- Multi-GPU infrastructure
- Profile and optimize

### For ML Engineers
- ML integration
- Feature engineering operators
- Model inference
- Approximate query processing

### For Systems Engineers
- Distributed execution
- Fault tolerance
- Monitoring and observability
- Cloud integration

---

## Release Schedule

### Version 0.9 (Next 3 months)
- New Mode feature complete
- TPC-H all queries passing
- Production-ready quality

### Version 1.0 (6 months)
- Stable API
- Comprehensive documentation
- Performance targets met
- Enterprise support

### Version 1.5 (12 months)
- Multi-GPU support
- Advanced SQL features
- Distributed execution (beta)

### Version 2.0 (18-24 months)
- Full distributed execution
- Cloud-native features
- ML integration
- Query optimizer

---

## How to Influence the Roadmap

1. **File Issues**: Report bugs, request features on GitHub
2. **Vote**: Upvote existing feature requests
3. **Contribute**: Submit PRs for priority items
4. **Discuss**: Join community discussions
5. **Sponsor**: Prioritize specific features

---

## Related Documents

- [Limitations](limitations.md) - Current limitations to be addressed
- [Performance Tips](performance-tips.md) - Optimize current version
- [System Overview](../02-architecture/system-overview.md) - Architecture context
- [Comparison to Other Systems](comparison-to-other-systems.md) - Competitive landscape

---

**Note**: This roadmap is aspirational and subject to change. Actual development priorities may shift based on user feedback, technical constraints, and resource availability.

**Last Updated**: February 2026
