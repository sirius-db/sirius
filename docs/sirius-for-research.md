# Sirius for Research

## Why use Sirius as your research platform

1. **Low barrier to entry** — Sirius bypasses the need to build a SQL frontend from scratch by leveraging the host system's parser and optimizer (e.g., DuckDB). This allows researchers to focus immediately on core innovations in the operators, execution engine, or buffer management rather than building the platform first.

2. **Modular operator design** — Implementations are highly interchangeable, allowing for a seamless transition between libcudf primitives and hand-tuned CUDA kernels, and supporting different compression schemes. This flexibility facilitates research on individual modules as well as system architecture.

3. **State-of-the-art benchmarks** — As of August 2026, Sirius leads ClickBench hot-run results, so a speedup measured on Sirius is a speedup over the state of the art. Sirius is integrated into DuckDB by default, so both systems serve as strong baselines that are easy to test against.

4. **Permissive and open licensing** — The Apache 2.0 license lets you publish results, release artifacts for reproducibility, and fork the architecture without legal friction, in either academic or industrial settings.

5. **Active collaborative ecosystem** — Joint development between NVIDIA and UW–Madison, alongside various external contributors, provides a professional review pipeline. Good contributions get merged into the main Sirius branch rather than stagnating in isolated forks.

## Open research problems in GPU databases

1. **GPU-native query optimization** — Sirius inherits plans from a host optimizer built around CPU cost models, so join orders and materialization choices are tuned for the wrong hardware. Open research questions include GPU-aware cost models and GPU-native query optimization.

2. **Efficient transaction support on GPU** — Transactions are latency-sensitive and synchronization-heavy, which is not what GPUs were originally designed for. They can either be supported natively on the GPU, or handled with an HTAP approach in which the CPU processes transactions and the GPU processes analytics.

3. **Efficient window function support** — Window functions are ubiquitous in analytical SQL, yet remain hard to run well on GPU. Frame evaluation carries sequential dependencies that must be recast as segmented scans, and partition-size skew breaks naive parallelization.

4. **Nested and semi-structured types** — LIST, STRUCT, and JSON break the fixed-stride assumptions GPU kernels rely on, making them hard for reasons distinct from flat relational operators.

5. **Vector search** — Analytical systems are increasingly expected to serve vector search alongside SQL. GPUs are a natural fit for the distance computations involved, but how to best support hybrid vector and classic SQL workloads on GPU remains an open question.

6. **UDF support and GPU-native UDFs** — UDFs are the most common reason a query gets stranded on CPU, and a single opaque UDF mid-pipeline can force a round trip that erases the GPU's advantage. The open problem is UDFs that fuse into surrounding operators rather than acting as barriers.

7. **GPU-optimized relational operators** — Relational operators are the building blocks of a query engine, and better performance is always desirable. Joins, group-by, sorting, aggregation, top-K, window functions, and string operators all reward optimization specifically for GPU hardware.

8. **Code generation and JIT compilation** — Fusing query fragments into single kernels produced huge gains on ClickBench q28, making it one of the largest demonstrated wins. Open questions are fusion granularity, amortizing compile latency on short queries, and how far fusion can go beyond individual operators.

9. **GPU-optimized shuffle and exchange** — Data exchange is critical for distributed GPU queries, making it a major bottleneck in scale-out. Existing libraries such as NCCL and NIXL were designed for ML workloads. Optimizing the communication layer for distributed GPU query processing is an open question.

10. **Concurrency and multi-tenancy** — Concurrency is essential for throughput and utilization, but multiple queries sharing one GPU compete for the same scarce device memory. Both mechanisms (memory reservation and isolation, admission control, spilling, preemption) and policies (how much memory each query gets, when to admit, whom to evict) remain open.

11. **Efficient out-of-core query execution** — GPU memory remains the binding constraint, so batching, partitioning, and spilling mechanisms/policies determine whether large workloads run at all. A key question is whether conventional CPU buffer management works in the GPU setting and what new mechanisms need to be invented.

12. **GPU data caching** — Caching in Sirius today is static and manual: the user decides up front what to pin on the GPU. Dynamic caching, along with the memory management it requires, is the open problem — admission and eviction policy, caching granularity, and placement across the memory hierarchy, etc. The format of cached data is also open: it can be raw bytes, compressed or encoded, pre-sorted or pre-partitioned, or even materialized results such as hash tables and query intermediates.

13. **GPU memory fragmentation** — GPU allocations reserve contiguous physical memory with no paging or virtual-memory remapping to fall back on, so fragmentation is far more damaging than on CPU: a query can fail on a large allocation despite ample free memory in aggregate.

14. **GPU-native compression and decompression** — Both table input data and query intermediate results need efficient compression and decompression on the GPU. Modern GPUs also ship dedicated hardware decompression engines, and how best to drive and optimize them for database workloads is an open question. Another opportunity is to operate directly on encoded data to avoid decompression altogether.

15. **GPU-native I/O optimization** — Data still largely reaches the GPU through a CPU-mediated scan, and this is where the cold-run gap lives. GPU-native Parquet decoding, GPUDirect Storage, plan-driven prefetching, and S3-over-RDMA are all directions worth pursuing.

16. **Computation pushdown** — I/O is a persistent bottleneck, and one way to shrink it is to push computation to where the data already lives: filters and projections into the CPU-side scan, into the SSD via computational storage, into the network or DPU, or into cloud object storage. Open questions are which operators are worth pushing down, how the optimizer decides given a GPU-aware cost model, and how to express pushdown uniformly across such heterogeneous targets.
