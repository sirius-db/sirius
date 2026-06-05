// Standalone multi-GPU Host->Device bandwidth probe for GB200 (2 Grace + 4 Blackwell).
//
// Goal: measure the ACHIEVABLE H2D ceiling and how it scales across GPUs, to explain why
// the q9 scan staging only reaches ~113 GB/s wall-aggregate. Mirrors cuCascade's copy method
// (cudaMemcpyAsync H2D on a per-worker stream + synchronize) but adds explicit per-GPU device
// selection and NUMA-local pinned-host placement.
//
// Topology on this box (from nvidia-smi topo / numactl):
//   GPU0,GPU1 -> Grace node 0 LPDDR (cpus 0-71)
//   GPU2,GPU3 -> Grace node 1 LPDDR (cpus 72-143)
// So GPU0+GPU1 share node-0 memory bandwidth; GPU2+GPU3 share node-1.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=native h2d_multigpu_bw.cu -o /tmp/h2d_bw -lnuma -lpthread
// Run:
//   /tmp/h2d_bw [chunk_MiB] [per_gpu_MiB] [iters]

#include <cuda_runtime.h>

#include <numa.h>
#include <numaif.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                      \
  do {                                                                                             \
    cudaError_t e = (x);                                                                           \
    if (e != cudaSuccess) {                                                                        \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      std::abort();                                                                                \
    }                                                                                              \
  } while (0)

// Host Grace NUMA node feeding each GPU (node 0 -> GPU0/1, node 1 -> GPU2/3).
static int host_node_for_gpu(int gpu) { return (gpu <= 1) ? 0 : 1; }
// A CPU on that Grace node to run the issuing thread on.
static int cpu_node_for_host(int node) { return node; }  // cpu nodes 0 and 1 are the Grace dies

struct Worker {
  int gpu;
  int host_node;
  void* d_buf{nullptr};
  void* h_buf{nullptr};
  uint64_t bytes_per_gpu;
  uint64_t chunk;
  int iters;
  double elapsed_s{0.0};  // measured wall on this worker between barrier and last sync
};

static std::atomic<int> g_ready{0};
static std::atomic<int> g_go{0};

static void run_worker(Worker* w, int nworkers)
{
  // Pin issuing thread to the host Grace node so the driver copy is issued locally.
  struct bitmask* mask = numa_allocate_cpumask();
  numa_node_to_cpus(cpu_node_for_host(w->host_node), mask);
  numa_sched_setaffinity(0, mask);
  numa_free_cpumask(mask);

  CK(cudaSetDevice(w->gpu));

  // Device buffer on this GPU.
  CK(cudaMalloc(&w->d_buf, w->bytes_per_gpu));

  // Pinned host buffer placed on the chosen Grace node: numa_alloc_onnode + register.
  w->h_buf = numa_alloc_onnode(w->bytes_per_gpu, w->host_node);
  if (!w->h_buf) {
    std::fprintf(stderr, "numa_alloc_onnode failed (gpu %d node %d)\n", w->gpu, w->host_node);
    std::abort();
  }
  std::memset(w->h_buf, 0x42, w->bytes_per_gpu);  // fault pages onto the node
  CK(cudaHostRegister(w->h_buf, w->bytes_per_gpu, cudaHostRegisterPortable));

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));

  // Warmup (also establishes the H2D path).
  CK(cudaMemcpyAsync(w->d_buf, w->h_buf, w->bytes_per_gpu, cudaMemcpyHostToDevice, stream));
  CK(cudaStreamSynchronize(stream));

  // Signal ready; wait for global go so all workers start together.
  g_ready.fetch_add(1);
  while (g_go.load() == 0) { /* spin */
  }

  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0, stream));
  for (int it = 0; it < w->iters; ++it) {
    uint64_t off = 0;
    while (off < w->bytes_per_gpu) {
      uint64_t n = std::min(w->chunk, w->bytes_per_gpu - off);
      CK(cudaMemcpyAsync(static_cast<char*>(w->d_buf) + off,
                         static_cast<char*>(w->h_buf) + off,
                         n,
                         cudaMemcpyHostToDevice,
                         stream));
      off += n;
    }
  }
  CK(cudaEventRecord(t1, stream));
  CK(cudaStreamSynchronize(stream));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, t0, t1));
  w->elapsed_s = ms / 1000.0;

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaStreamDestroy(stream);
  cudaHostUnregister(w->h_buf);
  numa_free(w->h_buf, w->bytes_per_gpu);
  CK(cudaFree(w->d_buf));
}

// Run one configuration: a set of (gpu, host_node) workers concurrently, report aggregate BW.
static void run_config(const std::string& name,
                       const std::vector<std::pair<int, int>>& gpus_nodes,
                       uint64_t bytes_per_gpu,
                       uint64_t chunk,
                       int iters)
{
  int n = static_cast<int>(gpus_nodes.size());
  std::vector<Worker> ws(n);
  for (int i = 0; i < n; ++i) {
    ws[i].gpu           = gpus_nodes[i].first;
    ws[i].host_node     = gpus_nodes[i].second;
    ws[i].bytes_per_gpu = bytes_per_gpu;
    ws[i].chunk         = chunk;
    ws[i].iters         = iters;
  }
  g_ready.store(0);
  g_go.store(0);

  std::vector<std::thread> threads;
  for (int i = 0; i < n; ++i)
    threads.emplace_back(run_worker, &ws[i], n);
  while (g_ready.load() < n) { /* wait for all set up */
  }
  g_go.store(1);
  for (auto& t : threads)
    t.join();

  // Aggregate: total bytes / max worker elapsed (they ran concurrently from the same go).
  double max_s       = 0.0;
  double total_bytes = 0.0;
  double sum_per_gpu = 0.0;
  for (auto& w : ws) {
    double gb = static_cast<double>(w.bytes_per_gpu) * w.iters;
    total_bytes += gb;
    max_s = std::max(max_s, w.elapsed_s);
    sum_per_gpu += gb / w.elapsed_s / 1e9;
  }
  double agg = total_bytes / max_s / 1e9;
  std::printf("%-34s  gpus=%d  agg=%7.1f GB/s   sum_per_gpu=%7.1f GB/s   (per-gpu avg=%6.1f)\n",
              name.c_str(),
              n,
              agg,
              sum_per_gpu,
              sum_per_gpu / n);
}

int main(int argc, char** argv)
{
  if (numa_available() < 0) {
    std::fprintf(stderr, "NUMA not available\n");
    return 1;
  }
  uint64_t chunk_mib   = (argc > 1) ? std::stoull(argv[1]) : 64;
  uint64_t per_gpu_mib = (argc > 2) ? std::stoull(argv[2]) : 4096;
  int iters            = (argc > 3) ? std::stoi(argv[3]) : 20;
  uint64_t chunk       = chunk_mib << 20;
  uint64_t per_gpu     = per_gpu_mib << 20;

  std::printf("chunk=%llu MiB  per_gpu=%llu MiB  iters=%d  (local NUMA placement)\n\n",
              (unsigned long long)chunk_mib,
              (unsigned long long)per_gpu_mib,
              iters);

  // 1) Single GPU peak (local).
  run_config("1 GPU (g0<-n0)", {{0, 0}}, per_gpu, chunk, iters);
  // 2) Two GPUs on the SAME Grace (share node-0 LPDDR) -> tests memory-bw sharing.
  run_config("2 GPU same-Grace (g0,g1<-n0)", {{0, 0}, {1, 0}}, per_gpu, chunk, iters);
  // 3) Two GPUs on DIFFERENT Grace (independent LPDDR) -> should scale ~2x.
  run_config("2 GPU cross-Grace (g0<-n0,g2<-n1)", {{0, 0}, {2, 1}}, per_gpu, chunk, iters);
  // 4) All 4 GPUs, each local to its Grace -> aggregate ceiling.
  run_config(
    "4 GPU all-local (g0,1<-n0;g2,3<-n1)", {{0, 0}, {1, 0}, {2, 1}, {3, 1}}, per_gpu, chunk, iters);
  // 5) Single GPU pulling REMOTE (g0 <- node1) -> local vs remote penalty.
  run_config("1 GPU remote (g0<-n1)", {{0, 1}}, per_gpu, chunk, iters);

  return 0;
}
