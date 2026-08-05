// hwsim WS5 diagnostic: can a second kernel co-run with a resident persistent
// spin kernel in the same context? Never blocks: polls cudaStreamQuery.
//
//   ./probe_concurrency <mode>
//     mode 0: spin grid = full wave w/ early-return gating (as smsteal_kernel)
//     mode 1: spin grid = only n_stolen*blocks_per_sm blocks, no gating
//     mode 2: like 1 but tiny spin blocks (1 block, 32 threads)
//     mode 3: like 1 but victim launched at higher stream priority
//
// FINDING (GB300, driver 595.58.03, CUDA 13.2): all four modes report
// "victim DID NOT RUN" by default, but pass with CUDA_MODULE_LOADING=EAGER.
// Root cause is lazy module loading (the CUDA 12+ default): the victim kernel's
// FIRST-EVER launch requires a module load that blocks while a persistent kernel
// is resident in the same context. Not a scheduler limitation -- warmed-up
// kernels co-run fine. Kept as a regression probe for that failure mode.

#include "throttle_kernels.cuh"

__global__ void tiny_victim(float* out) {
  float acc = 1.0f + threadIdx.x * 1e-6f;
  acc = hwsim_fma_burn(acc, 1 << 16);
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// Ungated spin: every block spins until *stop.
__global__ void spin_all(const volatile int* __restrict__ stop,
                         float* __restrict__ sink) {
  __shared__ int s_stop;
  float acc = 1.0f + (float)threadIdx.x * 1e-6f;
  do {
    if (threadIdx.x == 0) s_stop = *stop;
    __syncthreads();
    acc = hwsim_fma_burn(acc, 1 << 13);
    __syncthreads();
  } while (!s_stop);
  if (acc == 123.456f) *sink = acc;
}

int main(int argc, char** argv) {
  const int mode = argc > 1 ? atoi(argv[1]) : 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  ThrottleShape shape = throttle_shape((const void*)smsteal_kernel);
  printf("mode %d on %s (%d SMs, %d blk/SM)\n", mode, prop.name, shape.num_sms,
         shape.blocks_per_sm);

  int* stop_host;
  int* stop_dev;
  CUDA_CHECK(cudaHostAlloc(&stop_host, sizeof(int), cudaHostAllocMapped));
  *stop_host = 0;
  CUDA_CHECK(cudaHostGetDevicePointer(&stop_dev, stop_host, 0));
  unsigned int* coverage;
  CUDA_CHECK(cudaMalloc(&coverage, 1024 * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(coverage, 0, 1024 * sizeof(unsigned int)));
  float* sink;
  CUDA_CHECK(cudaMalloc(&sink, 1024 * sizeof(float)));

  cudaStream_t spin_s, victim_s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&spin_s, cudaStreamNonBlocking));
  if (mode == 3) {
    int lo, hi;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&victim_s, cudaStreamNonBlocking, hi));
    printf("victim stream priority %d (range lo=%d hi=%d)\n", hi, lo, hi);
  } else {
    CUDA_CHECK(cudaStreamCreateWithFlags(&victim_s, cudaStreamNonBlocking));
  }

  const int n_stolen = 3;
  if (mode == 0) {
    smsteal_kernel<<<shape.num_sms * shape.blocks_per_sm, shape.block_size, 0,
                     spin_s>>>(stop_dev, n_stolen, coverage, sink);
  } else if (mode == 1 || mode == 3) {
    spin_all<<<n_stolen * shape.blocks_per_sm, shape.block_size, 0, spin_s>>>(stop_dev,
                                                                              sink);
  } else {
    spin_all<<<1, 32, 0, spin_s>>>(stop_dev, sink);
  }
  CUDA_CHECK(cudaGetLastError());

  struct timespec ts{0, 100 * 1000 * 1000};
  nanosleep(&ts, nullptr);  // let the spin wave settle

  tiny_victim<<<4, 256, 0, victim_s>>>(sink);
  CUDA_CHECK(cudaGetLastError());

  // Poll for up to 5 seconds.
  bool done = false;
  for (int i = 0; i < 50; ++i) {
    cudaError_t q = cudaStreamQuery(victim_s);
    if (q == cudaSuccess) {
      done = true;
      break;
    }
    if (q != cudaErrorNotReady) {
      printf("victim stream error: %s\n", cudaGetErrorString(q));
      break;
    }
    nanosleep(&ts, nullptr);
  }
  printf("victim %s while spin kernel resident\n",
         done ? "COMPLETED" : "DID NOT RUN (5s)");

  *stop_host = 1;
  CUDA_CHECK(cudaStreamSynchronize(spin_s));
  CUDA_CHECK(cudaStreamSynchronize(victim_s));
  printf("spin kernel stopped cleanly; victim after stop: OK\n");
  return 0;
}
