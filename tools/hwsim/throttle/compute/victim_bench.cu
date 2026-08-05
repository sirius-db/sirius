// hwsim WS5: victim benchmark for calibrating GPU compute throttlers.
//
// Victims:
//   fma    - compute-bound: register-only FMA loop, negligible memory traffic.
//   saxpy  - memory-bound: streaming y = a*x + y over large arrays (HBM bandwidth).
//
// The bench repeatedly launches the victim kernel for --seconds and reports the
// median per-launch time and derived throughput on a machine-greppable RESULT line.
//
// Throttle co-run options (for calibration):
//   --co smsteal:F        in-process SM-steal spin kernel on a second stream
//   --co duty:F[:Pms]     in-process duty-cycle burst loop on a second stream
//   (cross-process: run ./throttle_compute in another shell instead)
//   --greenctx N          run the victim inside a CUDA green context restricted
//                         to ~N SMs (CUDA 12.4+; prints the actual granted count)
//
// Usage examples:
//   ./victim_bench --victim fma --seconds 1.5
//   ./victim_bench --victim saxpy --co smsteal:0.5
//   ./victim_bench --victim fma --greenctx 40

#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include "throttle_kernels.cuh"

// ---------------------------------------------------------------------------
// Victim kernels
__global__ void victim_fma_kernel(int iters, float* __restrict__ sink) {
  float acc = 1.0f + (float)(threadIdx.x + blockIdx.x * blockDim.x) * 1e-7f;
  acc = hwsim_fma_burn(acc, iters);
  if (acc == 123.456f) sink[0] = acc;
}

__global__ void victim_saxpy_kernel(const float* __restrict__ x, float* __restrict__ y,
                                    float a, size_t n, int passes) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int p = 0; p < passes; ++p) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
      y[i] = fmaf(a, x[i], y[i]);
    }
  }
}

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void sleep_ns(long ns) {
  if (ns <= 0) return;
  struct timespec ts{ns / 1000000000L, ns % 1000000000L};
  nanosleep(&ts, nullptr);
}

// ---------------------------------------------------------------------------
// In-process duty-cycle co-runner (host thread + dedicated stream).
struct DutyCoRunner {
  std::thread th;
  std::atomic<bool> stop{false};

  void start(double fraction, double period_ms) {
    th = std::thread([this, fraction, period_ms]() {
      ThrottleShape shape = throttle_shape((const void*)burst_kernel);
      const int grid = shape.num_sms * shape.blocks_per_sm;
      const long period_ns = (long)(period_ms * 1e6);
      const long busy_ns = (long)(fraction * period_ns);
      const long idle_ns = period_ns - busy_ns;
      float* sink = nullptr;
      CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      while (!stop.load(std::memory_order_relaxed) && busy_ns > 0) {
        burst_kernel<<<grid, shape.block_size, 0, stream>>>(
            (unsigned long long)busy_ns, sink);
        cudaStreamSynchronize(stream);
        sleep_ns(idle_ns);
      }
      cudaStreamDestroy(stream);
      cudaFree(sink);
    });
  }
  void join() {
    stop.store(true);
    if (th.joinable()) th.join();
  }
};

// ---------------------------------------------------------------------------
// Green context setup: restrict this thread's current context to ~want_sms SMs.
// Returns the actually-granted SM count (green context SM splits have hardware
// granularity, so the grant can differ from the request).
static int setup_green_context(int want_sms) {
  CU_CHECK(cuInit(0));
  CUdevice dev;
  CU_CHECK(cuDeviceGet(&dev, 0));

  CUdevResource sm_res;
  memset(&sm_res, 0, sizeof(sm_res));
  CU_CHECK(cuDeviceGetDevResource(dev, &sm_res, CU_DEV_RESOURCE_TYPE_SM));
  printf("greenctx: device reports %u SMs available for partitioning\n",
         sm_res.sm.smCount);

  CUdevResource split;
  CUdevResource remaining;
  memset(&split, 0, sizeof(split));
  memset(&remaining, 0, sizeof(remaining));
  unsigned int nb_groups = 1;
  CU_CHECK(cuDevSmResourceSplitByCount(&split, &nb_groups, &sm_res, &remaining,
                                       /*useFlags=*/0, /*minCount=*/want_sms));

  CUdevResourceDesc desc;
  CU_CHECK(cuDevResourceGenerateDesc(&desc, &split, 1));
  CUgreenCtx gctx;
  CU_CHECK(cuGreenCtxCreate(&gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));

  CUdevResource granted;
  memset(&granted, 0, sizeof(granted));
  CU_CHECK(cuGreenCtxGetDevResource(gctx, &granted, CU_DEV_RESOURCE_TYPE_SM));

  CUcontext ctx;
  CU_CHECK(cuCtxFromGreenCtx(&ctx, gctx));
  CU_CHECK(cuCtxSetCurrent(ctx));
  printf("greenctx: requested >=%d SMs, granted %u SMs\n", want_sms,
         granted.sm.smCount);
  return (int)granted.sm.smCount;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string victim = "fma";
  std::string co = "none";
  double seconds = 1.5;
  int greenctx_sms = -1;
  int fma_iters = 1 << 21;
  size_t saxpy_n = (size_t)256 << 20;  // 256M floats = 1 GiB per array
  int saxpy_passes = 8;
  int saxpy_blocks = 0;  // 0 = default 8 * num_sms

  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* what) {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", what);
        exit(2);
      }
      return argv[++i];
    };
    if (!strcmp(argv[i], "--victim")) victim = need("--victim");
    else if (!strcmp(argv[i], "--co")) co = need("--co");
    else if (!strcmp(argv[i], "--seconds")) seconds = atof(need("--seconds"));
    else if (!strcmp(argv[i], "--greenctx")) greenctx_sms = atoi(need("--greenctx"));
    else if (!strcmp(argv[i], "--fma-iters")) fma_iters = atoi(need("--fma-iters"));
    else if (!strcmp(argv[i], "--saxpy-mib"))
      saxpy_n = ((size_t)atoi(need("--saxpy-mib")) << 20) / sizeof(float);
    else if (!strcmp(argv[i], "--saxpy-passes")) saxpy_passes = atoi(need("--saxpy-passes"));
    else if (!strcmp(argv[i], "--saxpy-blocks")) saxpy_blocks = atoi(need("--saxpy-blocks"));
    else {
      fprintf(stderr,
              "usage: %s --victim fma|saxpy [--seconds S] [--co smsteal:F|duty:F[:Pms]] "
              "[--greenctx N] [--fma-iters I] [--saxpy-mib M] [--saxpy-passes P]\n",
              argv[0]);
      return 2;
    }
  }

  // Lazy module loading (the CUDA 12+ default) makes the first-ever launch of a
  // kernel BLOCK while a persistent kernel is resident in the same context, which
  // deadlocks the co-run calibration. Force eager loading (user can override).
  setenv("CUDA_MODULE_LOADING", "EAGER", /*overwrite=*/0);

  int granted_sms = -1;
  if (greenctx_sms > 0) {
    granted_sms = setup_green_context(greenctx_sms);
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  cudaStream_t vstream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&vstream, cudaStreamNonBlocking));

  // Set up the victim workload.
  float* sink = nullptr;
  float *dx = nullptr, *dy = nullptr;
  int fma_grid = 0, fma_block = 256;
  int saxpy_grid = 0, saxpy_block = 256;
  double work_per_launch = 0.0;  // FLOPs or bytes
  const char* unit = "";

  if (victim == "fma") {
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    fma_grid = 4 * prop.multiProcessorCount;
    work_per_launch = (double)fma_grid * fma_block * fma_iters * 2.0;  // FLOPs
    unit = "GFLOPS";
  } else if (victim == "saxpy") {
    CUDA_CHECK(cudaMalloc(&dx, saxpy_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, saxpy_n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dx, 0x3f, saxpy_n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dy, 0x3f, saxpy_n * sizeof(float)));
    saxpy_grid = saxpy_blocks > 0 ? saxpy_blocks : 8 * prop.multiProcessorCount;
    work_per_launch = 3.0 * saxpy_n * sizeof(float) * saxpy_passes;  // bytes moved
    unit = "GBps";
  } else {
    fprintf(stderr, "unknown --victim %s\n", victim.c_str());
    return 2;
  }

  auto launch_victim = [&]() {
    if (victim == "fma") {
      victim_fma_kernel<<<fma_grid, fma_block, 0, vstream>>>(fma_iters, sink);
    } else {
      victim_saxpy_kernel<<<saxpy_grid, saxpy_block, 0, vstream>>>(
          dx, dy, 1.0001f, saxpy_n, saxpy_passes);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(vstream));
  };

  // Warmup BEFORE starting any co-throttler: the victim kernel's first launch
  // must not race a resident spin kernel (see lazy-loading note above; warmup
  // also absorbs one-time init cost either way).
  for (int i = 0; i < 2; ++i) launch_victim();

  // Start in-process co-throttler if requested.
  SmStealHandle steal;
  DutyCoRunner duty;
  bool steal_on = false, duty_on = false;
  double co_fraction = 0.0, co_period_ms = 10.0;
  if (co != "none") {
    char kind[16] = {0};
    double f = 0.0, p = 10.0;
    if (sscanf(co.c_str(), "smsteal:%lf", &f) == 1) {
      strcpy(kind, "smsteal");
    } else if (sscanf(co.c_str(), "duty:%lf:%lf", &f, &p) >= 1) {
      strcpy(kind, "duty");
    } else {
      fprintf(stderr, "bad --co spec: %s\n", co.c_str());
      return 2;
    }
    co_fraction = f;
    co_period_ms = p;
    if (!strcmp(kind, "smsteal")) {
      steal.start(f, /*report_coverage=*/true);
      steal_on = true;
    } else {
      duty.start(f, p);
      duty_on = true;
    }
    sleep_ns(150 * 1000 * 1000);  // let the throttler settle before measuring
  }

  // Timed loop.
  std::vector<double> ms;
  const double t0 = now_s();
  while (now_s() - t0 < seconds) {
    const double a = now_s();
    launch_victim();
    ms.push_back((now_s() - a) * 1e3);
  }

  if (duty_on) duty.join();
  if (steal_on) steal.stop();

  std::sort(ms.begin(), ms.end());
  const double median = ms[ms.size() / 2];
  double mean = 0.0;
  for (double v : ms) mean += v;
  mean /= (double)ms.size();
  const double thpt = work_per_launch / (median * 1e-3) / 1e9;

  printf("RESULT victim=%s co=%s co_fraction=%.3f co_period_ms=%.1f greenctx=%d "
         "granted_sms=%d launches=%zu median_ms=%.3f mean_ms=%.3f thpt=%.1f unit=%s\n",
         victim.c_str(), co.c_str(), co_fraction, co_period_ms, greenctx_sms,
         granted_sms, ms.size(), median, mean, thpt, unit);
  return 0;
}
