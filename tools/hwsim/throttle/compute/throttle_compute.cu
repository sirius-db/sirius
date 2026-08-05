// hwsim WS5: standalone GPU compute throttler.
//
// Modes:
//   smsteal  - persistent spin kernel pinning N = round(fraction * num_sms) SMs at
//              full occupancy; the remaining SMs stay free. Verifies block placement
//              via %smid. NOTE: without MPS, a *separate process's* kernels time-slice
//              with this context instead of space-sharing SMs -- run the victim under
//              MPS (or use victim_bench --co for in-process calibration).
//   duty     - full-GPU burst kernel alternating busy/idle at --period-ms with busy
//              fraction --fraction.
//
// Stops cleanly on SIGINT/SIGTERM, or after --duration seconds if given.
//
// Usage:
//   ./throttle_compute --mode smsteal --fraction 0.5 [--duration 10]
//   ./throttle_compute --mode duty --fraction 0.5 [--period-ms 10] [--duration 10]

#include <csignal>
#include <cstring>
#include <string>

#include "throttle_kernels.cuh"

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

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

int main(int argc, char** argv) {
  std::string mode = "smsteal";
  double fraction = 0.5;
  double period_ms = 10.0;
  double duration_s = -1.0;  // <0: run until signal

  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* what) {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", what);
        exit(2);
      }
      return argv[++i];
    };
    if (!strcmp(argv[i], "--mode")) mode = need("--mode");
    else if (!strcmp(argv[i], "--fraction")) fraction = atof(need("--fraction"));
    else if (!strcmp(argv[i], "--period-ms")) period_ms = atof(need("--period-ms"));
    else if (!strcmp(argv[i], "--duration")) duration_s = atof(need("--duration"));
    else {
      fprintf(stderr,
              "usage: %s --mode smsteal|duty --fraction F [--period-ms P] "
              "[--duration S]\n",
              argv[0]);
      return 2;
    }
  }
  if (fraction < 0.0 || fraction > 1.0) {
    fprintf(stderr, "--fraction must be in [0,1]\n");
    return 2;
  }

  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);

  // See victim_bench.cu: avoid lazy-loading stalls against resident spin kernels.
  setenv("CUDA_MODULE_LOADING", "EAGER", /*overwrite=*/0);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("device: %s, %d SMs, cc %d.%d\n", prop.name, prop.multiProcessorCount,
         prop.major, prop.minor);

  const double t0 = now_s();
  auto should_stop = [&]() {
    return g_stop || (duration_s >= 0.0 && now_s() - t0 >= duration_s);
  };

  if (mode == "smsteal") {
    SmStealHandle h;
    h.start(fraction, /*report_coverage=*/true);
    printf("throttling (mode=smsteal fraction=%.3f); Ctrl-C to stop\n", fraction);
    fflush(stdout);
    while (!should_stop()) sleep_ns(50 * 1000 * 1000);
    h.stop();
    printf("stopped cleanly\n");
  } else if (mode == "duty") {
    ThrottleShape shape = throttle_shape((const void*)burst_kernel);
    const int grid = shape.num_sms * shape.blocks_per_sm;
    const long period_ns = (long)(period_ms * 1e6);
    const long busy_ns = (long)(fraction * period_ns);
    const long idle_ns = period_ns - busy_ns;
    float* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    printf("throttling (mode=duty fraction=%.3f period=%.1fms grid=%dx%d); "
           "Ctrl-C to stop\n",
           fraction, period_ms, grid, shape.block_size);
    fflush(stdout);
    while (!should_stop() && busy_ns > 0) {
      burst_kernel<<<grid, shape.block_size, 0, stream>>>(
          (unsigned long long)busy_ns, sink);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      sleep_ns(idle_ns);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(sink));
    printf("stopped cleanly\n");
  } else {
    fprintf(stderr, "unknown --mode %s (want smsteal or duty)\n", mode.c_str());
    return 2;
  }
  return 0;
}
