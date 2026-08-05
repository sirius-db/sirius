// membw_victim - tiny flat-out bandwidth benchmark, one per domain, used to
// measure how much bandwidth a "victim" workload actually achieves while a
// membw_eater runs in the background. Prints per-interval CSV plus a final
// grep-able summary line:
//
//   RESULT domain=<d> engine=<e> secs=<s> gbps=<g>
//
// Domains mirror membw_eater:
//   --domain hbm   D2D copy kernel (SM, default) or CE memcpy   [2x bytes]
//   --domain dram  host memcpy threads, NUMA-pinned             [2x bytes]
//   --domain c2c   pinned-host H2D (default) or D2H memcpy      [1x bytes]

#include "membw_common.h"

#include <string>
#include <thread>
#include <vector>

#ifndef NO_CUDA
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                              \
	do {                                                                                                              \
		cudaError_t err__ = (call);                                                                                   \
		if (err__ != cudaSuccess) {                                                                                   \
			fprintf(stderr, "membw_victim: CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__), __FILE__,          \
			        __LINE__, cudaGetErrorString(err__));                                                             \
			exit(3);                                                                                                  \
		}                                                                                                             \
	} while (0)
#endif

using namespace membw;

struct Opts {
	std::string domain;
	std::string engine = "";
	double secs = 3.0;
	double interval_s = 0.5;
	size_t chunk_mb = 0;
	size_t buf_mb = 0;
	int threads = 8;   // dram
	int cpu_start = 0; // dram
	int device = 0;
	bool quiet = false; // suppress per-interval rows

	bool domain_hbm() const {
		return domain == "hbm";
	}
};

static void Usage() {
	fprintf(stderr,
	        "usage: membw_victim --domain {hbm|dram|c2c} [options]\n"
	        "  --secs S       measurement duration (default 3)\n"
	        "  --engine E     hbm: sm (default) | ce;  c2c: h2d (default) | d2h\n"
	        "  --threads N    dram: memcpy threads (default 8)\n"
	        "  --cpu-start N  dram: first CPU to pin to (default 0)\n"
	        "  --chunk-mb N   bytes per iteration (default: hbm/c2c 256, dram 16)\n"
	        "  --buf-mb N     buffer size (default: hbm 1024, c2c 512, dram 64/thread)\n"
	        "  --interval S   per-interval CSV cadence (default 0.5)\n"
	        "  --quiet        only print the final RESULT line\n"
	        "  --device N     CUDA device (default 0)\n");
	exit(1);
}

static Opts Parse(int argc, char **argv) {
	Opts o;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto need = [&](const char *what) -> const char * {
			if (i + 1 >= argc) {
				fprintf(stderr, "membw_victim: %s needs a value\n", what);
				Usage();
			}
			return argv[++i];
		};
		if (a == "--domain") {
			o.domain = need("--domain");
		} else if (a == "--secs") {
			o.secs = atof(need("--secs"));
		} else if (a == "--engine") {
			o.engine = need("--engine");
		} else if (a == "--threads") {
			o.threads = atoi(need("--threads"));
		} else if (a == "--cpu-start") {
			o.cpu_start = atoi(need("--cpu-start"));
		} else if (a == "--chunk-mb") {
			o.chunk_mb = (size_t)atol(need("--chunk-mb"));
		} else if (a == "--buf-mb") {
			o.buf_mb = (size_t)atol(need("--buf-mb"));
		} else if (a == "--interval") {
			o.interval_s = atof(need("--interval"));
		} else if (a == "--quiet") {
			o.quiet = true;
		} else if (a == "--device") {
			o.device = atoi(need("--device"));
		} else {
			fprintf(stderr, "membw_victim: unknown arg %s\n", a.c_str());
			Usage();
		}
	}
	if (o.domain != "hbm" && o.domain != "dram" && o.domain != "c2c") {
		Usage();
	}
	if (o.domain == "hbm" && o.engine.empty()) {
		o.engine = "sm";
	}
	if (o.domain == "c2c" && o.engine.empty()) {
		o.engine = "h2d";
	}
	if (o.domain == "dram") {
		o.engine = "memcpy";
	}
	return o;
}

static void Result(const Opts &o, double secs, double gbps) {
	printf("RESULT domain=%s engine=%s secs=%.3f gbps=%.2f\n", o.domain.c_str(), o.engine.c_str(), secs, gbps);
	fflush(stdout);
}

// ----------------------------------------------------------------------------
#ifndef NO_CUDA
__global__ void CopyKernelStream(const uint4 *__restrict__ src, uint4 *__restrict__ dst, size_t n) {
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	for (; i < n; i += stride) {
		__stcs(&dst[i], __ldcs(&src[i]));
	}
}

static int RunGpu(Opts &o) {
	CUDA_CHECK(cudaSetDevice(o.device));
	size_t buf_bytes = (o.buf_mb ? o.buf_mb : (o.domain_hbm() ? 1024 : 512)) << 20;
	size_t chunk = (o.chunk_mb ? o.chunk_mb : 256) << 20;
	if (chunk > buf_bytes) {
		chunk = buf_bytes;
	}
	double factor = o.domain_hbm() ? 2.0 : 1.0;

	uint8_t *src, *dst;
	if (o.domain_hbm()) {
		CUDA_CHECK(cudaMalloc(&src, buf_bytes));
		CUDA_CHECK(cudaMalloc(&dst, buf_bytes));
		CUDA_CHECK(cudaMemset(src, 1, buf_bytes));
	} else {
		CUDA_CHECK(cudaMallocHost(&src, buf_bytes));
		memset(src, 1, buf_bytes);
		CUDA_CHECK(cudaMalloc(&dst, buf_bytes));
	}
	CUDA_CHECK(cudaMemset(dst, 0, buf_bytes));
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	int dev_sms = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(&dev_sms, cudaDevAttrMultiProcessorCount, o.device));
	int blocks = dev_sms * 8, tpb = 256;

	auto burst = [&](size_t off) {
		if (o.domain_hbm()) {
			if (o.engine == "sm") {
				CopyKernelStream<<<blocks, tpb, 0, stream>>>((const uint4 *)(src + off), (uint4 *)(dst + off),
				                                             chunk / sizeof(uint4));
			} else {
				CUDA_CHECK(cudaMemcpyAsync(dst + off, src + off, chunk, cudaMemcpyDeviceToDevice, stream));
			}
		} else if (o.engine == "d2h") {
			CUDA_CHECK(cudaMemcpyAsync(src + off, dst + off, chunk, cudaMemcpyDeviceToHost, stream));
		} else {
			CUDA_CHECK(cudaMemcpyAsync(dst + off, src + off, chunk, cudaMemcpyHostToDevice, stream));
		}
	};

	burst(0); // warm up
	CUDA_CHECK(cudaStreamSynchronize(stream));

	double t_start = NowS(), t_prev = t_start;
	double traffic = 0.0, traffic_prev = 0.0;
	size_t off = 0;
	while (!g_stop) {
		burst(off);
		CUDA_CHECK(cudaStreamSynchronize(stream));
		traffic += (double)chunk * factor;
		off += chunk;
		if (off + chunk > buf_bytes) {
			off = 0;
		}
		double t = NowS();
		if (!o.quiet && t - t_prev >= o.interval_s) {
			printf("interval,%.3f,%s,%s,%.2f\n", t - t_start, o.domain.c_str(), o.engine.c_str(),
			       (traffic - traffic_prev) / (t - t_prev) / 1e9);
			fflush(stdout);
			t_prev = t;
			traffic_prev = traffic;
		}
		if (t - t_start >= o.secs) {
			break;
		}
	}
	double secs = NowS() - t_start;
	Result(o, secs, traffic / secs / 1e9);
	cudaStreamDestroy(stream);
	if (o.domain_hbm()) {
		cudaFree(src);
	} else {
		cudaFreeHost(src);
	}
	cudaFree(dst);
	return 0;
}
#endif // !NO_CUDA

// ----------------------------------------------------------------------------
static int RunDram(Opts &o) {
	size_t buf_bytes = (o.buf_mb ? o.buf_mb : 64) << 20;
	size_t chunk = (o.chunk_mb ? o.chunk_mb : 16) << 20;
	if (chunk > buf_bytes) {
		chunk = buf_bytes;
	}
	std::atomic<uint64_t> traffic {0};
	std::atomic<bool> stop {false};

	auto worker = [&](int idx) {
		PinToCpu(o.cpu_start + idx);
		uint8_t *src = (uint8_t *)aligned_alloc(2 << 20, buf_bytes);
		uint8_t *dst = (uint8_t *)aligned_alloc(2 << 20, buf_bytes);
		if (!src || !dst) {
			fprintf(stderr, "membw_victim: alloc failed\n");
			stop.store(true);
			return;
		}
		memset(src, 1, buf_bytes);
		memset(dst, 0, buf_bytes);
		size_t off = 0;
		while (!stop.load(std::memory_order_relaxed) && !g_stop) {
			memcpy(dst + off, src + off, chunk);
			traffic.fetch_add((uint64_t)chunk * 2, std::memory_order_relaxed);
			off += chunk;
			if (off + chunk > buf_bytes) {
				off = 0;
			}
		}
		free(src);
		free(dst);
	};

	std::vector<std::thread> ws;
	for (int i = 0; i < o.threads; i++) {
		ws.emplace_back(worker, i);
	}
	// settle: let first-touch/allocation finish before timing
	SleepUntil(NowS() + 0.3);
	double t_start = NowS(), t_prev = t_start;
	uint64_t base = traffic.load(), traffic_prev = base;
	while (!g_stop) {
		SleepUntil(t_prev + o.interval_s);
		double t = NowS();
		uint64_t tr = traffic.load();
		if (!o.quiet) {
			printf("interval,%.3f,dram,memcpy,%.2f\n", t - t_start, (double)(tr - traffic_prev) / (t - t_prev) / 1e9);
			fflush(stdout);
		}
		t_prev = t;
		traffic_prev = tr;
		if (t - t_start >= o.secs) {
			break;
		}
	}
	double secs = NowS() - t_start;
	uint64_t total = traffic.load() - base;
	stop.store(true);
	for (auto &w : ws) {
		w.join();
	}
	Result(o, secs, (double)total / secs / 1e9);
	return 0;
}

int main(int argc, char **argv) {
	Opts o = Parse(argc, argv);
	InstallStopHandlers();
	if (o.domain == "dram") {
		return RunDram(o);
	}
#ifndef NO_CUDA
	return RunGpu(o);
#else
	fprintf(stderr, "membw_victim: built without CUDA (NO_CUDA); only --domain dram is available\n");
	return 4;
#endif
}
