// membw_eater - rate-controlled background bandwidth consumer ("eater") for
// three separately-controllable domains on GB300 (Grace + Blackwell):
//
//   --domain hbm   GPU HBM bandwidth   (SM copy kernel or CE cudaMemcpyAsync D2D)
//   --domain dram  host DRAM bandwidth (NUMA-pinned CPU memcpy threads)
//   --domain c2c   host<->device NVLink-C2C bandwidth (pinned H2D/D2H memcpyAsync)
//
// Closed-loop controller holds a target *consumed* GB/s (see traffic accounting
// in the README), absolute-deadline pacing, SIGINT/SIGTERM clean stop, CSV
// stats of achieved rate per interval.
//
// Build: see Makefile. The dram domain also builds without CUDA (-DNO_CUDA).

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
			fprintf(stderr, "membw_eater: CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__), __FILE__, __LINE__, \
			        cudaGetErrorString(err__));                                                                       \
			exit(3);                                                                                                  \
		}                                                                                                             \
	} while (0)
#endif

using namespace membw;

// ----------------------------------------------------------------------------
// options
// ----------------------------------------------------------------------------
struct Opts {
	std::string domain;        // hbm | dram | c2c
	std::string engine = "";   // hbm: sm|ce (default sm); c2c: h2d|d2h|both (default h2d)
	double target_gbps = 0.0;  // 0 = unlimited (flat out)
	double duration_s = 0.0;   // 0 = run until SIGINT/SIGTERM
	double interval_s = 0.5;   // CSV stats interval
	double pace_s = 0.002;     // desired burst period for auto chunk sizing
	size_t chunk_mb = 0;       // 0 = auto
	size_t buf_mb = 0;         // 0 = per-domain default
	int threads = 16;          // dram only
	int cpu_start = 36;        // dram only: pin threads to cpu_start..cpu_start+threads-1
	int device = 0;            // gpu domains
	std::string csv = "";      // "" or "-" = stdout

	bool domain_hbm() const {
		return domain == "hbm";
	}
};

static void Usage() {
	fprintf(stderr,
	        "usage: membw_eater --domain {hbm|dram|c2c} [options]\n"
	        "  --gbps N|max      target consumed GB/s (default: max = flat out)\n"
	        "  --duration S      stop after S seconds (default: run until SIGINT/SIGTERM)\n"
	        "  --engine E        hbm: sm (SM copy kernel, default) | ce (cudaMemcpyAsync D2D)\n"
	        "                    c2c: h2d (default) | d2h | both (alternating)\n"
	        "  --threads N       dram: memcpy threads (default 16)\n"
	        "  --cpu-start N     dram: first CPU to pin threads to (default 36)\n"
	        "  --chunk-mb N      bytes moved per burst (default: auto from target)\n"
	        "  --buf-mb N        buffer size (default: hbm 1024, c2c 512, dram 64/thread)\n"
	        "  --interval S      CSV stats interval seconds (default 0.5)\n"
	        "  --pace S          target burst period seconds for auto chunk (default 0.002)\n"
	        "  --csv PATH        write CSV stats to PATH (default stdout)\n"
	        "  --device N        CUDA device (default 0)\n"
	        "traffic accounting: hbm and dram count 2x bytes per copy (read+write);\n"
	        "c2c counts 1x bytes (each byte crosses the link once).\n");
	exit(1);
}

static Opts Parse(int argc, char **argv) {
	Opts o;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto need = [&](const char *what) -> const char * {
			if (i + 1 >= argc) {
				fprintf(stderr, "membw_eater: %s needs a value\n", what);
				Usage();
			}
			return argv[++i];
		};
		if (a == "--domain") {
			o.domain = need("--domain");
		} else if (a == "--gbps") {
			o.target_gbps = ParseGbps(need("--gbps"));
		} else if (a == "--duration") {
			o.duration_s = atof(need("--duration"));
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
		} else if (a == "--pace") {
			o.pace_s = atof(need("--pace"));
		} else if (a == "--csv") {
			o.csv = need("--csv");
		} else if (a == "--device") {
			o.device = atoi(need("--device"));
		} else {
			fprintf(stderr, "membw_eater: unknown arg %s\n", a.c_str());
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

// ----------------------------------------------------------------------------
// GPU domains (hbm, c2c)
// ----------------------------------------------------------------------------
#ifndef NO_CUDA

// Streaming grid-stride copy: __ldcs/__stcs bypass L2 persistence so traffic
// actually lands on HBM instead of being absorbed by the 100+ MB L2.
__global__ void CopyKernelStream(const uint4 *__restrict__ src, uint4 *__restrict__ dst, size_t n) {
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = (size_t)gridDim.x * blockDim.x;
	for (; i < n; i += stride) {
		__stcs(&dst[i], __ldcs(&src[i]));
	}
}

struct GpuEngine {
	Opts o;
	uint8_t *src = nullptr; // device (hbm) or pinned host (c2c)
	uint8_t *dst = nullptr; // device
	size_t buf_bytes = 0;
	size_t chunk = 0;
	double traffic_factor = 2.0;
	cudaStream_t stream;
	int blocks = 0, tpb = 256;

	void Setup() {
		CUDA_CHECK(cudaSetDevice(o.device));
		size_t buf_mb = o.buf_mb ? o.buf_mb : (o.domain_hbm() ? 1024 : 512);
		buf_bytes = buf_mb << 20;
		if (o.domain_hbm()) {
			traffic_factor = 2.0;
			CUDA_CHECK(cudaMalloc(&src, buf_bytes));
			CUDA_CHECK(cudaMalloc(&dst, buf_bytes));
			CUDA_CHECK(cudaMemset(src, 1, buf_bytes));
			CUDA_CHECK(cudaMemset(dst, 0, buf_bytes));
		} else { // c2c
			traffic_factor = 1.0;
			CUDA_CHECK(cudaMallocHost(&src, buf_bytes)); // pinned host
			memset(src, 1, buf_bytes);
			CUDA_CHECK(cudaMalloc(&dst, buf_bytes));
			CUDA_CHECK(cudaMemset(dst, 0, buf_bytes));
		}
		CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		size_t max_chunk = buf_bytes / 4 < (256u << 20) ? buf_bytes / 4 : (256u << 20);
		chunk = o.chunk_mb ? (o.chunk_mb << 20)
		                   : AutoChunk(o.target_gbps * 1e9, traffic_factor, o.pace_s, 4u << 20, max_chunk);
		if (chunk > buf_bytes) {
			chunk = buf_bytes;
		}
		int dev_sms = 0;
		CUDA_CHECK(cudaDeviceGetAttribute(&dev_sms, cudaDevAttrMultiProcessorCount, o.device));
		blocks = dev_sms * 8;
		// warm up
		Burst(0, 0);
		CUDA_CHECK(cudaStreamSynchronize(stream));
	}

	// One burst starting at byte offset `off` (16-byte aligned), alternating
	// direction index `k` for c2c "both" mode.
	void Burst(size_t off, uint64_t k) {
		if (o.domain_hbm()) {
			if (o.engine == "sm") {
				size_t n = chunk / sizeof(uint4);
				CopyKernelStream<<<blocks, tpb, 0, stream>>>((const uint4 *)(src + off), (uint4 *)(dst + off), n);
			} else { // ce
				CUDA_CHECK(cudaMemcpyAsync(dst + off, src + off, chunk, cudaMemcpyDeviceToDevice, stream));
			}
		} else { // c2c
			bool d2h = (o.engine == "d2h") || (o.engine == "both" && (k & 1));
			if (d2h) {
				CUDA_CHECK(cudaMemcpyAsync(src + off, dst + off, chunk, cudaMemcpyDeviceToHost, stream));
			} else {
				CUDA_CHECK(cudaMemcpyAsync(dst + off, src + off, chunk, cudaMemcpyHostToDevice, stream));
			}
		}
	}

	void Teardown() {
		cudaStreamDestroy(stream);
		if (o.domain_hbm()) {
			cudaFree(src);
		} else {
			cudaFreeHost(src);
		}
		cudaFree(dst);
	}
};

static int RunGpu(Opts &o) {
	GpuEngine eng;
	eng.o = o;
	eng.Setup();

	CsvWriter csv;
	csv.Open(o.csv);
	std::atomic<double> trim(1.0);
	Pacer pacer;
	pacer.Init((double)eng.chunk * eng.traffic_factor, o.target_gbps * 1e9, &trim);

	fprintf(stderr, "membw_eater: domain=%s engine=%s target=%s buf=%zuMB chunk=%zuMB\n", o.domain.c_str(),
	        o.engine.c_str(), o.target_gbps > 0 ? (std::to_string(o.target_gbps) + " GB/s").c_str() : "max",
	        eng.buf_bytes >> 20, eng.chunk >> 20);

	double t_start = NowS(), t_report = t_start + o.interval_s;
	double busy = 0.0;
	uint64_t bursts = 0, bursts_prev = 0;
	double traffic = 0.0, traffic_prev = 0.0, busy_prev = 0.0, t_prev = t_start;
	size_t off = 0;
	uint64_t k = 0;

	while (!g_stop) {
		double b0 = NowS();
		eng.Burst(off, k++);
		CUDA_CHECK(cudaStreamSynchronize(eng.stream));
		double b1 = NowS();
		busy += b1 - b0;
		bursts++;
		traffic += (double)eng.chunk * eng.traffic_factor;
		off += eng.chunk;
		if (off + eng.chunk > eng.buf_bytes) {
			off = 0;
		}
		if (b1 >= t_report) {
			double dt = b1 - t_prev;
			double ach = (traffic - traffic_prev) / dt / 1e9;
			double duty = (busy - busy_prev) / dt;
			uint64_t nb = bursts - bursts_prev;
			csv.Row(b1 - t_start, o.domain.c_str(), o.engine.c_str(), o.target_gbps, ach, nb,
			        nb ? (busy - busy_prev) * 1e3 / nb : 0.0, duty);
			UpdateTrim(trim, o.target_gbps * 1e9, ach * 1e9, duty);
			t_prev = b1;
			traffic_prev = traffic;
			busy_prev = busy;
			bursts_prev = bursts;
			t_report = b1 + o.interval_s;
		}
		if (o.duration_s > 0 && b1 - t_start >= o.duration_s) {
			break;
		}
		pacer.Wait();
	}
	double t_end = NowS();
	fprintf(stderr, "membw_eater: done. overall achieved %.2f GB/s over %.2fs (%" PRIu64 " bursts, duty %.2f)\n",
	        traffic / (t_end - t_start) / 1e9, t_end - t_start, bursts, busy / (t_end - t_start));
	csv.Close();
	eng.Teardown();
	return 0;
}

#endif // !NO_CUDA

// ----------------------------------------------------------------------------
// dram domain: NUMA-pinned CPU memcpy threads
// ----------------------------------------------------------------------------
struct DramShared {
	std::atomic<uint64_t> traffic {0}; // consumed bytes (2x memcpy size)
	std::atomic<uint64_t> bursts {0};
	std::atomic<uint64_t> busy_ns {0};
	std::atomic<double> trim {1.0};
	std::atomic<bool> stop {false};
};

static void DramWorker(const Opts *o, DramShared *sh, int idx, size_t buf_bytes, size_t chunk, double target_Bps) {
	PinToCpu(o->cpu_start + idx);
	// first-touch after pinning: pages land on this CPU's NUMA node (node 0 on
	// GB300 - all CPUs live there; use numactl for other layouts).
	uint8_t *src = (uint8_t *)aligned_alloc(2 << 20, buf_bytes);
	uint8_t *dst = (uint8_t *)aligned_alloc(2 << 20, buf_bytes);
	if (!src || !dst) {
		fprintf(stderr, "membw_eater: dram worker %d alloc failed\n", idx);
		sh->stop.store(true);
		return;
	}
	memset(src, 1, buf_bytes);
	memset(dst, 0, buf_bytes);

	Pacer pacer;
	pacer.Init((double)chunk * 2.0, target_Bps, &sh->trim);
	size_t off = 0;
	while (!sh->stop.load(std::memory_order_relaxed) && !g_stop) {
		double b0 = NowS();
		memcpy(dst + off, src + off, chunk);
		double b1 = NowS();
		sh->traffic.fetch_add((uint64_t)chunk * 2, std::memory_order_relaxed);
		sh->bursts.fetch_add(1, std::memory_order_relaxed);
		sh->busy_ns.fetch_add((uint64_t)((b1 - b0) * 1e9), std::memory_order_relaxed);
		off += chunk;
		if (off + chunk > buf_bytes) {
			off = 0;
		}
		pacer.Wait();
	}
	free(src);
	free(dst);
}

static int RunDram(Opts &o) {
	size_t buf_bytes = (o.buf_mb ? o.buf_mb : 64) << 20;
	double per_thread_Bps = o.target_gbps > 0 ? o.target_gbps * 1e9 / o.threads : 0.0;
	size_t max_chunk = buf_bytes / 4 < (32u << 20) ? buf_bytes / 4 : (32u << 20);
	size_t chunk =
	    o.chunk_mb ? (o.chunk_mb << 20) : AutoChunk(per_thread_Bps, 2.0, o.pace_s, 1u << 20, max_chunk);
	if (chunk > buf_bytes) {
		chunk = buf_bytes;
	}

	fprintf(stderr, "membw_eater: domain=dram threads=%d cpus=%d..%d target=%s buf=%zuMB/thr chunk=%zuMB\n",
	        o.threads, o.cpu_start, o.cpu_start + o.threads - 1,
	        o.target_gbps > 0 ? (std::to_string(o.target_gbps) + " GB/s").c_str() : "max", buf_bytes >> 20,
	        chunk >> 20);

	DramShared sh;
	std::vector<std::thread> ws;
	for (int i = 0; i < o.threads; i++) {
		ws.emplace_back(DramWorker, &o, &sh, i, buf_bytes, chunk, per_thread_Bps);
	}

	CsvWriter csv;
	csv.Open(o.csv);
	double t_start = NowS(), t_prev = t_start;
	uint64_t traffic_prev = 0, bursts_prev = 0, busy_prev = 0;
	while (!g_stop) {
		SleepUntil(t_prev + o.interval_s);
		double t = NowS();
		uint64_t traffic = sh.traffic.load(), bursts = sh.bursts.load(), busy = sh.busy_ns.load();
		double dt = t - t_prev;
		double ach = (double)(traffic - traffic_prev) / dt / 1e9;
		uint64_t nb = bursts - bursts_prev;
		double duty = (double)(busy - busy_prev) / 1e9 / dt / o.threads;
		csv.Row(t - t_start, "dram", "memcpy", o.target_gbps, ach, nb,
		        nb ? (double)(busy - busy_prev) / 1e6 / nb : 0.0, duty);
		UpdateTrim(sh.trim, o.target_gbps * 1e9, ach * 1e9, duty);
		t_prev = t;
		traffic_prev = traffic;
		bursts_prev = bursts;
		busy_prev = busy;
		if (o.duration_s > 0 && t - t_start >= o.duration_s) {
			break;
		}
		if (sh.stop.load()) {
			break;
		}
	}
	sh.stop.store(true);
	for (auto &w : ws) {
		w.join();
	}
	double t_end = NowS();
	fprintf(stderr, "membw_eater: done. overall achieved %.2f GB/s over %.2fs\n",
	        (double)sh.traffic.load() / (t_end - t_start) / 1e9, t_end - t_start);
	csv.Close();
	return 0;
}

// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
	Opts o = Parse(argc, argv);
	InstallStopHandlers();
	if (o.domain == "dram") {
		return RunDram(o);
	}
#ifndef NO_CUDA
	return RunGpu(o);
#else
	fprintf(stderr, "membw_eater: built without CUDA (NO_CUDA); only --domain dram is available\n");
	return 4;
#endif
}
