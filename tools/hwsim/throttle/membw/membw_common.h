// membw_common.h - shared plumbing for the hwsim memory-bandwidth eater/victim tools.
//
// Plain C++ (no CUDA) so the host-DRAM path builds with g++ alone (-DNO_CUDA).
// See README.md in this directory for usage.

#pragma once

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

namespace membw {

// ----------------------------------------------------------------------------
// time
// ----------------------------------------------------------------------------
inline double NowS() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

inline double EpochS() {
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Sleep until an absolute CLOCK_MONOTONIC deadline (seconds). Sleeps coarsely,
// then spins the last ~100us for accuracy at sub-ms periods.
inline void SleepUntil(double deadline_s) {
	for (;;) {
		double remain = deadline_s - NowS();
		if (remain <= 0) {
			return;
		}
		if (remain > 150e-6) {
			struct timespec ts;
			double coarse = deadline_s - 100e-6;
			ts.tv_sec = (time_t)coarse;
			ts.tv_nsec = (long)((coarse - (double)ts.tv_sec) * 1e9);
			clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
		}
		// spin remainder
		while (NowS() < deadline_s) {
		}
		return;
	}
}

// ----------------------------------------------------------------------------
// stop signal
// ----------------------------------------------------------------------------
inline volatile sig_atomic_t g_stop = 0;

inline void StopHandler(int) {
	g_stop = 1;
}

inline void InstallStopHandlers() {
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = StopHandler;
	sigaction(SIGINT, &sa, nullptr);
	sigaction(SIGTERM, &sa, nullptr);
}

// ----------------------------------------------------------------------------
// pacing: absolute-deadline pacer with catch-up clamp + integral trim
// ----------------------------------------------------------------------------
//
// Each burst moves `chunk` bytes of payload, which corresponds to
// `traffic = chunk * traffic_factor` bytes of consumed bandwidth on the domain
// (factor 2 for copies: read + write; factor 1 for a one-way link crossing).
// The pacer schedules bursts every `period = traffic / target_Bps` seconds
// using absolute deadlines, so controller/launch overhead does not accumulate
// into rate error. If bursts fall behind (saturation), the deadline is clamped
// to at most 2 periods of catch-up credit so the eater does not run flat-out
// for long stretches after a stall. `trim` is a small multiplicative
// correction (closed loop) applied from measured achieved rate.
struct Pacer {
	double period_s = 0.0; // seconds per burst at target rate
	double next_s = 0.0;   // absolute deadline of next burst start
	bool unlimited = true;
	std::atomic<double> *trim = nullptr; // shared closed-loop correction (may be null)

	void Init(double traffic_bytes_per_burst, double target_Bps, std::atomic<double> *trim_p = nullptr) {
		unlimited = target_Bps <= 0.0;
		period_s = unlimited ? 0.0 : traffic_bytes_per_burst / target_Bps;
		next_s = NowS();
		trim = trim_p;
	}

	// Call after each burst completes; blocks until the next burst should start.
	void Wait() {
		if (unlimited) {
			return;
		}
		double t = trim ? trim->load(std::memory_order_relaxed) : 1.0;
		double p = period_s / (t > 0.1 ? t : 0.1);
		next_s += p;
		double now = NowS();
		if (next_s < now - 2.0 * p) { // catch-up clamp
			next_s = now;
		}
		if (next_s > now) {
			SleepUntil(next_s);
		}
	}
};

// Closed-loop trim update, called once per stats interval by the reporter.
// Nudges the effective rate toward target; bounded so it can never run away.
// Skipped when saturated (achieved << target with duty ~1): no amount of trim
// can create bandwidth that isn't there.
inline void UpdateTrim(std::atomic<double> &trim, double target_Bps, double achieved_Bps, double duty) {
	if (target_Bps <= 0 || achieved_Bps <= 0) {
		return;
	}
	if (duty > 0.95) {
		return; // saturated: leave trim alone
	}
	double t = trim.load(std::memory_order_relaxed);
	double err = target_Bps / achieved_Bps; // >1 means we are slow
	// damped multiplicative step, per-interval step clamped to +/-5%
	double step = err;
	if (step > 1.05) step = 1.05;
	if (step < 0.95) step = 0.95;
	t *= step;
	if (t > 1.5) t = 1.5;
	if (t < 0.67) t = 0.67;
	trim.store(t, std::memory_order_relaxed);
}

// ----------------------------------------------------------------------------
// CSV stats
// ----------------------------------------------------------------------------
struct CsvWriter {
	FILE *f = nullptr;
	bool own = false;

	void Open(const std::string &path) {
		if (path.empty() || path == "-") {
			f = stdout;
			own = false;
		} else {
			f = fopen(path.c_str(), "w");
			if (!f) {
				fprintf(stderr, "membw: cannot open csv file %s: %s\n", path.c_str(), strerror(errno));
				exit(2);
			}
			own = true;
		}
		fprintf(f, "epoch_s,elapsed_s,domain,engine,target_gbps,achieved_gbps,bursts,avg_burst_ms,duty\n");
		fflush(f);
	}

	void Row(double elapsed_s, const char *domain, const char *engine, double target_gbps, double achieved_gbps,
	         uint64_t bursts, double avg_burst_ms, double duty) {
		fprintf(f, "%.3f,%.3f,%s,%s,%.2f,%.2f,%" PRIu64 ",%.3f,%.3f\n", EpochS(), elapsed_s, domain, engine,
		        target_gbps, achieved_gbps, bursts, avg_burst_ms, duty);
		fflush(f);
	}

	void Close() {
		if (f && own) {
			fclose(f);
		}
		f = nullptr;
	}
};

// ----------------------------------------------------------------------------
// misc helpers
// ----------------------------------------------------------------------------
inline void PinToCpu(int cpu) {
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	if (sched_setaffinity(0, sizeof(set), &set) != 0) {
		fprintf(stderr, "membw: warning: failed to pin to cpu %d: %s\n", cpu, strerror(errno));
	}
}

// Auto chunk size: aim for one burst per `pace_s` at the target rate so paced
// runs interleave with the victim at fine (~ms) granularity. Clamped so bursts
// stay big enough to amortize launch/syscall overhead and small enough to
// rotate inside the buffer.
inline size_t AutoChunk(double target_Bps, double traffic_factor, double pace_s, size_t min_chunk, size_t max_chunk) {
	if (target_Bps <= 0) {
		return max_chunk; // flat-out mode: use the biggest burst
	}
	double c = target_Bps * pace_s / traffic_factor;
	if (c < (double)min_chunk) {
		return min_chunk;
	}
	if (c > (double)max_chunk) {
		return max_chunk;
	}
	// round to 1 MiB
	size_t mb = (size_t)(c / (1 << 20));
	return (mb > 0 ? mb : 1) << 20;
}

inline double ParseGbps(const char *s) {
	if (strcmp(s, "max") == 0 || strcmp(s, "MAX") == 0) {
		return 0.0;
	}
	return atof(s);
}

} // namespace membw
