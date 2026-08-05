/*
 * io_load — O_DIRECT read-load generator for hwsim I/O throttling (Workstream 3).
 *
 * Two roles, one binary:
 *
 *   Injector (throttled):  --rate <GB/s>   (or --fraction F --baseline-gbps B)
 *       Issues O_DIRECT reads against a scratch file on the target device,
 *       discards the bytes, and holds a target aggregate rate with a token
 *       bucket. A closed-loop controller raises/lowers the active queue
 *       depth so the target is met even as device contention varies.
 *
 *   Victim / baseline (unlimited):  --rate 0   (default)
 *       Reads flat-out with N parallel sequential streams (like a parquet
 *       scan) and reports achieved bandwidth. Used to measure the device
 *       baseline and the victim's achieved rate under injection.
 *
 * All reads are O_DIRECT (page cache bypassed). Buffers are 4096-aligned,
 * request size is a multiple of 4096, offsets are multiples of the request
 * size — this satisfies O_DIRECT alignment on 512e/4Kn devices.
 *
 * Units: rates are decimal (1 MB/s = 1e6 B/s, 1 GB/s = 1e9 B/s).
 *
 * SIGINT/SIGTERM stop the run cleanly (summary + CSV still written).
 *
 * Build: make   (cc -O2 -pthread io_load.c -o io_load)
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ALIGN 4096
#define TICK_MS 100            /* controller / CSV tick */
#define BURST_TICKS 2          /* token bucket cap = BURST_TICKS * tick budget */
#define CTRL_WINDOW 5          /* ticks in the controller's moving average */

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

typedef struct {
    /* config */
    const char *path;
    double rate_bps;           /* 0 => unlimited */
    int threads;               /* max queue depth */
    size_t req_size;
    double duration_s;         /* measured window (after warmup) */
    double warmup_s;
    bool random_mode;
    const char *csv_path;
    /* runtime */
    int fd;
    uint64_t span;             /* file size rounded down to req_size multiple */
    uint64_t nblocks;
    _Atomic int64_t bucket;    /* token bucket, bytes */
    _Atomic uint64_t bytes_done;
    _Atomic int active_qd;
} ctx_t;

static ctx_t g;

typedef struct {
    int idx;
    uint64_t seed;
    pthread_t tid;
} worker_t;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void sleep_ns(long ns) {
    struct timespec ts = { ns / 1000000000L, ns % 1000000000L };
    nanosleep(&ts, NULL);
}

static uint64_t xorshift64(uint64_t x) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return x;
}

static void *worker(void *arg) {
    worker_t *w = (worker_t *)arg;
    void *buf = NULL;
    if (posix_memalign(&buf, ALIGN, g.req_size) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return NULL;
    }
    /* N parallel sequential streams: thread i starts at its own region and
     * strides forward by req_size, wrapping at the end of the file. */
    uint64_t off = (g.nblocks / (uint64_t)g.threads) * (uint64_t)w->idx * g.req_size;
    uint64_t rng = w->seed;

    while (!g_stop) {
        if (w->idx >= atomic_load_explicit(&g.active_qd, memory_order_relaxed)) {
            sleep_ns(1000000); /* parked by the controller */
            continue;
        }
        if (g.rate_bps > 0) { /* debit the token bucket before issuing */
            int64_t need = (int64_t)g.req_size;
            int64_t cur = atomic_load_explicit(&g.bucket, memory_order_relaxed);
            bool got = false;
            while (cur >= need) {
                if (atomic_compare_exchange_weak(&g.bucket, &cur, cur - need)) {
                    got = true;
                    break;
                }
            }
            if (!got) { sleep_ns(200000); continue; }
        }
        ssize_t n = pread(g.fd, buf, g.req_size, (off_t)off);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("pread");
            g_stop = 1;
            break;
        }
        atomic_fetch_add_explicit(&g.bytes_done, (uint64_t)n, memory_order_relaxed);
        if (g.random_mode) {
            rng = xorshift64(rng);
            off = (rng % g.nblocks) * g.req_size;
        } else {
            off += g.req_size;
            if (off >= g.span) off = 0;
        }
    }
    free(buf);
    return NULL;
}

/* Create a scratch file with O_DIRECT sequential writes. Real extents are
 * written on purpose: a bare fallocate() leaves unwritten extents that ext4
 * serves as zero-fill without ever touching the media, which would make read
 * "benchmarks" measure nothing. */
static int make_file(const char *path, double size_gib) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) {
        fprintf(stderr, "open(%s, O_DIRECT|O_CREAT): %s\n", path, strerror(errno));
        return 1;
    }
    const size_t chunk = 4 * 1024 * 1024;
    void *buf = NULL;
    if (posix_memalign(&buf, ALIGN, chunk) != 0) return 1;
    memset(buf, 0, chunk);
    uint64_t total = (uint64_t)(size_gib * 1024.0 * 1024.0 * 1024.0);
    total = total / chunk * chunk;
    double t0 = now_s();
    for (uint64_t done = 0; done < total && !g_stop; done += chunk) {
        ssize_t n = write(fd, buf, chunk);
        if (n != (ssize_t)chunk) { perror("write"); close(fd); return 1; }
    }
    fsync(fd);
    close(fd);
    free(buf);
    printf("MKFILE file=%s bytes=%" PRIu64 " elapsed_s=%.2f write_mbps=%.0f\n",
           path, total, now_s() - t0, (double)total / (now_s() - t0) / 1e6);
    return 0;
}

/* Drop PATH's pages from the page cache (owner-permission, no root needed).
 * Useful before validation runs: the injector only degrades *device* reads,
 * so a buffered victim (e.g. Sirius parquet scans) must start cold. */
static int evict_file(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "open(%s): %s\n", path, strerror(errno));
        return 1;
    }
    int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    if (rc != 0) {
        fprintf(stderr, "posix_fadvise(%s): %s\n", path, strerror(rc));
        return 1;
    }
    printf("EVICT file=%s ok\n", path);
    return 0;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --file PATH [options]\n"
        "  --file PATH          file to read with O_DIRECT (required)\n"
        "  --mkfile GIB         create PATH as a GIB-GiB scratch file (O_DIRECT writes) and exit\n"
        "  --evict              drop PATH from the page cache (fadvise DONTNEED) and exit\n"
        "  --rate GBPS          target aggregate rate, decimal GB/s; 0 = unlimited (default 0)\n"
        "  --fraction F         alternative to --rate: F * --baseline-gbps\n"
        "  --baseline-gbps B    measured baseline for --fraction\n"
        "  --threads N          max queue depth / parallel streams (default 8)\n"
        "  --req-kb K           request size in KiB, multiple of 4 (default 1024)\n"
        "  --duration S         measured seconds after warmup (default 10)\n"
        "  --warmup S           seconds excluded from the summary (default 0)\n"
        "  --rand               random offsets instead of sequential streams\n"
        "  --csv PATH           per-tick CSV: time_s,target_mbps,achieved_mbps,active_qd,req_kb,phase\n",
        argv0);
}

int main(int argc, char **argv) {
    double rate_gbps = 0, fraction = -1, baseline_gbps = 0, mkfile_gib = 0;
    bool do_evict = false;
    g.threads = 8;
    g.req_size = 1024 * 1024;
    g.duration_s = 10;
    g.warmup_s = 0;

    static struct option opts[] = {
        {"file", required_argument, 0, 'f'},
        {"mkfile", required_argument, 0, 'M'},
        {"evict", no_argument, 0, 'E'},
        {"rate", required_argument, 0, 'r'},
        {"fraction", required_argument, 0, 'F'},
        {"baseline-gbps", required_argument, 0, 'B'},
        {"threads", required_argument, 0, 't'},
        {"req-kb", required_argument, 0, 'k'},
        {"duration", required_argument, 0, 'd'},
        {"warmup", required_argument, 0, 'w'},
        {"rand", no_argument, 0, 'R'},
        {"csv", required_argument, 0, 'c'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}};
    int c;
    while ((c = getopt_long(argc, argv, "h", opts, NULL)) != -1) {
        switch (c) {
        case 'f': g.path = optarg; break;
        case 'M': mkfile_gib = atof(optarg); break;
        case 'E': do_evict = true; break;
        case 'r': rate_gbps = atof(optarg); break;
        case 'F': fraction = atof(optarg); break;
        case 'B': baseline_gbps = atof(optarg); break;
        case 't': g.threads = atoi(optarg); break;
        case 'k': g.req_size = (size_t)atoi(optarg) * 1024; break;
        case 'd': g.duration_s = atof(optarg); break;
        case 'w': g.warmup_s = atof(optarg); break;
        case 'R': g.random_mode = true; break;
        case 'c': g.csv_path = optarg; break;
        default: usage(argv[0]); return 2;
        }
    }
    if (!g.path || g.threads < 1 || g.req_size % ALIGN != 0 || g.req_size == 0) {
        usage(argv[0]);
        return 2;
    }
    if (do_evict) return evict_file(g.path);
    if (mkfile_gib > 0) {
        struct sigaction msa = {0};
        msa.sa_handler = on_signal;
        sigaction(SIGINT, &msa, NULL);
        sigaction(SIGTERM, &msa, NULL);
        return make_file(g.path, mkfile_gib);
    }
    if (fraction >= 0) {
        if (baseline_gbps <= 0) {
            fprintf(stderr, "--fraction requires --baseline-gbps\n");
            return 2;
        }
        rate_gbps = fraction * baseline_gbps;
    }
    g.rate_bps = rate_gbps * 1e9;

    g.fd = open(g.path, O_RDONLY | O_DIRECT);
    if (g.fd < 0) {
        fprintf(stderr, "open(%s, O_DIRECT): %s\n", g.path, strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(g.fd, &st) != 0) { perror("fstat"); return 1; }
    g.span = ((uint64_t)st.st_size / g.req_size) * g.req_size;
    g.nblocks = g.span / g.req_size;
    if (g.nblocks < (uint64_t)g.threads) {
        fprintf(stderr, "file too small: need >= threads*req_size (%d * %zu)\n",
                g.threads, g.req_size);
        return 1;
    }

    /* Throttled mode starts at a small queue depth and grows on demand;
     * unlimited mode uses all threads from the start. */
    atomic_store(&g.active_qd, g.rate_bps > 0 ? (g.threads < 2 ? g.threads : 2)
                                              : g.threads);
    /* pre-fill one tick's budget so the first 100 ms isn't a hole */
    atomic_store(&g.bucket, (int64_t)(g.rate_bps * TICK_MS / 1000.0));

    struct sigaction sa = {0};
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    FILE *csv = NULL;
    if (g.csv_path) {
        csv = fopen(g.csv_path, "w");
        if (!csv) { perror("fopen csv"); return 1; }
        fprintf(csv, "time_s,target_mbps,achieved_mbps,active_qd,req_kb,phase\n");
    }

    worker_t *ws = calloc((size_t)g.threads, sizeof(worker_t));
    for (int i = 0; i < g.threads; i++) {
        ws[i].idx = i;
        ws[i].seed = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)(i + 1) * 0xbf58476d1ce4e5b9ULL);
        pthread_create(&ws[i].tid, NULL, worker, &ws[i]);
    }

    const double tick_s = TICK_MS / 1000.0;
    const double target_mbps = g.rate_bps / 1e6;
    const int64_t tick_budget = (int64_t)(g.rate_bps * tick_s);
    const int64_t burst_cap = tick_budget * BURST_TICKS;

    double t0 = now_s(), t_meas_start = t0 + g.warmup_s;
    double t_end = t_meas_start + g.duration_s;
    uint64_t last_bytes = 0, meas_start_bytes = 0;
    double window[CTRL_WINDOW] = {0};
    int wpos = 0, wfill = 0, deficit_ticks = 0, surplus_ticks = 0;
    bool meas_started = (g.warmup_s <= 0);

    while (!g_stop) {
        sleep_ns((long)(tick_s * 1e9));
        double t = now_s();
        if (t >= t_end) break;

        if (g.rate_bps > 0) { /* refill token bucket, capped to bound bursts */
            int64_t cur = atomic_load(&g.bucket);
            int64_t next = cur + tick_budget;
            if (next > burst_cap) next = burst_cap;
            atomic_store(&g.bucket, next);
        }

        uint64_t bytes = atomic_load(&g.bytes_done);
        double tick_mbps = (double)(bytes - last_bytes) / tick_s / 1e6;
        last_bytes = bytes;
        if (!meas_started && t >= t_meas_start) {
            meas_started = true;
            meas_start_bytes = bytes;
            t_meas_start = t;
        }

        int qd = atomic_load(&g.active_qd);
        if (csv)
            fprintf(csv, "%.3f,%.1f,%.1f,%d,%zu,%s\n", t - t0, target_mbps,
                    tick_mbps, qd, g.req_size / 1024,
                    meas_started ? "measure" : "warmup");

        if (g.rate_bps > 0) { /* closed-loop queue-depth controller */
            window[wpos] = tick_mbps;
            wpos = (wpos + 1) % CTRL_WINDOW;
            if (wfill < CTRL_WINDOW) wfill++;
            double avg = 0;
            for (int i = 0; i < wfill; i++) avg += window[i];
            avg /= wfill;
            if (wfill == CTRL_WINDOW && avg < 0.97 * target_mbps) {
                if (++deficit_ticks >= 2 && qd < g.threads) {
                    atomic_store(&g.active_qd, qd + 1);
                    deficit_ticks = 0;
                }
                surplus_ticks = 0;
            } else if (wfill == CTRL_WINDOW && avg >= 0.995 * target_mbps) {
                if (++surplus_ticks >= 20 && qd > 2) {
                    atomic_store(&g.active_qd, qd - 1);
                    surplus_ticks = 0;
                }
                deficit_ticks = 0;
            } else {
                deficit_ticks = surplus_ticks = 0;
            }
        }
    }

    double t_stop = now_s();
    g_stop = 1;
    for (int i = 0; i < g.threads; i++) pthread_join(ws[i].tid, NULL);

    uint64_t total = atomic_load(&g.bytes_done);
    uint64_t meas_bytes = meas_started ? total - meas_start_bytes : 0;
    double meas_elapsed = meas_started ? t_stop - t_meas_start : 0;
    double achieved_mbps = meas_elapsed > 0 ? (double)meas_bytes / meas_elapsed / 1e6 : 0;

    printf("SUMMARY file=%s mode=%s pattern=%s req_kb=%zu threads=%d "
           "target_mbps=%.1f achieved_mbps=%.1f achieved_gbps=%.3f "
           "measured_s=%.2f bytes=%" PRIu64 " final_qd=%d\n",
           g.path, g.rate_bps > 0 ? "throttled" : "unlimited",
           g.random_mode ? "rand" : "seq", g.req_size / 1024, g.threads,
           target_mbps, achieved_mbps, achieved_mbps / 1000.0, meas_elapsed,
           meas_bytes, atomic_load(&g.active_qd));

    if (csv) fclose(csv);
    free(ws);
    close(g.fd);
    return 0;
}
