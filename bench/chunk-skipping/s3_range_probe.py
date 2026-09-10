#!/usr/bin/env python3
"""Does an object store reward reading only the bytes you need?

Takes a PRE-SIGNED GET url (no credentials here) and issues ranged GETs in patterns that
separate the three regimes a ranged read can land in:

  bandwidth-bound   time ~ bytes            -> pruning pays proportionally
  request-bound     time ~ request count    -> pruning HURTS, it makes more requests
  backend-unit      time ~ internal units   -> pruning pays only if skipped extents are big
                                               enough to skip whole units (CHUNK_SKIPPING_PLAN 7.2)

Reports useful GB/s and request count; request count matters on its own because S3 bills per GET.
"""
import argparse, http.client, random, statistics, threading, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

# stdlib only, so this runs on a bare instance with nothing installed. One keep-alive
# connection per worker thread, which is what a real reader would do -- reconnecting per
# range would measure TLS handshakes rather than the object store.
_local = threading.local()


def _conn(url):
    c = getattr(_local, "conn", None)
    if c is None:
        u = urlsplit(url)
        c = (
            http.client.HTTPSConnection(u.netloc, timeout=120)
            if u.scheme == "https"
            else http.client.HTTPConnection(u.netloc, timeout=120)
        )
        _local.conn = c
    return c


def _path(url):
    u = urlsplit(url)
    return u.path + ("?" + u.query if u.query else "")


def _get(url, start, length, retries=2):
    for attempt in range(retries + 1):
        try:
            c = _conn(url)
            t = time.perf_counter()
            c.request(
                "GET",
                _path(url),
                headers={
                    "Range": f"bytes={start}-{start+length-1}",
                    "Host": urlsplit(url).netloc,
                },
            )
            r = c.getresponse()
            body = r.read()
            if r.status not in (200, 206):
                raise RuntimeError(f"HTTP {r.status}: {body[:160]!r}")
            return len(body), (time.perf_counter() - t) * 1000
        except (http.client.HTTPException, OSError):
            try:
                _local.conn.close()
            except Exception:
                pass
            _local.conn = None
            if attempt == retries:
                raise


def run(url, ranges, concurrency):
    """ranges: [(offset,length)]. Returns (seconds, bytes, [latency_ms])."""
    lat, total = [], 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_get, url, o, l) for o, l in ranges]
        for f in futs:
            n, ms = f.result()
            total += n
            lat.append(ms)
    return time.perf_counter() - t0, total, lat


def scattered(size, piece, count, rng):
    """`count` non-overlapping ranges of `piece` bytes at random aligned offsets."""
    slots = size // piece
    if count > slots:
        count = slots
    return [(i * piece, piece) for i in sorted(rng.sample(range(slots), count))]


def contiguous(size, total, rng):
    """One run of `total` bytes at a random offset -- the same volume, zero fragmentation."""
    start = rng.randrange(0, max(1, size - total))
    return [(start, total)]


def fmt(label, secs, nbytes, nreq, lat):
    gbps = nbytes / secs / 1e9
    print(
        f"  {label:<34}{nbytes/1e6:8.0f} MB {nreq:5d} req {secs:7.2f}s "
        f"{gbps*8:7.2f} Gb/s {gbps:6.2f} GB/s  p50 {statistics.median(lat):6.0f}ms"
    )
    return gbps


def preflight(url):
    """Report where we are and where the bucket is -- cross-region silently changes the question
    from "how does S3 behave" to "how fast is the inter-region link", and the numbers look
    plausible either way."""
    bucket_region = "unknown"
    host = urlsplit(url).netloc
    for part in host.split("."):
        if part.startswith("us-") or part.startswith("eu-") or part.startswith("ap-"):
            bucket_region = part
    here = "not on EC2"
    try:
        c = http.client.HTTPConnection("169.254.169.254", timeout=2)
        c.request(
            "PUT",
            "/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        tok = c.getresponse().read().decode()
        c = http.client.HTTPConnection("169.254.169.254", timeout=2)
        c.request(
            "GET",
            "/latest/meta-data/placement/availability-zone",
            headers={"X-aws-ec2-metadata-token": tok},
        )
        here = c.getresponse().read().decode()
        c = http.client.HTTPConnection("169.254.169.254", timeout=2)
        c.request(
            "GET",
            "/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": tok},
        )
        here += " " + c.getresponse().read().decode()
    except Exception:
        pass
    print(
        f"== placement ==\n  reader: {here}\n  bucket host: {host} (region {bucket_region})"
    )
    if bucket_region != "unknown" and not here.startswith(bucket_region):
        print(
            "  !! reader and bucket look CROSS-REGION -- this measures the inter-region link,"
        )
        print("     not S3's own behaviour, and GETs are billed as egress.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--size", type=int, required=True, help="object size in bytes")
    ap.add_argument(
        "--budget-mb", type=int, default=256, help="useful MB per measurement point"
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--phase",
        default="all",
        choices=["all", "latency", "concurrency", "ranges", "striping"],
    )
    a = ap.parse_args()
    rng = random.Random(a.seed)
    preflight(a.url)
    budget = a.budget_mb * 1024 * 1024
    MB = 1024 * 1024

    if a.phase in ("all", "latency"):
        print("\n== latency floor (1 KB GETs, serial) ==")
        _, _, lat = run(
            a.url, [(rng.randrange(0, a.size - 4096), 1024) for _ in range(8)], 1
        )
        lat.sort()
        print(
            f"  min {lat[0]:.0f} ms  median {statistics.median(lat):.0f} ms  max {lat[-1]:.0f} ms"
        )
        print(
            "  (this is the per-request floor: a range smaller than latency x bandwidth is pure overhead)"
        )

    if a.phase in ("all", "concurrency"):
        print("\n== throughput vs concurrency (8 MB ranges) ==")
        for c in (1, 4, 16, 64, 128):
            rs = scattered(a.size, 8 * MB, max(c, budget // (8 * MB)), rng)
            secs, nb, lat = run(a.url, rs, c)
            fmt(f"concurrency {c}", secs, nb, len(rs), lat)

    if a.phase in ("all", "ranges"):
        print("\n== useful throughput vs range size (concurrency 64) -- finds R* ==")
        # Constant REQUEST COUNT, not constant bytes: with a fixed byte budget a 64 MB range
        # yields only 4 requests and measures 4-way concurrency rather than the range size.
        for piece_kb in (64, 256, 1024, 4096, 16384, 65536):
            piece = piece_kb * 1024
            if piece * 64 > a.size:
                continue
            rs = scattered(a.size, piece, 64, rng)
            secs, nb, lat = run(a.url, rs, 64)
            fmt(f"range {piece_kb:6d} KB x64", secs, nb, len(rs), lat)

    if a.phase in ("all", "striping"):
        print(
            "\n== fragmentation at EQUAL volume (concurrency 64) -- the 7.2 question =="
        )
        print(
            "  same bytes, increasing fragmentation. Flat => skipping is free at that grain."
        )
        print(
            "  NB only meaningful BELOW the saturation point found above: if the NIC is capped,"
        )
        print("  every arm hits the cap and fragmentation looks free when it is not.")
        for piece_kb in (65536, 16384, 4096, 1024, 256, 64):
            piece = piece_kb * 1024
            if piece > budget:
                continue
            rs = scattered(a.size, piece, budget // piece, rng)
            secs, nb, lat = run(a.url, rs, 64)
            fmt(f"{budget//piece:5d} x {piece_kb:6d} KB", secs, nb, len(rs), lat)
        # A single contiguous run is inherently serial; report it as the "no parallelism at all"
        # reference rather than as a fragmentation datapoint.
        rs = contiguous(a.size, min(budget, a.size // 2), rng)
        secs, nb, lat = run(a.url, rs, 1)
        fmt("1 contiguous run (serial ref)", secs, nb, len(rs), lat)

        print(
            "\n== does reading LESS take less time? (16 MB ranges, concurrency 64) =="
        )
        full = min(a.size, 1024 * MB)
        for frac in (0.05, 0.25, 0.50, 1.00):
            piece = 16 * MB
            count = max(1, int(full * frac) // piece)
            rs = scattered(a.size, piece, count, rng)
            secs, nb, lat = run(a.url, rs, 64)
            fmt(f"read {frac*100:5.0f}% of {full/1e6:.0f} MB", secs, nb, len(rs), lat)


if __name__ == "__main__":
    main()
