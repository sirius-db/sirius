#!/usr/bin/env python3
"""Summarizes a scripts/distributed_test_nixl_pairs.sh run: GB/s per phase, next to raw NIXL
writes between fabric-memory buffers with the same phases on two GB200 nodes."""

import pathlib
import re
import sys

# Raw NIXL (UCX cuda_ipc over MNNVL, 1.5 s per phase): total GB/s over the pairs that took part.
# The batches_* rows compare with one write of that size at a time; the exchange keeps several in
# flight.
REFERENCE = {
    "uni_1pair_1G": 771,
    "bidi_1pair_1G": 1535,
    "uni_4pair_1G": 3085,
    "bidi_4pair_1G": 6142,
    "uni_1pair_128M": 698,
    "bidi_1pair_128M": 1388,
    "uni_4pair_128M": 2813,
    "bidi_4pair_128M": 5598,
    "batches_64K": 4,
    "batches_256K": 17,
    "batches_1M": 63,
    "batches_4M": 196,
    "batches_16M": 417,
    "bidi_4pair_1M": 8 * 228,
}
# The bar the e2e test must clear: this share of the reference on one pair, one way, 1 GB.
BAR = ("uni_1pair_1G", 0.90)

PHASE = re.compile(
    r"PAIRS (\S+)\s+rank (\d+) <- (\d+): (\d+) bytes in ([\d.]+) s: ([\d.]+) GB/s"
)
VERIFY = re.compile(r"PAIRS verify rank (\d+) <- (\d+): .* (\d+) bad values")
PATH = re.compile(r"PAIRS rank (\d+) peer (\d+): (.*)")


def main():
    out = pathlib.Path(sys.argv[1])
    phases, order, verify, paths, ucx, failed = {}, [], {}, {}, set(), []
    for log in sorted(out.glob("rank*.log")):
        text = log.read_text(errors="replace")
        if "All tests passed" not in text:
            failed.append(log.name)
        for m in PHASE.finditer(text):
            tag = m.group(1)
            if tag not in phases:
                phases[tag] = {}
                order.append(tag)
            phases[tag][int(m.group(2))] = float(m.group(6))
        for m in VERIFY.finditer(text):
            verify[int(m.group(1))] = int(m.group(3))
        for m in PATH.finditer(text):
            paths[int(m.group(1))] = m.group(3).strip()
        for line in text.splitlines():
            if "from cuda/" in line and "to cuda/" in line and "put" in line:
                ucx.add(re.sub(r"^\[[^]]*\] \[[^]]*\]\s*", "", line).strip())
            elif "cuda_ipc" in line and "|" in line:
                ucx.add(re.sub(r"^\[[^]]*\] \[[^]]*\]\s*", "", line).strip())

    print(f"# [nixl_pairs] {out}\n")
    for name in ("config.txt", "ucx.txt"):
        if (out / name).exists():
            print("```\n" + (out / name).read_text().strip() + "\n```\n")
    print("| phase | total GB/s | per receiver | raw NIXL | share |")
    print("|---|---:|---|---:|---:|")
    for tag in order:
        per = phases[tag]
        total = sum(per.values())
        ref = REFERENCE.get(tag)
        share = f"{100 * total / ref:.0f}%" if ref else ""
        ranks = " ".join(f"r{r}={g:.0f}" for r, g in sorted(per.items()))
        print(f"| {tag} | {total:.1f} | {ranks} | {ref or ''} | {share} |")
    print()
    for r, p in sorted(paths.items()):
        print(f"- rank {r}: {p}")
    bad = sum(verify.values())
    print(f"\nverify: {bad} bad values across {len(verify)} receivers")
    print("\nUCX lines for puts between GPUs:")
    for line in sorted(ucx)[:12]:
        print(f"    {line}")
    tag, share = BAR
    got = sum(phases.get(tag, {}).values())
    passed = got >= share * REFERENCE[tag] and bad == 0 and not failed and verify
    if failed:
        print(f"\nfailed ranks: {', '.join(failed)}")
    print(
        f"\nRESULT: {'PASS' if passed else 'FAIL'} ({tag} {got:.1f} GB/s, bar "
        f"{share * REFERENCE[tag]:.0f} = {share:.0%} of {REFERENCE[tag]}; {bad} bad values)"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
