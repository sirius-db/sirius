# SPDX-License-Identifier: Apache-2.0
"""Plot the CSVs run_ingest_bench.sh writes: per-column GPU decode throughput for each engine.

Writes <table>.svg (grouped bars, log scale, whole-file TOTAL last) and prints a Markdown table of
throughput and compression ratio. Uses only the standard library.

Usage: plot_ingest_bench.py RESULTS_DIR [--out-dir DIR]
"""

import argparse
import csv
import math
from pathlib import Path

# engine -> (label, colour, throughput column)
ENGINES = {
    "parquet": ("cuDF Parquet", "#2a78d6", "decode_only_gbps_median"),
    "simpatico": ("Simpatico", "#eb6834", "decompress_gbps_median"),
    "vortex_gpu": ("Vortex GPU", "#3a9e5c", "decode_only_gbps_median"),
}


def load(results, engine, table):
    """{column: (gbps, ratio)} from the per-column CSV, plus TOTAL from the full-table one."""
    gbps_key = ENGINES[engine][2]
    out = {}
    for mode in ("per-column", "full-table"):
        path = results / f"{engine}_{table}_{mode}.csv"
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if (row["column"] == "TOTAL") == (mode == "full-table"):
                    out[row["column"]] = (float(row[gbps_key]), float(row["ratio"]))
    return out


def svg_chart(table, columns, data, path):
    W, H, ML, MR, MT, MB = 1100, 520, 70, 20, 50, 110
    x0, x1, y0, y1 = ML, W - MR, MT, H - MB
    values = [v[0] for d in data.values() for v in d.values() if v[0] > 0]
    lo = 10 ** math.floor(math.log10(min(values)))
    hi = 10 ** math.ceil(math.log10(max(values)))

    def y(v):
        return y1 - (y1 - y0) * (math.log10(max(v, lo)) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)
        )

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{ML}" y="24" font-size="16" font-weight="bold">{table}: GPU decode throughput (GB/s, log)</text>',
    ]
    decade = lo
    while decade <= hi:
        for m in (1, 2, 5):
            v = decade * m
            if v <= hi:
                out.append(
                    f'<line x1="{x0}" x2="{x1}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#ddd"/>'
                )
                out.append(
                    f'<text x="{x0 - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>'
                )
        decade *= 10

    engines = list(data)
    group = (x1 - x0) / len(columns)
    bar = group * 0.8 / len(engines)
    for i, col in enumerate(columns):
        gx = x0 + i * group + group * 0.1
        for j, engine in enumerate(engines):
            if col not in data[engine]:
                continue
            gbps, ratio = data[engine][col]
            label, colour, _ = ENGINES[engine]
            out.append(
                f'<rect x="{gx + j * bar:.1f}" y="{y(gbps):.1f}" width="{bar - 1:.1f}" '
                f'height="{y1 - y(gbps):.1f}" fill="{colour}">'
                f"<title>{col} {label}: {gbps:.0f} GB/s, ratio {ratio:.2f}x</title></rect>"
            )
        cx = x0 + (i + 0.5) * group
        out.append(
            f'<text x="{cx:.1f}" y="{y1 + 12}" text-anchor="end" '
            f'transform="rotate(-40 {cx:.1f} {y1 + 12})">{col}</text>'
        )
    for j, engine in enumerate(engines):
        label, colour, _ = ENGINES[engine]
        lx = x0 + j * 140
        out.append(
            f'<rect x="{lx}" y="{H - 22}" width="12" height="12" fill="{colour}"/>'
        )
        out.append(f'<text x="{lx + 18}" y="{H - 12}">{label}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results", type=Path)
    ap.add_argument(
        "--out-dir", type=Path, help="where to write the SVGs (default: RESULTS_DIR)"
    )
    args = ap.parse_args()
    out_dir = args.out_dir or args.results
    out_dir.mkdir(parents=True, exist_ok=True)

    for table in ("lineitem", "orders"):
        data = {e: d for e in ENGINES if (d := load(args.results, e, table))}
        if not data:
            continue
        columns = [c for c in next(iter(data.values())) if c != "TOTAL"]
        columns += [
            c for d in data.values() for c in d if c not in columns and c != "TOTAL"
        ]
        columns.append("TOTAL")
        svg_chart(table, columns, data, out_dir / f"{table}.svg")

        labels = [ENGINES[e][0] for e in data]
        print(f"\n### {table}\n")
        print(
            "| column | " + " | ".join(f"{l} GB/s | {l} ratio" for l in labels) + " |"
        )
        print("|---|" + "---:|---:|" * len(labels))
        for col in columns:
            cells = []
            for d in data.values():
                gbps, ratio = d.get(col, (float("nan"), float("nan")))
                cells += [f"{gbps:.0f}", f"{ratio:.2f}x"]
            print(f"| {col} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
