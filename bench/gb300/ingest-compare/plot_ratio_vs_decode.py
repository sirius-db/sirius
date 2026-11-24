#!/usr/bin/env python3
"""Plot PR 2717 compression ratio versus device-resident decode rate."""

import argparse
import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).parent

FORMATS = (
    ("parquet", "Parquet", "#2978b8", "decode_only_gbps_median"),
    ("simpatico", "Simpatico", "#db6c27", "decompress_gbps_median"),
    ("vortex_gpu", "Vortex", "#258d69", "decode_only_gbps_median"),
)
PANELS = (
    ("lineitem", 112, 0.7, 100),
    ("orders", 830, 0.7, 1_000_000),
)
PANEL_WIDTH = 585
TOP, BOTTOM = 166, 622
Y_MIN, Y_MAX = 20, 10_000


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def text(
    x, y, value, *, size=16, fill="#252d36", anchor="start", weight="normal", extra=""
):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" {extra}>'
        f"{html.escape(str(value))}</text>"
    )


def log_position(value, start, end, minimum, maximum):
    fraction = (math.log10(value) - math.log10(minimum)) / (
        math.log10(maximum) - math.log10(minimum)
    )
    return start + fraction * (end - start)


def x_position(value, left, minimum, maximum):
    return log_position(value, left, left + PANEL_WIDTH, minimum, maximum)


def y_position(value):
    return log_position(value, BOTTOM, TOP, Y_MIN, Y_MAX)


def point(x, y, color, label, *, whole_table=False):
    title = f"<title>{html.escape(label)}</title>"
    if whole_table:
        radius = 9
        path = f"M{x:.2f},{y-radius:.2f} L{x+radius:.2f},{y:.2f} L{x:.2f},{y+radius:.2f} L{x-radius:.2f},{y:.2f} Z"
        return (
            f'<path d="{path}" fill="white" stroke="{color}" stroke-width="3.5">'
            f"{title}</path>"
        )
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.1" fill="{color}" '
        f'fill-opacity="0.79" stroke="white" stroke-width="1.1">{title}</circle>'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "figures" / "ratio-vs-decode.svg"
    )
    parser.add_argument(
        "--png-output", type=Path, help="also render a PNG with system librsvg"
    )
    args = parser.parse_args()
    results = args.results_dir
    output = args.output
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="760" viewBox="0 0 1500 760">',
        '<rect width="1500" height="760" fill="white"/>',
        '<g font-family="DejaVu Sans, Arial, sans-serif">',
        text(
            112,
            73,
            "SF1000 • one partition per table • median of 10 measured runs",
            size=15,
            fill="#5d6670",
        ),
    ]

    # A single legend applies to both table panels.
    for index, (_, name, color, _) in enumerate(FORMATS):
        lx = 745 + index * 155
        svg += [point(lx, 68, color, name), text(lx + 14, 74, name, size=16)]
    svg += [
        point(1255, 68, "#69727c", "whole table", whole_table=True),
        text(1270, 74, "whole table", size=15),
    ]

    for table, left, x_min, x_max in PANELS:
        right = left + PANEL_WIDTH
        svg += [text(left, 137, table, size=21, weight="bold")]
        for rate in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000):
            y = y_position(rate)
            svg += [
                f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" '
                f'stroke="#e9edf1" stroke-width="1"/>'
            ]
            if table == "lineitem" and rate in (20, 100, 1000, 10000):
                svg += [
                    text(
                        left - 13,
                        y + 5,
                        f"{rate:,}",
                        size=13,
                        anchor="end",
                        fill="#5d6670",
                    )
                ]
        for exponent in range(
            math.ceil(math.log10(x_min)), math.floor(math.log10(x_max)) + 1
        ):
            ratio = 10**exponent
            x = x_position(ratio, left, x_min, x_max)
            svg += [
                f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{TOP}" y2="{BOTTOM}" '
                f'stroke="#e9edf1" stroke-width="1"/>',
                text(
                    x,
                    BOTTOM + 27,
                    f"{ratio:,}×",
                    size=13,
                    anchor="middle",
                    fill="#5d6670",
                ),
            ]
        svg += [
            f'<path d="M{left},{TOP} V{BOTTOM} H{right}" fill="none" '
            f'stroke="#5d6670" stroke-width="1.25"/>'
        ]

        # Plot the individual columns first, then the larger whole-table symbols.
        for fmt, name, color, rate_key in FORMATS:
            rows = read_rows(results / f"{fmt}_{table}_per-column.csv")
            for row in rows:
                if row["column"] == "TOTAL":
                    continue
                ratio, rate = float(row["ratio"]), float(row[rate_key])
                label = (
                    f"{table}.{row['column']} — {name}: {ratio:.2f}×, {rate:,.1f} GB/s"
                )
                svg += [
                    point(
                        x_position(ratio, left, x_min, x_max),
                        y_position(rate),
                        color,
                        label,
                    )
                ]

        for fmt, name, color, rate_key in FORMATS:
            row = read_rows(results / f"{fmt}_{table}_full-table.csv")[0]
            ratio, rate = float(row["ratio"]), float(row[rate_key])
            label = f"{table} whole table — {name}: {ratio:.2f}×, {rate:,.1f} GB/s"
            svg += [
                point(
                    x_position(ratio, left, x_min, x_max),
                    y_position(rate),
                    color,
                    label,
                    whole_table=True,
                )
            ]

        if table == "orders":
            # This nearly constant column is far outside the other ratios.
            row = next(
                row
                for row in read_rows(results / "vortex_gpu_orders_per-column.csv")
                if row["column"] == "o_shippriority"
            )
            x = x_position(float(row["ratio"]), left, x_min, x_max)
            y = y_position(float(row["decode_only_gbps_median"]))
            svg += [
                f'<path d="M{x-9:.1f},{y-7:.1f} L{x-45:.1f},{y-26:.1f}" '
                'fill="none" stroke="#58636e" stroke-width="1"/>',
                text(
                    x - 50,
                    y - 29,
                    "o_shippriority",
                    size=13,
                    anchor="end",
                    fill="#58636e",
                ),
            ]

        svg += [
            text(
                (left + right) / 2,
                687,
                "Compression ratio (×, log scale)",
                size=16,
                anchor="middle",
            )
        ]

    svg += [
        text(
            33,
            (TOP + BOTTOM) / 2,
            "Decode throughput (GB/s, log scale)",
            size=16,
            anchor="middle",
            extra='transform="rotate(-90 33 394)"',
        ),
        text(
            112,
            730,
            "Dots: individual columns; diamonds: whole-table result. Rates use decoded logical bytes.",
            size=14,
            fill="#5d6670",
        ),
        "</g></svg>",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n")
    print(output)
    if args.png_output:
        import gi

        gi.require_version("Rsvg", "2.0")
        from gi.repository import Rsvg

        args.png_output.parent.mkdir(parents=True, exist_ok=True)
        Rsvg.Handle.new_from_file(str(output)).get_pixbuf().savev(
            str(args.png_output), "png", [], []
        )
        print(args.png_output)


if __name__ == "__main__":
    main()
