#!/usr/bin/env python3
"""Render the saved GB300 host- and device-limit spill sweeps for a paper."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


GIB = 1024**3
SCALES = (1000, 3000, 10000)
HOST_CAPS = (32, 64, 128, 256, 450)
GPU_CAPS = (32, 64, 128, 256)
TRANSITIONS = ("gpu_to_host_bytes", "host_to_disk_bytes", "gpu_to_disk_bytes")
COLORS = {
    "runtime": "#262626",
    "gpu_to_host": "#0072B2",
    "host_to_disk": "#E69F00",
    "gpu_to_disk": "#D55E00",
}


@dataclass(frozen=True)
class Point:
    scale: int
    sweep: str
    cap: int
    runtime_s: float
    gpu_to_host_gib: float
    host_to_disk_gib: float
    gpu_to_disk_gib: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_point(scale: int, sweep: str, cap: int, repo: Path) -> Point:
    if sweep == "host":
        folder = repo / "bench/gb300-spill-sweep/results" / f"sf{scale}"
        query_file = folder / "per-query.csv"
        total_file = folder / "totals.csv"
        cap_field = "host_cap_gib"
    else:
        folder = repo / "bench/gb300-device-spill-sweep/results"
        stem = f"sf{scale}-gpu{cap}gb-host384gib"
        query_file = folder / f"{stem}-per-query.csv"
        total_file = folder / f"{stem}-totals.csv"
        cap_field = "gpu_cap_gb"

    queries = [row for row in read_csv(query_file) if int(row[cap_field]) == cap]
    totals = [row for row in read_csv(total_file) if int(row[cap_field]) == cap]
    if len(queries) != 22 or len(totals) != 1:
        raise ValueError(f"expected 22 queries and one total: {sweep} SF{scale} {cap}")
    if any(row["status"] != "success" for row in queries):
        raise ValueError(f"incomplete query: {sweep} SF{scale} {cap}")
    if len({row["query"] for row in queries}) != 22:
        raise ValueError(f"duplicate query: {sweep} SF{scale} {cap}")
    total = totals[0]
    if any(
        int(total[field]) != sum(int(row[field]) for row in queries)
        for field in TRANSITIONS
    ):
        raise ValueError(
            f"movement total disagrees with queries: {sweep} SF{scale} {cap}"
        )
    if any(
        int(total[field]) != 22
        for field in ("completed_queries", "successful_queries", "requested_queries")
    ):
        raise ValueError(f"incomplete total: {sweep} SF{scale} {cap}")

    return Point(
        scale=scale,
        sweep=sweep,
        cap=cap,
        runtime_s=sum(float(row["runtime_s"]) for row in queries),
        gpu_to_host_gib=int(total["gpu_to_host_bytes"]) / GIB,
        host_to_disk_gib=int(total["host_to_disk_bytes"]) / GIB,
        gpu_to_disk_gib=int(total["gpu_to_disk_bytes"]) / GIB,
    )


def write_data(points: list[Point], output: Path) -> None:
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "scale_factor",
                "sweep",
                "cap",
                "cap_unit",
                "runtime_s",
                "gpu_to_host_gib",
                "host_to_disk_gib",
                "gpu_to_disk_gib",
            )
        )
        for point in points:
            writer.writerow(
                (
                    point.scale,
                    point.sweep,
                    point.cap,
                    "GiB" if point.sweep == "host" else "GB",
                    f"{point.runtime_s:.6f}",
                    f"{point.gpu_to_host_gib:.6f}",
                    f"{point.host_to_disk_gib:.6f}",
                    f"{point.gpu_to_disk_gib:.6f}",
                )
            )


def render(points: list[Point], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(11.0, 6.3), facecolor="white")
    outer = GridSpec(
        2,
        3,
        figure=fig,
        left=0.07,
        right=0.99,
        bottom=0.13,
        top=0.91,
        wspace=0.34,
        hspace=0.45,
    )

    legend_handles = [
        plt.Line2D(
            [0], [0], color=COLORS["runtime"], marker="o", lw=1.8, label="Query time"
        ),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["gpu_to_host"], label="GPU→HOST"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=COLORS["host_to_disk"], label="HOST→DISK"
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS["gpu_to_disk"],
            hatch="////",
            edgecolor="#8E3C00",
            label="GPU→DISK",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.5,
        fontsize=10,
    )

    for row, sweep in enumerate(("host", "gpu")):
        for column, scale in enumerate(SCALES):
            row_points = [point for point in points if point.scale == scale]
            time_max = max(point.runtime_s for point in row_points) * 1.32
            selected = sorted(
                (point for point in row_points if point.sweep == sweep),
                key=lambda point: point.cap,
            )
            inner = outer[row, column].subgridspec(
                2, 1, height_ratios=(0.9, 1.15), hspace=0.08
            )
            time_ax = fig.add_subplot(inner[0])
            move_ax = fig.add_subplot(inner[1], sharex=time_ax)
            x = list(range(len(selected)))
            times = [point.runtime_s for point in selected]
            time_ax.plot(x, times, color=COLORS["runtime"], marker="o", ms=4.5, lw=1.8)
            time_ax.set_ylim(0, time_max)
            time_ax.set_xlim(-0.52, len(selected) - 0.48)
            if column == 0:
                time_ax.set_ylabel("Time (s)")
            time_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
            time_ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
            time_ax.tick_params(axis="x", bottom=False, labelbottom=False)
            for xi, runtime in zip(x, times):
                time_ax.annotate(
                    f"{runtime:,.0f}",
                    (xi, runtime),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            width = 0.30
            gpu_host = [point.gpu_to_host_gib for point in selected]
            host_disk = [point.host_to_disk_gib for point in selected]
            gpu_disk = [point.gpu_to_disk_gib for point in selected]
            move_ax.bar(
                [xi - width / 2 for xi in x],
                gpu_host,
                width=width,
                color=COLORS["gpu_to_host"],
                linewidth=0,
            )
            move_ax.bar(
                [xi + width / 2 for xi in x],
                host_disk,
                width=width,
                color=COLORS["host_to_disk"],
                linewidth=0,
            )
            move_ax.bar(
                [xi + width / 2 for xi in x],
                gpu_disk,
                bottom=host_disk,
                width=width,
                color=COLORS["gpu_to_disk"],
                hatch="////",
                edgecolor="#8E3C00",
                linewidth=0.5,
            )
            move_max = max(
                max(gpu_host), max(a + b for a, b in zip(host_disk, gpu_disk))
            )
            move_ax.set_ylim(0, move_max * 1.14 if move_max else 1)
            if column == 0:
                move_ax.set_ylabel("Moves (GiB)")
            move_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
            move_ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
            move_ax.set_axisbelow(True)
            move_ax.set_xticks(x, [str(point.cap) for point in selected])
            cap_label = "Host cap (GiB)" if sweep == "host" else "GPU cap (GB)"
            move_ax.set_xlabel(f"SF {scale:,} · {cap_label}")
            if not any(host_disk) and not any(gpu_disk):
                move_ax.text(
                    0.99,
                    0.94,
                    "No disk moves",
                    transform=move_ax.transAxes,
                    ha="right",
                    va="top",
                    color="#555555",
                    fontsize=9,
                )
    fig.text(
        0.07,
        0.025,
        "Host-cap sweep: GPU default, host pools scaled. GPU-cap sweep: host fixed at 384 GiB.",
        ha="left",
        fontsize=8.5,
        color="#444444",
    )
    stem = output_dir / "gb300-spill-sweeps"
    for suffix in ("pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), facecolor="white")
    svg = stem.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    points = [
        read_point(scale, sweep, cap, repo)
        for scale in SCALES
        for sweep, caps in (("host", HOST_CAPS), ("gpu", GPU_CAPS))
        for cap in caps
    ]
    write_data(points, output_dir / "gb300-spill-sweeps-data.csv")
    render(points, output_dir)
    print(f"Saved figure and aggregated data to {output_dir}")


if __name__ == "__main__":
    main()
