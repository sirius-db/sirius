#!/usr/bin/env python3
"""Render the saved GB300 concurrency and corrected ablation studies."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


SCALES = (100, 300, 1000)
STREAMS = (1, 2, 4, 8, 10)
COLORS = {100: "#0072B2", 300: "#009E73", 1000: "#D55E00"}
ABLATIONS = (
    (
        "Type narrowing first",
        "narrow-first-summary.csv",
        (
            "all_off",
            "type_narrow",
            "build_probe",
            "dynamic_filter",
            "compressed_pin",
            "fused_decode",
        ),
        (
            "Off",
            "Type\nnarrow",
            "BUILD\nPROBE",
            "Dynamic\nfilter",
            "Compressed\npin",
            "Fused\ndecode",
        ),
        "#0072B2",
    ),
    (
        "BUILD_PROBE first",
        "join-first-corrected-summary.csv",
        (
            "all_off",
            "build_probe",
            "dynamic_filter",
            "type_narrow",
            "compressed_pin",
            "fused_decode",
        ),
        (
            "Off",
            "BUILD\nPROBE",
            "Dynamic\nfilter",
            "Type\nnarrow",
            "Compressed\npin",
            "Fused\ndecode",
        ),
        "#009E73",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_concurrency(path: Path) -> dict[int, list[dict[str, str]]]:
    rows = read_csv(path)
    if len(rows) != len(SCALES) * len(STREAMS):
        raise ValueError("expected all 15 concurrency measurements")
    by_scale = {}
    for scale in SCALES:
        selected = sorted(
            (row for row in rows if int(row["scale_factor"]) == scale),
            key=lambda row: int(row["concurrency"]),
        )
        if tuple(int(row["concurrency"]) for row in selected) != STREAMS:
            raise ValueError(f"missing concurrency case at SF{scale}")
        for row in selected:
            streams = int(row["concurrency"])
            if int(row["query_executions"]) != 22 * streams:
                raise ValueError(f"wrong query count at SF{scale}, {streams} streams")
            calculated = int(row["query_executions"]) / float(row["wall_seconds"])
            if abs(calculated - float(row["queries_per_second"])) > 0.000001:
                raise ValueError(f"throughput disagrees with wall time: SF{scale}")
        by_scale[scale] = selected
    return by_scale


def check_ablation(path: Path, expected: tuple[str, ...]) -> list[float]:
    rows = read_csv(path)
    if len(rows) != 6 or tuple(row["optimization_added"] for row in rows) != expected:
        raise ValueError(f"unexpected ablation stages: {path}")
    if any(row["status"] != "success" for row in rows):
        raise ValueError(f"incomplete ablation stage: {path}")
    times = [float(row["query_total_s"]) for row in rows]
    for previous, current, row in zip(times, times[1:], rows[1:]):
        if abs(current - previous - float(row["change_from_previous_s"])) > 0.00001:
            raise ValueError(f"inconsistent ablation increment: {path}")
    return times


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def export(fig: plt.Figure, stem: Path) -> None:
    for suffix in ("pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), facecolor="white")
    svg = stem.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def concurrency_figure(by_scale: dict[int, list[dict[str, str]]], stem: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.19, top=0.77, wspace=0.25)
    for scale in SCALES:
        rates = [float(row["queries_per_second"]) for row in by_scale[scale]]
        for ax, values in zip(axes, (rates, [rate / rates[0] for rate in rates])):
            ax.plot(
                STREAMS,
                values,
                color=COLORS[scale],
                marker="o",
                markersize=5.5,
                linewidth=2,
                label=f"SF {scale:,}",
            )
    for ax in axes:
        ax.set_xlim(0.7, 10.3)
        ax.set_xticks(STREAMS)
        ax.set_xlabel("Concurrent query streams")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 29)
    axes[0].yaxis.set_major_locator(MultipleLocator(5))
    axes[0].set_title("A   Aggregate throughput", loc="left", weight="bold")
    axes[0].set_ylabel("Completed queries per second")
    axes[1].set_ylim(0.8, 2.25)
    axes[1].yaxis.set_major_locator(MultipleLocator(0.25))
    axes[1].set_title("B   Gain over one stream", loc="left", weight="bold")
    axes[1].set_ylabel("Throughput relative to one stream")
    fig.text(
        0.5,
        0.955,
        "TPC-H Parquet concurrency on GB300",
        ha="center",
        va="top",
        weight="bold",
        fontsize=14,
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.075,
        0.06,
        "One pass per point; randomized query order and predicates. Union of input columns pinned on host before timing.",
        ha="left",
        fontsize=8.8,
        color="#444444",
    )
    export(fig, stem)


def ablation_figure(results: list[list[float]], stem: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), facecolor="white", sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.24, top=0.80, wspace=0.14)
    for index, (ax, config, times) in enumerate(zip(axes, ABLATIONS, results)):
        title, _, _, labels, color = config
        positions = list(range(6))
        ax.plot(positions, times, marker="o", color=color, markersize=6, linewidth=2.2)
        for position, value in zip(positions, times):
            ax.annotate(
                f"{value:.2f}",
                (position, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )
        ax.set_xticks(positions, labels)
        ax.set_xlim(-0.25, 5.25)
        ax.set_ylim(0, 25.5)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_title(f"{chr(65 + index)}   {title}", loc="left", weight="bold")
        ax.set_xlabel("Optimization enabled at this step")
    axes[0].set_ylabel("Sum of 22 query times (s)")
    fig.text(
        0.5,
        0.955,
        "Corrected cumulative ablations · SF1k host pin",
        ha="center",
        va="top",
        weight="bold",
        fontsize=14,
    )
    fig.text(
        0.075,
        0.07,
        "Each stage runs once in a fresh process; pin/unpin excluded. Gains depend on the order and enabled context.",
        ha="left",
        fontsize=8.8,
        color="#444444",
    )
    export(fig, stem)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    destination = Path(__file__).resolve().parent
    by_scale = check_concurrency(repo / "bench/gb300-results/concurrency/summary.csv")
    results = [
        check_ablation(repo / "bench/gb300-ablation/results" / filename, expected)
        for _, filename, expected, _, _ in ABLATIONS
    ]
    concurrency_figure(by_scale, destination / "gb300-concurrency")
    ablation_figure(results, destination / "gb300-cumulative-ablation")
    print(f"Saved concurrency and ablation figures to {destination}")


if __name__ == "__main__":
    main()
