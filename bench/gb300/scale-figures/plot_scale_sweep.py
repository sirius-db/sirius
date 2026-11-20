#!/usr/bin/env python3
"""Plot the selected GB300 TPC-H scale sweep and completed DuckDB baselines."""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


SCALES = (100, 300, 1000, 3000, 10000)
DUCKDB_SCALES = (100, 300, 1000, 3000)
SERIES = (
    ("Sirius", "cold", "#0072B2", "o", "-"),
    ("Sirius", "hot", "#009E73", "o", "-"),
    ("Sirius", "pin", "#D55E00", "^", "-"),
    ("DuckDB", "cold", "#3C3C3C", "s", "--"),
    ("DuckDB", "hot", "#777777", "D", "--"),
)


@dataclass(frozen=True)
class Measurement:
    engine: str
    profile: str
    scale: int
    iteration: int
    time_s: float
    source: str


def load_sirius(repo: Path) -> list[Measurement]:
    manifest_path = repo / "bench/gb300/results/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result = []
    for scale in SCALES:
        for profile in ("cold", "hot", "pin"):
            item = manifest["results"][f"sf{scale}"][profile]
            if item["status"] != "complete":
                continue
            iteration = 1 if profile == "hot" else 0
            source = item["per_query_csv"]
            with (repo / source).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if profile == "pin":
                selected = [
                    row
                    for row in rows
                    if int(row["iteration"]) == iteration
                    and re.fullmatch(r"q(?:[1-9]|1[0-9]|2[0-2])", row["query"])
                ]
                times = [float(row["runtime_s"]) for row in selected]
                queries = {row["query"] for row in selected}
            else:
                times = [float(row[f"iteration_{iteration}_s"]) for row in rows]
                queries = {row["query"] for row in rows}
            if len(times) != 22 or len(queries) != 22:
                raise ValueError(f"incomplete Sirius result: SF{scale} {profile}")
            reported = item["query_time_s"][iteration]
            if abs(sum(times) - reported) > 0.0001:
                raise ValueError(f"Sirius manifest and query CSV disagree: {source}")
            result.append(
                Measurement("Sirius", profile, scale, iteration, reported, source)
            )
    return result


def load_duckdb(repo: Path) -> list[Measurement]:
    result = []
    base = repo / "bench/gb300/scale-figures/duckdb-snapshot"
    for scale in DUCKDB_SCALES:
        for profile in ("cold", "hot"):
            folder = base / f"sf{scale}" / profile
            metadata = json.loads((folder / "metadata.json").read_text())
            if metadata["duckdb_version"] != "2.0.0" or metadata["sirius_loaded"]:
                raise ValueError(f"unexpected DuckDB setup: {folder}")
            if metadata.get("status", "complete") != "complete":
                raise ValueError(f"incomplete DuckDB result: {folder}")
            with (folder / "runtimes.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            iteration = 1 if profile == "hot" else 0
            selected = [row for row in rows if int(row["iteration"]) == iteration]
            if len(selected) != 22 or len({row["query"] for row in selected}) != 22:
                raise ValueError(f"incomplete DuckDB result: {folder}")
            total = sum(float(row["runtime_s"]) for row in selected)
            if abs(total - metadata["query_time_s"][iteration]) > 0.0001:
                raise ValueError(f"DuckDB metadata and query CSV disagree: {folder}")
            source = f"bench/gb300/scale-figures/duckdb-snapshot/sf{scale}/{profile}/runtimes.csv"
            result.append(
                Measurement("DuckDB", profile, scale, iteration, total, source)
            )
    return result


def save_selected(measurements: list[Measurement], output: Path) -> None:
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ("engine", "profile", "scale_factor", "iteration", "query_time_s", "source")
        )
        for item in measurements:
            writer.writerow(
                (
                    item.engine,
                    item.profile,
                    item.scale,
                    item.iteration,
                    f"{item.time_s:.6f}",
                    item.source,
                )
            )


def plot(measurements: list[Measurement], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.2), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.975, bottom=0.21, top=0.85)
    for engine, profile, color, marker, style in SERIES:
        selected = sorted(
            (
                item
                for item in measurements
                if item.engine == engine and item.profile == profile
            ),
            key=lambda item: item.scale,
        )
        ax.plot(
            [item.scale for item in selected],
            [item.time_s for item in selected],
            color=color,
            linestyle=style,
            marker=marker,
            markersize=6.5,
            linewidth=2.1,
            label=f"{engine} {'host pin' if profile == 'pin' else profile}",
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(80, 12500)
    ax.set_ylim(1, 1500)
    ax.xaxis.set_major_locator(FixedLocator(SCALES))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_locator(FixedLocator((1, 10, 100, 1000)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(axis="y", which="major", color="#DADADA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("TPC-H scale factor")
    ax.set_ylabel("Sum of 22 query times (s)")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=5,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.3,
        fontsize=9.5,
    )
    fig.text(
        0.10,
        0.075,
        "Cold: first iteration   ·   Hot: second iteration   ·   Host pin: one iteration; pin/unpin excluded",
        ha="left",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.10,
        0.035,
        "DuckDB 2.0: SF100–SF3k; SF3k cold/hot caps 300/150 GiB. SF10k Sirius host pin unavailable.",
        ha="left",
        fontsize=9,
        color="#444444",
    )
    for suffix in ("pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), facecolor="white")
    svg = output.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(output.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parent
    measurements = load_sirius(repo) + load_duckdb(repo)
    save_selected(measurements, output_dir / "gb300-scale-sweep-data.csv")
    plot(measurements, output_dir / "gb300-scale-sweep")
    print(f"Saved figure and selected data to {output_dir}")


if __name__ == "__main__":
    main()
