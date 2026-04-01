from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

local_cache_dir = Path(".cache").resolve()
local_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(local_cache_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmedian_project.algorithms import KMedianInstance, Point, greedy_kmedian, local_search_kmedian


def generate_euclidean_instance(
    num_facilities: int,
    num_clients: int,
    seed: int,
    coordinate_limit: float = 100.0,
) -> KMedianInstance:
    rng = random.Random(seed)
    facilities = [
        Point(rng.uniform(0.0, coordinate_limit), rng.uniform(0.0, coordinate_limit))
        for _ in range(num_facilities)
    ]
    clients = [
        Point(rng.uniform(0.0, coordinate_limit), rng.uniform(0.0, coordinate_limit))
        for _ in range(num_clients)
    ]
    return KMedianInstance(facilities=facilities, clients=clients)


def ensure_directories(base_dir: Path) -> tuple[Path, Path]:
    data_dir = base_dir / "results"
    figures_dir = base_dir / "figures"
    mpl_config_dir = base_dir / ".mplconfig"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, figures_dir


def run_experiments(base_dir: Path, sizes: list[int], seeds: list[int], k_ratio: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for size in sizes:
        num_facilities = size
        num_clients = size
        k = max(2, int(k_ratio * num_facilities))
        if k >= num_facilities:
            k = num_facilities - 1

        for seed in seeds:
            instance = generate_euclidean_instance(num_facilities, num_clients, seed)
            distances = instance.distance_matrix()
            greedy_result = greedy_kmedian(distances, k)
            local_result = local_search_kmedian(distances, k, initial_open=greedy_result.open_facilities)

            rows.append(
                {
                    "seed": seed,
                    "num_facilities": num_facilities,
                    "num_clients": num_clients,
                    "k": k,
                    "greedy_cost": greedy_result.cost,
                    "greedy_runtime_seconds": greedy_result.runtime_seconds,
                    "local_search_cost": local_result.cost,
                    "local_search_runtime_seconds": local_result.runtime_seconds,
                    "local_search_total_runtime_seconds": (
                        greedy_result.runtime_seconds + local_result.runtime_seconds
                    ),
                    "cost_improvement_pct": (
                        100.0 * (greedy_result.cost - local_result.cost) / greedy_result.cost
                        if greedy_result.cost > 0
                        else 0.0
                    ),
                    "swap_iterations_to_convergence": local_result.swap_iterations,
                    "improving_swaps": local_result.improving_swaps,
                    "evaluated_swaps": local_result.evaluated_swaps,
                }
            )
    return rows


def save_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No experiment rows were generated.")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["num_facilities"])].append(row)

    by_size: list[dict[str, float]] = []
    for size, size_rows in sorted(grouped.items()):
        by_size.append(
            {
                "size": size,
                "avg_greedy_cost": mean(row["greedy_cost"] for row in size_rows),
                "avg_local_search_cost": mean(row["local_search_cost"] for row in size_rows),
                "avg_greedy_runtime_seconds": mean(row["greedy_runtime_seconds"] for row in size_rows),
                "avg_local_search_runtime_seconds": mean(
                    row["local_search_runtime_seconds"] for row in size_rows
                ),
                "avg_local_search_total_runtime_seconds": mean(
                    row["local_search_total_runtime_seconds"] for row in size_rows
                ),
                "avg_cost_improvement_pct": mean(row["cost_improvement_pct"] for row in size_rows),
                "avg_swap_iterations": mean(row["swap_iterations_to_convergence"] for row in size_rows),
                "avg_improving_swaps": mean(row["improving_swaps"] for row in size_rows),
                "avg_evaluated_swaps": mean(row["evaluated_swaps"] for row in size_rows),
            }
        )

    global_summary = {
        "num_trials": len(rows),
        "avg_greedy_cost": mean(row["greedy_cost"] for row in rows),
        "avg_local_search_cost": mean(row["local_search_cost"] for row in rows),
        "avg_greedy_runtime_seconds": mean(row["greedy_runtime_seconds"] for row in rows),
        "avg_local_search_runtime_seconds": mean(row["local_search_runtime_seconds"] for row in rows),
        "avg_local_search_total_runtime_seconds": mean(
            row["local_search_total_runtime_seconds"] for row in rows
        ),
        "avg_cost_improvement_pct": mean(row["cost_improvement_pct"] for row in rows),
        "avg_swap_iterations": mean(row["swap_iterations_to_convergence"] for row in rows),
        "avg_improving_swaps": mean(row["improving_swaps"] for row in rows),
        "avg_evaluated_swaps": mean(row["evaluated_swaps"] for row in rows),
    }
    return {"global_summary": global_summary, "by_size": by_size}


def save_summary(summary: dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def create_plots(summary: dict[str, object], figures_dir: Path) -> None:
    by_size = summary["by_size"]
    sizes = [item["size"] for item in by_size]

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, [item["avg_greedy_cost"] for item in by_size], marker="o", label="Greedy baseline")
    plt.plot(
        sizes,
        [item["avg_local_search_cost"] for item in by_size],
        marker="s",
        label="Local search",
    )
    plt.xlabel("Number of facilities / clients")
    plt.ylabel("Average assignment cost")
    plt.title("Solution Cost Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "cost_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        sizes,
        [item["avg_greedy_runtime_seconds"] for item in by_size],
        marker="o",
        label="Greedy runtime",
    )
    plt.plot(
        sizes,
        [item["avg_local_search_total_runtime_seconds"] for item in by_size],
        marker="s",
        label="Greedy + local search runtime",
    )
    plt.xlabel("Number of facilities / clients")
    plt.ylabel("Average runtime (seconds)")
    plt.title("Runtime Growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "runtime_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(sizes, [item["avg_swap_iterations"] for item in by_size], width=8)
    plt.xlabel("Number of facilities / clients")
    plt.ylabel("Average passes until convergence")
    plt.title("Local Search Convergence")
    plt.tight_layout()
    plt.savefig(figures_dir / "swap_iterations.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(sizes, [item["avg_cost_improvement_pct"] for item in by_size], width=8)
    plt.xlabel("Number of facilities / clients")
    plt.ylabel("Average improvement over greedy (%)")
    plt.title("Cost Improvement from Local Search")
    plt.tight_layout()
    plt.savefig(figures_dir / "improvement_pct.png", dpi=200)
    plt.close()


def build_report_table(summary: dict[str, object], output_path: Path) -> None:
    lines = [
        "size,avg_greedy_cost,avg_local_search_cost,avg_greedy_runtime_seconds,"
        "avg_local_search_total_runtime_seconds,avg_cost_improvement_pct,avg_swap_iterations"
    ]
    for item in summary["by_size"]:
        lines.append(
            ",".join(
                [
                    str(item["size"]),
                    f"{item['avg_greedy_cost']:.4f}",
                    f"{item['avg_local_search_cost']:.4f}",
                    f"{item['avg_greedy_runtime_seconds']:.6f}",
                    f"{item['avg_local_search_total_runtime_seconds']:.6f}",
                    f"{item['avg_cost_improvement_pct']:.2f}",
                    f"{item['avg_swap_iterations']:.2f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report_table_tex(summary: dict[str, object], output_path: Path) -> None:
    lines = []
    for item in summary["by_size"]:
        lines.append(
            "        "
            + " & ".join(
                [
                    str(item["size"]),
                    f"{item['avg_greedy_cost']:.2f}",
                    f"{item['avg_local_search_cost']:.2f}",
                    f"{item['avg_greedy_runtime_seconds']:.4f}",
                    f"{item['avg_local_search_total_runtime_seconds']:.4f}",
                    f"{item['avg_cost_improvement_pct']:.2f}",
                    f"{item['avg_swap_iterations']:.2f}",
                ]
            )
            + r" \\"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_detailed_results_tex(rows: list[dict[str, float]], output_path: Path) -> None:
    lines = []
    for row in rows:
        lines.append(
            "        "
            + " & ".join(
                [
                    str(int(row["num_facilities"])),
                    str(int(row["seed"])),
                    str(int(row["k"])),
                    f"{row['greedy_cost']:.2f}",
                    f"{row['local_search_cost']:.2f}",
                    f"{row['cost_improvement_pct']:.2f}",
                    str(int(row["swap_iterations_to_convergence"])),
                    str(int(row["improving_swaps"])),
                    str(int(row["evaluated_swaps"])),
                ]
            )
            + r" \\"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metric k-median experiments.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[20, 40, 60, 80, 100],
        help="Problem sizes for the number of facilities and clients.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Random seeds for repeated trials.",
    )
    parser.add_argument(
        "--k-ratio",
        type=float,
        default=0.2,
        help="Fraction of facilities to open.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where results and figures are stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir, figures_dir = ensure_directories(args.output_dir)
    rows = run_experiments(args.output_dir, args.sizes, args.seeds, args.k_ratio)
    summary = summarize(rows)

    save_csv(rows, results_dir / "experiment_results.csv")
    save_summary(summary, results_dir / "summary.json")
    build_report_table(summary, results_dir / "summary_table.csv")
    build_report_table_tex(summary, results_dir / "summary_table.tex")
    build_detailed_results_tex(rows, results_dir / "detailed_results.tex")
    create_plots(summary, figures_dir)

    print(f"Saved detailed results to {results_dir / 'experiment_results.csv'}")
    print(f"Saved summary JSON to {results_dir / 'summary.json'}")
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
