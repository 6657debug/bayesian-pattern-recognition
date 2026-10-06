"""Repeat the VBLL comparison over stratified split and initialization seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vbll_digits_classification as vbll


ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (11, 23, 42, 57, 99, 123, 2024, 2025, 3024, 4096)
MODELS = ("deterministic_mlp", "vbll")
METRICS = (
    "accuracy",
    "nll",
    "brier_score",
    "ece_15_bins",
    "error_detection_auroc",
    "mean_predictive_entropy_nats",
)
PREFERRED_DIRECTION = {
    "accuracy": "higher",
    "nll": "lower",
    "brier_score": "lower",
    "ece_15_bins": "lower",
    "error_detection_auroc": "higher",
}


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("provide at least two comma-separated seeds")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def summarize(runs: list[dict]) -> dict:
    summaries = {}
    paired_differences = {}
    win_counts = {}
    for model in MODELS:
        summaries[model] = {}
        for metric in METRICS:
            values = np.asarray(
                [run["test_metrics"][model][metric] for run in runs], dtype=float
            )
            summaries[model][metric] = {
                "mean": float(values.mean()),
                "sample_standard_deviation": float(values.std(ddof=1)),
            }
    for metric, direction in PREFERRED_DIRECTION.items():
        differences = np.asarray(
            [
                run["test_metrics"]["vbll"][metric]
                - run["test_metrics"]["deterministic_mlp"][metric]
                for run in runs
            ],
            dtype=float,
        )
        paired_differences[metric] = {
            "definition": "VBLL minus deterministic MLP",
            "mean": float(differences.mean()),
            "sample_standard_deviation": float(differences.std(ddof=1)),
        }
        better = differences < 0 if direction == "lower" else differences > 0
        win_counts[metric] = {
            "vbll_better_runs": int(better.sum()),
            "total_runs": len(runs),
            "preferred_direction": direction,
        }
    return {
        "per_model_test_metrics": summaries,
        "paired_test_metric_differences": paired_differences,
        "vbll_better_run_counts": win_counts,
    }


def save_figure(runs: list[dict], path: Path) -> None:
    panels = (
        ("accuracy", "Accuracy (%)", 100.0),
        ("nll", "Negative log likelihood", 1.0),
        ("ece_15_bins", "Expected calibration error", 1.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8))
    x_positions = np.array([0.0, 1.0])
    colors = {"deterministic_mlp": "#64748b", "vbll": "#0f766e"}
    labels = {"deterministic_mlp": "MLP", "vbll": "VBLL"}
    for axis, (metric, title, multiplier) in zip(axes, panels):
        values_by_model = {
            model: np.asarray(
                [run["test_metrics"][model][metric] * multiplier for run in runs]
            )
            for model in MODELS
        }
        for run_idx in range(len(runs)):
            axis.plot(
                x_positions,
                [values_by_model[model][run_idx] for model in MODELS],
                color="#CBD5E1",
                linewidth=0.8,
                alpha=0.85,
                zorder=1,
            )
        for model_idx, model in enumerate(MODELS):
            values = values_by_model[model]
            jitter = np.linspace(-0.055, 0.055, len(values))
            axis.scatter(
                np.full(len(values), model_idx) + jitter,
                values,
                s=17,
                color=colors[model],
                alpha=0.7,
                zorder=2,
            )
            axis.errorbar(
                model_idx,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="D",
                color=colors[model],
                markersize=5,
                capsize=3,
                linewidth=1.4,
                zorder=3,
            )
        axis.set_xticks(x_positions, [labels[m] for m in MODELS])
        axis.set_title(title, fontsize=9.5)
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8)
        axis.margins(x=0.18)
    fig.suptitle(
        f"Paired results across {len(runs)} stratified random splits (mean ± sample SD)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(seeds: list[int]) -> dict:
    runs = []
    representative_result = None
    representative_seed = 42 if 42 in seeds else seeds[0]
    for index, seed in enumerate(seeds, start=1):
        print(f"[{index}/{len(seeds)}] Running seed {seed}...", flush=True)
        result = vbll.main(seed=seed, write_outputs=False)
        record = {
            "seed": seed,
            "split_sizes": result["experiment"]["split"],
            "best_epochs": {
                "deterministic_mlp": result["experiment"]["network"][
                    "mlp_best_epoch_by_validation_nll"
                ],
                "vbll": result["experiment"]["network"][
                    "vbll_best_epoch_by_validation_predictive_nll"
                ],
            },
            "validation_metrics": result["validation_metrics"],
            "test_metrics": result["test_metrics"],
            "runtime_seconds": result["experiment"]["runtime_seconds"],
        }
        runs.append(record)
        if seed == representative_seed:
            representative_result = result
    robustness = {
        "protocol": {
            "design": "Repeated stratified train/validation/test holdout; each seed changes the split, initialization, minibatch order, and posterior predictive samples.",
            "seeds": seeds,
            "runs": len(seeds),
            "summary_statistic": "Arithmetic mean and sample standard deviation (ddof=1) across runs.",
            "comparison": "Paired by seed: both methods use the same split and the same deterministic feature network within each run.",
            "interpretation_limit": "Different test splits overlap in source examples, so the run-to-run standard deviation is descriptive and is not an independent-sample confidence interval.",
        },
        "summary": summarize(runs),
        "runs": runs,
    }
    figure_path = ROOT / "multiseed_metrics.png"
    save_figure(runs, figure_path)
    output = {
        "experiment": representative_result["experiment"],
        "validation_metrics": representative_result["validation_metrics"],
        "test_metrics": representative_result["test_metrics"],
        "notes": [
            f"The top-level experiment, validation_metrics, and test_metrics show representative seed {representative_seed}.",
            "The multi_seed_robustness section reports repeated stratified holdout results and descriptive between-run variability.",
            "Each run uses validation data for checkpoint selection; its test split is evaluated only after the run's training choices are fixed.",
        ],
        "multi_seed_robustness": robustness,
    }
    results_path = ROOT / "experiment_results.json"
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(robustness["summary"], indent=2))
    print(f"Saved robustness results: {results_path}")
    print(f"Saved robustness figure: {figure_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=list(DEFAULT_SEEDS),
        help="comma-separated unique seeds (default: ten predeclared seeds)",
    )
    args = parser.parse_args()
    main(args.seeds)
