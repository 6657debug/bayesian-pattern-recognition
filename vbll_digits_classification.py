"""AI-generated reproduction of the discriminative VBLL objective on digits.

This implements the sampling-free variational training bound for classification
from Harrison, Willes, and Snoek (ICLR 2024), then estimates predictive
probabilities by Monte Carlo samples from the fitted diagonal Gaussian head.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import sklearn
import torch
from torch import nn
from torch.nn import functional as F


SEED = 42
TEST_SIZE = 0.25
VALIDATION_FRACTION_OF_TRAIN_POOL = 0.20
HIDDEN_UNITS = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
FEATURE_WEIGHT_DECAY = 1e-4
PRIOR_VARIANCE = 1.0
VALIDATION_MC_SAMPLES = 64
TEST_MC_SAMPLES = 500
ECE_BINS = 15


class DeterministicMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_layer = nn.Linear(64, HIDDEN_UNITS)
        self.classifier = nn.Linear(HIDDEN_UNITS, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.feature_layer(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class VariationalBayesianHead(nn.Module):
    """Independent diagonal Gaussian posterior for each class weight row."""

    def __init__(self, initial_mean: torch.Tensor) -> None:
        super().__init__()
        self.weight_mean = nn.Parameter(torch.zeros(10, HIDDEN_UNITS + 1))
        self.log_weight_variance = nn.Parameter(
            torch.full((10, HIDDEN_UNITS + 1), math.log(0.1))
        )
        with torch.no_grad():
            self.weight_mean.copy_(initial_mean)

    def head_moments(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        variance = self.log_weight_variance.clamp(-12.0, 4.0).exp()
        mean_logits = F.linear(features, self.weight_mean)
        logit_variances = F.linear(features.square(), variance)
        return mean_logits, logit_variances

    def kl_to_prior(self) -> torch.Tensor:
        variance = self.log_weight_variance.clamp(-12.0, 4.0).exp()
        prior_variance = PRIOR_VARIANCE
        return 0.5 * (
            (variance + self.weight_mean.square()) / prior_variance
            - 1.0
            + math.log(prior_variance)
            - self.log_weight_variance.clamp(-12.0, 4.0)
        ).sum()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head_moments(features)


def set_reproducible_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        if index == 0:
            in_bin = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            in_bin = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if np.any(in_bin):
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )
    return ece


def evaluate(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    errors = (predictions != labels).astype(np.int32)
    auroc = (
        float(roc_auc_score(errors, entropy))
        if np.unique(errors).size > 1
        else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "nll": float(-np.log(clipped[np.arange(labels.size), labels]).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15_bins": expected_calibration_error(probabilities, labels),
        "error_detection_auroc": auroc,
        "mean_predictive_entropy_nats": float(entropy.mean()),
    }


def deterministic_predictive_probabilities(
    model: DeterministicMLP, x: np.ndarray
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(x, dtype=torch.float32)), dim=1).cpu().numpy()


def train_deterministic_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[DeterministicMLP, int, float]:
    set_reproducible_seed(SEED)
    model = DeterministicMLP()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=FEATURE_WEIGHT_DECAY
    )
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_validation_t = torch.tensor(x_validation, dtype=torch.float32)
    y_validation_t = torch.tensor(y_validation, dtype=torch.long)
    generator = torch.Generator().manual_seed(SEED)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(y_train_t), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_train_t[batch]), y_train_t[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                F.cross_entropy(model(x_validation_t), y_validation_t).item()
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Deterministic baseline training produced no checkpoint.")
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def predictive_probabilities_from_samples(
    model: VariationalBayesianHead,
    features: np.ndarray,
    epsilon: torch.Tensor,
) -> np.ndarray:
    model.eval()
    features_t = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        mean = model.weight_mean
        std = model.log_weight_variance.clamp(-12.0, 4.0).mul(0.5).exp()
        sampled_weights = mean.unsqueeze(0) + epsilon * std.unsqueeze(0)
        logits = torch.einsum("bd,skd->sbk", features_t, sampled_weights)
        probabilities = torch.softmax(logits, dim=-1).mean(dim=0)
    return probabilities.cpu().numpy()


def variational_classification_bound(
    model: VariationalBayesianHead,
    features: torch.Tensor,
    y: torch.Tensor,
    total_training_examples: int,
) -> torch.Tensor:
    mean_logits, logit_variances = model.head_moments(features)
    expected_log_likelihood_bound = (
        mean_logits.gather(1, y[:, None]).squeeze(1)
        - torch.logsumexp(mean_logits + 0.5 * logit_variances, dim=1)
    ).mean()
    # The minibatch average estimates the data term; KL/N gives the full-data ELBO scale.
    return -expected_log_likelihood_bound + model.kl_to_prior() / total_training_examples


def train_vbll(
    features_train: np.ndarray,
    y_train: np.ndarray,
    features_validation: np.ndarray,
    y_validation: np.ndarray,
    initial_mean: torch.Tensor,
) -> tuple[VariationalBayesianHead, int, float]:
    set_reproducible_seed(SEED + 1)
    model = VariationalBayesianHead(initial_mean)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    features_train_t = torch.tensor(features_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    generator = torch.Generator().manual_seed(SEED + 1)
    validation_epsilon = torch.randn(
        (VALIDATION_MC_SAMPLES, 10, HIDDEN_UNITS + 1),
        generator=torch.Generator().manual_seed(SEED + 10),
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(y_train_t), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = variational_classification_bound(
                model, features_train_t[batch], y_train_t[batch], len(y_train_t)
            )
            loss.backward()
            optimizer.step()
        validation_probabilities = predictive_probabilities_from_samples(
            model, features_validation, validation_epsilon
        )
        validation_loss = evaluate(validation_probabilities, y_validation)["nll"]
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("VBLL training produced no checkpoint.")
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def save_metrics_figure(
    metrics: dict[str, dict[str, float]],
    path: Path,
) -> None:
    names = ["MLP", "VBLL"]
    metric_keys = ["accuracy", "nll", "ece_15_bins"]
    titles = ["Test accuracy", "Test NLL (nats)", "Test ECE (15 bins)"]
    colors = ["#64748b", "#0f766e"]
    figure, axes = plt.subplots(1, 3, figsize=(8.7, 3.5))
    for axis, key, title in zip(axes, metric_keys, titles):
        values = [metrics["deterministic_mlp"][key], metrics["vbll"][key]]
        display_values = [value * 100.0 for value in values] if key == "accuracy" else values
        bars = axis.bar(names, display_values, color=colors, width=0.58)
        axis.set_title(title, fontsize=10, pad=9)
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
        for bar, value in zip(bars, display_values):
            label = f"{value:.2f}%" if key == "accuracy" else f"{value:.3f}"
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        if key == "accuracy":
            axis.set_ylim(90, 100)
        else:
            axis.set_ylim(0, max(display_values) * 1.3)
    figure.suptitle("Digits test set comparison (450 images)", fontsize=12, y=1.03)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    set_reproducible_seed(SEED)
    output_dir = Path(__file__).resolve().parent
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    x_train_pool, x_test, y_train_pool, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_pool,
        y_train_pool,
        test_size=VALIDATION_FRACTION_OF_TRAIN_POOL,
        random_state=SEED,
        stratify=y_train_pool,
    )

    mlp, mlp_best_epoch, mlp_validation_nll = train_deterministic_mlp(
        x_train, y_train, x_validation, y_validation
    )
    with torch.no_grad():
        def frozen_features(values: np.ndarray) -> np.ndarray:
            hidden = mlp.features(torch.tensor(values, dtype=torch.float32))
            augmented = torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=1)
            return augmented.cpu().numpy()

        features_train = frozen_features(x_train)
        features_validation = frozen_features(x_validation)
        features_test = frozen_features(x_test)
        initial_mean = torch.cat(
            [mlp.classifier.weight.detach(), mlp.classifier.bias.detach()[:, None]], dim=1
        )
    vbll, vbll_best_epoch, vbll_validation_nll = train_vbll(
        features_train,
        y_train,
        features_validation,
        y_validation,
        initial_mean,
    )
    test_epsilon = torch.randn(
        (TEST_MC_SAMPLES, 10, HIDDEN_UNITS + 1),
        generator=torch.Generator().manual_seed(SEED + 20),
    )
    validation_epsilon = torch.randn(
        (VALIDATION_MC_SAMPLES, 10, HIDDEN_UNITS + 1),
        generator=torch.Generator().manual_seed(SEED + 10),
    )
    validation_vbll_probabilities = predictive_probabilities_from_samples(
        vbll, features_validation, validation_epsilon
    )
    test_vbll_probabilities = predictive_probabilities_from_samples(
        vbll, features_test, test_epsilon
    )
    validation_mlp_probabilities = deterministic_predictive_probabilities(mlp, x_validation)
    test_mlp_probabilities = deterministic_predictive_probabilities(mlp, x_test)

    results = {
        "experiment": {
            "algorithm": "Variational Bayesian Last Layers (VBLL), discriminative classification",
            "paper": "Harrison, Willes, and Snoek, ICLR 2024",
            "dataset": "scikit-learn digits (8x8 grayscale handwritten digits)",
            "dataset_samples": int(len(y)),
            "class_count": int(np.unique(y).size),
            "split": {
                "train": int(len(y_train)),
                "validation": int(len(y_validation)),
                "test": int(len(y_test)),
                "random_seed": SEED,
                "stratified": True,
            },
            "input_scaling": "pixel values divided by 16 to map 0..16 into 0..1",
            "network": {
                "architecture": f"64 -> {HIDDEN_UNITS} ReLU -> 10",
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "feature_weight_decay": FEATURE_WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "maximum_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "mlp_best_epoch_by_validation_nll": mlp_best_epoch,
                "mlp_best_validation_nll": mlp_validation_nll,
                "vbll_best_epoch_by_validation_predictive_nll": vbll_best_epoch,
                "vbll_best_validation_predictive_nll": vbll_validation_nll,
            },
            "vbll": {
                "prior": f"independent zero-mean Gaussian, variance {PRIOR_VARIANCE}",
                "posterior": "independent diagonal Gaussian weight row for each class",
                "classification_objective": "sampling-free Jensen lower bound from Theorem 2, plus analytic KL divided by training-set size",
                "training_mode": "post-training variational fit on frozen MLP features; posterior mean initialized from the deterministic MLP head",
                "logit_noise_variance": 0.0,
                "validation_predictive_samples": VALIDATION_MC_SAMPLES,
                "test_predictive_samples": TEST_MC_SAMPLES,
            },
            "metrics": [
                "accuracy",
                "negative log likelihood (NLL)",
                "multiclass Brier score",
                "15-bin expected calibration error (ECE)",
                "error-detection AUROC using predictive entropy",
            ],
            "software_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "matplotlib": matplotlib.__version__,
                "pytorch": torch.__version__,
            },
            "runtime_seconds": time.perf_counter() - started,
        },
        "validation_metrics": {
            "deterministic_mlp": evaluate(validation_mlp_probabilities, y_validation),
            "vbll": evaluate(validation_vbll_probabilities, y_validation),
        },
        "test_metrics": {
            "deterministic_mlp": evaluate(test_mlp_probabilities, y_test),
            "vbll": evaluate(test_vbll_probabilities, y_test),
        },
        "notes": [
            "The deterministic MLP and VBLL use the same data split and feature-network width.",
            "The validation set selected checkpoints; the held-out test set was evaluated after training choices were fixed.",
            "VBLL training uses the paper's deterministic bound; posterior predictive probabilities use Monte Carlo samples from q(W).",
            "All reported values are from this fixed-seed run; the small dataset limits generalization claims.",
        ],
    }
    figure_path = output_dir / "test_metrics.png"
    save_metrics_figure(results["test_metrics"], figure_path)
    results_path = output_dir / "experiment_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved results: {results_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
