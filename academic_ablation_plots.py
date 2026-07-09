"""Publication-style plots shared by the Chinese and English entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib

# Use a non-interactive backend so command-line export is stable on servers and CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

Language = Literal["zh", "en"]

# Okabe-Ito-inspired, color-vision-deficiency-friendly palette.
COLORS = {
    "actual": "#6C757D",
    "complete": "#009E73",
    "vmd": "#0072B2",
    "woa": "#E69F00",
    "baseline": "#CC79A7",
}
LINESTYLES = {
    "complete": "-",
    "vmd": "--",
    "woa": "-.",
    "baseline": ":",
}

TEXT = {
    "en": {
        "actual": "Actual",
        "cleaning_factor": "Cleaning factor",
        "error": "Prediction error",
        "time": "Time step",
        "prediction_panel": "Prediction comparison",
        "error_panel": "Residual comparison",
        "model": "Model",
        "mape": "MAPE (%)",
        "mae": "MAE",
        "mse": "MSE",
        "prediction": "Prediction",
        "residual": "Residual",
    },
    "zh": {
        "actual": "真实值",
        "cleaning_factor": "清洁因子",
        "error": "预测误差",
        "time": "时间步",
        "prediction_panel": "预测结果对比",
        "error_panel": "预测误差对比",
        "model": "模型",
        "mape": "MAPE（%）",
        "mae": "MAE",
        "mse": "MSE",
        "prediction": "预测值",
        "residual": "误差",
    },
}


@dataclass(frozen=True)
class ModelResult:
    name: str
    time: pd.Series
    predicted: pd.Series
    actual: pd.Series
    error: pd.Series
    mae: float
    mape: float
    mse: float


def configure_matplotlib(language: Language) -> None:
    """Apply journal-like typography and line defaults."""

    font_family = ["Times New Roman"]
    if language == "zh":
        font_family = ["SimSun", "Times New Roman"]
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "savefig.dpi": 600,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def find_result_file(file_name: str) -> Path:
    """Find result CSV files from common project locations."""

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / file_name,
        script_dir / file_name,
        script_dir / "results" / file_name,
        script_dir.parent / "results" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {file_name}. Searched:\n"
        + "\n".join(str(path) for path in candidates)
    )


def load_model_result(file_name: str, model_name: str) -> ModelResult:
    """Load one model result and calculate its evaluation metrics."""

    data = pd.read_csv(find_result_file(file_name))
    missing = {"time", "predicted", "true"}.difference(data.columns)
    if missing:
        raise ValueError(f"{file_name} is missing columns: {', '.join(sorted(missing))}")

    clean = pd.DataFrame(
        {
            "time": data["time"],
            "predicted": pd.to_numeric(data["predicted"], errors="coerce"),
            "actual": pd.to_numeric(data["true"], errors="coerce"),
        }
    ).dropna(subset=["predicted", "actual"]).reset_index(drop=True)
    predicted = clean["predicted"]
    actual = clean["actual"]
    error = predicted - actual
    return ModelResult(
        name=model_name,
        time=clean["time"],
        predicted=predicted,
        actual=actual,
        error=error,
        mae=float(mean_absolute_error(actual, predicted)),
        mape=float(mean_absolute_percentage_error(actual, predicted)),
        mse=float(mean_squared_error(actual, predicted)),
    )


def validate_actual_series(results: Iterable[ModelResult]) -> None:
    """Ensure ablation models are evaluated against the same target sequence."""

    results = list(results)
    reference = results[0]
    for result in results[1:]:
        if len(result.actual) != len(reference.actual) or not np.allclose(
            result.actual.to_numpy(),
            reference.actual.to_numpy(),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Actual values differ between {reference.name} and {result.name}; "
                "the ablation comparison would not be valid."
            )


def _axis_limits(series: Iterable[pd.Series], margin: float) -> tuple[float, float]:
    values = pd.concat([item.reset_index(drop=True) for item in series], ignore_index=True)
    lower, upper = float(values.min()), float(values.max())
    span = upper - lower
    padding = span * margin if span else max(abs(lower) * margin, 1e-3)
    return lower - padding, upper + padding


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9DEE2", linewidth=0.55, alpha=0.75)
    axis.grid(axis="x", color="#E9ECEF", linewidth=0.4, alpha=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.7)


def _metric_table(axis: plt.Axes, results: list[ModelResult], language: Language) -> None:
    labels = TEXT[language]
    axis.axis("off")
    rows = [
        [
            result.name,
            f"{result.mape * 100:.3f}",
            f"{result.mae:.5f}",
            f"{result.mse:.2e}",
        ]
        for result in results
    ]
    table = axis.table(
        cellText=rows,
        colLabels=[labels["model"], labels["mape"], labels["mae"], labels["mse"]],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.42, 0.19, 0.19, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.18)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DEE3")
        cell.set_linewidth(0.55)
        cell.set_facecolor("#F2F5F7" if row == 0 else "#FFFFFF")
        if row == 0:
            cell.set_text_props(weight="bold")


def _save_figure(fig: Figure, save_path: str | Path | None, show: bool) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix:
            fig.savefig(path, dpi=600, bbox_inches="tight")
            if suffix == ".png":
                fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        else:
            fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ablation(
    results: list[ModelResult],
    language: Language,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Create a two-panel ablation figure with a compact metric table."""

    configure_matplotlib(language)
    validate_actual_series(results)
    labels = TEXT[language]
    x = np.arange(len(results[0].actual))

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.05, 1, 0.48])
    prediction_axis = fig.add_subplot(grid[0])
    error_axis = fig.add_subplot(grid[1], sharex=prediction_axis)
    table_axis = fig.add_subplot(grid[2])

    prediction_axis.plot(
        x,
        results[0].actual,
        label=labels["actual"],
        color=COLORS["actual"],
        linewidth=1.6,
        alpha=0.86,
    )
    roles = ["complete", "vmd", "woa", "baseline"]
    for role, result in zip(roles, results):
        prediction_axis.plot(
            x,
            result.predicted,
            label=result.name,
            color=COLORS[role],
            linestyle=LINESTYLES[role],
            linewidth=1.8 if role == "complete" else 1.3,
        )
        error_axis.plot(
            x,
            result.error,
            label=result.name,
            color=COLORS[role],
            linestyle=LINESTYLES[role],
            linewidth=1.7 if role == "complete" else 1.2,
        )

    prediction_axis.set_ylabel(labels["cleaning_factor"])
    prediction_axis.set_ylim(
        *_axis_limits(
            [results[0].actual, *(result.predicted for result in results)],
            margin=0.04,
        )
    )
    prediction_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
    )
    prediction_axis.text(
        0.01,
        0.96,
        f"(a) {labels['prediction_panel']}",
        transform=prediction_axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    error_axis.axhline(0, color="#4B5563", linewidth=0.75, alpha=0.8, zorder=0)
    error_axis.set_xlabel(labels["time"])
    error_axis.set_ylabel(labels["error"])
    error_axis.set_ylim(*_axis_limits([result.error for result in results], margin=0.1))
    error_axis.text(
        0.01,
        0.96,
        f"(b) {labels['error_panel']}",
        transform=error_axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    _style_axis(prediction_axis)
    _style_axis(error_axis)
    plt.setp(prediction_axis.get_xticklabels(), visible=False)
    _metric_table(table_axis, results, language)
    _save_figure(fig, save_path, show)
    return fig


def plot_single_result(
    result: ModelResult,
    language: Language,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Create a compact prediction/residual figure for one model."""

    configure_matplotlib(language)
    labels = TEXT[language]
    x = np.arange(len(result.actual))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, constrained_layout=True)

    axes[0].plot(x, result.actual, label=labels["actual"], color=COLORS["actual"], linewidth=1.6)
    axes[0].plot(
        x,
        result.predicted,
        label=result.name,
        color=COLORS["complete"],
        linewidth=1.8,
    )
    axes[0].set_ylabel(labels["cleaning_factor"])
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
    axes[0].text(
        0.01, 0.96, f"(a) {labels['prediction']}", transform=axes[0].transAxes,
        ha="left", va="top", fontweight="bold",
    )
    axes[0].set_ylim(*_axis_limits([result.actual, result.predicted], margin=0.04))

    axes[1].plot(x, result.error, color=COLORS["complete"], linewidth=1.5)
    axes[1].axhline(0, color="#4B5563", linewidth=0.75, alpha=0.8)
    axes[1].set_xlabel(labels["time"])
    axes[1].set_ylabel(labels["error"])
    axes[1].text(
        0.01, 0.96, f"(b) {labels['residual']}", transform=axes[1].transAxes,
        ha="left", va="top", fontweight="bold",
    )
    axes[1].set_ylim(*_axis_limits([result.error], margin=0.1))
    for axis in axes:
        _style_axis(axis)

    metrics = (
        f"MAPE = {result.mape * 100:.3f}%   "
        f"MAE = {result.mae:.5f}   MSE = {result.mse:.2e}"
    )
    axes[1].text(
        0.99,
        0.04,
        metrics,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#D8DEE3", "alpha": 0.9},
    )
    _save_figure(fig, save_path, show)
    return fig


def tsmixer_results() -> list[ModelResult]:
    return [
        load_model_result("VMD+TSMixer+WOA_result.csv", "VMD+TSMixer+WOA"),
        load_model_result("VMD+TSMixer_result.csv", "VMD+TSMixer"),
        load_model_result("TSMixer+WOA_result.csv", "TSMixer+WOA"),
        load_model_result("TSMixer_result.csv", "TSMixer"),
    ]


def tide_results() -> list[ModelResult]:
    return [
        load_model_result("VMD+TiDE+WOA_result.csv", "VMD+TiDE+WOA"),
        load_model_result("VMD+TiDE_result.csv", "VMD+TiDE"),
        load_model_result("TiDE+WOA_result.csv", "TiDE+WOA"),
        load_model_result("TiDE_result.csv", "TiDE"),
    ]
