"""English Matplotlib visualizations for ablation-study results.

This module reads the existing prediction-result CSV files and generates
publication-/portfolio-ready Matplotlib figures with English labels.

The original CSV file names are preserved for compatibility with the current
PSBAF project:

- TSMixer_result.csv
- TSMixer+WOA_result.csv
- VMD+TSMixer_result.csv
- VMD+TSMixer+WOA_result.csv
- TiDE_result.csv
- TiDE+WOA_result.csv
- VMD+TiDE_result.csv
- VMD+TiDE+WOA_result.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


# Use a standard English-compatible font and keep minus signs readable.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class ModelResult:
    """Container for one model's prediction results and error metrics."""

    name: str
    time: pd.Series
    predicted: pd.Series
    actual: pd.Series
    error: pd.Series
    mae: float
    mape: float
    mse: float


def _candidate_dirs() -> list[Path]:
    """Return likely directories where result CSV files may be stored."""

    script_dir = Path(__file__).resolve().parent
    return [
        Path.cwd(),
        script_dir,
        script_dir.parent,
        script_dir / "results",python ablation_study_english.py --plot tide --save-dir figures --no-show
        script_dir.parent / "results",
    ]


def find_result_file(file_name: str) -> Path:
    """Find a result CSV file in the project root or results directories."""

    for directory in _candidate_dirs():
        candidate = directory / file_name
        if candidate.exists():
            return candidate

    searched = "\n".join(str(directory / file_name) for directory in _candidate_dirs())
    raise FileNotFoundError(
        f"Cannot find required result file: {file_name}\n"
        f"Searched paths:\n{searched}"
    )


def load_model_result(file_name: str, model_name: str) -> ModelResult:
    """Load one result CSV file and calculate MAE, MAPE, and MSE."""

    file_path = find_result_file(file_name)
    result_df = pd.read_csv(file_path)

    required_columns = {"time", "predicted", "true"}
    missing_columns = required_columns.difference(result_df.columns)
    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    predicted = pd.to_numeric(result_df["predicted"], errors="coerce")
    actual = pd.to_numeric(result_df["true"], errors="coerce")
    valid_mask = predicted.notna() & actual.notna()

    if not valid_mask.all():
        dropped = int((~valid_mask).sum())
        print(f"Warning: dropped {dropped} invalid rows from {file_path.name}.")

    predicted = predicted[valid_mask].reset_index(drop=True)
    actual = actual[valid_mask].reset_index(drop=True)
    time = result_df.loc[valid_mask, "time"].reset_index(drop=True)
    error = predicted - actual

    return ModelResult(
        name=model_name,
        time=time,
        predicted=predicted,
        actual=actual,
        error=error,
        mae=mean_absolute_error(actual, predicted),
        mape=mean_absolute_percentage_error(actual, predicted),
        mse=mean_squared_error(actual, predicted),
    )


def _axis_limits(series_list: Iterable[pd.Series], margin: float = 0.02) -> tuple[float, float]:
    """Calculate y-axis limits with a proportional margin."""

    values = pd.concat([series.reset_index(drop=True) for series in series_list], ignore_index=True)
    min_value = float(values.min())
    max_value = float(values.max())

    if min_value == max_value:
        padding = abs(min_value) * margin if min_value != 0 else margin
        return min_value - padding, max_value + padding

    padding = (max_value - min_value) * margin
    return min_value - padding, max_value + padding


def _print_metrics(results: Iterable[ModelResult]) -> None:
    """Print metrics in a compact, English table-like format."""

    print("\nEvaluation metrics")
    print("-" * 72)
    print(f"{'Model':<28} {'MAPE':>12} {'MAE':>12} {'MSE':>12}")
    print("-" * 72)

    for result in results:
        print(
            f"{result.name:<28} "
            f"{round(Decimal(result.mape), 10):>12} "
            f"{round(Decimal(result.mae), 10):>12} "
            f"{round(Decimal(result.mse), 10):>12}"
        )

    print("-" * 72)


def _save_or_show(fig: plt.Figure, save_path: str | Path | None, show: bool) -> None:
    """Save and/or display a Matplotlib figure."""

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_prediction_and_error(
    results: list[ModelResult],
    actual_source: ModelResult,
    title_prefix: str,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create a two-panel prediction and error comparison figure."""

    x = range(len(actual_source.time))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: prediction comparison.
    axes[0].plot(
        x,
        actual_source.actual.values,
        label="Actual cleaning factor",
        color="grey",
        linewidth=2,
        alpha=0.7,
    )

    line_styles = ["-", "--", "-.", ":"]
    colors = ["orange", "blue", "green", "black", "red", "cyan", "purple", "goldenrod"]

    for idx, result in enumerate(results):
        axes[0].plot(
            x,
            result.predicted.values,
            label=f"{result.name} prediction",
            color=colors[idx % len(colors)],
            linewidth=2,
            linestyle=line_styles[idx % len(line_styles)],
        )

    axes[0].set_ylabel("Cleaning factor", fontsize=14)
    axes[0].legend(loc="upper right", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(
        *_axis_limits([actual_source.actual] + [result.predicted for result in results], margin=0.03)
    )
    axes[0].text(
        0.5,
        -0.16,
        f"A. Prediction comparison in the {title_prefix} ablation study",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )

    # Panel B: prediction-error comparison.
    for idx, result in enumerate(results):
        axes[1].plot(
            x,
            result.error.values,
            label=f"{result.name} error",
            color=colors[idx % len(colors)],
            linewidth=2,
            linestyle=line_styles[idx % len(line_styles)],
        )

    axes[1].set_xlabel("Time step", fontsize=14)
    axes[1].set_ylabel("Prediction error", fontsize=14)
    axes[1].legend(loc="upper right", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(*_axis_limits([result.error for result in results], margin=0.12))
    axes[1].text(
        0.5,
        -0.16,
        f"B. Prediction-error comparison in the {title_prefix} ablation study",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.62)

    _print_metrics(results)
    _save_or_show(fig, save_path=save_path, show=show)
    return fig


def plot_tsmixer_ablation(
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the TSMixer ablation-study comparison."""

    vmd_tsmixer = load_model_result("VMD+TSMixer_result.csv", "VMD+TSMixer")
    vmd_tsmixer_woa = load_model_result("VMD+TSMixer+WOA_result.csv", "VMD+TSMixer+WOA")
    tsmixer_woa = load_model_result("TSMixer+WOA_result.csv", "TSMixer+WOA")
    tsmixer = load_model_result("TSMixer_result.csv", "TSMixer")

    return _plot_prediction_and_error(
        results=[vmd_tsmixer_woa, vmd_tsmixer, tsmixer_woa, tsmixer],
        actual_source=vmd_tsmixer_woa,
        title_prefix="TSMixer",
        save_path=save_path,
        show=show,
    )


def plot_tide_ablation(
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the TiDE ablation-study comparison."""

    vmd_tide = load_model_result("VMD+TiDE_result.csv", "VMD+TiDE")
    vmd_tide_woa = load_model_result("VMD+TiDE+WOA_result.csv", "VMD+TiDE+WOA")
    tide_woa = load_model_result("TiDE+WOA_result.csv", "TiDE+WOA")
    tide = load_model_result("TiDE_result.csv", "TiDE")

    return _plot_prediction_and_error(
        results=[vmd_tide_woa, vmd_tide, tide_woa, tide],
        actual_source=vmd_tide_woa,
        title_prefix="TiDE",
        save_path=save_path,
        show=show,
    )


def plot_single_model_result(
    file_name: str,
    model_name: str,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the prediction and error curves for a single trained model."""

    result = load_model_result(file_name, model_name)
    x = range(len(result.time))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(
        x,
        result.actual.values,
        label="Actual cleaning factor",
        color="grey",
        linewidth=2,
        alpha=0.7,
    )
    axes[0].plot(
        x,
        result.predicted.values,
        label=f"{result.name} prediction",
        color="orange" if "TSMixer" in result.name else "red",
        linewidth=2,
    )
    axes[0].set_ylabel("Cleaning factor", fontsize=14)
    axes[0].legend(loc="upper right", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(*_axis_limits([result.actual, result.predicted], margin=0.03))
    axes[0].text(
        0.5,
        -0.16,
        f"A. Prediction result after {result.name} training",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )

    axes[1].plot(
        x,
        result.error.values,
        label=f"{result.name} error",
        color="orange" if "TSMixer" in result.name else "red",
        linewidth=2,
    )
    axes[1].set_xlabel("Time step", fontsize=14)
    axes[1].set_ylabel("Prediction error", fontsize=14)
    axes[1].legend(loc="upper right", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(*_axis_limits([result.error], margin=0.12))
    axes[1].text(
        0.5,
        -0.16,
        f"B. Prediction-error curve after {result.name} training",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.62)

    _print_metrics([result])
    _save_or_show(fig, save_path=save_path, show=show)
    return fig


def plot_vmd_tide_woa_training(
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the VMD+TiDE+WOA training result."""

    return plot_single_model_result(
        file_name="VMD+TiDE+WOA_result.csv",
        model_name="VMD+TiDE+WOA",
        save_path=save_path,
        show=show,
    )


def plot_vmd_tsmixer_woa_training(
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the VMD+TSMixer+WOA training result."""

    return plot_single_model_result(
        file_name="VMD+TSMixer+WOA_result.csv",
        model_name="VMD+TSMixer+WOA",
        save_path=save_path,
        show=show,
    )


# Backward-compatible function names used by the original script.
def TSMixer() -> plt.Figure:
    """Backward-compatible wrapper for the TSMixer ablation plot."""

    return plot_tsmixer_ablation()


def TiDE() -> plt.Figure:
    """Backward-compatible wrapper for the TiDE ablation plot."""

    return plot_tide_ablation()


def TiDE_train() -> plt.Figure:
    """Backward-compatible wrapper for the VMD+TiDE+WOA training plot."""

    return plot_vmd_tide_woa_training()


def TSM_train() -> plt.Figure:
    """Backward-compatible wrapper for the VMD+TSMixer+WOA training plot."""

    return plot_vmd_tsmixer_woa_training()


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Generate English Matplotlib figures for PSBAF ablation-study results."
    )
    parser.add_argument(
        "--plot",
        choices=["tsmixer", "tide", "vmd-tide-woa", "vmd-tsmixer-woa", "all"],
        default="tsmixer",
        help="Select which figure to generate.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory for saving figures as PNG files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening an interactive Matplotlib window.",
    )

    args = parser.parse_args()
    show = not args.no_show

    def output_path(file_stem: str) -> Path | None:
        if args.save_dir is None:
            return None
        return args.save_dir / f"{file_stem}.png"

    if args.plot in {"tsmixer", "all"}:
        plot_tsmixer_ablation(
            save_path=output_path("tsmixer_ablation"),
            show=show,
        )

    if args.plot in {"tide", "all"}:
        plot_tide_ablation(
            save_path=output_path("tide_ablation"),
            show=show,
        )

    if args.plot in {"vmd-tide-woa", "all"}:
        plot_vmd_tide_woa_training(
            save_path=output_path("vmd_tide_woa_training"),
            show=show,
        )

    if args.plot in {"vmd-tsmixer-woa", "all"}:
        plot_vmd_tsmixer_woa_training(
            save_path=output_path("vmd_tsmixer_woa_training"),
            show=show,
        )


if __name__ == "__main__":
    main()