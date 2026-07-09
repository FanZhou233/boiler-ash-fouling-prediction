"""Generate publication-style English ablation-study figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from academic_ablation_plots import (
    load_model_result,
    plot_ablation,
    plot_single_result,
    tide_results,
    tsmixer_results,
)


def plot_tsmixer_ablation(save_path=None, show=True):
    return plot_ablation(tsmixer_results(), "en", save_path, show)


def plot_tide_ablation(save_path=None, show=True):
    return plot_ablation(tide_results(), "en", save_path, show)


def plot_vmd_tide_woa_training(save_path=None, show=True):
    result = load_model_result("VMD+TiDE+WOA_result.csv", "VMD+TiDE+WOA")
    return plot_single_result(result, "en", save_path, show)


def plot_vmd_tsmixer_woa_training(save_path=None, show=True):
    result = load_model_result("VMD+TSMixer+WOA_result.csv", "VMD+TSMixer+WOA")
    return plot_single_result(result, "en", save_path, show)


TSMixer = plot_tsmixer_ablation
TiDE = plot_tide_ablation
TiDE_train = plot_vmd_tide_woa_training
TSM_train = plot_vmd_tsmixer_woa_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate academic English figures.")
    parser.add_argument(
        "--plot",
        choices=["tsmixer", "tide", "vmd-tide-woa", "vmd-tsmixer-woa", "all"],
        default="tsmixer",
    )
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    show = not args.no_show

    def output(name: str):
        return args.save_dir / f"{name}.png" if args.save_dir else None

    if args.plot in {"tsmixer", "all"}:
        plot_tsmixer_ablation(output("tsmixer_ablation"), show)
    if args.plot in {"tide", "all"}:
        plot_tide_ablation(output("tide_ablation"), show)
    if args.plot in {"vmd-tide-woa", "all"}:
        plot_vmd_tide_woa_training(output("vmd_tide_woa_training"), show)
    if args.plot in {"vmd-tsmixer-woa", "all"}:
        plot_vmd_tsmixer_woa_training(output("vmd_tsmixer_woa_training"), show)


if __name__ == "__main__":
    main()
