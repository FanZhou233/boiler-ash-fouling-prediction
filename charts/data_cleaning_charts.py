"""Pyecharts visualizations for wavelet denoising."""

from __future__ import annotations

from pyecharts import options as opts
from pyecharts.charts import Line

from charts.theme import ACTUAL_COLOR, BLUE, CYAN, global_line_opts, init_opts, line_style
from data_cleaning import load_clean_factor_data, residual_energy_ratio, smoothness_index


def _denoising_chart(
    time_index,
    original_signal,
    denoised_signal,
    series_name: str,
    color: str,
) -> str:
    """Build a consistently styled denoising comparison chart."""

    signal_min = min(float(original_signal.min()), float(denoised_signal.min()))
    signal_max = max(float(original_signal.max()), float(denoised_signal.max()))
    padding = max((signal_max - signal_min) * 0.08, 0.01)

    chart = (
        Line(init_opts())
        .add_xaxis(time_index.tolist())
        .add_yaxis(
            "原始信号 / Original",
            original_signal.tolist(),
            is_symbol_show=False,
            is_smooth=False,
            linestyle_opts=line_style(ACTUAL_COLOR, width=1, opacity=0.58),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name,
            denoised_signal.tolist(),
            is_symbol_show=False,
            is_smooth=True,
            linestyle_opts=line_style(color, width=2),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            **global_line_opts(
                "采样点 / Sample",
                "清洁因子 / Cleaning factor",
                signal_min - padding,
                signal_max + padding,
            )
        )
    )
    return chart.render_embed()


def denoised_soft_chart() -> tuple[str, float, float]:
    """Render the soft-threshold denoising chart and quality indicators."""

    denoised_soft, _, _, original_signal, time_index = load_clean_factor_data()
    chart = _denoising_chart(
        time_index,
        original_signal,
        denoised_soft,
        "软阈值去噪 / Soft threshold",
        CYAN,
    )
    return (
        chart,
        residual_energy_ratio(original_signal, denoised_soft),
        smoothness_index(denoised_soft),
    )


def denoised_hard_chart() -> tuple[str, float, float]:
    """Render the hard-threshold denoising chart and quality indicators."""

    _, denoised_hard, _, original_signal, time_index = load_clean_factor_data()
    chart = _denoising_chart(
        time_index,
        original_signal,
        denoised_hard,
        "硬阈值去噪 / Hard threshold",
        BLUE,
    )
    return (
        chart,
        residual_energy_ratio(original_signal, denoised_hard),
        smoothness_index(denoised_hard),
    )
