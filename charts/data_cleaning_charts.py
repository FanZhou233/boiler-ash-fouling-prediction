"""Pyecharts visualizations for wavelet denoising."""

from __future__ import annotations

from pyecharts import options as opts
from pyecharts.charts import Line

from data_cleaning import load_clean_factor_data, residual_energy_ratio, smoothness_index


def denoised_soft_chart() -> tuple[str, float, float]:
    """Render the soft-threshold denoising chart and quality indicators."""

    denoised_soft, _, _, original_signal, time_index = load_clean_factor_data()
    chart = (
        Line()
        .add_xaxis(time_index.tolist())
        .add_yaxis(
            "原始信号",
            original_signal.tolist(),
            is_smooth=True,
            color="gray",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            "软阈值去噪",
            denoised_soft.tolist(),
            is_smooth=True,
            color="blue",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="小波去噪 - 软阈值"),
            xaxis_opts=opts.AxisOpts(name="检查点"),
            yaxis_opts=opts.AxisOpts(
                name="清洁因子",
                min_=round(min(original_signal), 2),
                max_=round(max(original_signal), 2),
            ),
            datazoom_opts=opts.DataZoomOpts(),
        )
    )
    rer_soft = residual_energy_ratio(original_signal, denoised_soft)
    smooth_soft = smoothness_index(denoised_soft)
    return chart.render_embed(), rer_soft, smooth_soft


def denoised_hard_chart() -> tuple[str, float, float]:
    """Render the hard-threshold denoising chart and quality indicators."""

    _, denoised_hard, _, original_signal, time_index = load_clean_factor_data()
    chart = (
        Line()
        .add_xaxis(time_index.tolist())
        .add_yaxis(
            "原始信号",
            original_signal.tolist(),
            is_smooth=True,
            color="gray",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            "硬阈值去噪",
            denoised_hard.tolist(),
            is_smooth=True,
            color="red",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="小波去噪 - 硬阈值"),
            xaxis_opts=opts.AxisOpts(name="检查点"),
            yaxis_opts=opts.AxisOpts(
                name="清洁因子",
                min_=round(min(original_signal), 2) - 0.03,
                max_=round(max(original_signal), 2) + 0.03,
            ),
            datazoom_opts=opts.DataZoomOpts(),
        )
    )
    rer_hard = residual_energy_ratio(original_signal, denoised_hard)
    smooth_hard = smoothness_index(denoised_hard)
    return chart.render_embed(), rer_hard, smooth_hard
