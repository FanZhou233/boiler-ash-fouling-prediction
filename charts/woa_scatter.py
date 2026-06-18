"""3D scatter charts for WOA hyperparameter-search history."""

from __future__ import annotations

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Scatter3D

COLOR_RANGE = ["#1710c0", "#4575b4", "#74add1", "#abd9e9", "#fee090", "#f46d43", "#a50026"]


def _render_scatter(csv_file: str, title: str) -> str:
    """Render a WOA 3D scatter chart from an optimization-history CSV file."""

    data = pd.read_csv(csv_file)
    scatter_data = data[["input_chunk_length", "output_chunk_length", "hidden_size", "MAPE"]].values.tolist()
    mape_values = data["MAPE"]

    chart = (
        Scatter3D()
        .add(
            series_name="",
            data=scatter_data,
            xaxis3d_opts=opts.Axis3DOpts(name="Input Chunk Length"),
            yaxis3d_opts=opts.Axis3DOpts(name="Output Chunk Length"),
            zaxis3d_opts=opts.Axis3DOpts(name="Hidden Size"),
        )
        .set_series_opts(
            symbol_size=5,
            opacity=0.7,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            visualmap_opts=opts.VisualMapOpts(
                min_=float(mape_values.min()),
                max_=float(mape_values.max()),
                dimension=3,
                pos_top="center",
                pos_left="left",
                range_color=COLOR_RANGE,
            ),
        )
    )
    return chart.render_embed()


def woa_3d_scatter() -> str:
    """Render the TSMixer WOA hyperparameter-search chart."""

    return _render_scatter("optimization_history.csv", "WOA使用MAPE量化的TSMixer优化超参数3D散点图")


def woa_3d_scatter_tide() -> str:
    """Render the TiDE WOA hyperparameter-search chart."""

    return _render_scatter("tide_optimization_history.csv", "WOA使用MAPE量化的TiDE模型优化超参数3D散点图")


# Backward-compatible aliases for older templates/imports.
WOA_3D_scatter = woa_3d_scatter
WOA_3D_scatter_TiDE = woa_3d_scatter_tide
