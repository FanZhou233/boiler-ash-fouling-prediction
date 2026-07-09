"""3D scatter charts for WOA hyperparameter-search history."""

from __future__ import annotations

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Scatter3D

from charts.theme import FONT_FAMILY, TEXT_COLOR, init_opts

COLOR_RANGE = ["#173B57", "#256B87", "#20A9A3", "#40D9CF", "#F2D479", "#F2A93B"]


def _axis_3d(name: str) -> opts.Axis3DOpts:
    return opts.Axis3DOpts(
        name=name,
        textstyle_opts=opts.TextStyleOpts(color=TEXT_COLOR, font_family=FONT_FAMILY),
        axislabel_opts=opts.LabelOpts(color=TEXT_COLOR, font_family=FONT_FAMILY),
        axisline_opts=opts.AxisLineOpts(
            linestyle_opts=opts.LineStyleOpts(color="#AFC0C8")
        ),
        splitline_opts=opts.SplitLineOpts(
            is_show=True,
            linestyle_opts=opts.LineStyleOpts(color="#E8EFF2", opacity=0.8),
        ),
    )


def _render_scatter(csv_file: str) -> str:
    """Render a WOA 3D scatter chart from an optimization-history CSV file."""

    data = pd.read_csv(csv_file)
    scatter_data = data[
        ["input_chunk_length", "output_chunk_length", "hidden_size", "MAPE"]
    ].values.tolist()
    mape_values = data["MAPE"]

    chart = (
        Scatter3D(init_opts("520px"))
        .add(
            series_name="MAPE",
            data=scatter_data,
            xaxis3d_opts=_axis_3d("Input length"),
            yaxis3d_opts=_axis_3d("Output length"),
            zaxis3d_opts=_axis_3d("Hidden size"),
            grid3d_opts=opts.Grid3DOpts(
                width=105,
                height=105,
                depth=105,
                rotate_speed=8,
                is_rotate=False,
            ),
        )
        .set_series_opts(
            symbol_size=7,
            opacity=0.78,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(
                background_color="rgba(6, 20, 31, 0.94)",
                border_color="#20C7BD",
                textstyle_opts=opts.TextStyleOpts(color="#FFFFFF", font_family=FONT_FAMILY),
            ),
            visualmap_opts=opts.VisualMapOpts(
                min_=float(mape_values.min()),
                max_=float(mape_values.max()),
                dimension=3,
                pos_top="middle",
                pos_left="2%",
                item_width=12,
                item_height=150,
                range_color=COLOR_RANGE,
                textstyle_opts=opts.TextStyleOpts(color=TEXT_COLOR, font_family=FONT_FAMILY),
            ),
        )
    )
    return chart.render_embed()


def woa_3d_scatter() -> str:
    """Render the TSMixer WOA hyperparameter-search chart."""

    return _render_scatter("optimization_history.csv")


def woa_3d_scatter_tide() -> str:
    """Render the TiDE WOA hyperparameter-search chart."""

    return _render_scatter("tide_optimization_history.csv")


WOA_3D_scatter = woa_3d_scatter
WOA_3D_scatter_TiDE = woa_3d_scatter_tide
