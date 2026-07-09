"""Shared visual options for dashboard charts."""

from __future__ import annotations

from pyecharts import options as opts

FONT_FAMILY = "Inter, Segoe UI, Microsoft YaHei, sans-serif"
TEXT_COLOR = "#536671"
AXIS_COLOR = "#9CB0BA"
GRID_COLOR = "#E8EFF2"
ACTUAL_COLOR = "#7C8F99"
CYAN = "#20C7BD"
BLUE = "#4F83FF"
NAVY = "#173B57"
AMBER = "#F2A93B"
VIOLET = "#8067DC"
ERROR_RED = "#E76F6F"

MODEL_COLORS = {
    "actual": ACTUAL_COLOR,
    "baseline": NAVY,
    "vmd": BLUE,
    "woa": AMBER,
    "complete": CYAN,
}


def init_opts(height: str = "440px") -> opts.InitOpts:
    """Return responsive chart initialization options."""

    return opts.InitOpts(
        width="100%",
        height=height,
        bg_color="#FFFFFF",
        animation_opts=opts.AnimationOpts(animation_duration=450),
    )


def axis_opts(name: str, min_: float | None = None, max_: float | None = None) -> opts.AxisOpts:
    """Return the shared 2D axis style."""

    return opts.AxisOpts(
        name=name,
        min_=min_,
        max_=max_,
        name_textstyle_opts=opts.TextStyleOpts(color=TEXT_COLOR, font_family=FONT_FAMILY),
        axislabel_opts=opts.LabelOpts(color=TEXT_COLOR, font_family=FONT_FAMILY),
        axisline_opts=opts.AxisLineOpts(
            linestyle_opts=opts.LineStyleOpts(color=AXIS_COLOR, width=1)
        ),
        axistick_opts=opts.AxisTickOpts(is_show=False),
        splitline_opts=opts.SplitLineOpts(
            is_show=True,
            linestyle_opts=opts.LineStyleOpts(color=GRID_COLOR, width=1),
        ),
    )


def global_line_opts(
    x_name: str,
    y_name: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> dict:
    """Return shared global options for line charts."""

    return {
        "tooltip_opts": opts.TooltipOpts(
            trigger="axis",
            axis_pointer_type="cross",
            background_color="rgba(6, 20, 31, 0.94)",
            border_color="#20C7BD",
            textstyle_opts=opts.TextStyleOpts(color="#FFFFFF", font_family=FONT_FAMILY),
        ),
        "legend_opts": opts.LegendOpts(
            pos_top="2%",
            pos_left="center",
            textstyle_opts=opts.TextStyleOpts(color=TEXT_COLOR, font_family=FONT_FAMILY),
        ),
        "xaxis_opts": axis_opts(x_name),
        "yaxis_opts": axis_opts(y_name, y_min, y_max),
        "datazoom_opts": [
            opts.DataZoomOpts(
                type_="inside",
                range_start=0,
                range_end=100,
            ),
            opts.DataZoomOpts(
                type_="slider",
                pos_bottom="2%",
                is_show_detail=False,
                is_show_data_shadow=False,
            ),
        ],
        "toolbox_opts": opts.ToolboxOpts(
            is_show=True,
            pos_right="2%",
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    title="Save / 保存",
                    background_color="#FFFFFF",
                ),
                restore=opts.ToolBoxFeatureRestoreOpts(title="Reset / 重置"),
                data_zoom=opts.ToolBoxFeatureDataZoomOpts(
                    zoom_title="Zoom / 缩放",
                    back_title="Back / 返回",
                ),
            ),
        ),
    }


def line_style(color: str, width: int = 2, opacity: float = 1) -> opts.LineStyleOpts:
    """Return the shared series line style."""

    return opts.LineStyleOpts(color=color, width=width, opacity=opacity)
