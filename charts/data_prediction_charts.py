"""Pyecharts visualizations for prediction and ablation results."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from charts.theme import (
    ACTUAL_COLOR,
    AMBER,
    BLUE,
    CYAN,
    ERROR_RED,
    NAVY,
    global_line_opts,
    init_opts,
    line_style,
)


@dataclass
class ModelResult:
    """Prediction values and evaluation metrics loaded from a result CSV."""

    predicted: pd.Series
    actual: pd.Series
    errors: pd.Series
    mae: float
    mape: float
    mse: float


def _load_model_data(file_name: str) -> ModelResult:
    data = pd.read_csv(file_name)
    values = pd.DataFrame(
        {
            "predicted": pd.to_numeric(data["predicted"], errors="coerce"),
            "actual": pd.to_numeric(data["true"], errors="coerce"),
        }
    ).dropna()
    predicted = values["predicted"]
    actual = values["actual"]
    errors = predicted - actual
    return ModelResult(
        predicted=predicted,
        actual=actual,
        errors=errors,
        mae=mean_absolute_error(actual, predicted),
        mape=mean_absolute_percentage_error(actual, predicted),
        mse=mean_squared_error(actual, predicted),
    )


def _prediction_chart(
    actual: pd.Series,
    series: list[tuple[str, pd.Series, str, int]],
) -> str:
    """Build a model prediction-comparison chart."""

    all_values = [actual, *(values for _, values, _, _ in series)]
    lower = min(float(values.min()) for values in all_values)
    upper = max(float(values.max()) for values in all_values)
    padding = max((upper - lower) * 0.08, 0.005)

    chart = (
        Line(init_opts("470px"))
        .add_xaxis(list(range(len(actual))))
        .add_yaxis(
            "真实值 / Actual",
            actual.tolist(),
            is_symbol_show=False,
            is_smooth=False,
            linestyle_opts=opts.LineStyleOpts(
                color=ACTUAL_COLOR,
                width=2,
                opacity=0.72,
                type_="dashed",
            ),
            label_opts=opts.LabelOpts(is_show=False),
        )
    )
    for name, values, color, width in series:
        chart.add_yaxis(
            name,
            values.tolist(),
            is_symbol_show=False,
            is_smooth=True,
            linestyle_opts=line_style(color, width=width),
            label_opts=opts.LabelOpts(is_show=False),
        )
    chart.set_global_opts(
        **global_line_opts(
            "时间点 / Time",
            "清洁因子 / Cleaning factor",
            lower - padding,
            upper + padding,
        )
    )
    return chart.render_embed()


def _error_chart(series: list[tuple[str, pd.Series, str, int]]) -> str:
    """Build a prediction-error comparison chart."""

    chart = Line(init_opts("420px")).add_xaxis(list(range(len(series[0][1]))))
    for name, values, color, width in series:
        chart.add_yaxis(
            name,
            values.tolist(),
            is_symbol_show=False,
            is_smooth=True,
            linestyle_opts=line_style(color, width=width),
            label_opts=opts.LabelOpts(is_show=False),
        )
    chart.set_global_opts(**global_line_opts("时间点 / Time", "误差 / Error"))
    return chart.render_embed()


def _metric(value: float) -> float:
    return round(float(value), 10)


def ablation_experiment_chart():
    """Render TSMixer and TiDE ablation charts and return their metrics."""

    vmd_tsmixer = _load_model_data("VMD+TSMixer_result.csv")
    vmd_tsmixer_woa = _load_model_data("VMD+TSMixer+WOA_result.csv")
    tsmixer_woa = _load_model_data("TSMixer+WOA_result.csv")
    tsmixer = _load_model_data("TSMixer_result.csv")

    vmd_tide = _load_model_data("VMD+TiDE_result.csv")
    vmd_tide_woa = _load_model_data("VMD+TiDE+WOA_result.csv")
    tide_woa = _load_model_data("TiDE+WOA_result.csv")
    tide = _load_model_data("TiDE_result.csv")

    tsmixer_chart = _prediction_chart(
        vmd_tsmixer_woa.actual,
        [
            ("TSMixer", tsmixer.predicted, NAVY, 1),
            ("VMD + TSMixer", vmd_tsmixer.predicted, BLUE, 2),
            ("TSMixer + WOA", tsmixer_woa.predicted, AMBER, 2),
            ("VMD + TSMixer + WOA", vmd_tsmixer_woa.predicted, CYAN, 3),
        ],
    )
    tsmixer_error_chart = _error_chart(
        [
            ("TSMixer error", tsmixer.errors, ERROR_RED, 1),
            ("VMD + TSMixer error", vmd_tsmixer.errors, BLUE, 2),
            ("TSMixer + WOA error", tsmixer_woa.errors, AMBER, 2),
            ("Complete model error", vmd_tsmixer_woa.errors, CYAN, 3),
        ]
    )

    tide_chart = _prediction_chart(
        vmd_tide_woa.actual,
        [
            ("TiDE", tide.predicted, NAVY, 1),
            ("VMD + TiDE", vmd_tide.predicted, BLUE, 2),
            ("TiDE + WOA", tide_woa.predicted, AMBER, 2),
            ("VMD + TiDE + WOA", vmd_tide_woa.predicted, CYAN, 3),
        ],
    )
    tide_error_chart = _error_chart(
        [
            ("TiDE error", tide.errors, ERROR_RED, 1),
            ("VMD + TiDE error", vmd_tide.errors, BLUE, 2),
            ("TiDE + WOA error", tide_woa.errors, AMBER, 2),
            ("Complete model error", vmd_tide_woa.errors, CYAN, 3),
        ]
    )

    return (
        tsmixer_chart,
        tsmixer_error_chart,
        tide_chart,
        tide_error_chart,
        _metric(vmd_tsmixer_woa.mape),
        _metric(vmd_tsmixer.mape),
        _metric(tsmixer_woa.mape),
        _metric(tsmixer.mape),
        _metric(vmd_tsmixer_woa.mae),
        _metric(vmd_tsmixer.mae),
        _metric(tsmixer_woa.mae),
        _metric(tsmixer.mae),
        _metric(vmd_tsmixer_woa.mse),
        _metric(vmd_tsmixer.mse),
        _metric(tsmixer_woa.mse),
        _metric(tsmixer.mse),
        _metric(vmd_tide_woa.mape),
        _metric(vmd_tide.mape),
        _metric(tide_woa.mape),
        _metric(tide.mape),
        _metric(vmd_tide_woa.mae),
        _metric(vmd_tide.mae),
        _metric(tide_woa.mae),
        _metric(tide.mae),
        _metric(vmd_tide_woa.mse),
        _metric(vmd_tide.mse),
        _metric(tide_woa.mse),
        _metric(tide.mse),
    )
