"""Pyecharts visualizations for prediction and ablation results."""

from decimal import Decimal
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def WOA_result_chart():
    def TSMixer():
        TSMixer_prediction = pd.read_csv('TSMixer+WOA_result.csv')
        predicted_values = pd.to_numeric(TSMixer_prediction['predicted'], errors='coerce')
        TSM_true_values = pd.to_numeric(TSMixer_prediction['true'], errors='coerce')
        time = TSMixer_prediction['time']
        errors = predicted_values - TSM_true_values

        # Calculate error metrics using scikit-learn.
        TSMixer_mae = mean_absolute_error(TSM_true_values, predicted_values)
        TSMixer_mape = mean_absolute_percentage_error(TSM_true_values, predicted_values)

        return time, predicted_values, TSM_true_values, TSMixer_mae, TSMixer_mape, errors

    def tide():
        tide_prediction = pd.read_csv('TiDE+WOA_result.csv')
        tide_predicted_values = pd.to_numeric(tide_prediction['predicted'], errors='coerce')
        true_values = pd.to_numeric(tide_prediction['true'], errors='coerce')
        time = tide_prediction['time']
        errors = tide_predicted_values - true_values

        tide_mae = mean_absolute_error(true_values, tide_predicted_values)
        tide_mape = mean_absolute_percentage_error(true_values, tide_predicted_values)

        return time, tide_predicted_values, true_values, tide_mae, tide_mape, errors

    def VMD():
        vmd_prediction = pd.read_csv('VMD+TSMixer+WOA_result.csv')
        VMD_predicted_values = pd.to_numeric(vmd_prediction['predicted'], errors='coerce')
        true_values = pd.to_numeric(vmd_prediction['true'], errors='coerce')
        time = vmd_prediction['time']
        errors = VMD_predicted_values - true_values

        VMD_mae = mean_absolute_error(true_values, VMD_predicted_values)
        VMD_mape = mean_absolute_percentage_error(true_values, VMD_predicted_values)

        return time, VMD_predicted_values, true_values, VMD_mae, VMD_mape, errors

    time, tide_predicted_values, true_values, tide_mae, tide_mape, TiDE_errors = tide()
    time, TSM_predicted_values, TSM_true_values, TSMixer_mae, TSMixer_mape, TSMixer_errors = TSMixer()
    time, VMD_predicted_values, true_values, VMD_mae, VMD_mape, VMD_errors = VMD()

    TSMixer_chart = (
        Line()
        .add_xaxis([i for i in range(len(time))])
        .add_yaxis("清洁因子", TSM_true_values.values.tolist(), is_smooth=False, color="grey",
                   linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.7), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("TSMixer预测值", TSM_predicted_values.values.tolist(), is_smooth=False, color="blue",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("TiDE预测值", tide_predicted_values.values.tolist(), is_smooth=True, color="green",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("VMD+TSMixer预测值", VMD_predicted_values.values.tolist(), is_smooth=True, color="red",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="TSMixer/TiDE/VMD+TSMixer利用WOA优化全局参数预测结果对比"),
            xaxis_opts=opts.AxisOpts(name="检查点"),
            yaxis_opts=opts.AxisOpts(name="清洁因子", min_=round(min(TSM_predicted_values), 5) * 0.98,
                                     max_=round(max(TSM_predicted_values), 5) * 1.02),
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="10%")
        )
    )
    errors_chart = (
        Line()
        .add_xaxis([i for i in range(len(time))])
        .add_yaxis("TSMixer误差", TSMixer_errors.values.tolist(), is_smooth=False, color="blue",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("TiDE误差", TiDE_errors.values.tolist(), is_smooth=True, color="green",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("WOA+TSMixer误差", VMD_errors.values.tolist(), is_smooth=True, color="red",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="TSMixer/TiDE/VMD+TSMixer利用WOA优化全局参数误差对比"),
            xaxis_opts=opts.AxisOpts(name="检查点"),
            yaxis_opts=opts.AxisOpts(name="误差", min_=round(min(TSMixer_errors), 5) * 0.98,
                                     max_=round(max(TiDE_errors), 5) * 1.02),
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="10%")
        )
    )
    return TSMixer_chart.render_embed(), errors_chart.render_embed(), round(Decimal(TSMixer_mape), 10), round(
        Decimal(tide_mape), 10), round(Decimal(VMD_mape), 10), round(Decimal(TSMixer_mae), 10), round(Decimal(tide_mae),
                                                                                                      10), round(
        Decimal(VMD_mae), 10),


def without_WOA_chart():
    def with_WOA():
        # Load prediction results.
        TSMixer_prediction = pd.read_csv('TSMixer+WOA_result.csv')
        predicted_values = pd.to_numeric(TSMixer_prediction['predicted'], errors='coerce')
        TSM_true_values = pd.to_numeric(TSMixer_prediction['true'], errors='coerce')
        time = TSMixer_prediction['time']
        errors = predicted_values - TSM_true_values
        tide_mean_error = errors.mean()
        with_mae = mean_absolute_error(TSM_true_values, predicted_values)
        with_mape = mean_absolute_percentage_error(TSM_true_values, predicted_values)

        return time, predicted_values, TSM_true_values, tide_mean_error, errors, with_mape, with_mae

    def without_WOA():
        TSMixer_prediction = pd.read_csv('TSMixer_result.csv')
        predicted_values = pd.to_numeric(TSMixer_prediction['predicted'], errors='coerce')
        TSM_true_values = pd.to_numeric(TSMixer_prediction['true'], errors='coerce')
        time = TSMixer_prediction['time']
        errors = predicted_values - TSM_true_values
        tide_mean_error = errors.mean()
        without_mae = mean_absolute_error(TSM_true_values, predicted_values)
        without_mape = mean_absolute_percentage_error(TSM_true_values, predicted_values)

        return time, predicted_values, TSM_true_values, tide_mean_error, errors, without_mape, without_mae

    _, WOA_prediction, WOA_true_values, WOA_mean_error, with_WOA_errors, with_mape, with_mae = with_WOA()
    time, prediction, true_values, mean_error, without_WOA_errors, without_mape, without_mae = without_WOA()
    WOA_chart = (
        Line()
        .add_xaxis([i for i in range(len(time))])
        .add_yaxis("清洁因子", WOA_true_values.values.tolist(), is_smooth=False, color="grey",
                   linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.7), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("WOA参数优化", WOA_prediction.values.tolist(), is_smooth=False, color="blue",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("手动调参", prediction.values.tolist(), is_smooth=True, color="red",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="TSMixer利用WOA优化全局参数与手动调参预测结果对比"),
            xaxis_opts=opts.AxisOpts(name="检查点"),
            yaxis_opts=opts.AxisOpts(name="清洁因子", min_=round(min(WOA_prediction), 5) - 0.003,
                                     max_=round(max(WOA_prediction), 5) + 0.003),
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="10%")

        )
    )
    errors_chart = (
        Line()
        .add_xaxis([i for i in range(len(time))])
        .add_yaxis("WOA全局优化TSMixer误差", with_WOA_errors.values.tolist(), is_smooth=False, color="blue",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("手动调参TSMixer误差", without_WOA_errors.values.tolist(), is_smooth=True, color="red",
                   linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="WOA全局优化TSMixer与手动调参误差对比"),
            xaxis_opts=opts.AxisOpts(name="检查点"),
            yaxis_opts=opts.AxisOpts(name="误差", min_=round(min(without_WOA_errors), 5) - 0.003,
                                     max_=round(max(without_WOA_errors), 5) + 0.003),
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="10%")
        )
    )
    return WOA_chart.render_embed(), errors_chart.render_embed(), round(Decimal(with_mape), 10), round(
        Decimal(without_mape), 10), round(Decimal(with_mae), 10), round(Decimal(without_mae), 10)


def ablation_experiment_chart():
    def load_model_data(file_name):
        model_data = pd.read_csv(file_name)
        predicted_values = pd.to_numeric(model_data['predicted'], errors='coerce')
        true_values = pd.to_numeric(model_data['true'], errors='coerce')
        time = model_data['time']
        errors = predicted_values - true_values
        mae = mean_absolute_error(true_values, predicted_values)
        mape = mean_absolute_percentage_error(true_values, predicted_values)
        mse = mean_squared_error(true_values, predicted_values)
        return time, predicted_values, true_values, mae, mape, mse, errors

    # Load TSMixer model result files.
    time_VMD_TSMixer, VMD_TSMixer_predicted_values, VMD_TSMixer_true_values, VMD_TSMixer_mae, VMD_TSMixer_mape, VMD_TSMixer_mse, VMD_TSMixer_errors = load_model_data('VMD+TSMixer_result.csv')
    time_VMD_TSMixer_WOA, VMD_TSMixer_WOA_predicted_values, VMD_TSMixer_WOA_true_values, VMD_TSMixer_WOA_mae, VMD_TSMixer_WOA_mape, VMD_TSMixer_WOA_mse, VMD_TSMixer_WOA_errors = load_model_data('VMD+TSMixer+WOA_result.csv')
    time_TSMixer_WOA, TSMixer_WOA_predicted_values, TSMixer_WOA_true_values, TSMixer_WOA_mae, TSMixer_WOA_mape, TSMixer_WOA_mse, TSMixer_WOA_errors = load_model_data('TSMixer+WOA_result.csv')
    time_TSMixer, TSMixer_predicted_values, TSMixer_true_values, TSMixer_mae, TSMixer_mape, TSMixer_mse, TSMixer_errors = load_model_data('TSMixer_result.csv')

    # Load TiDE model result files.
    time_VMD_TiDE, VMD_TiDE_predicted_values, VMD_TiDE_true_values, VMD_TiDE_mae, VMD_TiDE_mape, VMD_TiDE_mse, VMD_TiDE_errors = load_model_data('VMD+TiDE_result.csv')
    time_VMD_TiDE_WOA, VMD_TiDE_WOA_predicted_values, VMD_TiDE_WOA_true_values, VMD_TiDE_WOA_mae, VMD_TiDE_WOA_mape, VMD_TiDE_WOA_mse, VMD_TiDE_WOA_errors = load_model_data('VMD+TiDE+WOA_result.csv')
    time_TiDE_WOA, TiDE_WOA_predicted_values, TiDE_WOA_true_values, TiDE_WOA_mae, TiDE_WOA_mape, TiDE_WOA_mse, TiDE_WOA_errors = load_model_data('TiDE+WOA_result.csv')
    time_TiDE, TiDE_predicted_values, TiDE_true_values, TiDE_mae, TiDE_mape, TiDE_mse, TiDE_errors = load_model_data('TiDE_result.csv')

    # Create the TSMixer ablation prediction-comparison chart.
    TSMixer_chart = (
        Line()
        .add_xaxis([i for i in range(len(time_VMD_TSMixer_WOA))])
        .add_yaxis("真实值（清洁因子）", VMD_TSMixer_true_values.values.tolist(), is_smooth=False, color="grey", linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.7), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("VMD+TSMixer预测值", VMD_TSMixer_predicted_values.values.tolist(), is_smooth=True, color="blue", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("VMD+TSMixer+WOA预测值", VMD_TSMixer_WOA_predicted_values.values.tolist(), is_smooth=True, color="orange", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("TSMixer预测值", TSMixer_predicted_values.values.tolist(), is_smooth=False, color="black", linestyle_opts=opts.LineStyleOpts(width=2,), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("WOA+TSMixer预测值", TSMixer_WOA_predicted_values.values.tolist(), is_smooth=False, color="green", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="A.TSMixer模型消融试验预测结果对比"),
            xaxis_opts=opts.AxisOpts(name="时间点"),
            yaxis_opts=opts.AxisOpts(name="清洁因子",max_=max(VMD_TSMixer_true_values)*1.025,min_=min(VMD_TSMixer_true_values)*0.975),
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="10%")
        )
    )

    # Create the TSMixer ablation error-comparison chart.
    TSMixer_error_chart = (
        Line()
        .add_xaxis([i for i in range(len(time_VMD_TSMixer_WOA))])
        .add_yaxis("VMD+TSMixer误差", VMD_TSMixer_errors.values.tolist(), is_smooth=False, color="blue", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("VMD+TSMixer+WOA误差", VMD_TSMixer_WOA_errors.values.tolist(), is_smooth=True, color="orange", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("TSMixer误差", TSMixer_errors.values.tolist(), is_smooth=False, color="black", linestyle_opts=opts.LineStyleOpts(width=2, ), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("WOA+TSMixer误差", TSMixer_WOA_errors.values.tolist(), is_smooth=False, color="green", linestyle_opts=opts.LineStyleOpts(width=2, ), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="B.TSMixer模型消融试验误差对比"),
            xaxis_opts=opts.AxisOpts(name="时间点"),
            yaxis_opts=opts.AxisOpts(name="误差"),
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="10%")
        )
    )

    # Create the TiDE ablation prediction-comparison chart.
    TiDE_chart = (
        Line()
        .add_xaxis([i for i in range(len(time_VMD_TiDE_WOA))])
        .add_yaxis("真实值（清洁因子）", VMD_TiDE_true_values.values.tolist(), is_smooth=False, color="grey", linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.7), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("VMD+TiDE预测值", VMD_TiDE_predicted_values.values.tolist(), is_smooth=True, color="cyan", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("VMD+TiDE+WOA预测值", VMD_TiDE_WOA_predicted_values.values.tolist(), is_smooth=True, color="red", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("TiDE预测值", TiDE_predicted_values.values.tolist(), is_smooth=False, color="purple", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("WOA+TiDE预测值", TiDE_WOA_predicted_values.values.tolist(), is_smooth=False, color="goldenrod", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="A.TiDE模型消融试验预测结果对比"),
            xaxis_opts=opts.AxisOpts(name="时间点"),
            yaxis_opts=opts.AxisOpts(name="清洁因子",max_=max(VMD_TSMixer_true_values)*1.025,min_=min(VMD_TSMixer_true_values)*0.975),
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="10%"),
        )
    )

    # Create the TiDE ablation error-comparison chart.
    TiDE_error_chart = (
        Line()
        .add_xaxis([i for i in range(len(time_VMD_TiDE_WOA))])
        .add_yaxis("VMD+TiDE误差", VMD_TiDE_errors.values.tolist(), is_smooth=False, color="cyan", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("VMD+TiDE+WOA误差", VMD_TiDE_WOA_errors.values.tolist(), is_smooth=True, color="red", linestyle_opts=opts.LineStyleOpts(width=2), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("TiDE误差", TiDE_errors.values.tolist(), is_smooth=False, color="purple", linestyle_opts=opts.LineStyleOpts(width=2,), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("WOA+TiDE误差", TiDE_WOA_errors.values.tolist(), is_smooth=False, color="goldenrod", linestyle_opts=opts.LineStyleOpts(width=2,), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="B.TiDE模型消融试验误差对比"),
            xaxis_opts=opts.AxisOpts(name="时间点"),
            yaxis_opts=opts.AxisOpts(name="误差"),
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="10%")
        )
    )

    # Return MAPE, MAE, and MSE values for the eight ablation experiments.
    return TSMixer_chart.render_embed(), TSMixer_error_chart.render_embed(), TiDE_chart.render_embed(), TiDE_error_chart.render_embed(), \
           round(Decimal(VMD_TSMixer_WOA_mape), 10), round(Decimal(VMD_TSMixer_mape), 10), round(Decimal(TSMixer_WOA_mape), 10), round(Decimal(TSMixer_mape), 10), \
           round(Decimal(VMD_TSMixer_WOA_mae), 10), round(Decimal(VMD_TSMixer_mae), 10), round(Decimal(TSMixer_WOA_mae), 10), round(Decimal(TSMixer_mae), 10), \
           round(Decimal(VMD_TSMixer_WOA_mse), 10), round(Decimal(VMD_TSMixer_mse), 10), round(Decimal(TSMixer_WOA_mse), 10), round(Decimal(TSMixer_mse), 10), \
           round(Decimal(VMD_TiDE_WOA_mape), 10), round(Decimal(VMD_TiDE_mape), 10), round(Decimal(TiDE_WOA_mape), 10), round(Decimal(TiDE_mape), 10), \
           round(Decimal(VMD_TiDE_WOA_mae), 10), round(Decimal(VMD_TiDE_mae), 10), round(Decimal(TiDE_WOA_mae), 10), round(Decimal(TiDE_mae), 10), \
           round(Decimal(VMD_TiDE_WOA_mse), 10), round(Decimal(VMD_TiDE_mse), 10), round(Decimal(TiDE_WOA_mse), 10), round(Decimal(TiDE_mse), 10)


