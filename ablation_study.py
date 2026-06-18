"""Normalized script generated from ablation_study.py."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from decimal import Decimal
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimSun']  # Configure Chinese font rendering for Matplotlib labels.
plt.rcParams['axes.unicode_minus'] = False  # Render minus signs correctly in Matplotlib.


def TSMixer():
    # Load a model result file and calculate metrics.
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

    # Read result data using the same names as the CSV columns.
    time_VMD_TSMixer, VMD_TSMixer_predicted_values, VMD_TSMixer_true_values, VMD_TSMixer_mae, VMD_TSMixer_mape, VMD_TSMixer_mse, VMD_TSMixer_errors = load_model_data(
        'VMD+TSMixer_result.csv')
    time_VMD_TSMixer_WOA, VMD_TSMixer_WOA_predicted_values, VMD_TSMixer_WOA_true_values, VMD_TSMixer_WOA_mae, VMD_TSMixer_WOA_mape, VMD_TSMixer_WOA_mse, VMD_TSMixer_WOA_errors = load_model_data(
        'VMD+TSMixer+WOA_result.csv')
    time_TSMixer_WOA, TSMixer_WOA_predicted_values, TSMixer_WOA_true_values, TSMixer_WOA_mae, TSMixer_WOA_mape, TSMixer_WOA_mse, TSMixer_WOA_errors = load_model_data(
        'TSMixer+WOA_result.csv')
    time_TSMixer, TSMixer_predicted_values, TSMixer_true_values, TSMixer_mae, TSMixer_mape, TSMixer_mse, TSMixer_errors = load_model_data(
        'TSMixer_result.csv')

    # Use a consistent numeric x-axis index.
    x = range(len(time_VMD_TSMixer_WOA))

    # Create prediction and error plots.
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    x = range(len(time_VMD_TSMixer_WOA))

    # Create prediction and error plots.
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # First panel: prediction comparison.
    axs[0].plot(x, VMD_TSMixer_true_values.values, label='真实值（清洁因子）', color='grey', linewidth=2, alpha=0.7)
    axs[0].plot(x, VMD_TSMixer_WOA_predicted_values.values, label='VMD+TSMixer+WOA预测值', color='orange', linewidth=2, linestyle='-')
    axs[0].plot(x, VMD_TSMixer_predicted_values.values, label='VMD+TSMixer预测值', color='blue', linewidth=2, linestyle='--')
    axs[0].plot(x, TSMixer_WOA_predicted_values.values, label='WOA+TSMixer预测值', color='green', linewidth=2, linestyle='-.')
    axs[0].plot(x, TSMixer_predicted_values.values, label='TSMixer预测值', color='black', linewidth=2, linestyle=':')
    axs[0].set_ylabel("清洁因子", fontsize=14)
    axs[0].legend(loc='upper right', fontsize=12)
    axs[0].grid(True)
    axs[0].set_ylim(
        min(VMD_TSMixer_WOA_predicted_values.min(),
            VMD_TSMixer_predicted_values.min(),
            TSMixer_WOA_predicted_values.min()) * 0.98,
        max(VMD_TSMixer_WOA_predicted_values.max(),
            VMD_TSMixer_predicted_values.max(),
            TSMixer_WOA_predicted_values.max()) * 1.02
    )
    # Add a panel caption below the subplot.
    axs[0].text(
        0.5, -0.15, "A. TSMixer模型消融试验预测结果对比",
        transform=axs[0].transAxes,
        ha="center", va="top",
        fontsize=14
    )

    # Second panel: error comparison.
    axs[1].plot(x, VMD_TSMixer_WOA_errors.values, label='VMD+TSMixer+WOA误差', color='orange', linewidth=2)
    axs[1].plot(x, VMD_TSMixer_errors.values, label='VMD+TSMixer误差', color='blue', linewidth=2)
    axs[1].plot(x, TSMixer_WOA_errors.values, label='WOA+TSMixer误差', color='green', linewidth=2)
    axs[1].plot(x, TSMixer_errors.values, label='TSMixer误差', color='black', linewidth=2)
    axs[1].set_xlabel("时间点", fontsize=14)
    axs[1].set_ylabel("预测误差", fontsize=14)
    axs[1].legend(loc='upper right', fontsize=12)
    axs[1].grid(True)
    axs[1].set_ylim(
        min(VMD_TSMixer_WOA_errors.min(),
            VMD_TSMixer_errors.min(),
            TSMixer_errors.min()) * 1.15,
        max(VMD_TSMixer_WOA_errors.max(),
            VMD_TSMixer_errors.max(),
            TSMixer_errors.max()) * 1.15
    )
    # Add a panel caption below the subplot.
    axs[1].text(
        0.5, -0.15, "B. TSMixer消融试验的预测误差对比",
        transform=axs[1].transAxes,
        ha="center", va="top",
        fontsize=14
    )

    # Adjust the layout to prevent panel captions from being clipped.
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    plt.show()


    # Print error metrics rounded to 10 decimal places.
    print("VMD+TSMixer+WOA MAPE:", round(Decimal(VMD_TSMixer_WOA_mape), 10))
    print("VMD+TSMixer MAPE:", round(Decimal(VMD_TSMixer_mape), 10))
    print("WOA+TSMixer MAPE:", round(Decimal(TSMixer_WOA_mape), 10))
    print("TSMixer MAPE:", round(Decimal(TSMixer_mape), 10))

    print("VMD+TSMixer+WOA MAE:", round(Decimal(VMD_TSMixer_WOA_mae), 10))
    print("VMD+TSMixer MAE:", round(Decimal(VMD_TSMixer_mae), 10))
    print("WOA+TSMixer MAE:", round(Decimal(TSMixer_WOA_mae), 10))
    print("TSMixer MAE:", round(Decimal(TSMixer_mae), 10))

    print("VMD+TSMixer+WOA MSE:", round(Decimal(VMD_TSMixer_WOA_mse), 10))
    print("VMD+TSMixer MSE:", round(Decimal(VMD_TSMixer_mse), 10))
    print("WOA+TSMixer MSE:", round(Decimal(TSMixer_WOA_mse), 10))
    print("TSMixer MSE:", round(Decimal(TSMixer_mse), 10))




def TiDE():
    # Load a model result file and calculate metrics.
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

    time_VMD_TiDE, VMD_TiDE_predicted_values, VMD_TiDE_true_values, VMD_TiDE_mae, VMD_TiDE_mape, VMD_TiDE_mse, VMD_TiDE_errors = load_model_data(
        'VMD+TiDE_result.csv')
    time_VMD_TiDE_WOA, VMD_TiDE_WOA_predicted_values, VMD_TiDE_WOA_true_values, VMD_TiDE_WOA_mae, VMD_TiDE_WOA_mape, VMD_TiDE_WOA_mse, VMD_TiDE_WOA_errors = load_model_data(
        'VMD+TiDE+WOA_result.csv')
    time_TiDE_WOA, TiDE_WOA_predicted_values, TiDE_WOA_true_values, TiDE_WOA_mae, TiDE_WOA_mape, TiDE_WOA_mse, TiDE_WOA_errors = load_model_data(
        'TiDE+WOA_result.csv')
    time_TiDE, TiDE_predicted_values, TiDE_true_values, TiDE_mae, TiDE_mape, TiDE_mse, TiDE_errors = load_model_data(
        'TiDE_result.csv')
    # Use a consistent numeric x-axis index.
    x = range(len(time_VMD_TiDE_WOA))

    # Create prediction and error plots.
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # First panel: prediction comparison.
    axs[0].plot(x, VMD_TiDE_true_values.values, label='真实值（清洁因子）', color='grey', linewidth=2, alpha=0.7)
    axs[0].plot(x, VMD_TiDE_WOA_predicted_values.values, label='VMD+TiDE+WOA预测值', color='red', linewidth=2, linestyle='-')
    axs[0].plot(x, VMD_TiDE_predicted_values.values, label='VMD+TiDE预测值', color='cyan', linewidth=2, linestyle='--')
    axs[0].plot(x, TiDE_WOA_predicted_values.values, label='WOA+TiDE预测值', color='goldenrod', linewidth=2, linestyle='-.')
    axs[0].plot(x, TiDE_predicted_values.values, label='TiDE预测值', color='purple', linewidth=2, linestyle=':')
    axs[0].set_ylabel("清洁因子", fontsize=14)
    axs[0].legend(loc='upper right', fontsize=12)
    axs[0].grid(True)
    axs[0].set_ylim(
        min(VMD_TiDE_WOA_predicted_values.min(),
            VMD_TiDE_predicted_values.min(),
            TiDE_WOA_predicted_values.min()) * 0.98,
        max(VMD_TiDE_WOA_predicted_values.max(),
            VMD_TiDE_predicted_values.max(),
            TiDE_WOA_predicted_values.max()) * 1.02
    )
    # Add a panel caption below the subplot.
    axs[0].text(
        0.5, -0.15, "A. TiDE模型消融试验预测结果对比",
        transform=axs[0].transAxes,
        ha="center", va="top",
        fontsize=14
    )

    # Second panel: error comparison.
    axs[1].plot(x, VMD_TiDE_WOA_errors.values, label='VMD+TiDE+WOA误差', color='red', linewidth=2)
    axs[1].plot(x, VMD_TiDE_errors.values, label='VMD+TiDE误差', color='cyan', linewidth=2)
    axs[1].plot(x, TiDE_WOA_errors.values, label='WOA+TiDE误差', color='goldenrod', linewidth=2)
    axs[1].plot(x, TiDE_errors.values, label='TiDE误差', color='purple', linewidth=2)
    axs[1].set_xlabel("时间点", fontsize=14)
    axs[1].set_ylabel("预测误差", fontsize=14)
    axs[1].legend(loc='upper right', fontsize=12)
    axs[1].grid(True)
    axs[1].set_ylim(
        min(VMD_TiDE_WOA_errors.min(),
            VMD_TiDE_errors.min(),
            TiDE_errors.min()) * 1.15,
        max(VMD_TiDE_WOA_errors.max(),
            VMD_TiDE_errors.max(),
            TiDE_errors.max()) * 1.15
    )
    # Add a panel caption below the subplot.
    axs[1].text(
        0.5, -0.15, "B. TiDE消融试验的预测误差对比",
        transform=axs[1].transAxes,
        ha="center", va="top",
        fontsize=14
    )

    # Adjust the layout to prevent panel captions from being clipped.
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    plt.show()


def TiDE_train():
    # Load a model result file and calculate metrics.
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

    time_VMD_TiDE_WOA, VMD_TiDE_WOA_predicted_values, VMD_TiDE_WOA_true_values, VMD_TiDE_WOA_mae, VMD_TiDE_WOA_mape, VMD_TiDE_WOA_mse, VMD_TiDE_WOA_errors = load_model_data(
        'VMD+TiDE+WOA_result.csv')

    # Use a consistent numeric x-axis index.
    x = range(len(time_VMD_TiDE_WOA))

    # Create prediction and error plots.
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # First panel: prediction comparison.
    axs[0].plot(x, VMD_TiDE_WOA_true_values.values, label='真实值（清洁因子）', color='grey', linewidth=2, alpha=0.7)
    axs[0].plot(x, VMD_TiDE_WOA_predicted_values.values, label='VMD+TSMixer+WOA预测值', color='red', linewidth=2,
                linestyle='-')
    axs[0].set_ylabel("清洁因子", fontsize=14)
    axs[0].legend(loc='upper right', fontsize=12)
    axs[0].grid(True)
    axs[0].set_ylim(
        min(VMD_TiDE_WOA_predicted_values.min(), VMD_TiDE_WOA_true_values.min()) * 0.98,
        max(VMD_TiDE_WOA_predicted_values.max(), VMD_TiDE_WOA_true_values.max()) * 1.02
    )
    # Add a panel caption below the subplot.
    axs[0].text(
        0.5, -0.15, "A. VMD+TiDE+WOA训练后预测结果",
        transform=axs[0].transAxes,
        ha="center", va="top",
        fontsize=14
    )

    # Second panel: error comparison.
    axs[1].plot(x, VMD_TiDE_WOA_errors.values, label='VMD+TiDE+WOA误差', color='red', linewidth=2)
    axs[1].set_xlabel("时间点", fontsize=14)
    axs[1].set_ylabel("预测误差", fontsize=14)
    axs[1].legend(loc='upper right', fontsize=12)
    axs[1].grid(True)
    axs[1].set_ylim(
        VMD_TiDE_WOA_errors.min() * 1.15,
        VMD_TiDE_WOA_errors.max() * 1.15
    )
    # Add a panel caption below the subplot.
    axs[1].text(
        0.5, -0.15, "B. VMD+TiDE+WOA训练后的预测误差对比",
        transform=axs[1].transAxes,
        ha="center", va="top",
        fontsize=14
    )

    # Adjust the layout to prevent panel captions from being clipped.
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)  # Fine-tune subplot spacing as needed.
    plt.show()


def TSM_train():
    # Load a model result file and calculate metrics.
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

    time_VMD_TiDE_WOA, VMD_TiDE_WOA_predicted_values, VMD_TiDE_WOA_true_values, VMD_TiDE_WOA_mae, VMD_TiDE_WOA_mape, VMD_TiDE_WOA_mse, VMD_TiDE_WOA_errors = load_model_data(
        'VMD+TSMixer+WOA_result.csv')

    # Use a consistent numeric x-axis index.
    x = range(len(time_VMD_TiDE_WOA))

    # Create prediction and error plots.
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # First panel: prediction comparison.
    axs[0].plot(x, VMD_TiDE_WOA_true_values.values, label='真实值（清洁因子）', color='grey', linewidth=2, alpha=0.7)
    axs[0].plot(x, VMD_TiDE_WOA_predicted_values.values, label='VMD+TSMixer+WOA预测值', color='orange', linewidth=2, linestyle='-')
    axs[0].set_ylabel("清洁因子", fontsize=14)
    axs[0].legend(loc='upper right', fontsize=12)
    axs[0].grid(True)
    axs[0].set_ylim(
        min(VMD_TiDE_WOA_predicted_values.min(), VMD_TiDE_WOA_true_values.min()) * 0.98,
        max(VMD_TiDE_WOA_predicted_values.max(), VMD_TiDE_WOA_true_values.max()) * 1.02
    )
    # Add a panel caption below the subplot.
    axs[0].text(
        0.5, -0.15, "A. VMD+TSMixer+WOA训练后预测结果",
        transform=axs[0].transAxes,
        ha="center", va="top",
        fontsize=14
    )


    axs[1].plot(x, VMD_TiDE_WOA_errors.values, label='VMD+TSMixer+WOA误差', color='orange', linewidth=2)
    axs[1].set_xlabel("时间点", fontsize=14)
    axs[1].set_ylabel("预测误差", fontsize=14)
    axs[1].legend(loc='upper right', fontsize=12)
    axs[1].grid(True)
    axs[1].set_ylim(
        VMD_TiDE_WOA_errors.min() * 1.15,
        VMD_TiDE_WOA_errors.max() * 1.15
    )
    # Add a panel caption below the subplot.
    axs[1].text(
        0.5, -0.15, "B. VMD+TSMixer+WOA训练后的预测误差对比",
        transform=axs[1].transAxes,
        ha="center", va="top",
        fontsize=14
    )

    # Adjust the layout to prevent panel captions from being clipped.
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)  # Fine-tune subplot spacing as needed.
    plt.show()



# Function calls.
# TSMixer()
# TiDE()
#
# TiDE_train()
# TSM_train()
# TiDE_train()
# TiDE()
TSMixer()
