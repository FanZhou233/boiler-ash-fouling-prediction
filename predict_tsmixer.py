"""Normalized script generated from data_prediction_TSMIxer.py."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from functools import reduce
from darts import TimeSeries
from darts.models import TSMixerModel
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from mealpy import IntegerVar

# Enable high-precision matrix multiplication and keep deterministic settings where possible.
torch.set_float32_matmul_precision('high')

# Part 1: Prepare and normalize the data.

from data_cleaning import load_clean_factor_data

# Load the original signal.
soft, _, _, _, _ = load_clean_factor_data()
time_all = np.arange(2672, 4172)
signal_all = soft[2672:4172]

# Split the signal into training and validation segments.
train_len = int(len(signal_all) * 0.9)
train_signal = signal_all[:train_len]
val_signal = signal_all[train_len:]
train_time = time_all[:train_len]
val_time = time_all[train_len:]

# Build Darts TimeSeries objects.
train_series = TimeSeries.from_times_and_values(pd.Series(train_time), train_signal)
val_series = TimeSeries.from_times_and_values(pd.Series(val_time), val_signal)

# Normalize the signal with MinMaxScaler.
scaler = Scaler(MinMaxScaler())
train_series = scaler.fit_transform(train_series)
val_series = scaler.transform(val_series)

# Part 2: Define one-step historical forecasting.

def prediction(model, scaler, return_series=False):
    full_series = train_series.concatenate(val_series)
    preds = model.historical_forecasts(
        series=full_series,
        start=len(train_series),
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=False,
    )
    preds = reduce(lambda x, y: x.concatenate(y), preds)
    preds = scaler.inverse_transform(preds)
    true = scaler.inverse_transform(val_series)
    if return_series:
        return preds, true
    return mape(true, preds)

# Part 3: Set manual hyperparameters and train the model.

# Manual hyperparameter settings.
input_chunk_length = 48  # Input sequence length.
output_chunk_length = 12  # Output sequence length.
hidden_size = 128  # Hidden layer size.

# Create and train the model.
model = TSMixerModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    hidden_size=hidden_size,
    dropout=0.1,
    n_epochs=10,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-3},
    loss_fn=torch.nn.MSELoss(),
    random_state=42,
)

model.fit(train_series, verbose=False)

# Part 4: Predict and visualize results.

# Generate predictions with the trained model.
preds, trues = prediction(model, scaler, return_series=True)
preds = preds.slice_intersect(trues)

# Visualize prediction results.
plt.figure(figsize=(10, 5))
trues.plot(label="true", lw=2)
preds.plot(label="predicted", lw=2)
plt.title(f"TSMixer 单步预测效果 (input_chunk_length={input_chunk_length}, output_chunk_length={output_chunk_length}, hidden_size={hidden_size})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save prediction results as a CSV file.
df_preds = preds.pd_dataframe().rename(columns={preds.columns[0]: "predicted"})
df_trues = trues.pd_dataframe().rename(columns={trues.columns[0]: "true"})

# Merge aligned prediction and ground-truth series before saving.
df_result = pd.concat([df_preds, df_trues], axis=1)
df_result.to_csv("TSMixer_prediction_without_WOA_result.csv", encoding="utf-8-sig")
print("Prediction results saved to TSMixer_prediction_without_WOA_result.csv")
