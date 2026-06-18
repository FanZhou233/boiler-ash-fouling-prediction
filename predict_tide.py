"""Normalized script generated from data_prediction_TiDE.py."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from functools import reduce
from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler

torch.set_float32_matmul_precision('high')

from data_cleaning import load_clean_factor_data

# Prepare and normalize the data.
soft, _, _, _, _ = load_clean_factor_data()
time_all = np.arange(2672, 4172)
signal_all = soft[2672:4172]

train_len = int(len(signal_all) * 0.9)
train_signal = signal_all[:train_len]
val_signal = signal_all[train_len:]
train_time = time_all[:train_len]
val_time = time_all[train_len:]

train_series = TimeSeries.from_times_and_values(pd.Series(train_time), train_signal)
val_series = TimeSeries.from_times_and_values(pd.Series(val_time), val_signal)

scaler = Scaler(MinMaxScaler())
train_series = scaler.fit_transform(train_series)
val_series = scaler.transform(val_series)


# Prediction helper function.
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


# Fixed hyperparameters for the baseline model.
input_chunk_length = 48
output_chunk_length = 12
hidden_size = 256

# Train the model with fixed hyperparameters.
model = TiDEModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    hidden_size=hidden_size,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dropout=0.1,
    n_epochs=30,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-3},
    loss_fn=torch.nn.MSELoss(),
    random_state=42,
)

model.fit(train_series, verbose=True)

# Predict and visualize.
preds, trues = prediction(model, scaler, return_series=True)
preds = preds.slice_intersect(trues)

plt.figure(figsize=(10, 5))
trues.plot(label="true", lw=2)
preds.plot(label="predicted", lw=2)
plt.title("TiDE 固定参数预测结果")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save results.
df_preds = preds.pd_dataframe().rename(columns={preds.columns[0]: "predicted"})
df_trues = trues.pd_dataframe().rename(columns={trues.columns[0]: "true"})
df_result = pd.concat([df_preds, df_trues], axis=1)
df_result.to_csv("TiDE_result.csv", encoding="utf-8-sig")
print("Prediction results saved to TiDE_result.csv")
