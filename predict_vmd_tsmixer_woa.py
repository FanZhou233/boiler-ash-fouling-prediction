"""Normalized script generated from data_prediction_VMD+TSMixer+WOA.py."""

import numpy as np
import pandas as pd
from darts.models import TSMixerModel
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from functools import reduce
from mealpy.swarm_based.WOA import OriginalWOA
from mealpy import IntegerVar
from darts.metrics import mape as darts_mape
import torch

from data_cleaning import load_clean_factor_data
from data_decomposition_vmd import decompose_signal


# Utility functions.
torch.set_float32_matmul_precision('high')
def to_int_params(sol, lb, ub):
    """四舍五入并裁剪超参到整数范围"""
    rounded = [int(round(x)) for x in sol]
    return [
        int(np.clip(rounded[0], lb[0], ub[0])),                    # ic
        int(np.clip(rounded[1], max(1, lb[1]), ub[1])),  # The output chunk length must be at least 1.
        int(np.clip(rounded[2], lb[2], ub[2]))                     # hs
    ]

def safe_mape(true_ts, pred_ts, eps=1e-6):
    """仅对 |y_true|>eps 的点计算 MAPE"""
    y_true = true_ts.values().flatten()
    y_pred = pred_ts.values().flatten()
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan  # No valid points are available.
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Main workflow.

soft, _, _, _, _ = load_clean_factor_data()
signal_all = soft[2672:4172]
modes, _ = decompose_signal(signal_all, K=8)

train_len = int(len(signal_all) * 0.9)
time_all = np.arange(2672, 4172)

mode_preds = np.zeros((modes.shape[0], len(signal_all) - train_len))
mode_trues = signal_all[train_len:]

# Hyperparameter bounds.
lb = [12, 1, 32]
ub = [96, 24, 512]

for k, mode_k in enumerate(modes):
    # Skip forecasting when the mode amplitude is extremely small.
    if np.mean(np.abs(mode_k[:train_len])) < 1e-3:
        print(f"Mode {k + 1} has negligible amplitude; zero prediction is used.")
        continue

    # Build and normalize the Darts series.
    train_k, val_k = mode_k[:train_len], mode_k[train_len:]
    ts_train = TimeSeries.from_times_and_values(pd.Series(time_all[:train_len]), train_k)
    ts_val   = TimeSeries.from_times_and_values(pd.Series(time_all[train_len:]), val_k)
    scaler   = Scaler(MinMaxScaler())
    ts_train = scaler.fit_transform(ts_train)
    ts_val   = scaler.transform(ts_val)

    # Wrap Darts historical forecasting.
    def predict_series(model):
        full = ts_train.concatenate(ts_val)
        raws = model.historical_forecasts(
            series=full,
            start=len(ts_train),
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False,
        )
        pred = reduce(lambda x, y: x.concatenate(y), raws)
        pred = scaler.inverse_transform(pred)
        return pred.slice_intersect(ts_val)

    # Objective function returning safe MAPE.
    def record_obj(sol):
        ic, oc, hs = to_int_params(sol, lb, ub)
        try:
            m = TSMixerModel(
                input_chunk_length=ic,
                output_chunk_length=oc,
                hidden_size=hs,
                dropout=0.1,
                n_epochs=10,
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={'lr': 1e-3},
                loss_fn=torch.nn.MSELoss(),
                random_state=42,
            )
            m.fit(ts_train, verbose=False)
            pred_ts = predict_series(m)
            err = safe_mape(scaler.inverse_transform(ts_val), pred_ts)
            return err if not np.isnan(err) else 1e3
        except Exception:
            return 1e6

    # Run WOA optimization.
    problem = {
        "obj_func": record_obj,
        "bounds": IntegerVar(lb=lb, ub=ub),
        "minmax": "min"
    }
    woa = OriginalWOA(epoch=10, pop_size=5, verbose=False)
    best = woa.solve(problem)

    ic, oc, hs = to_int_params(best.solution, lb, ub)
    print(f"Mode {k + 1} best parameters: ic={ic}, oc={oc}, hs={hs}")

    # Retrain for more epochs using the best hyperparameters.
    final = TSMixerModel(
        input_chunk_length=ic,
        output_chunk_length=oc,
        hidden_size=hs,
        dropout=0.1,
        n_epochs=30,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={'lr': 1e-3},
        loss_fn=torch.nn.MSELoss(),
        random_state=42,
    )
    final.fit(ts_train, verbose=False)
    pred_k = predict_series(final).pd_series().values.flatten()
    mode_preds[k] = pred_k

# Merge modal forecasts and evaluate the final signal.
merged_preds = mode_preds.sum(axis=0)
final_mape = safe_mape(
    TimeSeries.from_times_and_values(pd.Series(time_all[train_len:]), mode_trues),
    TimeSeries.from_times_and_values(pd.Series(time_all[train_len:]), merged_preds)
)
print(f"Final merged signal Safe-MAPE: {final_mape:.2f}%")
# Merge modal predictions and visualize the final reconstructed signal.
plt.figure(figsize=(10, 4))
plt.plot(time_all[train_len:], mode_trues, label="真实信号")
plt.plot(time_all[train_len:], merged_preds, label="合并后预测")
plt.legend()
plt.grid(True)
plt.title("VMD+TSMixer+WOA 分解后逐模态预测并合并")
plt.tight_layout()
plt.show()

df_results = pd.DataFrame({
    'time': time_all[train_len:],  # Time for the test set
    'predicted': merged_preds,     # Merged predictions
    'true': mode_trues             # Actual true values
})

df_results.to_csv('VMD_prediction_result.csv', index=False)
