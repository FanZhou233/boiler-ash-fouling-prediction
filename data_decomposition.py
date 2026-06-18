"""Visualize VMD modes and instantaneous frequencies."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from data_cleaning import load_clean_factor_data
from data_decomposition_vmd import decompose_signal

SIGNAL_START = 2672
SIGNAL_END = 4172
N_MODES = 8


def plot_vmd_decomposition() -> None:
    """Plot VMD modes and a 3D instantaneous-frequency waterfall."""

    denoised_soft, _, _, _, _ = load_clean_factor_data()
    time_all = np.arange(SIGNAL_START, SIGNAL_END)
    signal_all = denoised_soft[SIGNAL_START:SIGNAL_END]
    modes, freqs = decompose_signal(signal_all, k=N_MODES)

    plt.rcParams["font.sans-serif"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    freqs = np.asarray(freqs)
    if freqs.ndim > 1:
        freqs = freqs[:, -1]

    plt.figure(figsize=(10, 8))
    for idx in range(modes.shape[0]):
        plt.subplot(modes.shape[0], 1, idx + 1)
        plt.plot(modes[idx])
        plt.title(f"Mode {idx + 1}  (center freq ≈ {freqs[idx]:.3f})")
        plt.tight_layout()
    plt.show()

    n_modes, n_points = modes.shape
    dt = 1
    instantaneous_frequency = np.zeros((n_modes, n_points))
    for idx in range(n_modes):
        analytic_signal = hilbert(modes[idx])
        phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency[idx, :-1] = np.diff(phase) / (2 * np.pi * dt)
        instantaneous_frequency[idx, -1] = instantaneous_frequency[idx, -2]

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    for idx in range(n_modes):
        x_values = time_all
        y_values = np.full(n_points, idx)
        z_values = instantaneous_frequency[idx]
        ax.plot(x_values, y_values, z_values, linewidth=1)

    ax.set_xlabel("采样点")
    ax.set_ylabel("IMF 分量")
    ax.set_zlabel("瞬时频率 (Hz)")
    ax.set_title("VMD 分解瞬时频率瀑布图")
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_vmd_decomposition()
