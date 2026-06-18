"""Wavelet denoising utilities for the boiler clean-factor signal."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.io

DATA_FILE = Path("CF1.mat")
MAT_KEY = "fai_sm"
DEFAULT_WAVELET = "db4"


def load_clean_factor_data(
    file_path: str | Path = DATA_FILE,
    mat_key: str = MAT_KEY,
    wavelet: str = DEFAULT_WAVELET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the clean-factor signal and return several wavelet-denoised variants.

    Returns:
        denoised_soft: Signal processed with soft thresholding.
        denoised_hard: Signal processed with hard thresholding.
        denoised_fixed: Signal processed with the fixed Sqtwolog threshold.
        original_signal: Original one-dimensional signal.
        time_index: Integer sample index.
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file '{file_path}' was not found. Place CF1.mat in the project root "
            "or update DATA_FILE in data_cleaning.py."
        )

    mat_data = scipy.io.loadmat(file_path)
    original_signal = mat_data[mat_key].flatten()
    time_index = np.arange(len(original_signal))

    max_level = pywt.dwt_max_level(len(original_signal), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(original_signal, wavelet, level=max_level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    universal_threshold = sigma * np.sqrt(2 * np.log(len(original_signal)))

    coeffs_soft = [coeffs[0]] + [
        pywt.threshold(coef, universal_threshold, mode="soft") for coef in coeffs[1:]
    ]
    denoised_soft = pywt.waverec(coeffs_soft, wavelet)

    coeffs_hard = [coeffs[0]] + [
        pywt.threshold(coef, universal_threshold, mode="hard") for coef in coeffs[1:]
    ]
    denoised_hard = pywt.waverec(coeffs_hard, wavelet)

    coeffs_fixed = [coeffs[0]] + [
        pywt.threshold(coef, universal_threshold, mode="hard") for coef in coeffs[1:]
    ]
    denoised_fixed = pywt.waverec(coeffs_fixed, wavelet)

    min_len = min(
        len(original_signal),
        len(denoised_soft),
        len(denoised_hard),
        len(denoised_fixed),
    )
    return (
        denoised_soft[:min_len],
        denoised_hard[:min_len],
        denoised_fixed[:min_len],
        original_signal[:min_len],
        time_index[:min_len],
    )


def residual_energy_ratio(original_signal: np.ndarray, denoised_signal: np.ndarray) -> float:
    """Calculate the residual energy ratio after denoising."""

    numerator = np.sum((original_signal - denoised_signal) ** 2)
    denominator = np.sum(original_signal**2)
    return float(numerator / denominator)


def smoothness_index(signal: np.ndarray) -> float:
    """Calculate a smoothness index from first-order differences."""

    return float(np.sum(np.diff(signal) ** 2))


def plot_denoising_comparison() -> None:
    """Plot soft-threshold and hard-threshold denoising results."""

    denoised_soft, denoised_hard, _, original_signal, time_index = load_clean_factor_data()

    plt.rcParams["font.sans-serif"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(time_index, original_signal, label="原始信号", color="gray", alpha=0.7)
    axes[0].plot(time_index, denoised_soft, label="软阈值去噪", color="blue")
    axes[0].set_xlabel("检查点")
    axes[0].set_ylabel("清洁因子")
    axes[0].set_ylim([0.52, 0.64])
    axes[0].set_xlim([2672, 4172])
    axes[0].legend()
    axes[0].text(
        0.5,
        -0.15,
        "A. 小波去噪 - 软阈值",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=12,
    )

    axes[1].plot(time_index, original_signal, label="原始信号", color="gray", alpha=0.7)
    axes[1].plot(time_index, denoised_hard, label="硬阈值去噪", color="red")
    axes[1].set_xlabel("检查点")
    axes[1].set_ylabel("清洁因子")
    axes[1].set_xlim([2672, 4172])
    axes[1].set_ylim([0.52, 0.64])
    axes[1].legend()
    axes[1].text(
        0.5,
        -0.15,
        "B. 小波去噪 - 硬阈值",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Backward-compatible alias for the original graduation-design scripts.
dataCleaning = load_clean_factor_data


if __name__ == "__main__":
    plot_denoising_comparison()
