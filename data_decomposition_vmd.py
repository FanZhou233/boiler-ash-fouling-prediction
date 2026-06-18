"""VMD decomposition wrapper."""

from __future__ import annotations

import numpy as np
from vmdpy import VMD


def decompose_signal(
    data: np.ndarray,
    k: int = 5,
    alpha: float = 20,
    tau: float = 0.0,
    dc: int = 0,
    init: int = 1,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a one-dimensional signal using Variational Mode Decomposition.

    Args:
        data: Input signal.
        k: Number of VMD modes.
        alpha: Bandwidth constraint parameter.
        tau: Noise-tolerance parameter.
        dc: Whether to keep the first mode at DC.
        init: Initialization strategy used by vmdpy.
        tol: Convergence tolerance.

    Returns:
        A tuple containing the decomposed modes and center-frequency history.
    """

    signal = np.asarray(data, dtype=float).flatten()
    signal = np.nan_to_num(signal)
    if signal.ndim != 1:
        raise ValueError(f"VMD input must be one-dimensional, got ndim={signal.ndim}.")

    modes, modes_hat, omega = VMD(signal, alpha, tau, k, dc, init, tol)
    return modes, omega


# Backward-compatible alias for the original script name.
decomposeSignal = decompose_signal
