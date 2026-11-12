"""
Preprocessing module for LSC spectrum analysis.

Responsibilities:
  - Subtract blank count rates from sample gross count rates (Mode A: classical blank subtraction)
  - Propagate Poisson counting uncertainties (var = gross/t_s^2 + blank/t_b^2)
  - Ensure numerical stability and clamp non-physical negative rates
"""
from __future__ import annotations
import numpy as np


def subtract_blank_poisson(
    sample_counts: np.ndarray | list[float],
    sample_time: float,
    blank_counts: np.ndarray | list[float] | None = None,
    blank_time: float | None = None,
    min_variance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Mode A blank subtraction and Poisson variance error propagation.

    Parameters:
      sample_counts : Raw counts array from sample (1024-channel)
      sample_time   : Measurement live time for sample in seconds (> 0)
      blank_counts  : Raw counts array from blank (1024-channel, or None if no blank)
      blank_time    : Measurement live time for blank in seconds (or None)
      min_variance  : Clamping threshold for zero/negative variance floor

    Returns:
      rate_net  : Net count rate spectrum array (cps, len = 1024), clamped >= 0.0
      sigma_net : Combined Poisson standard uncertainty array (cps, len = 1024)
    """
    s_counts = np.asarray(sample_counts, dtype=float)
    t_s = float(sample_time)
    if t_s <= 0.0:
        raise ValueError(f"Sample live time must be positive, got {t_s}")

    s_rate = s_counts / t_s
    s_var = s_counts / (t_s ** 2)

    if blank_counts is not None and blank_time is not None and float(blank_time) > 0.0:
        b_counts = np.asarray(blank_counts, dtype=float)
        t_b = float(blank_time)
        if b_counts.shape != s_counts.shape:
            raise ValueError(f"Blank shape {b_counts.shape} does not match sample shape {s_counts.shape}")
        b_rate = b_counts / t_b
        b_var = b_counts / (t_b ** 2)

        rate_net = np.maximum(s_rate - b_rate, 0.0)
        total_var = s_var + b_var
    else:
        rate_net = np.maximum(s_rate, 0.0)
        total_var = s_var

    sigma_net = np.sqrt(np.maximum(total_var, min_variance))
    return rate_net, sigma_net
