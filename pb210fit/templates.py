from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Betashape data directory within package
DATA_DIR = Path(__file__).resolve().parent / "data" / "betashape"


def load_betashape_spectrum(bs_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Parse LNHB BetaShape .bs file and return (E_keV, dNdE)."""
    p = Path(bs_path)
    if not p.exists():
        raise FileNotFoundError(f"BetaShape file not found: {p}")

    E, dN, started = [], [], False
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if t.startswith("E(keV)"):
            started = True
            continue
        if not started or not t:
            continue
        parts = t.split()
        if len(parts) >= 2:
            try:
                E.append(float(parts[0]))
                dN.append(float(parts[1]))
            except ValueError:
                continue

    E_arr = np.array(E, dtype=float)
    dN_arr = np.array(dN, dtype=float)
    
    total_sum = dN_arr.sum()
    if total_sum > 0:
        dN_arr = dN_arr / total_sum
    return E_arr, dN_arr


def get_bi210_physical_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Return LNHB BetaShape physical beta spectrum for Bi-210 (E_max = 1162.2 keV)."""
    bs_file = DATA_DIR / "Bi210" / "beta-_Bi210_tot.bs"
    if not bs_file.exists():
        bs_file = DATA_DIR / "Bi210" / "beta-_Bi210_trans0.bs"
    return load_betashape_spectrum(bs_file)


def get_pb210_deposit_spectrum(n_bins: int = 1200, P_esc_xl: float = 0.23, P_esc_g: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct Pb-210 complete energy deposition spectrum from LNHB decay scheme (nanosecond coincidence).
    """
    BR1, BR2 = 0.802, 0.198
    E_LEVEL = 46.539
    A_TOTAL = (84.0 - 4.25) / 4.25            # 18.76 (total IC coefficient)
    F_IC = A_TOTAL / (1.0 + A_TOTAL)          # 0.9494
    F_L = 14.2 / 18.7                         # L-shell fraction among IC (~0.7594)
    OMEGA_L = 0.39                            # L-shell fluorescence yield
    E_XL = 11.5                               # Mean L X-ray energy (keV)

    bs_trans1 = DATA_DIR / "Pb210" / "beta-_Pb210_trans1.bs"
    bs_trans0 = DATA_DIR / "Pb210" / "beta-_Pb210_trans0.bs"
    
    E1, w1 = load_betashape_spectrum(bs_trans1)
    E2, w2 = load_betashape_spectrum(bs_trans0)

    f_xesc = F_L * OMEGA_L * P_esc_xl
    chans = [
        (E1 + E_LEVEL, BR1 * F_IC * (1.0 - f_xesc) * w1),
        (E1 + (E_LEVEL - E_XL), BR1 * F_IC * f_xesc * w1),
        (E1 + E_LEVEL, BR1 * (1.0 - F_IC) * (1.0 - P_esc_g) * w1),
        (E1, BR1 * (1.0 - F_IC) * P_esc_g * w1),
        (E2, BR2 * w2),
    ]

    e_grid = np.linspace(0.1, 70.0, n_bins)
    de = e_grid[1] - e_grid[0]
    spec = np.zeros(n_bins)

    for E_arr, w_arr in chans:
        dE_arr = np.gradient(E_arr)
        density = w_arr / np.maximum(dE_arr, 1e-6)
        interp_density = np.interp(e_grid, E_arr, density, left=0.0, right=0.0)
        spec += interp_density * de

    total_sum = spec.sum()
    if total_sum > 0:
        spec /= total_sum

    return e_grid, spec


_PB210_BASIS_PAIR_CACHE: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def get_pb210_deposit_basis_pair(
    n_bins: int = 1200,
    P_esc_g: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return cached endpoint basis pair (e_grid, dN_pb_0, dN_pb_1) for Pb-210 L-X-ray escape probability P_esc_xl:
      - dN_pb_0 : P_esc_xl = 0.0 (100% of 11.5 keV L-X-rays absorbed in 20 mL cocktail)
      - dN_pb_1 : P_esc_xl = 1.0 (100% of 11.5 keV L-X-rays escape from 20 mL cocktail)
    For any P_esc_xl in [0, 1], the exact energy deposition spectrum is:
      dN_pb(P_esc_xl) = (1.0 - P_esc_xl) * dN_pb_0 + P_esc_xl * dN_pb_1
    """
    key = (int(n_bins), round(float(P_esc_g), 5))
    if key not in _PB210_BASIS_PAIR_CACHE:
        e_grid, dN_0 = get_pb210_deposit_spectrum(n_bins=n_bins, P_esc_xl=0.0, P_esc_g=P_esc_g)
        _,      dN_1 = get_pb210_deposit_spectrum(n_bins=n_bins, P_esc_xl=1.0, P_esc_g=P_esc_g)
        _PB210_BASIS_PAIR_CACHE[key] = (e_grid, dN_0, dN_1)
    return _PB210_BASIS_PAIR_CACHE[key]


A_MFG = 362.0
B_MFG = 173.0
E0_MFG = 3.0
# Po-210 5.304 MeV alpha electron-equivalent unquenched energy E_alpha_eq (keV)
# such that at nominal Q = 0.265, ch = 362 * log10(0.265 * (E_ALPHA_EQ + 3.0)) - 173 = 574.0
E_ALPHA_EQ = (10.0 ** ((574.0 + B_MFG) / A_MFG)) / 0.265 - E0_MFG



def build_template_steps(
    E_tab: np.ndarray,
    dNdE: np.ndarray,
    Q: float,
    sigma_0: float,
    ch_0: float = 85.0,
    n_channels: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    3-Step First-Principles Forward Transformation:
      Step 1: Jacobian continuous coordinate transform (E -> ch)
      Step 2: Gaussian optical resolution convolution (sigma(E) broadening)
      Step 3: Triple Coincidence TDCR turn-on discrimination
    """
    CH = np.arange(n_channels, dtype=float)
    E_safe = np.maximum(E_tab, 0.0)
    ch_centers = A_MFG * np.log10(np.maximum(Q * (E_safe + E0_MFG), 1e-6)) - B_MFG
    sigma_ch = np.maximum(sigma_0 / np.sqrt(np.maximum(Q * (E_safe + E0_MFG), 1e-3)), 1.0)

    # Step 1: Continuous Jacobian transform
    E_ch = (10.0 ** ((CH + B_MFG) / A_MFG)) / max(Q, 1e-6) - E0_MFG
    dnde_interp = np.interp(E_ch, E_tab, dNdE, left=0.0, right=0.0)
    jacobian = (np.log(10.0) / A_MFG) * np.maximum(E_ch + E0_MFG, 0.0)
    step1_raw = dnde_interp * jacobian
    s1 = step1_raw.sum()
    if s1 > 0:
        step1_raw /= s1

    # Step 2: Gaussian convolution matrix (Area = 1.0 per energy point)
    diff = CH[:, None] - ch_centers[None, :]
    gauss_matrix = np.exp(-0.5 * (diff / sigma_ch[None, :]) ** 2) / (np.sqrt(2.0 * np.pi) * sigma_ch[None, :])
    step2_gauss = gauss_matrix @ dNdE
    s2 = step2_gauss.sum()
    if s2 > 0:
        step2_gauss /= s2

    # Step 3: TDCR turn-on efficiency
    u = 10.0 ** ((CH - ch_0) / A_MFG)
    p = 1.0 - np.exp(- (u ** 2.0))
    eff_tdcr = np.clip(p ** 3.0, 0.0, 1.0)
    step3_final = step2_gauss * eff_tdcr

    return step1_raw, step2_gauss, step3_final, eff_tdcr


def build_template_from_spectrum(
    E_tab: np.ndarray,
    dNdE: np.ndarray,
    Q: float,
    sigma_0: float,
    ch_0: float = 85.0,
    n_channels: int = 1024,
) -> np.ndarray:
    """Return final instrument template spectrum T(ch)."""
    _, _, final_template, _ = build_template_steps(E_tab, dNdE, Q, sigma_0, ch_0, n_channels)
    return final_template


def load_bi210_template(template_path: str | Path | None = None) -> dict:
    """
    Backward-compatible template loader.
    """
    E_bi, dN_bi = get_bi210_physical_spectrum()
    E_pb, dN_pb = get_pb210_deposit_spectrum()

    T_bi_nom = build_template_from_spectrum(E_bi, dN_bi, Q=0.2646, sigma_0=326.8)
    T_pb_nom = build_template_from_spectrum(E_pb, dN_pb, Q=0.2646, sigma_0=326.8)

    return {
        "template": T_bi_nom,
        "energy": np.arange(1024, dtype=float),
        "spectrum": T_bi_nom,
        "E_bi": E_bi,
        "dN_bi": dN_bi,
        "E_pb": E_pb,
        "dN_pb": dN_pb,
    }


def build_warped_blank_template(
    blank_spectrum: np.ndarray,
    k_qpe: float,
    ch_0: float = 85.0,
    n_channels: int = 1024,
) -> np.ndarray:
    """
    Construct Mode B quench-warped empirical Blank basis template T_blank^corr(ch).

    In HIDEX logarithmic response ch(E; Q) = 362*log10(Q*(E+3)) - 173, a change in quench
    by ratio k_qpe = Q_sample / Q_blank induces a uniform channel shift:
      delta_ch = 362.0 * log10(k_qpe)

    Parameters:
      blank_spectrum : Measured blank counts or rate array (len = n_channels)
      k_qpe          : Quench scale ratio Q_sample / Q_blank (> 0)
      ch_0           : Low-energy hardware turn-on threshold (default 85.0)
      n_channels     : Number of channels (default 1024)

    Returns:
      Normalized warped template array (len = n_channels, sum = 1.0)
    """
    raw_bkg = np.asarray(blank_spectrum, dtype=float)
    if k_qpe <= 0 or not np.isfinite(k_qpe):
        k_qpe = 1.0

    delta_ch = 362.0 * np.log10(max(k_qpe, 1e-4))
    ch_grid = np.arange(n_channels, dtype=float)

    # Inverse map: to find value at ch_grid, query raw_bkg at (ch_grid - delta_ch)
    src_ch = ch_grid - delta_ch
    warped = np.interp(src_ch, ch_grid, raw_bkg, left=0.0, right=0.0)

    # Apply hardware turn-on threshold
    turn_on = np.clip((ch_grid - ch_0) / 10.0, 0.0, 1.0) ** 2.0
    warped = np.maximum(warped * turn_on, 0.0)

    total = warped.sum()
    if total > 0:
        warped /= total
    return warped


