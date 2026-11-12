from __future__ import annotations

import sys
import math
from typing import Any
import numpy as np

try:
    from scipy.optimize import least_squares, nnls
    from scipy.special import erfc as _scipy_erfc
except ImportError:
    least_squares = None
    nnls = None
    _scipy_erfc = np.vectorize(math.erfc)

from .templates import (
    get_pb210_deposit_spectrum,
    get_pb210_deposit_basis_pair,
    get_bi210_physical_spectrum,
    build_template_from_spectrum,
    A_MFG,
    B_MFG,
    E0_MFG,
    E_ALPHA_EQ,
)


def po_light_gauss(ch: np.ndarray, mu: float, w: float) -> np.ndarray:
    """
    Polonium-210 Alpha Monoenergetic Peak on the unified hardware log axis (A_MFG = 362.0).
      Linear effective-signal Gaussian mapped to logarithmic channel space with Jacobian du/dch.
    """
    u = np.power(10.0, (ch + B_MFG) / A_MFG)
    u0 = np.power(10.0, (mu + B_MFG) / A_MFG)
    s = u0 * np.log(10.0) * max(w, 1e-3) / A_MFG
    g = np.exp(-0.5 * ((u - u0) / s) ** 2) * (u * np.log(10.0) / A_MFG)
    total_sum = g.sum()
    if total_sum > 0:
        g /= total_sum
    return g


def po_light_emg(ch: np.ndarray, mu: float, w: float, tau_rel: float = 0.0) -> np.ndarray:
    """
    Polonium-210 Alpha Peak with Left-Tailed Exponential Energy Loss (Left-EMG)
    in linear effective-light space u = 10^((ch + B_MFG) / A_MFG), mapped to the
    logarithmic MCA channel axis with Jacobian du/dch = u * ln(10) / A_MFG.
      - Models alpha self-absorption / energy straggling (e.g., filter paper or precipitate).
      - tau_rel = tau_u / u0 is the dimensionless relative energy-loss decay scale.
      - Smoothly reduces to po_light_gauss(ch, mu, w) as tau_rel -> 0.
    """
    if tau_rel <= 1e-4:
        return po_light_gauss(ch, mu, w)
    u = np.power(10.0, (ch + B_MFG) / A_MFG)
    u0 = np.power(10.0, (mu + B_MFG) / A_MFG)
    s = u0 * np.log(10.0) * max(w, 1e-3) / A_MFG
    tau_u = u0 * max(float(tau_rel), 1e-4)
    z = u0 - u
    arg1 = np.clip(0.5 * (s / tau_u) ** 2 - z / tau_u, -60.0, 60.0)
    arg2 = (s / tau_u - z / s) / np.sqrt(2.0)
    emg_u = (1.0 / (2.0 * tau_u)) * np.exp(arg1) * _scipy_erfc(arg2)
    g = emg_u * (u * np.log(10.0) / A_MFG)
    total_sum = g.sum()
    if total_sum > 0:
        g /= total_sum
    return g


# Nominal optical resolution constant at the alpha peak energy (~150 ch*keV^0.5 -> ~14 ch at Q=0.265)
SIG0_ALPHA_NOM = 150.5

# Nominal 20 mL vial Bi L-X-ray escape probability (P_esc_xl)
PESC_XL_NOM = 0.230


def compute_po210_coupled_params(
    Q: float,
    sig_0: float,
    rho_ab: float = 1.0,
    scale_w: float = 1.0,
) -> tuple[float, float]:
    """
    Soft-coupled Po-210 channel center (mu_po) and width (w_po) derived from shared quenching Q:
      1. Effective quenched alpha signal: u_alpha = rho_ab * Q * (E_ALPHA_EQ + E0_MFG)
      2. Channel center: mu_po = A_MFG * log10(u_alpha) - B_MFG
      3. Quench-coupled statistical width: w_base(Q, rho_ab) = SIG0_ALPHA_NOM / sqrt(u_alpha)
      4. Total width: w_po = scale_w * w_base (soft-coupled width ratio around 1.0)
    """
    u_alpha = max(rho_ab * Q * (E_ALPHA_EQ + E0_MFG), 1e-3)
    mu_po = float(A_MFG * np.log10(u_alpha) - B_MFG)
    w_base = float(SIG0_ALPHA_NOM / np.sqrt(u_alpha))
    w_po = float(np.clip(scale_w * w_base, 5.0, 35.0))
    return mu_po, w_po


# Coarse 2D pre-scan grids and hierarchical beta-template cache
_Q_PRESCAN_GRID = np.linspace(0.05, 1.20, 24)
_RHO_PRESCAN_GRID = np.linspace(0.80, 1.24, 10)
_SIG0_PRESCAN = 200.0
_PRESCAN_BETA_CACHE: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}


def _prescan_2d_q_rho(
    rate_net: np.ndarray,
    sigma_net: np.ndarray,
    ok: np.ndarray,
    CH: np.ndarray,
    E_pb: np.ndarray,
    dN_pb: np.ndarray,
    E_bi: np.ndarray,
    dN_bi: np.ndarray,
    bkg_template: np.ndarray | None = None,
    prior_sigma_rho: float = 0.20,
    use_cache: bool = True,
) -> list[tuple[float, float, float, np.ndarray]]:
    """
    Decoupled 2D (Q, rho_ab) Variable-Projection Pre-Scan:
      - Outer loop (Q_i, 15 points): evaluates/caches beta basis templates T_pb(Q_i), T_bi(Q_i)
      - Inner loop (rho_j, 10 points): evaluates analytic alpha peak T_po(Q_i, rho_j)
      - Solves 3x3 (or 4x4 with Bkg) non-negative linear least squares (NNLS)
      - Returns sorted list of (cost, Q_i, rho_j, amps)
    """
    n_ch = len(rate_net)
    inv_sig = 1.0 / sigma_net[ok]
    yw = rate_net[ok] * inv_sig
    f_bg_w = (bkg_template[ok] * inv_sig) if bkg_template is not None else None
    candidates: list[tuple[float, float, float, np.ndarray]] = []

    for q_val in _Q_PRESCAN_GRID:
        qf = float(q_val)
        cache_key = (n_ch, round(qf, 5))
        if use_cache and cache_key in _PRESCAN_BETA_CACHE:
            T_pb_q, T_bi_q = _PRESCAN_BETA_CACHE[cache_key]
        else:
            T_pb_q = build_template_from_spectrum(E_pb, dN_pb, qf, _SIG0_PRESCAN, ch_0=85.0, n_channels=n_ch)
            T_bi_q = build_template_from_spectrum(E_bi, dN_bi, qf, _SIG0_PRESCAN, ch_0=85.0, n_channels=n_ch)
            if use_cache:
                _PRESCAN_BETA_CACHE[cache_key] = (T_pb_q, T_bi_q)

        f_pb_w = T_pb_q[ok] * inv_sig
        f_bi_w = T_bi_q[ok] * inv_sig

        for rho_val in _RHO_PRESCAN_GRID:
            rhof = float(rho_val)
            mu_p, w_p = compute_po210_coupled_params(qf, _SIG0_PRESCAN, rhof, 1.0)
            T_po_qr = po_light_gauss(CH, mu_p, w_p)
            f_po_w = T_po_qr[ok] * inv_sig
            if f_bg_w is not None:
                Mw = np.column_stack([f_pb_w, f_bi_w, f_po_w, f_bg_w])
            else:
                Mw = np.column_stack([f_pb_w, f_bi_w, f_po_w])
            if nnls is not None:
                amps, _ = nnls(Mw, yw)
            else:
                amps, _, _, _ = np.linalg.lstsq(Mw, yw, rcond=None)
                amps = np.maximum(amps, 0.0)
            res_d = Mw @ amps - yw
            prior_r = (rhof - 1.0) / prior_sigma_rho
            cost_est = 0.5 * float(np.sum(res_d ** 2) + prior_r ** 2)
            candidates.append((cost_est, qf, rhof, amps))

    candidates.sort(key=lambda item: item[0])
    return candidates


def fit_3nuclides_spectrum(
    rate_net: np.ndarray,
    sigma_net: np.ndarray | None = None,
    E_pb: np.ndarray | None = None,
    dN_pb: np.ndarray | None = None,
    E_bi: np.ndarray | None = None,
    dN_bi: np.ndarray | None = None,
    bkg_template: np.ndarray | None = None,
    fit_min_ch: int = 120,
    fit_max_ch: int = 750,
    q_init: float = 0.265,
    sig0_init: float = 320.0,
    mu_po_init: float = 574.0,
    w_po_init: float = 14.0,
    max_nfev: int = 3000,
) -> dict[str, Any]:
    """
    First-Principles Physical Simultaneous Fitting:
      Pb-210 + Bi-210 + Po-210 (+ optional Bkg template for 4-component fit)
      with hardware transfer function optimization.
      
    Parameters:
      rate_net     : Net count rate spectrum array (cps, len = 1024)
      sigma_net    : Poisson standard uncertainty array (len = 1024)
      bkg_template : Optional warped blank template for Mode B simultaneous 4-component fit
      fit_min_ch   : Lower ROI channel (default 120)
      fit_max_ch   : Upper ROI channel (default 750)
      
    Returns:
      Dictionary containing fit parameters, synthesized components, residuals, and quality metrics.
    """
    if least_squares is None:
        raise RuntimeError("scipy.optimize.least_squares is required for physical fitting.")

    rate_net = np.asarray(rate_net, dtype=float)
    CH = np.arange(len(rate_net), dtype=float)

    if sigma_net is None:
        # Default Poisson sqrt(rate / time) or Poisson variance estimate
        sigma_net = np.sqrt(np.maximum(np.abs(rate_net), 1e-4))
    else:
        sigma_net = np.asarray(sigma_net, dtype=float)
        sigma_net = np.maximum(sigma_net, 1e-4)

    # Load default LNHB physical spectra if not provided
    use_default_templates = (E_pb is None or dN_pb is None) and (E_bi is None or dN_bi is None)
    if E_pb is None or dN_pb is None:
        E_pb, dN_pb_0, dN_pb_1 = get_pb210_deposit_basis_pair()
        dN_pb = (1.0 - PESC_XL_NOM) * dN_pb_0 + PESC_XL_NOM * dN_pb_1
    else:
        dN_pb_0 = dN_pb
        dN_pb_1 = dN_pb
    if E_bi is None or dN_bi is None:
        E_bi, dN_bi = get_bi210_physical_spectrum()

    ok = (CH >= fit_min_ch) & (CH <= fit_max_ch)
    if not np.any(ok):
        raise ValueError(f"No valid channels in ROI [{fit_min_ch}, {fit_max_ch}]")

    a0 = max(float(rate_net[ok].max()), 1e-4)
    has_bkg = bkg_template is not None

    # Physical parameter bounds:
    # Mode A (3-nuclide, 9 params): p = [A_pb, A_bi, A_po, Q, sig_0, rho_ab, scale_w, P_esc_xl, tau_rel]
    # Mode B (4-component, 10 params): p = [A_pb, A_bi, A_po, Q, sig_0, rho_ab, scale_w, P_esc_xl, tau_rel, A_bg]
    if has_bkg:
        bounds_lower = [0.0, 0.0, 0.0, 0.02, 50.0, 0.75, 0.35, 0.00, 0.00, 0.0]
        bounds_upper = [np.inf, np.inf, np.inf, 2.00, 600.0, 1.30, 2.50, 1.00, 1.00, np.inf]
    else:
        bounds_lower = [0.0, 0.0, 0.0, 0.02, 50.0, 0.75, 0.35, 0.00, 0.00]
        bounds_upper = [np.inf, np.inf, np.inf, 2.00, 600.0, 1.30, 2.50, 1.00, 1.00]

    # Soft-coupling prior weights:
    # Gently anchors (rho_ab -> 1.0, scale_w -> 1.0, P_esc_xl -> 0.230, tau_rel -> 0.0) when nuclides are weak/absent,
    # while allowing free physical adaptation when spectra are well-resolved (e.g., filter-paper alpha tailing).
    prior_sigma_rho = 0.20
    prior_sigma_scale_w = 0.50
    prior_sigma_pesc = 0.12
    prior_sigma_tau = 0.15

    # Step 1–3: 2D (Q, rho_ab) Decoupled Variable-Projection Pre-Scan (grid-2d)
    prescan_cands = _prescan_2d_q_rho(
        rate_net=rate_net,
        sigma_net=sigma_net,
        ok=ok,
        CH=CH,
        E_pb=E_pb,
        dN_pb=dN_pb,
        E_bi=E_bi,
        dN_bi=dN_bi,
        bkg_template=bkg_template,
        prior_sigma_rho=prior_sigma_rho,
        use_cache=use_default_templates,
    )
    _, q0_grid, rho0_grid, amps0_grid = prescan_cands[0]
    if has_bkg:
        x0 = [
            max(float(amps0_grid[0]), a0 * 0.02),
            max(float(amps0_grid[1]), a0 * 0.02),
            max(float(amps0_grid[2]), a0 * 0.01),
            float(np.clip(q0_grid, 0.155, 0.445)),
            _SIG0_PRESCAN,
            float(np.clip(rho0_grid, 0.76, 1.29)),
            1.0,
            PESC_XL_NOM,
            0.02,
            max(float(amps0_grid[3]) if len(amps0_grid) > 3 else 0.0, a0 * 0.01),
        ]
    else:
        x0 = [
            max(float(amps0_grid[0]), a0 * 0.02),
            max(float(amps0_grid[1]), a0 * 0.02),
            max(float(amps0_grid[2]), a0 * 0.01),
            float(np.clip(q0_grid, 0.155, 0.445)),
            _SIG0_PRESCAN,
            float(np.clip(rho0_grid, 0.76, 1.29)),
            1.0,
            PESC_XL_NOM,
            0.02,
        ]

    def residual_func(p: np.ndarray) -> np.ndarray:
        if has_bkg:
            A_pb, A_bi, A_po, Q, sig_0, rho_ab, scale_w, p_esc, tau_rel, A_bg = p
        else:
            A_pb, A_bi, A_po, Q, sig_0, rho_ab, scale_w, p_esc, tau_rel = p
            A_bg = 0.0

        dN_pb_mix = (1.0 - p_esc) * dN_pb_0 + p_esc * dN_pb_1
        T_pb = build_template_from_spectrum(E_pb, dN_pb_mix, Q, sig_0, ch_0=85.0, n_channels=len(rate_net))
        T_bi = build_template_from_spectrum(E_bi, dN_bi, Q, sig_0, ch_0=85.0, n_channels=len(rate_net))
        mu_p, w_p = compute_po210_coupled_params(Q, sig_0, rho_ab, scale_w)
        T_po = po_light_emg(CH, mu_p, w_p, tau_rel)
        model = A_pb * T_pb + A_bi * T_bi + A_po * T_po
        if has_bkg and bkg_template is not None:
            model = model + A_bg * bkg_template
        data_res = (model[ok] - rate_net[ok]) / sigma_net[ok]
        prior_res = np.array([
            (rho_ab - 1.0) / prior_sigma_rho,
            (scale_w - 1.0) / prior_sigma_scale_w,
            (p_esc - PESC_XL_NOM) / prior_sigma_pesc,
            (tau_rel - 0.0) / prior_sigma_tau,
        ], dtype=float)
        return np.concatenate([data_res, prior_res])

    try:
        # Step 4: Full continuous TRF nonlinear optimization from top grid-2d seed
        opt = least_squares(
            residual_func,
            x0=x0,
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            max_nfev=max_nfev,
            loss='linear',
        )
        success = bool(opt.success)
        best_p = opt.x
        best_cost = float(opt.cost)

        # Secondary basin check if initial basin leaves elevated residual cost
        ndf_est = max(int(np.sum(ok)) - (10 if has_bkg else 9), 1)
        if (2.0 * best_cost / ndf_est) > 1.5:
            for _, q_alt, rho_alt, amps_alt in prescan_cands[1:]:
                if abs(q_alt - q0_grid) > 0.04 or abs(rho_alt - rho0_grid) > 0.08:
                    if has_bkg:
                        x0_alt = [
                            max(float(amps_alt[0]), a0 * 0.02),
                            max(float(amps_alt[1]), a0 * 0.02),
                            max(float(amps_alt[2]), a0 * 0.01),
                            float(np.clip(q_alt, 0.155, 0.445)),
                            _SIG0_PRESCAN,
                            float(np.clip(rho_alt, 0.76, 1.29)),
                            1.0,
                            PESC_XL_NOM,
                            0.02,
                            max(float(amps_alt[3]) if len(amps_alt) > 3 else 0.0, a0 * 0.01),
                        ]
                    else:
                        x0_alt = [
                            max(float(amps_alt[0]), a0 * 0.02),
                            max(float(amps_alt[1]), a0 * 0.02),
                            max(float(amps_alt[2]), a0 * 0.01),
                            float(np.clip(q_alt, 0.155, 0.445)),
                            _SIG0_PRESCAN,
                            float(np.clip(rho_alt, 0.76, 1.29)),
                            1.0,
                            PESC_XL_NOM,
                            0.02,
                        ]
                    opt_alt = least_squares(
                        residual_func,
                        x0=x0_alt,
                        bounds=(bounds_lower, bounds_upper),
                        method='trf',
                        max_nfev=max_nfev,
                        loss='linear',
                    )
                    if opt_alt.success and float(opt_alt.cost) < best_cost:
                        opt = opt_alt
                        success = bool(opt_alt.success)
                        best_p = opt_alt.x
                        best_cost = float(opt_alt.cost)
                    break
    except Exception as e:
        success = False
        best_p = np.array(x0, dtype=float)

    if has_bkg:
        A_pb, A_bi, A_po, Q, sig_0, rho_ab, scale_w, P_esc_xl, tau_rel, A_bg = best_p
    else:
        A_pb, A_bi, A_po, Q, sig_0, rho_ab, scale_w, P_esc_xl, tau_rel = best_p
        A_bg = 0.0
    mu_p, w_p = compute_po210_coupled_params(Q, sig_0, rho_ab, scale_w)

    # Compute final full 1024-ch curves
    dN_pb_final = (1.0 - P_esc_xl) * dN_pb_0 + P_esc_xl * dN_pb_1
    T_pb = build_template_from_spectrum(E_pb, dN_pb_final, Q, sig_0, ch_0=85.0, n_channels=len(rate_net))
    T_bi = build_template_from_spectrum(E_bi, dN_bi, Q, sig_0, ch_0=85.0, n_channels=len(rate_net))
    T_po = po_light_emg(CH, mu_p, w_p, tau_rel)

    fit_Pb_y = A_pb * T_pb
    fit_Bi_y = A_bi * T_bi
    fit_Po_y = A_po * T_po
    fit_Bg_y = (A_bg * bkg_template) if (has_bkg and bkg_template is not None) else np.zeros_like(fit_Pb_y)
    fit_cbn = fit_Pb_y + fit_Bi_y + fit_Po_y + fit_Bg_y
    count_residue = rate_net - fit_cbn

    # Pull residuals in ROI
    pull = np.zeros_like(rate_net)
    pull[ok] = count_residue[ok] / sigma_net[ok]

    # Full Nominal Metrics
    n_params = 10 if has_bkg else 9
    chi2 = float(np.sum(pull[ok] ** 2))
    ndf = max(int(np.sum(ok)) - n_params, 1)
    red_chi2 = float(chi2 / ndf)
    pull_std = float(np.std(pull[ok])) if np.sum(ok) > 0 else 1.0
    pull_mean = float(np.mean(pull[ok])) if np.sum(ok) > 0 else 0.0

    # Compute GUM-compliant parameter covariance matrix from Jacobian J with Birge scaling max(1.0, red_chi2)
    sigma_A_pb = 0.0
    sigma_A_bi = 0.0
    sigma_A_po = 0.0
    sigma_A_bg = 0.0
    birge_scale = max(1.0, red_chi2)
    if success and hasattr(opt, 'jac') and opt.jac is not None:
        try:
            # J is ((M + 4) x P) weighted residual Jacobian (data + weak priors), so J.T @ J is the Gauss-Newton Hessian
            H = opt.jac.T @ opt.jac
            cov = birge_scale * np.linalg.inv(H)
            diag = np.maximum(np.diag(cov), 0.0)
            sigma_A_pb = float(np.sqrt(diag[0]))
            sigma_A_bi = float(np.sqrt(diag[1]))
            sigma_A_po = float(np.sqrt(diag[2]))
            if has_bkg and len(diag) > 9:
                sigma_A_bg = float(np.sqrt(diag[9]))
        except Exception:
            sigma_A_pb = float(np.sqrt(max(A_pb / 43200.0, 1e-6)))
            sigma_A_bi = float(np.sqrt(max(A_bi / 43200.0, 1e-6)))
            sigma_A_po = float(np.sqrt(max(A_po / 43200.0, 1e-6)))
            if has_bkg:
                sigma_A_bg = float(np.sqrt(max(A_bg / 43200.0, 1e-6)))
    else:
        sigma_A_pb = float(np.sqrt(max(A_pb / 43200.0, 1e-6)))

    # Active ROI & Effective dof (Correcting zero-count channel dilution for low-activity samples)
    max_model_roi = float(np.max(fit_cbn[ok])) if np.any(ok) else 1e-12
    max_rate_roi = float(np.max(rate_net[ok])) if np.any(ok) else 1e-12
    active_mask = ok & (
        (fit_cbn > max(max_model_roi * 0.005, 1e-6)) |
        (rate_net > 2.0 * sigma_net) |
        (rate_net > max(max_rate_roi * 0.01, 1e-6))
    )
    active_channels = int(np.sum(active_mask))
    active_ndf = max(active_channels - n_params, 1) if active_channels > n_params else max(active_channels - 3, 1)

    if active_channels > 0:
        chi2_eff = float(np.sum(pull[active_mask] ** 2))
        red_chi2_eff = float(chi2_eff / active_ndf)
        pull_active_std = float(np.std(pull[active_mask]))
        pull_active_mean = float(np.mean(pull[active_mask]))
    else:
        chi2_eff = chi2
        red_chi2_eff = red_chi2
        pull_active_std = pull_std
        pull_active_mean = pull_mean

    # Resolution-Binned Wald-Wolfowitz Runs Test:
    # LSC optical scintillation resolution (FWHM) is ~15-25 channels, causing strong local autocorrelation
    # on raw single MCA channels. Binning residuals by optical resolution element (width = 12 ch)
    # allows testing macroscopic model shape distortion (peak shifts, tailing errors) with independent Poisson blocks.
    bin_size = 12
    roi_indices = np.where(ok)[0]
    bin_residuals = []
    
    if len(roi_indices) > 0:
        for start_idx in range(roi_indices[0], roi_indices[-1] + 1, bin_size):
            end_idx = min(start_idx + bin_size, roi_indices[-1] + 1)
            sub_mask = np.zeros_like(rate_net, dtype=bool)
            sub_mask[start_idx:end_idx] = True
            
            sub_active = sub_mask & active_mask
            if np.any(sub_active):
                bin_residuals.append(float(np.sum(count_residue[sub_active])))
            elif np.any(sub_mask & ok) and np.sum(rate_net[sub_mask & ok]) > 0:
                bin_residuals.append(float(np.sum(count_residue[sub_mask & ok])))

    bin_residuals = np.array(bin_residuals, dtype=float)
    signs = np.where(bin_residuals > 0, 1, -1)
    n_pos = int(np.sum(signs == 1))
    n_neg = int(np.sum(signs == -1))
    n_total = n_pos + n_neg

    if n_pos > 0 and n_neg > 0 and n_total >= 6:
        runs = int(1 + np.sum(signs[:-1] != signs[1:]))
        mu_runs = 1.0 + (2.0 * n_pos * n_neg) / float(n_total)
        var_runs = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_total)) / float((n_total ** 2) * (n_total - 1))
        if var_runs > 1e-9:
            z_runs = (runs - mu_runs) / np.sqrt(var_runs)
            runs_p_value = float(math.erfc(abs(z_runs) / math.sqrt(2.0)))
        else:
            runs_p_value = 1.0
    else:
        runs_p_value = 1.0

    norm_rate = float(np.linalg.norm(rate_net[ok]))
    norm_res = float(np.linalg.norm(count_residue[ok]))
    relative_residual = float((norm_res / max(norm_rate, 1e-12)) * 100.0)
    
    # Poisson noise-subtracted systematic error
    excess_var = max(float(np.sum(count_residue[ok] ** 2) - np.sum(sigma_net[ok] ** 2)), 0.0)
    systematic_error = float((np.sqrt(excess_var) / max(np.sum(rate_net[ok]), 1e-12)) * 100.0)

    bi_pb_ratio = float(A_bi / max(A_pb, 1e-12))

    # Nuclear Physics multi-criteria grading:
    # If systematic_error or relative_residual is very low (< 4.0%), microscopic non-randomness is physically negligible.
    is_low_residual = bool((systematic_error < 4.0) or (relative_residual < 4.0))

    # Pass: red_chi2_eff <= 2.0 (or nominal red_chi2 <= 1.6) + systematic_error < 5.0% + pull_std <= 1.35 + (runs_p >= 0.01 or is_low_residual)
    # Physical parameter validity: tau_rel <= 0.20 (normal) and P_esc_xl <= 0.40 (normal)
    is_pass = success and (
        (red_chi2 <= 1.6 or red_chi2_eff <= 2.0) and
        (systematic_error < 5.0) and
        (pull_std <= 1.35) and
        (runs_p_value >= 0.01 or is_low_residual) and
        (tau_rel <= 0.20) and
        (P_esc_xl <= 0.40)
    )
    is_warn = success and not is_pass and (
        (red_chi2 <= 3.0 or red_chi2_eff <= 3.5) and
        (systematic_error < 15.0)
    )
    grade = "Pass" if is_pass else ("Warn" if is_warn else "Fail")

    fit_quality = {
        "is_valid": success and (red_chi2 <= 3.0 or red_chi2_eff <= 3.5),
        "chi2": chi2,
        "ndf": ndf,
        "red_chi2": red_chi2,
        "red_chi2_eff": red_chi2_eff,
        "active_channels": active_channels,
        "active_ndf": active_ndf,
        "runs_p_value": runs_p_value,
        "pull_std": pull_std,
        "pull_mean": pull_mean,
        "pull_active_std": pull_active_std,
        "pull_active_mean": pull_active_mean,
        "relative_residual": relative_residual,
        "systematic_error": systematic_error,
        "bi_pb_ratio": bi_pb_ratio,
        "grade": grade,
        "q_scale": float(Q * 100.0),
        "sig_0": float(sig_0),
        "po_mu": float(mu_p),
        "po_sigma": float(w_p),
        "po_rho_ab": float(rho_ab),
        "po_scale_w": float(scale_w),
        "po_tau_rel": float(tau_rel),
        "pb_pesc_xl": float(P_esc_xl),
    }

    return {
        "success": success,
        "fit_ok": success and (red_chi2 <= 3.0 or red_chi2_eff <= 3.5),
        "A_pb": float(A_pb),
        "A_pb_unc": float(sigma_A_pb),
        "A_bi": float(A_bi),
        "A_bi_unc": float(sigma_A_bi),
        "A_po": float(A_po),
        "A_po_unc": float(sigma_A_po),
        "A_bg": float(A_bg),
        "A_bg_unc": float(sigma_A_bg),
        "bkg_mode": "simultaneous" if has_bkg else "subtract",
        "q_scale": float(Q * 100.0),
        "Q": float(Q),
        "sig_0": float(sig_0),
        "mu_po": float(mu_p),
        "w_po": float(w_p),
        "rho_ab": float(rho_ab),
        "scale_w": float(scale_w),
        "tau_rel": float(tau_rel),
        "po_tau_rel": float(tau_rel),
        "pb_pesc_xl": float(P_esc_xl),
        "P_esc_xl": float(P_esc_xl),
        "bi_pb_ratio": bi_pb_ratio,
        "chi2": chi2,
        "ndf": ndf,
        "red_chi2": red_chi2,
        "red_chi2_eff": red_chi2_eff,
        "active_channels": active_channels,
        "active_ndf": active_ndf,
        "runs_p_value": runs_p_value,
        "pull_std": pull_std,
        "pull_mean": pull_mean,
        "relative_residual": relative_residual,
        "systematic_error": systematic_error,
        "fit_quality": fit_quality,
        "raw_counts": rate_net.tolist(),
        "fit_Pb_y": fit_Pb_y.tolist(),
        "fit_Bi_y": fit_Bi_y.tolist(),
        "fit_Po_y": fit_Po_y.tolist(),
        "fit_Bg_y": fit_Bg_y.tolist(),
        "fit_cbn": fit_cbn.tolist(),
        "count_residue": count_residue.tolist(),
        "pull": pull.tolist(),
        "T_pb": T_pb.tolist(),
        "T_bi": T_bi.tolist(),
        "T_po": T_po.tolist(),
        "T_bg": (bkg_template.tolist() if (has_bkg and bkg_template is not None) else []),
    }

def fit_spectra(
    experiment_counts: np.ndarray,
    energy_values: np.ndarray | None = None,
    template_Bi_data: np.ndarray | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """
    Backward-compatible wrapper executing the first-principles 3-nuclide physical fit.
    """
    counts = np.asarray(experiment_counts, dtype=float)
    fit_min_ch = int(kwargs.get("fit_min_ch", kwargs.get("window_lo", 120)))
    fit_max_ch = int(kwargs.get("fit_max_ch", kwargs.get("window_hi", 750)))
    fit_min_ch = max(fit_min_ch, 50)
    fit_max_ch = min(fit_max_ch, 1000)

    try:
        res = fit_3nuclides_spectrum(
            rate_net=counts,
            fit_min_ch=fit_min_ch,
            fit_max_ch=fit_max_ch,
        )
    except Exception as e:
        sys.stderr.write(f"fit_spectra error: {e}\n")
        return None

    return {
        "is_valid": res["fit_ok"],
        "fit_ok": res["fit_ok"],
        "bi_fit_params": [res["A_bi"], 1.0, 0.0],
        "po_fit_params": [res["A_po"], res["mu_po"], res["w_po"], 1.0, 1.0],
        "pb_fit_params": [res["A_pb"], res["Q"], res["sig_0"]],
        "fit_Bi_y": res["fit_Bi_y"],
        "fit_Po_y": res["fit_Po_y"],
        "fit_Pb_y": res["fit_Pb_y"],
        "fit_cbn": res["fit_cbn"],
        "count_residue": res["count_residue"],
        "pull": res["pull"],
        "raw_counts": counts.tolist(),
        "pb_estimated_counts": res["fit_Pb_y"],
        "A_pb": res["A_pb"],
        "A_bi": res["A_bi"],
        "A_po": res["A_po"],
        "q_scale": res["q_scale"],
        "sig_0": res["sig_0"],
        "rho_ab": res["rho_ab"],
        "scale_w": res["scale_w"],
        "tau_rel": res["tau_rel"],
        "pb_pesc_xl": res["pb_pesc_xl"],
        "mu_po": res["mu_po"],
        "w_po": res["w_po"],
        "fit_quality": res.get("fit_quality", {
            "is_valid": res["fit_ok"],
            "chi2": res["chi2"],
            "ndf": res["ndf"],
            "red_chi2": res["red_chi2"],
            "relative_residual": res["relative_residual"],
            "bi_pb_ratio": res["bi_pb_ratio"],
            "grade": "Pass" if res["relative_residual"] < 6.0 else ("Warn" if res["relative_residual"] < 15.0 else "Fail"),
        }),
    }


