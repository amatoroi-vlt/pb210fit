"""
Radiochemistry domain module for Pb-210 activity and concentration analysis.

Responsibilities:
  - Chemical yield (Lead recovery fraction Y) calculation from gravimetric precipitate mass
  - Detection efficiency (epsilon, CPS/Bq) calculation from Standard measurement
  - Carrier Pb-210 background correction
  - Radioactive decay correction factor exp(+lambda * Delta_t)
  - True Activity (Bq) and Specific Activity / Concentration (Bq/kg) derivation
  - Rigorous GUM (Guide to the Expression of Uncertainty in Measurement) error propagation
  
Applicability:
  - Used symmetrically by BOTH:
      (1) Traditional 2-Window / Tailing Factor method (pb_signal_cps = A_net - B_net * TF)
      (2) Physical First-Principles Fitting method (pb_signal_cps = A_pb)
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Physical constants for Pb-210
PB210_HALF_LIFE_YEARS = 22.20
DAYS_PER_YEAR = 365.25
PB210_HALF_LIFE_DAYS = PB210_HALF_LIFE_YEARS * DAYS_PER_YEAR  # 8108.55 days
PB210_LAMBDA_DAYS = np.log(2.0) / PB210_HALF_LIFE_DAYS         # ~8.54835e-5 day^-1

# Molecular stoichiometry: Pb (207.2) / PbSO4 (295.2)
PB_TO_PBSO4_RATIO = 207.2 / 295.2  # ~0.701897


@dataclass(frozen=True)
class ChemicalRecoveryResult:
    precipitate_mass_g: float | None
    precipitate_mass_unc_g: float
    yield_frac: float | None
    yield_unc: float | None


@dataclass(frozen=True)
class DetectorEfficiencyResult:
    eff: float
    eff_unc: float
    source_activity_bq: float
    source_activity_unc: float
    corrected_activity: float
    corrected_unc: float
    std_yield_frac: float | None
    yield_corrected_activity: float
    yield_corrected_unc: float
    pb_net_cps: float
    pb_net_unc: float


@dataclass(frozen=True)
class ConcentrationResult:
    pb_signal_cps: float
    pb_signal_unc: float
    carrier_correction_cps: float | None
    carrier_correction_bq: float | None
    pb_net_corrected_cps: float
    pb_net_corrected_unc: float
    activity_raw: float
    activity_raw_unc: float
    yield_frac: float | None
    yield_frac_unc: float | None
    activity_bq: float
    activity_unc: float
    conc_meas_bq_kg: float
    conc_meas_unc: float
    decay_days_sampling: float | None
    conc_sampling_bq_kg: float | None
    conc_sampling_unc: float | None
    decay_days_ref: float | None
    conc_ref_bq_kg: float | None
    conc_ref_unc: float | None


def calculate_chemical_recovery(
    filter_before_g: float | str | None,
    filter_after_g: float | str | None,
    carrier_added_g: float | str | None,
    carrier_conc: float = 0.01,
    filter_before_unc_g: float | str = 0.0001,
    filter_after_unc_g: float | str = 0.0001,
    carrier_added_unc_g: float | str = 0.0001,
    carrier_conc_unc: float = 0.0,
) -> ChemicalRecoveryResult:
    """
    Calculate lead chemical recovery fraction Y from precipitate mass (PbSO4) and carrier lead added.
    """
    try:
        fb = float(filter_before_g) if filter_before_g is not None else float("nan")
        fa = float(filter_after_g) if filter_after_g is not None else float("nan")
        ca = float(carrier_added_g) if carrier_added_g is not None else float("nan")
    except (ValueError, TypeError):
        return ChemicalRecoveryResult(None, 0.0, None, None)

    if not (np.isfinite(fb) and np.isfinite(fa) and np.isfinite(ca) and ca > 0 and carrier_conc > 0):
        return ChemicalRecoveryResult(None, 0.0, None, None)

    ppt_mass = fa - fb
    if ppt_mass <= 0:
        return ChemicalRecoveryResult(ppt_mass, 0.0, None, None)

    s_fb = float(filter_before_unc_g or 0.0)
    s_fa = float(filter_after_unc_g or 0.0)
    s_ca = float(carrier_added_unc_g or 0.0)
    s_cc = float(carrier_conc_unc or 0.0)

    s_ppt = np.sqrt(s_fb ** 2 + s_fa ** 2)

    # Lead mass in precipitate = ppt_mass * (207.2 / 295.2)
    # Carrier lead added = ca * carrier_conc
    y = (ppt_mass * PB_TO_PBSO4_RATIO) / (ca * carrier_conc)

    rel_ppt = s_ppt / ppt_mass if ppt_mass > 0 else 0.0
    rel_ca = s_ca / ca if ca > 0 else 0.0
    rel_cc = s_cc / carrier_conc if carrier_conc > 0 else 0.0
    y_unc = y * np.sqrt(rel_ppt ** 2 + rel_ca ** 2 + rel_cc ** 2)

    return ChemicalRecoveryResult(
        precipitate_mass_g=ppt_mass,
        precipitate_mass_unc_g=s_ppt,
        yield_frac=float(y),
        yield_unc=float(y_unc),
    )


def calculate_decay_factor(elapsed_days: float) -> float:
    """
    Calculate radioactive decay correction factor exp(+lambda * elapsed_days).
    Multiplies measured activity to obtain activity at an earlier date.
    """
    return float(np.exp(PB210_LAMBDA_DAYS * float(elapsed_days)))


def calculate_sample_concentration(
    pb_signal_cps: float,
    pb_signal_unc: float,
    efficiency: float,
    eff_unc: float,
    sample_mass_kg: float | None = None,
    sample_mass_unc_kg: float = 0.0,
    yield_frac: float | None = None,
    yield_frac_unc: float | None = None,
    precipitate_mass_g: float | None = None,
    carrier_correction_type: str = "none",  # "none", "lsc_cps", "gamma_bq", "na_bq"
    carrier_pb210_cps_g: float = 0.0,
    carrier_pb210_bq_g: float = 0.0,
    decay_days_sampling: float | None = None,
    decay_days_ref: float | None = None,
) -> ConcentrationResult:
    """
    Derive True Activity (Bq) and Specific Concentration (Bq/kg) with uncertainty propagation.
    Compatible with both 2-Window pb_net_cps and Fitter A_pb.
    """
    cps = float(pb_signal_cps)
    s_cps = float(pb_signal_unc)
    eff = float(efficiency)
    s_eff = float(eff_unc)

    carrier_cps: float | None = None
    carrier_bq: float | None = None
    pb_net_corr = cps
    pb_net_corr_unc = s_cps

    # Carrier correction
    if carrier_correction_type == "lsc_cps" and precipitate_mass_g is not None and carrier_pb210_cps_g > 0:
        carrier_cps = precipitate_mass_g * carrier_pb210_cps_g
        pb_net_corr = cps - carrier_cps
    elif carrier_correction_type in ("gamma_bq", "na_bq") and precipitate_mass_g is not None and carrier_pb210_bq_g > 0:
        carrier_bq = precipitate_mass_g * carrier_pb210_bq_g

    # Activity raw (before chemical yield)
    if eff > 0:
        act_raw = pb_net_corr / eff
        rel_cps = pb_net_corr_unc / abs(pb_net_corr) if pb_net_corr != 0 else 0.0
        rel_eff = s_eff / eff if eff > 0 else 0.0
        act_raw_unc = abs(act_raw) * np.sqrt(rel_cps ** 2 + rel_eff ** 2)

        if carrier_bq is not None:
            act_raw = max(0.0, act_raw - carrier_bq)
    else:
        act_raw = 0.0
        act_raw_unc = 0.0

    # True Activity (yield-corrected)
    yf = yield_frac if (yield_frac is not None and yield_frac > 0) else 1.0
    yf_unc = yield_frac_unc if (yield_frac_unc is not None and yield_frac_unc > 0) else 0.0

    act_true = act_raw / yf
    rel_act_raw = act_raw_unc / abs(act_raw) if act_raw != 0 else 0.0
    rel_yf = yf_unc / yf if yf > 0 else 0.0
    act_true_unc = abs(act_true) * np.sqrt(rel_act_raw ** 2 + rel_yf ** 2)

    # Concentration @ measurement date
    m_kg = float(sample_mass_kg) if (sample_mass_kg is not None and sample_mass_kg > 0) else None
    s_m_kg = float(sample_mass_unc_kg or 0.0)

    if m_kg is not None and m_kg > 0:
        conc_meas = act_true / m_kg
        rel_act = act_true_unc / abs(act_true) if act_true != 0 else 0.0
        rel_m = s_m_kg / m_kg if m_kg > 0 else 0.0
        conc_meas_unc = abs(conc_meas) * np.sqrt(rel_act ** 2 + rel_m ** 2)
    else:
        conc_meas = 0.0
        conc_meas_unc = 0.0

    # Decay corrections
    conc_samp = None
    conc_samp_unc = None
    if decay_days_sampling is not None and conc_meas > 0:
        f_samp = calculate_decay_factor(decay_days_sampling)
        conc_samp = conc_meas * f_samp
        conc_samp_unc = conc_meas_unc * f_samp

    conc_ref = None
    conc_ref_unc = None
    if decay_days_ref is not None and conc_meas > 0:
        f_ref = calculate_decay_factor(decay_days_ref)
        conc_ref = conc_meas * f_ref
        conc_ref_unc = conc_meas_unc * f_ref

    return ConcentrationResult(
        pb_signal_cps=cps,
        pb_signal_unc=s_cps,
        carrier_correction_cps=carrier_cps,
        carrier_correction_bq=carrier_bq,
        pb_net_corrected_cps=pb_net_corr,
        pb_net_corrected_unc=pb_net_corr_unc,
        activity_raw=act_raw,
        activity_raw_unc=act_raw_unc,
        yield_frac=yield_frac,
        yield_frac_unc=yield_frac_unc,
        activity_bq=act_true,
        activity_unc=act_true_unc,
        conc_meas_bq_kg=conc_meas,
        conc_meas_unc=conc_meas_unc,
        decay_days_sampling=decay_days_sampling,
        decay_days_ref=decay_days_ref,
        conc_sampling_bq_kg=conc_samp,
        conc_sampling_unc=conc_samp_unc,
        conc_ref_bq_kg=conc_ref,
        conc_ref_unc=conc_ref_unc,
    )


def calculate_radiochemistry_batch(payload: dict) -> dict:
    """
    Execute complete batch radiochemical concentration calculations from application payload.
    Supports both 'fitting' and 'two_window' modes.
    Returns:
      {
        "ok": bool,
        "efficiency": dict (EffAutoResult or EffManualResult),
        "concentrations": dict[int, SampleConcentrationDetail],
      }
    """
    settings = payload.get("settings", {})
    mode = payload.get("mode", "fitting")  # "fitting" or "two_window"
    
    conc_src = float(settings.get("source_conc_bq_g") or 0.0)
    conc_src_unc = float(settings.get("source_conc_bq_g_unc") or 0.0)
    source_date_str = settings.get("source_date")
    ref_date_str = settings.get("ref_date")
    corr_type = settings.get("carrier_pb210_correction", "none")
    carrier_conc = float(settings.get("carrier_conc") or 0.01)
    carrier_conc_unc = float(settings.get("carrier_conc_unc") or 0.0)
    carrier_pb210_cps_g = float(settings.get("carrier_pb210_cps_g") or 0.0)
    carrier_pb210_bq_g = float(settings.get("carrier_pb210_bq_g") or 0.0)
    carrier_pb210_bq_g_unc = float(settings.get("carrier_pb210_bq_g_unc") or 0.0)

    # 1. Determine Efficiency (Auto from Std or Manual)
    eff_manual = float(settings.get("eff_manual") or 0.0)
    eff_manual_unc = float(settings.get("eff_manual_unc") or 0.0)
    
    std_info = payload.get("standard")
    eff_result = None
    efficiency_val = 0.0
    eff_unc_val = 0.0

    if std_info and conc_src > 0:
        added_g = float(std_info.get("source_added_g") or 0.0)
        added_unc_g = float(std_info.get("source_added_g_unc") or 0.0)
        std_time = float(std_info.get("time") or 1.0)
        std_end_unix = float(std_info.get("end_time_unix") or 0.0)
        std_meas_ms = std_end_unix * 1000.0 if std_end_unix > 0 else datetime.now().timestamp() * 1000.0

        # Source activity
        source_act = conc_src * added_g
        rel_src = (conc_src_unc / conc_src) if conc_src > 0 else 0.0
        rel_add = (added_unc_g / added_g) if added_g > 0 else 0.0
        source_act_unc = source_act * np.sqrt(rel_src**2 + rel_add**2)

        # Decay correction from source calibration date to measurement date
        decay_days = 0.0
        decay_factor = 1.0
        if source_date_str:
            try:
                src_ms = datetime.fromisoformat(source_date_str.replace("Z", "+00:00")).timestamp() * 1000.0
                decay_days = max(0.0, (std_meas_ms - src_ms) / 86400000.0)
                decay_factor = float(np.exp(-PB210_LAMBDA_DAYS * decay_days))
            except Exception:
                pass
        
        corr_act = source_act * decay_factor
        corr_act_unc = source_act_unc * decay_factor

        # Standard chemical recovery
        std_rec = calculate_chemical_recovery(
            filter_before_g=std_info.get("filter_before_g"),
            filter_after_g=std_info.get("filter_after_g"),
            carrier_added_g=std_info.get("carrier_added_g"),
            carrier_conc=carrier_conc,
            carrier_conc_unc=carrier_conc_unc,
        )
        std_yf = std_rec.yield_frac if (std_rec.yield_frac is not None and std_rec.yield_frac > 0) else 1.0
        yield_corr_act = corr_act * std_yf
        yield_corr_act_unc = corr_act_unc * std_yf

        # Standard Pb signal (from fitting A_pb or 2-window pb_net)
        std_pb_cps = float(std_info.get("pb_signal_cps") or 0.0)
        std_pb_unc = float(std_info.get("pb_signal_unc") or 0.0)

        # Carrier correction on Standard
        carrier_corr_bq = None
        carrier_corr_act = None
        carrier_corr_cps = None
        carrier_corr_pb = None

        eff_pb_net = std_pb_cps
        eff_act = yield_corr_act
        eff_act_unc = yield_corr_act_unc

        ppt_mass_std = std_rec.precipitate_mass_g
        if corr_type in ("gamma_bq", "na_bq") and ppt_mass_std is not None and carrier_pb210_bq_g > 0:
            carrier_corr_bq = ppt_mass_std * carrier_pb210_bq_g
            carrier_corr_act = yield_corr_act + carrier_corr_bq
            eff_act = carrier_corr_act
        elif corr_type == "lsc_cps" and ppt_mass_std is not None and carrier_pb210_cps_g > 0:
            carrier_corr_cps = ppt_mass_std * carrier_pb210_cps_g
            carrier_corr_pb = std_pb_cps - carrier_corr_cps
            eff_pb_net = carrier_corr_pb

        if eff_pb_net > 0 and eff_act > 0:
            efficiency_val = eff_pb_net / eff_act
            rel_pb = (std_pb_unc / eff_pb_net) if eff_pb_net > 0 else 0.0
            rel_act = (eff_act_unc / eff_act) if eff_act > 0 else 0.0
            eff_unc_val = efficiency_val * np.sqrt(rel_pb**2 + rel_act**2)

            eff_result = {
                "method": "auto",
                "eff": efficiency_val,
                "unc": eff_unc_val,
                "source_conc_bq_g": conc_src,
                "source_conc_bq_g_unc": conc_src_unc,
                "source_added_g": added_g,
                "source_added_g_unc": added_unc_g,
                "source_activity_bq": source_act,
                "source_activity_unc": source_act_unc,
                "decay_days": decay_days,
                "decay_factor": decay_factor,
                "corrected_activity": corr_act,
                "corrected_unc": corr_act_unc,
                "std_yield_frac": std_rec.yield_frac,
                "yield_corrected_activity": yield_corr_act,
                "yield_corrected_unc": yield_corr_act_unc,
                "ppt_mass_std": ppt_mass_std,
                "carrier_correction_bq": carrier_corr_bq,
                "carrier_corrected_activity": carrier_corr_act,
                "carrier_correction_cps": carrier_corr_cps,
                "carrier_corrected_pb_net": carrier_corr_pb,
                "pb_net": std_pb_cps,
                "pb_net_unc": std_pb_unc,
            }

    if eff_result is None and eff_manual > 0:
        efficiency_val = eff_manual
        eff_unc_val = eff_manual_unc
        eff_result = {
            "method": "manual",
            "eff": efficiency_val,
            "unc": eff_unc_val,
        }

    # 2. Calculate Concentrations for each Sample
    samples_list = payload.get("samples", [])
    conc_map = {}

    for s in samples_list:
        s_idx = int(s.get("sample_index"))
        d_name = str(s.get("display_name") or s.get("well_name") or f"Sample {s_idx}")
        w_name = str(s.get("well_name") or "")
        time_s = float(s.get("time") or 43200.0)
        end_unix = float(s.get("end_time_unix") or 0.0)
        meas_ms = end_unix * 1000.0 if end_unix > 0 else datetime.now().timestamp() * 1000.0

        pb_signal_cps = float(s.get("pb_signal_cps") or 0.0)
        pb_signal_unc = float(s.get("pb_signal_unc") or 0.0)

        # Chemical recovery
        rec = calculate_chemical_recovery(
            filter_before_g=s.get("filter_before_g"),
            filter_after_g=s.get("filter_after_g"),
            carrier_added_g=s.get("carrier_added_g"),
            carrier_conc=carrier_conc,
            carrier_conc_unc=carrier_conc_unc,
        )

        # Decay days to sampling date
        decay_days_samp = 0.0
        samp_date = str(s.get("sampling_date") or "")
        if samp_date:
            try:
                samp_ms = datetime.fromisoformat(samp_date.replace("Z", "+00:00")).timestamp() * 1000.0
                decay_days_samp = max(0.0, (meas_ms - samp_ms) / 86400000.0)
            except Exception:
                decay_days_samp = 0.0

        # Decay days to reference date
        decay_days_ref = 0.0
        if ref_date_str:
            try:
                ref_ms = datetime.fromisoformat(ref_date_str.replace("Z", "+00:00")).timestamp() * 1000.0
                decay_days_ref = max(0.0, (meas_ms - ref_ms) / 86400000.0)
            except Exception:
                decay_days_ref = 0.0

        mass_g = float(s.get("mass") or 0.0)
        mass_unc_g = float(s.get("mass_unc") or 0.0)
        mass_kg = (mass_g / 1000.0) if mass_g > 0 else None
        mass_unc_kg = (mass_unc_g / 1000.0) if mass_unc_g > 0 else 0.0

        detail_res = calculate_sample_concentration(
            pb_signal_cps=pb_signal_cps,
            pb_signal_unc=pb_signal_unc,
            efficiency=efficiency_val,
            eff_unc=eff_unc_val,
            sample_mass_kg=mass_kg,
            sample_mass_unc_kg=mass_unc_kg,
            yield_frac=rec.yield_frac,
            yield_frac_unc=rec.yield_unc,
            precipitate_mass_g=rec.precipitate_mass_g,
            carrier_correction_type=corr_type,
            carrier_pb210_cps_g=carrier_pb210_cps_g,
            carrier_pb210_bq_g=carrier_pb210_bq_g,
            decay_days_sampling=decay_days_samp if decay_days_samp > 0 else None,
            decay_days_ref=decay_days_ref if decay_days_ref > 0 else None,
        )

        conc_map[s_idx] = {
            "sample_index": s_idx,
            "display_name": d_name,
            "well_name": w_name,
            "time": time_s,
            "pb_signal_cps": detail_res.pb_signal_cps,
            "pb_signal_unc": detail_res.pb_signal_unc,
            "efficiency": efficiency_val,
            "eff_unc": eff_unc_val,
            "activity_raw": detail_res.activity_raw,
            "activity_raw_unc": detail_res.activity_raw_unc,
            "ppt_mass": rec.precipitate_mass_g,
            "ppt_mass_unc": rec.precipitate_mass_unc_g,
            "carrier_correction_cps": detail_res.carrier_correction_cps,
            "yield_frac": detail_res.yield_frac,
            "yield_frac_unc": detail_res.yield_frac_unc,
            "activity_bq": detail_res.activity_bq,
            "activity_unc": detail_res.activity_unc,
            "mass_g": mass_g,
            "mass_unc": mass_unc_g if mass_unc_g > 0 else None,
            "conc_bq_kg": detail_res.conc_meas_bq_kg,
            "conc_unc": detail_res.conc_meas_unc,
            "sampling_date": samp_date,
            "decay_days_sampling": decay_days_samp,
            "conc_sampling_bq_kg": detail_res.conc_sampling_bq_kg,
            "conc_sampling_unc": detail_res.conc_sampling_unc,
            "decay_days_ref": decay_days_ref,
            "conc_ref_bq_kg": detail_res.conc_ref_bq_kg,
            "conc_ref_unc": detail_res.conc_ref_unc,
        }

    return {
        "ok": True,
        "efficiency": eff_result,
        "concentrations": conc_map,
    }

