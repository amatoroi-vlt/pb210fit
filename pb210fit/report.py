from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from . import config


FLAG_DESCRIPTIONS = {
    "high_residual_rmse": ("Residual RMSE high", "Overall residual noise level is high."),
    "po_not_fitted": ("Po not fitted", "Po component was not fit as a free parameter."),
    "failed_gof_red_chi2": ("GOF red-chi2 fail", "Reduced chi-square is outside acceptance range."),
    "failed_gof_p_value": ("GOF p-value fail", "Goodness-of-fit p-value is below threshold."),
    "residual_nonrandom_runs": ("Runs test fail", "Residual sign sequence is non-random."),
    "residual_mean_large": ("Residual mean large", "Residual mean magnitude is too large."),
    "bi_params_at_boundary": ("Bi boundary reach", "Bi-210 fitting parameters reached their limits."),
    "po_params_at_boundary": ("Po boundary reach", "Po-210 fitting parameters reached their limits."),
    "invalid_po_errors": ("Invalid Po errors", "The uncertainty matrix for Po-210 could not be calculated."),
}

# Tier-1 flags (numerical/boundary issues) → red; Tier-2 (statistical) → orange
_TIER1_FLAGS = {"bi_params_at_boundary", "po_params_at_boundary", "invalid_po_errors", "po_not_fitted"}
_TIER2_FLAGS = {"high_residual_rmse", "failed_gof_red_chi2", "failed_gof_p_value", "residual_nonrandom_runs", "residual_mean_large"}


def flag_severity(flag):
    if flag in _TIER1_FLAGS:
        return "critical"
    if flag in _TIER2_FLAGS:
        return "warning"
    return "warning"


def grade_color(grade):
    if grade in ("review", "physical_review"):
        return "#f97316"
    if grade in ("warning", "numerical_fail"):
        return "#ef4444"
    if grade == "good":
        return "#16a34a"
    return "#64748b"


def write_diagnostic_reports(
    output_dir,
    run_id,
    sample,
    fit_result,
    channels,
    window_suggestion,
    standard_prefit,
    context,
    file_prefix,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectrum_file_name = f"{file_prefix}_spectrum.html"
    residual_file_name = f"{file_prefix}_residual.html"
    qc_report_file_name = f"{file_prefix}_qc_report.html"
    combined_report_file_name = f"{file_prefix}_combined.html"

    spectrum_path = output_dir / spectrum_file_name
    residual_path = output_dir / residual_file_name
    qc_report_path = output_dir / qc_report_file_name
    combined_report_path = output_dir / combined_report_file_name

    write_spectrum_html(spectrum_path, run_id, sample, fit_result, channels, window_suggestion, context)
    write_residual_html(residual_path, run_id, sample, fit_result, channels, window_suggestion, context)
    write_qc_report_html(qc_report_path, run_id, sample, fit_result, window_suggestion, standard_prefit, context)
    write_combined_report_html(combined_report_path, run_id, sample, fit_result, channels, window_suggestion, standard_prefit, context)

    return {
        "spectrum_file_name": spectrum_file_name,
        "residual_file_name": residual_file_name,
        "qc_report_file_name": qc_report_file_name,
        "combined_report_file_name": combined_report_file_name,
        "spectrum_path": str(spectrum_path),
        "residual_path": str(residual_path),
        "qc_report_path": str(qc_report_path),
        "combined_report_path": str(combined_report_path),
    }


def _build_spectrum_figure(run_id, sample, fit_result, channels, window_suggestion):
    pb = np.asarray(fit_result.get("fit_Pb_y", []), dtype=float)
    bi = np.asarray(fit_result.get("fit_Bi_y", []), dtype=float)
    po = np.asarray(fit_result.get("fit_Po_y", []), dtype=float)
    bg = np.asarray(fit_result.get("fit_Bg_y", []), dtype=float)
    combined = np.asarray(fit_result.get("fit_cbn", []), dtype=float)
    raw = np.asarray(fit_result.get("raw_counts", []), dtype=float)
    quality = fit_result.get("fit_quality", {})
    bkg_mode = fit_result.get("bkg_mode", "subtract")

    fig = go.Figure()

    if bkg_mode == "simultaneous":
        net_raw = raw - bg if (len(raw) == len(channels) and len(bg) == len(channels)) else raw
        model_3nuclides = combined - bg if (len(combined) == len(channels) and len(bg) == len(channels)) else combined

        # --- 윗줄 (기본 표시 5개) ---
        # 1. 실측 Net 스펙트럼
        fig.add_trace(go.Scatter(x=channels, y=net_raw, mode="lines", name="실측 Net 스펙트럼", line=dict(color="#0f172a", width=2.0)))
        # 2. 3핵종 합성 모델
        fig.add_trace(go.Scatter(x=channels, y=model_3nuclides, mode="lines", name="3핵종 합성 모델", line=dict(color="#475569", width=1.6, dash="dot")))
        # 3. Pb-210 물리 모델
        if len(pb) == len(channels):
            fig.add_trace(go.Scatter(x=channels, y=pb, mode="lines", name="Pb-210 물리 모델", line=dict(color="#8b5cf6", width=1.8)))
        # 4. Bi-210 물리 모델
        if len(bi) == len(channels):
            fig.add_trace(go.Scatter(x=channels, y=bi, mode="lines", name="Bi-210 물리 모델", line=dict(color="#3b82f6", width=1.8)))
        # 5. Po-210 물리 모델
        if len(po) == len(channels):
            fig.add_trace(go.Scatter(x=channels, y=po, mode="lines", name="Po-210 물리 모델", line=dict(color="#ef4444", width=1.8)))

        # --- 아랫줄 (기본 미표시 3개) ---
        # 6. 실측 Gross 스펙트럼
        fig.add_trace(go.Scatter(x=channels, y=raw, mode="lines", name="실측 Gross 스펙트럼", visible="legendonly", line=dict(color="#0f172a", width=2.0)))
        # 7. Total 물리 합성 모델
        fig.add_trace(go.Scatter(x=channels, y=combined, mode="lines", name="Total 물리 합성 모델", visible="legendonly", line=dict(color="#475569", width=1.6, dash="dot")))
        # 8. Blank 배경 모델
        if len(bg) == len(channels):
            fig.add_trace(go.Scatter(x=channels, y=bg, mode="lines", name="Blank 배경 모델", visible="legendonly", line=dict(color="#64748b", dash="dash", width=1.8)))
    else:
        # Mode A
        fig.add_trace(go.Scatter(x=channels, y=raw, mode="lines", name="실측 Net 스펙트럼", line=dict(color="#0f172a", width=2.0)))
        if len(combined) == len(channels):
            fig.add_trace(go.Scatter(x=channels, y=combined, mode="lines", name="3핵종 합성 모델", line=dict(color="#475569", dash="dot", width=1.6)))
        if len(pb) == len(channels):
            fig.add_trace(go.Scatter(x=channels, y=pb, mode="lines", name="Pb-210 물리 모델", line=dict(color="#8b5cf6", width=1.8)))
        if len(bi) == len(channels):
            fig.add_trace(go.Scatter(x=channels, y=bi, mode="lines", name="Bi-210 물리 모델", line=dict(color="#3b82f6", width=1.8)))
        if len(po) == len(channels):
            fig.add_trace(go.Scatter(x=channels, y=po, mode="lines", name="Po-210 물리 모델", line=dict(color="#ef4444", width=1.8)))

    add_window_shapes(fig, window_suggestion)

    fig.update_layout(
        title=(
            f"Pb-210 Spectrum Fit — {escape(sample['display_name'])}"
            f"<br><sup>grade={escape(str(quality.get('grade', 'n/a')))}</sup>"
        ),
        template="plotly_white",
        xaxis={"title": "Channel"},
        yaxis={"title": "Counts"},
        legend=dict(orientation="h", x=0, y=-0.28, font=dict(size=11), entrywidth=0.19, entrywidthmode="fraction"),
        margin={"l": 70, "r": 40, "t": 80, "b": 80},
    )
    return fig


def _build_residual_figure(run_id, sample, fit_result, channels, window_suggestion):
    residue = np.asarray(fit_result["count_residue"], dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=channels, y=residue, mode="lines", name="Weighted residual"))
    fig.add_hline(y=0, line_dash="solid", line_color="#475569")
    fig.add_hline(y=2, line_dash="dot", line_color="#ef4444")
    fig.add_hline(y=-2, line_dash="dot", line_color="#ef4444")
    add_window_shapes(fig, window_suggestion)
    fig.update_layout(
        title=f"Pb-210 Weighted Residual — {escape(sample['display_name'])}",
        template="plotly_white",
        xaxis={"title": "Channel"},
        yaxis={"title": "Weighted residual"},
        margin={"l": 70, "r": 40, "t": 70, "b": 60},
    )
    return fig


def write_spectrum_html(output_path, run_id, sample, fit_result, channels, window_suggestion, context):
    fig = _build_spectrum_figure(run_id, sample, fit_result, channels, window_suggestion)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def write_residual_html(output_path, run_id, sample, fit_result, channels, window_suggestion, context):
    fig = _build_residual_figure(run_id, sample, fit_result, channels, window_suggestion)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def write_qc_report_html(output_path, run_id, sample, fit_result, window_suggestion, standard_prefit, context):
    quality = fit_result.get("fit_quality", {})
    attempts = fit_result.get("fit_attempts", {})
    standard_quality = standard_prefit.get("fit_quality", {})
    flags = quality.get("flags") or []

    rows = [
        ("Grade", quality.get("grade")),
        ("Tier 1", quality.get("tier1_status")),
        ("Tier 2", quality.get("tier2_status")),
        ("Tier 3", quality.get("tier3_status")),
        ("Component mode", quality.get("component_mode")),
        ("Validation window", quality.get("fit_validation_window")),
        ("Residual RMSE", quality.get("residue_rmse")),
        ("Reduced chi-square (nominal)", quality.get("red_chi2") or quality.get("red_chi2_global")),
        ("Reduced chi-square (effective)", quality.get("red_chi2_eff")),
        ("Active channels", quality.get("active_channels")),
        ("Runs p-value", quality.get("runs_p_value")),
        ("Pull Std / Mean", f"σ={quality.get('pull_std', 0):.2f}, μ={quality.get('pull_mean', 0):.2f}" if quality.get('pull_std') is not None else None),
        ("Systematic error (%)", quality.get("systematic_error")),
        ("Bi stretch", quality.get("bi_stretch")),
        ("Bi shift", quality.get("bi_shift")),
        ("Po mu", quality.get("po_mu")),
        ("Po sigma", quality.get("po_sigma")),
        ("Po tailing (tau_rel)", quality.get("po_tau_rel")),
    ]
    attempt_rows = [
        ("Bi attempts", attempts.get("bi_attempt_count")),
        ("Po attempts", attempts.get("po_attempt_count")),
        ("Refine attempts", attempts.get("refine_attempt_count")),
        ("Selected score", attempts.get("selected_score")),
        ("Selected penalty", attempts.get("selected_penalty")),
    ]

    bkg_section_html = ""
    if fit_result.get("bkg_mode") == "simultaneous":
        a_bg = fit_result.get("A_bg", 0.0)
        a_bg_unc = fit_result.get("A_bg_unc", 0.0)
        a_pb = fit_result.get("A_pb", 0.0)
        a_bi = fit_result.get("A_bi", 0.0)
        a_po = fit_result.get("A_po", 0.0)
        tot_a = a_pb + a_bi + a_po + a_bg
        bg_pct = (a_bg / tot_a * 100.0) if tot_a > 0 else 0.0
        rel_unc = (a_bg_unc / a_bg * 100.0) if a_bg > 0 else 0.0
        bkg_section_html = (
            "<h2>Blank Background Optimization (Mode B)</h2>"
            + html_table(
                ["Metric", "Value"],
                [
                    ("Blank Scale Factor (A_bg)", f"{a_bg:.5f} ± {a_bg_unc:.5f} CPS"),
                    ("Background Share in Total Model", f"{bg_pct:.2f} %"),
                    ("Optimization Quality", f"Good (rel unc: {rel_unc:.1f}%)" if rel_unc < 30 else f"High Uncertainty ({rel_unc:.1f}%)"),
                    ("Blank Spectrum Alignment", "QPE Warping Applied"),
                ],
            )
        )

    std_display = context.get("standard_display_name")
    has_std = bool(std_display and std_display not in ("None", "미지정 (N/A)"))

    output_path = Path(output_path)
    output_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'>",
                "<style>",
                "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#172033;background:#f8fafc}",
                "main{max-width:1120px;margin:auto;background:#fff;border:1px solid #dbe3ef;padding:24px}",
                "h1{font-size:24px;margin:0 0 8px} h2{font-size:17px;margin:28px 0 10px}",
                "table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #dbe3ef;padding:8px;text-align:left}th{background:#eef4fb}",
                ".flag-critical{border-left:4px solid #ef4444;background:#fff7f7;padding:10px;margin:8px 0}",
                ".flag-warning{border-left:4px solid #f97316;background:#fff8f1;padding:10px;margin:8px 0}",
                ".ok{border-left:4px solid #16a34a;background:#f0fdf4;padding:10px;margin:8px 0}.muted{color:#667085}",
                "</style></head><body><main>",
                f"<h1>Pb-210 QC Report — {escape(sample['display_name'])}</h1>",
                f"<p class='muted'>run={escape(run_id)} · target policy={escape(context['target_role_policy'])} · qpe policy={escape(context['qpe_policy'])}</p>",
                "<h2>Run Selection</h2>",
                html_table(
                    ["Item", "Value"],
                    [
                        ("Input rows", selection_text(context.get("input_rows"))),
                        ("Result targets", selection_text(context.get("target_rows"))),
                        ("Standard anchor", context.get("standard_display_name")),
                        ("Blank subtraction", context.get("blank_display_name")),
                    ],
                ),
                "<h2>Quality Metrics</h2>",
                html_table(["Metric", "Value"], rows),
                *( [bkg_section_html] if bkg_section_html else [] ),
                "<h2>Fit Attempts</h2>",
                html_table(["Attempt", "Value"], attempt_rows),
                "<h2>Standard Anchor</h2>",
                html_table(
                    ["Metric", "Value"],
                    [
                        ("Std Po mu", standard_quality.get("po_mu") if has_std else "N/A (Std 미지정)"),
                        ("Std Po sigma", standard_quality.get("po_sigma") if has_std else "N/A (Std 미지정)"),
                        ("Std red-chi2", standard_quality.get("red_chi2_global") if has_std else "N/A (Std 미지정)"),
                        ("Std GOF p-value", standard_quality.get("gof_p_value") if has_std else "N/A (Std 미지정)"),
                    ],
                ),
                "<h2>Window / Fit Range</h2>",
                html_table(
                    ["Window", "Channel range", "Energy range"],
                    [
                        ("A", range_text(window_suggestion, "A_ch"), range_text(window_suggestion, "A_e")),
                        ("B", range_text(window_suggestion, "B_ch"), range_text(window_suggestion, "B_e")),
                        ("C", range_text(window_suggestion, "C_ch"), range_text(window_suggestion, "C_e")),
                        ("D", range_text(window_suggestion, "D_ch"), range_text(window_suggestion, "D_e")),
                    ],
                ),
                "<h2>Active Quality Flags</h2>",
                flags_html(flags, quality),
                "</main></body></html>",
            ]
        ),
        encoding="utf-8",
    )


def write_combined_report_html(output_path, run_id, sample, fit_result, channels, window_suggestion, standard_prefit, context):
    quality = fit_result.get("fit_quality", {})
    attempts = fit_result.get("fit_attempts", {})
    standard_quality = standard_prefit.get("fit_quality", {})
    flags = quality.get("flags") or []
    grade = quality.get("grade", "n/a")
    gcolor = grade_color(grade)

    spectrum_fig = _build_spectrum_figure(run_id, sample, fit_result, channels, window_suggestion)
    residual_fig = _build_residual_figure(run_id, sample, fit_result, channels, window_suggestion)

    spectrum_div = spectrum_fig.to_html(full_html=False, include_plotlyjs="cdn")
    residual_div = residual_fig.to_html(full_html=False, include_plotlyjs=False)

    quality_rows = [
        ("Grade", quality.get("grade")),
        ("Tier 1", quality.get("tier1_status")),
        ("Tier 2", quality.get("tier2_status")),
        ("Tier 3", quality.get("tier3_status")),
        ("Component mode", quality.get("component_mode")),
        ("Validation window", quality.get("fit_validation_window")),
        ("Residual RMSE", quality.get("residue_rmse")),
        ("Reduced chi-square", quality.get("red_chi2_global")),
        ("GOF p-value", quality.get("gof_p_value")),
        ("Runs p-value", quality.get("runs_p_value")),
        ("Bi stretch", quality.get("bi_stretch")),
        ("Bi shift", quality.get("bi_shift")),
        ("Po mu", quality.get("po_mu")),
        ("Po sigma", quality.get("po_sigma")),
        ("Po tailing (tau_rel)", quality.get("po_tau_rel")),
    ]
    attempt_rows = [
        ("Bi attempts", attempts.get("bi_attempt_count")),
        ("Po attempts", attempts.get("po_attempt_count")),
        ("Refine attempts", attempts.get("refine_attempt_count")),
    ]

    bkg_section_html = ""
    if fit_result.get("bkg_mode") == "simultaneous":
        a_bg = fit_result.get("A_bg", 0.0)
        a_bg_unc = fit_result.get("A_bg_unc", 0.0)
        a_pb = fit_result.get("A_pb", 0.0)
        a_bi = fit_result.get("A_bi", 0.0)
        a_po = fit_result.get("A_po", 0.0)
        tot_a = a_pb + a_bi + a_po + a_bg
        bg_pct = (a_bg / tot_a * 100.0) if tot_a > 0 else 0.0
        rel_unc = (a_bg_unc / a_bg * 100.0) if a_bg > 0 else 0.0
        bkg_section_html = (
            "<h2>Blank Background Optimization (Mode B)</h2>"
            + html_table(
                ["Metric", "Value"],
                [
                    ("Blank Scale Factor (A_bg)", f"{a_bg:.5f} ± {a_bg_unc:.5f} CPS"),
                    ("Background Share in Total Model", f"{bg_pct:.2f} %"),
                    ("Optimization Quality", f"Good (rel unc: {rel_unc:.1f}%)" if rel_unc < 30 else f"High Uncertainty ({rel_unc:.1f}%)"),
                    ("Blank Spectrum Alignment", "QPE Warping Applied"),
                ],
            )
        )

    std_display = context.get("standard_display_name")
    has_std = bool(std_display and std_display not in ("None", "미지정 (N/A)"))

    output_path = Path(output_path)
    output_path.write_text(
        "\n".join([
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;color:#172033;background:#f8fafc}",
            "main{max-width:1200px;margin:auto;padding:16px}",
            "h2{font-size:15px;margin:20px 0 8px;color:#334155}",
            ".grade-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:13px;font-weight:700;"
            f"background:{gcolor}22;color:{gcolor};border:1px solid {gcolor}55;margin-bottom:10px}}",
            ".section{background:#fff;border:1px solid #dbe3ef;border-radius:8px;padding:16px;margin-bottom:16px}",
            "table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}",
            "td,th{border:1px solid #e2e8f0;padding:7px 10px;text-align:left}th{background:#f1f5f9;font-weight:600}",
            ".flag-critical{border-left:4px solid #ef4444;background:#fff1f2;padding:10px 12px;margin:6px 0;border-radius:0 6px 6px 0}",
            ".flag-warning{border-left:4px solid #f97316;background:#fff8f1;padding:10px 12px;margin:6px 0;border-radius:0 6px 6px 0}",
            ".ok{border-left:4px solid #16a34a;background:#f0fdf4;padding:10px 12px;margin:6px 0;border-radius:0 6px 6px 0}",
            ".muted{color:#64748b;font-size:12px}",
            ".two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}",
            "</style></head><body><main>",
            f"<span class='grade-badge'>Grade: {escape(str(grade))}</span>",
            f"<p class='muted'>std={escape(context.get('standard_display_name',''))} · blank={escape(context.get('blank_display_name',''))}</p>",
            "<div class='section'>",
            spectrum_div,
            "</div>",
            "<div class='section'>",
            residual_div,
            "</div>",
            "<div class='section'>",
            "<h2>Quality Metrics</h2>",
            "<div class='two-col'>",
            html_table(["Metric", "Value"], quality_rows[:7]),
            html_table(["Metric", "Value"], quality_rows[7:]),
            "</div>",
            *( [bkg_section_html] if bkg_section_html else [] ),
            "<h2>Standard Anchor</h2>",
            html_table(
                ["Metric", "Value"],
                [
                    ("Std Po mu", standard_quality.get("po_mu") if has_std else "N/A (Std 미지정)"),
                    ("Std Po sigma", standard_quality.get("po_sigma") if has_std else "N/A (Std 미지정)"),
                    ("Std red-chi2", standard_quality.get("red_chi2_global") if has_std else "N/A (Std 미지정)"),
                    ("Std GOF p-value", standard_quality.get("gof_p_value") if has_std else "N/A (Std 미지정)"),
                ],
            ),
            "<h2>Window / Fit Range</h2>",
            html_table(
                ["Window", "Channel range", "Energy range"],
                [
                    ("A", range_text(window_suggestion, "A_ch"), range_text(window_suggestion, "A_e")),
                    ("B", range_text(window_suggestion, "B_ch"), range_text(window_suggestion, "B_e")),
                    ("C", range_text(window_suggestion, "C_ch"), range_text(window_suggestion, "C_e")),
                    ("D", range_text(window_suggestion, "D_ch"), range_text(window_suggestion, "D_e")),
                ],
            ),
            "<h2>Fit Attempts</h2>",
            html_table(["Attempt", "Value"], attempt_rows),
            "<h2>Active Quality Flags</h2>",
            flags_html(flags, quality),
            "</div>",
            "</main></body></html>",
        ]),
        encoding="utf-8",
    )


def add_window_shapes(fig, suggestion):
    colors = {"A": "#fde68a", "B": "#bfdbfe", "C": "#fecaca", "D": "#bbf7d0"}
    for key, color in colors.items():
        bounds = suggestion.get(f"{key}_ch")
        if not bounds:
            continue
        fig.add_vrect(
            x0=bounds[0],
            x1=bounds[1],
            fillcolor=color,
            opacity=0.18,
            layer="below",
            line_width=0,
            annotation_text=key,
            annotation_position="top left",
        )


def html_table(headers, rows):
    head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(format_value(item))}</td>" for item in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def selection_text(summary):
    if not isinstance(summary, dict):
        return "-"
    role_counts = summary.get("role_counts") or {}
    if not role_counts:
        return f"{summary.get('total', 0)} rows"
    parts = [f"{role}={count}" for role, count in sorted(role_counts.items())]
    return f"{summary.get('total', 0)} rows ({', '.join(parts)})"


def flags_html(flags, quality):
    if not flags:
        return "<div class='ok'>No active flags.</div>"
    items = []
    for flag in flags:
        title, description = FLAG_DESCRIPTIONS.get(flag, (flag, "No description available."))
        severity = flag_severity(flag)
        css_class = "flag-critical" if severity == "critical" else "flag-warning"
        items.append(
            f"<div class='{css_class}'><b>{escape(flag)}</b> ({escape(title)})<br>"
            f"{escape(description)}<br><span class='muted'>{escape(flag_evidence(flag, quality))}</span></div>"
        )
    return "".join(items)


def flag_evidence(flag, quality):
    if flag == "high_residual_rmse":
        return f"criterion: residue_rmse <= {config.fit_qc_residue_rmse_max}; actual={fmt(quality.get('residue_rmse'))}"
    if flag == "failed_gof_red_chi2":
        return f"criterion: {config.fit_qc_red_chi2_min} <= red_chi2 <= {config.fit_qc_red_chi2_max}; actual={fmt(quality.get('red_chi2_global'))}"
    if flag == "failed_gof_p_value":
        return f"criterion: gof_p_value >= {config.fit_qc_gof_p_min}; actual={fmt(quality.get('gof_p_value'))}"
    if flag == "residual_nonrandom_runs":
        return f"criterion: runs_p_value >= {config.fit_qc_runs_p_min}; actual={fmt(quality.get('runs_p_value'))}"
    if flag == "residual_mean_large":
        return f"criterion: |residue_mean| <= {config.fit_qc_abs_mean_max}; actual={fmt(abs(float(quality.get('residue_mean') or 0)))}"
    return "criterion: see fitting quality metrics"


def range_text(suggestion, key):
    values = suggestion.get(key)
    if not values:
        return "n/a"
    return f"{fmt(values[0])} - {fmt(values[1])}"


def format_value(value):
    if isinstance(value, float):
        return fmt(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(item) for item in value)
    if value is None:
        return "n/a"
    return str(value)


def fmt(value):
    try:
        number = float(value)
        if np.isfinite(number):
            return f"{number:.5g}"
    except Exception:
        pass
    return "n/a"
