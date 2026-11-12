"""Default fitting constants.

The original research runner used a process-wide config module populated from
environment variables. The web app keeps the same constant names for the
vendored fitter while passing run-specific values through JSON.
"""

init_Bi = [1, 1, 0]
lowerLimP_bi = [0, 0.1, -10]
upperLimP_bi = [1e6, 10.0, 100]

init_Po = [1, 100.0, 10, 1.5, 3.0]
lowerLimP_po = [0, 1, 1.0, 0.2, 1.01]
upperLimP_po = [1e6, 1000, 100.0, 8.0, 30.0]

Eb_min, Eb_max = 23.2960, 58.6673
Ed_min, Ed_max = 153.0882, 351.5
E_min_po, E_max_po = Eb_min, Ed_max
Ea_min, Ea_max = 1.0, Eb_min

enable_auto_window_suggestion = True
auto_window_sg_window_length = 11
auto_window_sg_polyorder = 2
auto_window_a_half_width_ch = 120.0
auto_window_c_half_width_ch = 70.0

enable_physical_constraints = True
po_mu_ref_tolerance = 12.0
apply_po_sigma_constraint = True
po_sigma_ref_scale_min = 0.5
po_sigma_ref_scale_max = 1.6
apply_po_alpha_constraint = True
po_alpha_ref_scale_min = 0.5
po_alpha_ref_scale_max = 2.0
apply_po_n_constraint = True
po_n_ref_scale_min = 0.5
po_n_ref_scale_max = 2.0

enable_two_component_refinement = True
two_component_refine_min_improve_rmse = 0.0
standard_prefit_enable_two_component_refinement = False
standard_prefit_max_po_initial_guesses = 48
standard_prefit_po_early_stop_red_chi2 = 1.2

fit_qc_red_chi2_min = 0.3
fit_qc_red_chi2_max = 2.0
fit_qc_gof_p_min = 0.01
fit_qc_runs_p_min = 0.01
fit_qc_mean_p_min = 0.01
fit_qc_abs_mean_max = 0.35
fit_qc_residue_rmse_max = 2.5

enable_three_component_fit = False
strict_three_component_fit = False
pb_template_min_area = 10.0
upperLimP_pb = 1e6

