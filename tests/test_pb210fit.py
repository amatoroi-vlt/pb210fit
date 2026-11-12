"""
Unit tests for refactored Pb-210 modules:
  - pb210fit.preprocessing (Mode A blank subtraction & Poisson error propagation)
  - pb210fit.templates (Mode B quench-warped empirical blank template)
  - pb210fit.radiochemistry (Common radiochemical analysis for both 2-Window and fitting)
  - analyze.py integration
"""
import unittest
import sys
from pathlib import Path

# Add server/python to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pb210fit.preprocessing import subtract_blank_poisson
from pb210fit.templates import build_warped_blank_template
from pb210fit.fitter import fit_3nuclides_spectrum
from pb210fit.radiochemistry import (
    calculate_chemical_recovery,
    calculate_decay_factor,
    calculate_sample_concentration,
    PB210_HALF_LIFE_DAYS,
    PB_TO_PBSO4_RATIO,
)


class TestPreprocessing(unittest.TestCase):
    def test_subtract_blank_poisson_basic(self):
        # 1024-channel synthetic data
        sample_counts = np.full(1024, 100.0)
        sample_time = 100.0  # gross rate = 1.0 cps, var_s = 100 / 10000 = 0.01

        blank_counts = np.full(1024, 20.0)
        blank_time = 50.0   # blank rate = 0.4 cps, var_b = 20 / 2500 = 0.008

        rate_net, sigma_net = subtract_blank_poisson(
            sample_counts=sample_counts,
            sample_time=sample_time,
            blank_counts=blank_counts,
            blank_time=blank_time,
        )

        self.assertEqual(len(rate_net), 1024)
        self.assertEqual(len(sigma_net), 1024)

        # Expected net rate = 1.0 - 0.4 = 0.6 cps
        np.testing.assert_allclose(rate_net, 0.6, rtol=1e-5)

        # Expected variance = 0.01 + 0.008 = 0.018 -> sigma = sqrt(0.018) ~ 0.134164
        expected_sigma = np.sqrt(0.018)
        np.testing.assert_allclose(sigma_net, expected_sigma, rtol=1e-5)

    def test_subtract_blank_clamping(self):
        # When blank rate > sample rate, net rate must clamp to 0.0
        sample_counts = np.array([10.0, 20.0])
        sample_time = 100.0  # rates: 0.1, 0.2 cps

        blank_counts = np.array([50.0, 10.0])
        blank_time = 100.0   # rates: 0.5, 0.1 cps

        rate_net, sigma_net = subtract_blank_poisson(
            sample_counts=sample_counts,
            sample_time=sample_time,
            blank_counts=blank_counts,
            blank_time=blank_time,
        )

        # Ch 0: 0.1 - 0.5 = -0.4 -> clamped to 0.0
        # Ch 1: 0.2 - 0.1 = +0.1
        self.assertAlmostEqual(rate_net[0], 0.0)
        self.assertAlmostEqual(rate_net[1], 0.1)
        # Uncertainties must remain strictly positive
        self.assertTrue(np.all(sigma_net > 0))

    def test_no_blank_fallback(self):
        sample_counts = np.full(1024, 400.0)
        sample_time = 200.0  # rate = 2.0 cps, var = 400/40000 = 0.01, sigma = 0.1

        rate_net, sigma_net = subtract_blank_poisson(
            sample_counts=sample_counts,
            sample_time=sample_time,
            blank_counts=None,
            blank_time=None,
        )

        np.testing.assert_allclose(rate_net, 2.0, rtol=1e-5)
        np.testing.assert_allclose(sigma_net, 0.1, rtol=1e-5)


class TestTemplatesModeB(unittest.TestCase):
    def test_build_warped_blank_template(self):
        # Gaussian peak at channel 250
        chs = np.arange(1024, dtype=float)
        bkg = np.exp(-0.5 * ((chs - 250.0) / 20.0) ** 2)

        # k_qpe = 1.0 (no shift)
        t_base = build_warped_blank_template(bkg, k_qpe=1.0)
        self.assertEqual(len(t_base), 1024)
        self.assertAlmostEqual(t_base.sum(), 1.0, places=5)
        self.assertEqual(np.argmax(t_base), 250)

        # k_qpe = 1.2 (higher light output -> shifts right: delta_ch = 362 * log10(1.2) ~ +28.66 ch)
        t_up = build_warped_blank_template(bkg, k_qpe=1.2)
        peak_up = np.argmax(t_up)
        self.assertGreater(peak_up, 250)
        self.assertAlmostEqual(peak_up, 279, delta=2)
        self.assertAlmostEqual(t_up.sum(), 1.0, places=5)

        # k_qpe = 0.8 (lower light output / more quenched -> shifts left)
        t_down = build_warped_blank_template(bkg, k_qpe=0.8)
        peak_down = np.argmax(t_down)
        self.assertLess(peak_down, 250)
        self.assertAlmostEqual(t_down.sum(), 1.0, places=5)


class TestRadiochemistry(unittest.TestCase):
    def test_chemical_recovery(self):
        # Filter before = 0.0800 g, after = 0.1100 g -> ppt = 0.0300 g
        # carrier added = 2.000 g, carrier conc = 0.0100 (10 mg Pb/g) -> carrier Pb = 0.0200 g
        # Stoichiometry ratio: 207.2 / 295.2 = 0.701897
        # Expected Y = (0.0300 * 0.701897) / (2.000 * 0.0100) = 0.0210569 / 0.0200 = 1.0528
        res = calculate_chemical_recovery(
            filter_before_g=0.0800,
            filter_after_g=0.1100,
            carrier_added_g=2.000,
            carrier_conc=0.0100,
            filter_before_unc_g=0.0001,
            filter_after_unc_g=0.0001,
            carrier_added_unc_g=0.001,
            carrier_conc_unc=0.0001,
        )

        self.assertIsNotNone(res.yield_frac)
        expected_y = (0.0300 * PB_TO_PBSO4_RATIO) / 0.0200
        self.assertAlmostEqual(res.yield_frac, expected_y, places=4)
        self.assertGreater(res.yield_unc, 0.0)

    def test_decay_factor(self):
        # 1 half life (8108.55 days) -> decay factor must be exactly 2.0
        f_half = calculate_decay_factor(PB210_HALF_LIFE_DAYS)
        self.assertAlmostEqual(f_half, 2.0, places=5)

        # 0 days -> factor 1.0
        self.assertAlmostEqual(calculate_decay_factor(0.0), 1.0, places=6)

    def test_sample_concentration_symmetry(self):
        # Works identically for either Mode 1 (2-Window) or Mode 2 (Fitter)
        pb_cps = 0.500
        pb_cps_unc = 0.010
        eff = 0.400       # 40% efficiency (CPS/Bq)
        eff_unc = 0.008
        mass_kg = 2.500   # 2.5 kg water/soil sample
        yield_frac = 0.85 # 85% chemical yield

        res = calculate_sample_concentration(
            pb_signal_cps=pb_cps,
            pb_signal_unc=pb_cps_unc,
            efficiency=eff,
            eff_unc=eff_unc,
            sample_mass_kg=mass_kg,
            yield_frac=yield_frac,
            decay_days_sampling=365.25, # 1 year decay elapsed
        )

        # Raw activity = 0.500 / 0.400 = 1.25 Bq
        self.assertAlmostEqual(res.activity_raw, 1.25, places=4)

        # True activity = 1.25 / 0.85 = 1.470588 Bq
        self.assertAlmostEqual(res.activity_bq, 1.25 / 0.85, places=4)

        # Conc @ measurement date = 1.470588 / 2.5 = 0.588235 Bq/kg
        self.assertAlmostEqual(res.conc_meas_bq_kg, (1.25 / 0.85) / 2.5, places=4)

        # Conc @ sampling date: 1 year decay factor exp(lambda * 365.25) ~ exp(ln2 / 22.2) ~ 1.0317
        expected_decay_f = np.exp(np.log(2.0) / 22.2)
        self.assertAlmostEqual(res.conc_sampling_bq_kg, res.conc_meas_bq_kg * expected_decay_f, places=4)
        self.assertGreater(res.conc_meas_unc, 0.0)


class TestFitterSimultaneousBkg(unittest.TestCase):
    def test_fit_with_bkg_template(self):
        # Synthetic flat Bkg + Gaussian Po-210 test
        np.random.seed(42)
        rate = np.zeros(1024, dtype=float)
        bkg = np.zeros(1024, dtype=float)
        bkg[120:750] = 0.05
        # Add bkg and a small peak
        rate += bkg
        ch = np.arange(1024)
        rate += 1.0 * np.exp(-0.5 * ((ch - 574.0) / 14.0) ** 2)

        res = fit_3nuclides_spectrum(
            rate_net=rate,
            bkg_template=bkg,
            fit_min_ch=120,
            fit_max_ch=750,
            max_nfev=500,
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["bkg_mode"], "simultaneous")
        self.assertIn("A_bg", res)
        self.assertIn("fit_Bg_y", res)
        self.assertGreater(res["A_bg"], 0.5)


if __name__ == "__main__":
    unittest.main()
