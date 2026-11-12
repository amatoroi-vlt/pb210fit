# pb210fit

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Report](https://img.shields.io/badge/Research_Report-First_Principles-green.svg)](https://amatoroi-vlt.github.io/reports/Research/report-first-principles-chain.html)

**First-Principles Multi-Nuclide Spectral Deconvolution Engine for Liquid Scintillation Spectrometry ($^{210}\text{Pb}$, $^{210}\text{Bi}$, $^{210}\text{Po}$)**.

---

## 🔬 Overview & Theoretical Foundation

`pb210fit` is an open-source scientific Python engine dedicated to resolving overlapping beta and alpha spectra in liquid scintillation counting (LSC). It models the full radioactive decay chain:

$$\text{Pb-210} \xrightarrow{\beta^-} \text{Bi-210} \xrightarrow{\beta^-} \text{Po-210} \xrightarrow{\alpha} \text{Pb-206 (stable)}$$

Rather than relying on empirical calibration curves or arbitrary multi-Gaussian approximations, `pb210fit` solves the deconvolution problem from **nuclear first principles**:

1. **LNHB Physical Beta Spectra**: Utilizes official French National Metrology Laboratory (LNHB) *BetaShape* tables to construct theoretical energy distributions for $^{210}\text{Pb}$ ($E_{\beta,\max} = 17.0\text{ keV}, 63.5\text{ keV}$) and $^{210}\text{Bi}$ ($E_{\beta,\max} = 1162.1\text{ keV}$).
2. **Bi L-X Ray Escape Modeling ($P_{\text{esc,XL}}$)**: Accounts for the geometric escape probability of Bi L-shell X-rays ($\sim 11.5\text{ keV}$) emitted during internal conversion (IC) in finite LSC vial volumes.
3. **Left-EMG Alpha Straggling ($\tau_{\text{rel}}$)**: Employs a Left-Exponentially Modified Gaussian (Left-EMG) distribution to model the asymmetric low-energy energy loss and straggling of $^{210}\text{Po}$ $5.304\text{ MeV}$ alpha particles in heterogeneous or precipitated samples.
4. **Decoupled 2D Pre-Scanning**: Combines a variable-projection grid search over quenching ($Q$) and $\alpha/\beta$ light output ratio ($\rho_{\alpha\beta}$) with constrained Trust Region Reflective (TRF) non-linear optimization.
5. **Rigorous Quality Control Metrics**: Computes reduced $\chi^2_{\text{eff}}$, Wald-Wolfowitz Runs test $p$-values, normalized Pull residuals, and systematic error percentages to automatically classify fit quality.

> 📄 **Full Interactive Report & Derivations**:  
> Read the complete interactive derivation and physics report at:  
> **[First-Principles Decay Chain Deconvolution Report](https://amatoroi-vlt.github.io/reports/Research/report-first-principles-chain.html)**

---

## 🚀 Key Features

* **Dual Background Modes**:
  * **Mode A**: Standard Poisson blank subtraction with full covariance and variance propagation.
  * **Mode B**: Simultaneous multi-component fitting with quench-warped empirical blank templates.
* **Radiochemical Analysis**:
  * Gravimetric chemical yield calculation (e.g., $\text{PbSO}_4$ precipitate stoichiometry).
  * Analytical radioactive decay correction for sampling, separation, and counting dates.
  * Activity concentration ($\text{Bq/kg}$, $\text{Bq/L}$) and Minimum Detectable Activity (MDA) estimation according to ISO 11929 / Currie limits.

---

## 📦 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/amatoroi-vlt/pb210fit.git
cd pb210fit
pip install -e .
```

### Requirements
* Python $\ge$ 3.9
* `numpy` $\ge$ 1.22.0
* `scipy` $\ge$ 1.8.0
* `pandas` $\ge$ 1.4.0

---

## 💡 Quickstart Example

```python
import numpy as np
from pb210fit.fitter import fit_3nuclides_spectrum
from pb210fit.radiochemistry import calculate_sample_concentration

# 1. Load your 1024-channel net count rate spectrum (CPS)
channels = np.arange(1024)
net_rate = ...   # 1D numpy array of net count rates (CPS)

# 2. Perform 3-nuclide first-principles deconvolution
result = fit_3nuclides_spectrum(
    rate_net=net_rate,
    fit_min_ch=120,
    fit_max_ch=750,
)

if result["success"]:
    print(f"Fit Grade: {result['fit_quality']['grade']}")
    print(f"Pb-210 Rate: {result['A_pb']:.4f} ± {result['A_pb_unc']:.4f} CPS")
    print(f"Bi-210 Rate: {result['A_bi']:.4f} ± {result['A_bi_unc']:.4f} CPS")
    print(f"Po-210 Rate: {result['A_po']:.4f} ± {result['A_po_unc']:.4f} CPS")
    print(f"Quenching Q: {result['q_scale']:.2f} %")
    print(f"Po-210 Tailing (tau_rel): {result['tau_rel']:.3f}")

# 3. Calculate sample concentration with chemical yield & decay correction
conc = calculate_sample_concentration(
    pb_signal_cps=result["A_pb"],
    pb_signal_unc=result["A_pb_unc"],
    efficiency=0.420,             # CPS / Bq
    sample_mass_kg=2.50,          # kg
    yield_frac=0.88,              # 88% chemical recovery
    decay_days_sampling=120.0,    # Days elapsed since sampling
)

print(f"Concentration at sampling: {conc.conc_sampling_bq_kg:.3f} ± {conc.conc_sampling_unc:.3f} Bq/kg")
```

---

## 🧪 Running Tests

Unit tests are implemented with synthetic Poisson spectra and analytical benchmarks:

```bash
python -m unittest tests/test_pb210fit.py
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
