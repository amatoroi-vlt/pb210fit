"""Vendored Pb-210 fitting and radiochemistry engine for the local web app."""

from .fitter import fit_3nuclides_spectrum
from .preprocessing import subtract_blank_poisson
from .templates import (
    build_template_from_spectrum,
    build_warped_blank_template,
    get_bi210_physical_spectrum,
    get_pb210_deposit_spectrum,
)
from .radiochemistry import (
    calculate_chemical_recovery,
    calculate_decay_factor,
    calculate_sample_concentration,
)

__all__ = [
    "fit_3nuclides_spectrum",
    "subtract_blank_poisson",
    "build_template_from_spectrum",
    "build_warped_blank_template",
    "get_bi210_physical_spectrum",
    "get_pb210_deposit_spectrum",
    "calculate_chemical_recovery",
    "calculate_decay_factor",
    "calculate_sample_concentration",
]
