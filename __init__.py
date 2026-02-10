"""
Greeks Library - BSM Option Pricing and Greeks.

Main module: bsm.py - Contains all helper functions that are not core to the focus of the project.

Usage:
    from greeks_lib.bsm import bsm_price, bsm_delta, bsm_get_greeks, bsm_iv
"""

from .bsm import (
    # Data fetching
    fetch_fred_rf,
    fetch_option_chains,
    # Core components
    bsm_d1,
    bsm_d2,
    # Pricing
    bsm_price,
    # Greeks
    bsm_delta,
    bsm_gamma,
    bsm_vega,
    bsm_theta,
    bsm_rho,
    bsm_get_greeks,
    # IV
    bsm_iv,
    bsm_iv_vectorized,
    # Visualization
    plot_greeks_surface,
    plot_all_greeks_surface,
)

__version__ = "1.0.0"
__all__ = [
    "fetch_fred_rf", "fetch_option_chains",
    "bsm_d1", "bsm_d2", "bsm_price",
    "bsm_delta", "bsm_gamma", "bsm_vega", "bsm_theta", "bsm_rho", "bsm_get_greeks",
    "bsm_iv", "bsm_iv_vectorized",
    "plot_greeks_surface",
    "plot_all_greeks_surface"
]
