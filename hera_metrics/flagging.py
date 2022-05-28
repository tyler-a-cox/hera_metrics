"""
File for filtering radio frequency interference
"""

import numpy as np
from hera_filters import dspec

DPSS_DEFAULTS_1D = {
    "suppression_factors": [1e-9],
    "eigenval_cutoff": [1e-12],
    "max_contiguous_edge_flags": 10,
}
DFT_DEFAULTS_1D = {
    "suppression_factors": [1e-9],
    "fundamental_period": np.nan,
    "max_contiguous_edge_flags": 10,
}

basis_defaults = {"DPSS": DPSS_DEFAULTS_1D, "DFT": DFT_DEFAULTS_1D}

flag_estimation_algorithms = {
    "derivative": derivative,
    "rewlse": _robust_least_squares,
    "least_squares": _least_squares,
}


def _least_squares():
    """
    """
    pass


def _robust_least_squares():
    """
    """
    pass


def _derivative():
    """
    """
    pass


def flag_waterfall(
    data,
    filter_center,
    filter_half_width,
    basis_options=None,
    filter_dims=1,
    mode="dpss_solve",
):
    """
    Filter radio frequency interference using

    Parameters:
    ----------
    data : np.ndarray


    """
    pass


def flag_integration(uvdata, pols=["xx", "yy"], bls=None, include_autos=True):
    """
    Flag an entire integration based on the number

    Parameters:
    -----------
    uvdata : pyuvdata object
        Data
    pols : list of strings
        Polarizations to include in estimate of array-wide flags
    include_autos: bool, default=True
        Include autos in the estimate of the array-wide flags
    """
    pass
