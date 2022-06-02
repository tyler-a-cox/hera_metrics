"""
File for filtering radio frequency interference
"""

import numpy as np
from hera_filters import dspec
from scipy import optimize

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

def estimate_solution_matrix(data, filter_center, filter_half_width, basis='dpss', basis_options=None):
    """
    """
    def _least_median_squares(sol_mat):
        """
        """
        return np.median(np.abs(y - X @ sol_mat))

    X, _ = hera_filters.dspec.dpss_operator()
    x0 = X.T @ y
    solution = optimize.minimize(_least_median_squares, x0)
    return solution

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


def load_data(uvdata):
    """
    Reformat data into a bls x time x frequency array

    Parameters:
    ---------
    uvdata: UVData object
        pass

    Returns:
    -------
    data: np.ndarray
        Array of data unpacked from uvdata object
    """
    pass


def solve_model(
    freqs,
    data,
    filter_centers,
    filter_half_widths,
    wgts=None,
    robust=True,
    model_comps=None,
    basis="dpss",
    method="solve",
    distribution="normal",
    combine_wgts_method="mean",
    nsig=2.5,
    **basis_options,
):
    """
    Estimate a model of your data using discrete prolate spheroidal sequences

    Parameters:
    ----------
    freqs: np.ndarray
        pass
    data: np.ndarray
        pass
    filter_centers: list
        pass
    filter_half_widths: list
        pass
    wgts: np.ndarray, default=None
        Optional weight array to pass in. This method assumes that weights grid is shared between baselines
    robust: bool, default=True
        Optionally use robust regression techniques to estimate a model of your data. Uses the robust and efficient
        weighted least squares estimator to estimate weights and a model for a given basis
    model_comps: np.ndarray, default=None
        pass
    basis: str, default='dpss'
        Modeling basis
    distribution: str, default='normal'
        Distribution used for estimating
    method: str, default='solve'
        Method used to solve for model components
    combine_wgts_method: str, default='mean'
        Method used to combine weights array

    Returns:
    -------
    model: np.ndarray
        Model of your data
    """
    assert distribution in [
        "normal",
        "rayleigh",
    ], f"Distribution {distribution} not supported"

    # Maybe move this so that we don't have to pass in freqs
    design_mat, nterms = dspec.dpss_operator(
        freqs, filter_centers, filter_half_widths, **basis_options
    )

    # If model components are already provided for this design matrix,
    if model_comps is None:
        if wgts is None and basis == "dpss":
            model_comps = jnp.einsum("ij,ki->jk", design_mat, data, optimize=True)

        elif wgts is not None and method == "solve":
            XTX = jnp.einsum("ij,i,il->lj", design_mat, wgts, design_mat, optimize=True)
            XTWy = jnp.einsum("ij,ki->jk", design_mat, wgts * data, optimize=True)
            model_comps = jnp.linalg.solve(XTX, XTWy)

        else:
            raise ValueError(f"Method {method} not currently supported")

    # Compute model
    model = jnp.einsum("ij,jk->ki", design_mat, model_comps, optimize=True)

    if robust:
        # Compute residuals
        res = data - model

        # Compute modified z-score
        sigma = np.median(
            np.abs(res - np.median(res, keepdims=True, axis=1)), axis=1, keepdims=True
        )
        res = res / sigma * 0.675

        # Compute absolute value and sort
        absr = np.sort(np.abs(res), axis=1)

        # Identify outliers using distribution
        idx = absr >= nsig
        temp = stats.norm.cdf(absr) - np.arange(absr.shape[1]) / absr.shape[1]
        temp = np.where(idx, temp, 0)
        temp = np.where(temp < 0, 0, temp)
        d = np.max(temp, axis=1)
        t = absr[
            np.arange(absr.shape[0]),
            absr.shape[1] - 1 - np.array(np.floor(absr.shape[1] * d), dtype=int),
        ]

        # Compute model weights given robust regression
        model_wgts = np.where(np.abs(res) <= t[:, None], 1, 0)
        model_wgts = combine_weights(model_wgts, method=combine_wgts_method)

        # Compute model using model weights
        model = solve_model(
            freqs,
            data,
            filter_centers,
            filter_half_widths,
            wgts=model_wgts,
            robust=False,
            basis=basis,
            method=method,
            **basis_options,
        )

    return model


def identify_outliers(data, model, nsig=5, noise_model=None):
    """
    Identify outliers in your data given a model. Uses medians to estimate z-score

    Parameters:
    ----------
    data: np.ndarray
        Data array
    model: np.ndarray
        Estimated model of the data with RFI removed
    nsig: float, default=5

    noise_model: np.ndarray, default=None
        Estimate of the noise. Usually computed from the auto-correlations
    """
    assert data.shape == model.shape, "Model shape incompatible with data"

    # Calculate residuals
    res = np.abs(data - model)

    if noise_model is None:
        # Estimate standard deviation from residuals
        sigma = np.median(res, axis=1, keepdims=True) / 0.675
    else:
        sigma = np.copy(noise_model)

    wgts = np.array(
        (res - np.median(res, axis=1, keepdims=True)) / sigma < nsig, dtype=float
    )
    return wgts


def combine_weights(wgts, axis=0, method="mean", threshold=0.8):
    """
    Parameters:
    ----------
    wgts: np.ndarray
        Array to
    axis: int, default=0
        Axis to collapse weights down on
    method: str, default='mean'

    """
    assert method in ["quadrature", "mean"]

    if method == "quadrature":
        pass

    elif method == "mean":
        combined_wgts = np.mean(wgts, axis=axis)

    combined_wgts = np.where(combined_wgts < threshold, 0, 1)

    return combined_wgts


def flag_data(
    freqs,
    data,
    narrow_filter_width=[1 / 70e6],
    wide_filter_width=[1 / 10e6],
    narrow_nsig=5,
    wide_nsig=5,
    incoherent_average=False,
    estimate_noise=False,
    niter=2,
    robust_second_pass=False,
    final_flagging=True,
    **basis_options,
):
    """
    Parameters:
    ----------
    narrow_filter_width:
        Filter width of first pass narrow filter
    wide_filter_width:
        Filter width of second pass wide filter. Should at least be as wide as your longest baseline
    narrow_nsig: float
        Number of standard deviations to flag outliers at in the first pass step
    wide_nsig: float
        Number of standard deviations to flag outliers at in the second step
    incoherent_average: bool, default=False
        Average data incoherently after the wide filter width filtering stage. Seeks to identify residual
        RFI at or below the noise level.
    estimate_noise: bool, default=False
        Use auto correlations to estimate the noise level
    niter: int, default=2
        Number of iterations to attempt to remove RFI. Future parameter that may or may not be used
    robust_second_pass: bool, default=False,
        Whether or not to robustly flag after the first step
    """
    filter_center = [0]

    # First pass filtering
    model = solve_model(
        freqs, data, filter_center, narrow_filter_width, robust=True, **basis_options
    )
    model_wgts = identify_outliers(data, model, nsig=narrow_nsig)
    model_wgts = combine_weights(model_wgts)

    # Second Pass
    model = solve_model(
        freqs,
        data,
        filter_center,
        wide_filter_width,
        wgts=model_wgts,
        robust=robust_second_pass,
        **basis_options,
    )
    model_wgts = identify_outliers(data, model, nsig=wide_nsig)
    model_wgts = combine_weights(model_wgts)

    if incoherent_average:
        # Compute final model
        model = solve_model(
            freqs,
            data,
            filter_center,
            wide_filter_width,
            wgts=model_wgts,
            robust=False,
            **basis_options,
        )
        res = np.mean(np.abs(model - data), axis=0)

        # Make 2D to work on modeling functions
        res = res.reshape(1, -1)

        # First pass filtering
        model = solve_model(
            freqs, res, filter_center, narrow_filter_width, robust=True, **basis_options
        )
        model_wgts = identify_outliers(res, model, nsig=6)
        model_wgts = combine_weights(model_wgts)

        # Second Pass
        model = solve_model(
            freqs,
            res,
            filter_center,
            wide_filter_width,
            wgts=model_wgts,
            robust=robust_second_pass,
            **basis_options,
        )
        model_wgts = identify_outliers(res, model, nsig=6)
        model_wgts = combine_weights(model_wgts)

    model = solve_model(
        freqs,
        data,
        filter_center,
        wide_filter_width,
        wgts=model_wgts,
        robust=False,
        **basis_options,
    )
    return model, model_wgts