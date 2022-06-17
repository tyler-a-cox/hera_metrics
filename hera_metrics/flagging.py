"""
File for filtering radio frequency interference
"""
from .utils import *
import tqdm
import numpy as np
from scipy import stats
from scipy import optimize
from hera_filters import dspec
from jax import numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

dpss_basis_defaults = {"eigenval_cutoff": [1e-3]}
stats_distributions = {"gaussian": stats.norm.cdf, "rayleigh": stats.rayleigh.cdf}


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


def _m_estimators(estimator="huber"):
    """
    Uses M-estimators to estimate regression parameters
    """
    assert estimator.lower() in ["huber", "arctan", ""]
    pass


def _maximum_correntropy(x):
    """
    Uses the maximum correntropy technique to estimate regression parameters
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
    update_weights=True,
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
    update_weights: bool, default=True
        If weights are provided and robust method is True, provided weights will be
        updated.

    Returns:
    -------
    model: np.ndarray
        Model of your data
    """
    assert distribution in [
        "normal",
        "rayleigh",
    ], f"Distribution {distribution} not supported"

    if len(basis_options) == 0:
        basis_options = dpss_basis_defaults

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

        # Combine with previous weights if previous weights given
        if wgts is not None and update_weights:
            wgts = ~wgts.astype(bool)
            wgts |= ~model_wgts.astype(bool)
            wgts = (~wgts).astype(float)

        else:
            wgts = model_wgts

        # Compute model using model weights
        model = solve_model(
            freqs,
            data,
            filter_centers,
            filter_half_widths,
            wgts=wgts,
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


def combine_weights(wgts, axis=0, method="mean", threshold=0.9):
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


def identify_rfi_waterfall(
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
    update_weights=True,
    combine_weights_threshold=0.8,
    **basis_options,
):
    """
    Same identify_rfi, but works on an individual waterfall
    """
    pass


def combine_waterfall_flags(waterfall_flags):
    """
    Combines a set weights of 2D waterfall plots to create a set of shared flags
    """
    pass


def estimate_weights(data, nsig=20):
    """
    Parameters:
    ----------
    data: np.ndarray
        Data
    nsig: float
        Number of sigma to flag outliers

    Returns:
    -------
    weights: 
    """
    sigma = (
        np.median(
            np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1, keepdims=True
        )
        / 0.675
    )
    mod_zscore = data / sigma
    weights = np.nanmedian(np.abs(mod_zscore) < nsig, axis=0)
    return weights


def flag_time_integration(
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
    update_weights=True,
    combine_weights_threshold=0.8,
    wgts=None,
    method="rewlse",
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
    update_weights: bool, default=False
        Update weights after each iteration. Otherwise, replace weights with new iteration
    update_weights_threshold: float, default=0.8
    method: str, default='rewlse'
        Robust regression method used to fit DPSS matrix

    Returns:
    -------
    model: np.ndarray
        Model of the data
    model_wgts: np.ndarray
        Weights applied to data
    """
    assert method.lower() in [
        "m-estimator",
        "rewlse",
        "maximum-correntropy",
    ], "Method not defined"

    filter_center = [0]
    filter_half_widths = np.linspace(
        1 / narrow_filter_width[0], 1 / wide_filter_width[0], niter
    )

    if len(basis_options) == 0:
        basis_options = dpss_basis_defaults

    # First pass filtering
    model = solve_model(
        freqs,
        data,
        filter_center,
        [1 / filter_half_widths[0]],
        robust=True,
        update_weights=update_weights,
        wgts=wgts,
        **basis_options,
    )
    outliers = identify_outliers(data, model, nsig=wide_nsig)
    model_wgts = combine_weights(outliers)

    for ni in range(1, niter):
        model = solve_model(
            freqs,
            data,
            filter_center,
            [1 / filter_half_widths[ni]],
            wgts=model_wgts,
            robust=robust_second_pass,
            update_weights=update_weights,
            **basis_options,
        )
        outliers = identify_outliers(data, model, nsig=wide_nsig)

        if update_weights:
            new_wgts = combine_weights(outliers, threshold=combine_weights_threshold)
            model_wgts = ~model_wgts.astype(bool)
            model_wgts |= ~new_wgts.astype(bool)
            model_wgts = (~model_wgts).astype(float)
        else:
            model_wgts = combine_weights(outliers, threshold=combine_weights_threshold)

    if incoherent_average:
        # Compute f model
        model = solve_model(
            freqs,
            data,
            filter_center,
            wide_filter_width,
            wgts=model_wgts,
            robust=False,
            update_weights=update_weights,
            **basis_options,
        )
        res = np.abs(model - data) * model_wgts
        noise = np.median(res, axis=1, keepdims=True)
        res = res / noise
        res = np.nanmedian(res, axis=0, keepdims=True)

        # First pass filtering
        model = solve_model(
            freqs, res, filter_center, narrow_filter_width, robust=True, **basis_options
        )
        outliers = identify_outliers(res, model, nsig=wide_nsig)

        if update_weights:
            new_wgts = combine_weights(outliers, threshold=combine_weights_threshold)
            model_wgts = ~model_wgts.astype(bool)
            model_wgts |= ~new_wgts.astype(bool)
            model_wgts = (~model_wgts).astype(float)
        else:
            model_wgts = combine_weights(outliers, threshold=combine_weights_threshold)

        for ni in range(1, niter):
            model = solve_model(
                freqs,
                res,
                filter_center,
                [1 / filter_half_widths[ni]],
                wgts=model_wgts,
                robust=robust_second_pass,
                update_weights=update_weights,
                **basis_options,
            )
            outliers = identify_outliers(res, model, nsig=narrow_nsig)

            if update_weights:
                new_wgts = combine_weights(
                    outliers, threshold=combine_weights_threshold
                )
                model_wgts = ~model_wgts.astype(bool)
                model_wgts |= ~new_wgts.astype(bool)
                model_wgts = (~model_wgts).astype(float)

            else:
                model_wgts = combine_weights(
                    outliers, threshold=combine_weights_threshold
                )

    model = solve_model(
        freqs,
        data,
        filter_center,
        wide_filter_width,
        wgts=model_wgts,
        robust=False,
        update_weights=update_weights,
        **basis_options,
    )
    return model, model_wgts


def flag_data(
    uvdata,
    narrow_filter_width=[1 / 70e6],
    wide_filter_width=[1 / 10e6],
    narrow_nsig=5,
    wide_nsig=5,
    incoherent_average=False,
    estimate_noise=False,
    niter=2,
    robust_second_pass=False,
    update_weights=True,
    combine_weights_threshold=0.8,
    wgts=None,
    **basis_options,
):
    """
    Parameters:
    ----------
    files:
        Files to read in
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
    update_weights: bool, default=False
        Update weights after each iteration. Otherwise, replace weights with new iteration

    Returns:
    -------
    model: np.ndarray
        Model of the data
    model_wgts: np.ndarray
        Weights applied to data
    """
    data, freqs, times = load_data(uvdata)
    cache = {}
    flags, models = [], []

    for i in tqdm.tqdm(range(data.shape[1])):
        model, wgts = flag_time_integration(
            freqs,
            data[:, i, :],
            narrow_filter_width=narrow_filter_width,
            wide_filter_width=wide_filter_width,
            cache=cache,
            narrow_nsig=narrow_nsig,
            wide_nsig=wide_nsig,
            niter=niter,
            robust_second_pass=robust_second_pass,
            incoherent_average=incoherent_average,
            update_weights=update_weights,
        )
        flags.append(wgts)
        models.append(model)

    return flags, models
