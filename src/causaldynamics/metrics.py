import inspect
from typing import Union

import dysts.flows as flows
import numpy as np
import torch
from scipy.linalg import expm
from scipy.signal import periodogram
from scipy.spatial.distance import pdist
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance as _wasserstein_distance_1d


def _check_shapes(true, pred):
    if true.shape != pred.shape:
        raise ValueError(
            f"true and pred must have the same shape, got {true.shape} and {pred.shape}"
        )


def _nrmse_per_timestep(true, pred, eps=1e-12):
    """
    Per-trajectory, per-timestep NRMSE, RMS-combined across dimensions after
    each dimension's squared error is normalized by that dimension's variance.

    Parameters
    ----------
    true : ndarray, shape (K, T, D)
    pred : ndarray, shape (K, T, D)
    eps : float
        Added to each dimension's variance to avoid division by zero.

    Returns
    -------
    ndarray, shape (K, T)
    """
    _check_shapes(true, pred)
    var_d = np.var(true, axis=(0, 1)) + eps  # (D,)
    sq_err = (true - pred) ** 2  # (K, T, D)
    return np.sqrt(np.mean(sq_err / var_d, axis=-1))  # (K, T)


def valid_prediction_time(true, pred, threshold=1.0):
    """
    Valid prediction time (VPT): the timestep at which the NRMSE between
    true and predicted trajectories first exceeds `threshold`, averaged
    over the K trajectories. Trajectories that never exceed the threshold
    are counted as diverging at the last timestep (T - 1).

    Parameters
    ----------
    true : ndarray, shape (K, T, D)
    pred : ndarray, shape (K, T, D)
    threshold : float, default=1.0
        NRMSE threshold above which a prediction is considered invalid.

    Returns
    -------
    float
        Mean valid prediction time (in timestep units) across the K trajectories.
    """
    nrmse_kt = _nrmse_per_timestep(true, pred)  # (K, T)
    T = nrmse_kt.shape[1]
    exceeded = nrmse_kt > threshold

    vpt_per_traj = np.where(
        exceeded.any(axis=1), np.argmax(exceeded, axis=1), T - 1
    )
    return float(np.mean(vpt_per_traj))


def nrmse_vs_time(true, pred):
    """
    NRMSE at each timestep, averaged over the K trajectories.

    Parameters
    ----------
    true : ndarray, shape (K, T, D)
    pred : ndarray, shape (K, T, D)

    Returns
    -------
    ndarray, shape (T,)
    """
    nrmse_kt = _nrmse_per_timestep(true, pred)  # (K, T)
    return np.mean(nrmse_kt, axis=0)


def log_spectral_distance(true, pred, eps=1e-12):
    """
    Log-spectral distance between true and predicted trajectories.

    For each trajectory and dimension, the power spectral density (PSD) is
    estimated along the time axis (periodogram), and the RMS difference of
    the log-PSDs across frequencies is computed. The result is averaged
    across dimensions D and trajectories K.

    Parameters
    ----------
    true : ndarray, shape (K, T, D)
    pred : ndarray, shape (K, T, D)
    eps : float
        Added to the PSD before taking the log, to avoid log(0).

    Returns
    -------
    float
    """
    _check_shapes(true, pred)
    _, psd_true = periodogram(true, axis=1)  # (K, F, D)
    _, psd_pred = periodogram(pred, axis=1)  # (K, F, D)

    log_ratio = 10 * np.log10(psd_true + eps) - 10 * np.log10(psd_pred + eps)
    lsd_kd = np.sqrt(np.mean(log_ratio**2, axis=1))  # (K, D), RMS over frequency
    return float(np.mean(lsd_kd))


def wasserstein_distance(true, pred, t_burn=500, multi_dimensional=True, eps=1e-12):
    """
    Wasserstein distance between the distributions of true and predicted
    states, pooling all K trajectories and all timesteps after a burn-in
    period `t_burn` (to compare long-run / attractor statistics rather than
    the transient).

    Each dimension is standardized by the standard deviation of the true
    pooled samples along that dimension before the distance is computed, so
    that dimensions with a larger absolute scale (e.g. `z` vs. `x` in
    Lorenz) don't dominate the result -- analogous to the per-dimension
    variance normalization used in `nrmse_vs_time`.

    Parameters
    ----------
    true : ndarray, shape (K, T, D)
    pred : ndarray, shape (K, T, D)
    t_burn : int, default=500
        Number of initial timesteps discarded before pooling samples.
    multi_dimensional : bool, default=True
        If True, compute a single D-dimensional 2-Wasserstein distance
        between the pooled point clouds using exact optimal transport
        (requires the `POT` package, `pip install pot`). This solves an
        exact OT problem with an N x N cost matrix, where
        N = K * (T - t_burn), so it can become slow / memory-heavy for
        large N.
        If False, compute the 1-D Wasserstein distance independently for
        each dimension (via `scipy.stats.wasserstein_distance`) and average
        the result over D.
    eps : float
        Added to each dimension's standard deviation before dividing, to
        avoid division by zero.

    Returns
    -------
    float
    """
    _check_shapes(true, pred)
    D = true.shape[2]
    true_pool = true[:, t_burn:, :].reshape(-1, D)
    pred_pool = pred[:, t_burn:, :].reshape(-1, D)

    std_d = np.std(true_pool, axis=0) + eps  # (D,)
    true_pool = true_pool / std_d
    pred_pool = pred_pool / std_d

    if multi_dimensional:
        try:
            import ot
        except ImportError as e:
            raise ImportError(
                "multi_dimensional=True requires the 'POT' package. "
                "Install it with `pip install pot`, or call with "
                "multi_dimensional=False to use the per-dimension "
                "scipy-only fallback instead."
            ) from e

        cost = ot.dist(true_pool, pred_pool, metric="sqeuclidean")
        a = np.full(true_pool.shape[0], 1.0 / true_pool.shape[0])
        b = np.full(pred_pool.shape[0], 1.0 / pred_pool.shape[0])
        w2_squared = ot.emd2(a, b, cost)
        return float(np.sqrt(w2_squared))

    return float(
        np.mean(
            [
                _wasserstein_distance_1d(true_pool[:, d], pred_pool[:, d])
                for d in range(D)
            ]
        )
    )


def _gmm_log_density(x, means, bandwidth, eps=1e-12):
    """
    Log-density of an isotropic-Gaussian-kernel mixture, evaluated at each
    row of `x`.

    One mixture component is placed at each row of `means`, all sharing
    the same scalar standard deviation `bandwidth` in every dimension
    (i.e. this is a Gaussian KDE). Pairwise squared distances are computed
    via the `||a-b||^2 = |a|^2 + |b|^2 - 2 a.b` expansion rather than a
    materialized (N, M, D) difference tensor, so memory scales as O(N*M)
    instead of O(N*M*D).

    Parameters
    ----------
    x : ndarray, shape (N, D)
        Points to evaluate the density at.
    means : ndarray, shape (M, D)
        Mixture component means.
    bandwidth : float
        Shared standard deviation of every component, in every dimension.
    eps : float
        Added to `bandwidth` before dividing/logging, to avoid
        division by zero.

    Returns
    -------
    ndarray, shape (N,)
    """
    D = x.shape[1]
    sq_dists = (
        np.sum(x**2, axis=1)[:, None]
        + np.sum(means**2, axis=1)[None, :]
        - 2 * x @ means.T
    )  # (N, M)
    sq_dists = np.clip(sq_dists, 0, None)  # guard against tiny negative roundoff

    log_kernel = -0.5 * sq_dists / (bandwidth**2 + eps)  # (N, M)
    log_norm = -D * np.log(bandwidth + eps) - 0.5 * D * np.log(2 * np.pi)
    return logsumexp(log_kernel, axis=1) + log_norm - np.log(means.shape[0])


def _subsample(x, max_samples, rng):
    if x.shape[0] > max_samples:
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        return x[idx]
    return x

from scipy.special import logsumexp
 
 
def _gmm_log_density(query_points, centers, sigmas, eps=1e-12):
    """
    Log-density of a Gaussian mixture at a set of query points.
 
    Parameters
    ----------
    query_points : (n, d) array
        Points at which to evaluate the mixture density.
    centers : (T, d) array
        Mixture component means (the trajectory points).
    sigmas : (T,) array
        Per-component isotropic standard deviations.
    eps : float
        Floor on sigma to avoid division by zero at repeated/duplicate points.
 
    Returns
    -------
    (n,) array of log p(x) for each query point.
    """
    T, d = centers.shape
    sigmas = np.maximum(sigmas, eps)
 
    # Squared distances between each query point and each component center:
    # (n, T)
    sq_dists = np.sum(
        (query_points[:, None, :] - centers[None, :, :]) ** 2, axis=-1
    )
 
    # log N(x; mu_t, sigma_t^2 I) for each (query, component) pair
    log_norm_const = -0.5 * d * np.log(2 * np.pi) - d * np.log(sigmas)  # (T,)
    log_kernel = -0.5 * sq_dists / (sigmas[None, :] ** 2)  # (n, T)
    log_components = log_norm_const[None, :] + log_kernel  # (n, T)
 
    # log( (1/T) * sum_t exp(log_components) ) via log-sum-exp for stability
    return logsumexp(log_components, axis=1) - np.log(T)
 

def _dstsp_pair(true_traj, pred_traj, n_samples=None, eps=1e-12, rng=None):
    """
    Estimate Dstsp = D_KL(p_true || q_pred) for a single pair of trajectories.
 
    Parameters
    ----------
    true_traj : (T1, d) array_like
        Ground-truth trajectory (points on the true attractor).
    pred_traj : (T2, d) array_like
        Model-generated trajectory (points on the reconstructed attractor).
        Does not need to be the same length as true_traj, and does not
        need to be time-aligned with it.
    n_samples : int, optional
        Number of Monte Carlo sample points to draw from the true
        distribution for the KL estimate. Defaults to min(T1, 2000).
        Samples are drawn (without replacement) from true_traj itself,
        since true_traj's points are already samples from p_true.
    eps : float
        Numerical floor for bandwidths, to guard against zero step sizes
        (e.g. repeated points).
    rng : np.random.Generator
        Random generator used for subsampling.
 
    Returns
    -------
    float
        Estimated KL divergence D_KL(p_true || q_pred). Larger values mean
        the predicted attractor's density diverges more from the true one
        (in particular, penalizes the model for placing too little density
        where the true attractor actually has mass).
    """
    true_traj = np.asarray(true_traj, dtype=float)
    pred_traj = np.asarray(pred_traj, dtype=float)
 
    if true_traj.ndim == 1:
        true_traj = true_traj[:, None]
    if pred_traj.ndim == 1:
        pred_traj = pred_traj[:, None]
 
    if true_traj.shape[1] != pred_traj.shape[1]:
        raise ValueError("true_traj and pred_traj must have the same dimensionality")
 
    T1 = true_traj.shape[0]
    T2 = pred_traj.shape[0]
    if T1 < 2 or T2 < 2:
        raise ValueError("Each trajectory needs at least 2 points to estimate a local bandwidth")
 
    # Adaptive per-point bandwidth: local step size along each trajectory.
    # sigma_t = ||x_t - x_{t-1}||, with the first point reusing the second
    # point's step size so every point has a defined bandwidth.
    def local_step_sizes(traj):
        steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)  # (T-1,)
        return np.concatenate([[steps[0]], steps])  # (T,)
 
    sigma_true = local_step_sizes(true_traj)
    sigma_pred = local_step_sizes(pred_traj)
 
    # Monte Carlo samples from p_true: subsample the true trajectory itself.
    if rng is None:
        rng = np.random.default_rng()
    if n_samples is None:
        n_samples = min(T1, 2000)
    n_samples = min(n_samples, T1)
    sample_idx = rng.choice(T1, size=n_samples, replace=False)
    samples = true_traj[sample_idx]
 
    log_p = _gmm_log_density(samples, true_traj, sigma_true, eps=eps)
    log_q = _gmm_log_density(samples, pred_traj, sigma_pred, eps=eps)
 
    return float(np.mean(log_p - log_q))
 
 
def dstsp(true_traj, pred_traj, n_samples=None, eps=1e-12, random_state=None,
          return_individual=False):
    """
    Estimate Dstsp = D_KL(p_true || q_pred), the state-space divergence
    between true and predicted attractor reconstructions.
 
    Accepts either a single pair of trajectories, shape (T, K), or a batch
    of N trajectory pairs, shape (N, T, K) — e.g. N different initial
    conditions for the same dynamical system. In the batched case, Dstsp is
    computed independently for each of the N pairs (by index: true_traj[i]
    against pred_traj[i]) and the results are averaged.
 
    Parameters
    ----------
    true_traj : (T, K) or (N, T, K) array_like
        Ground-truth trajectory/trajectories.
    pred_traj : (T2, K) or (N, T2, K) array_like
        Model-generated trajectory/trajectories, paired by index with
        true_traj. T2 may differ from T (trajectories need not be the same
        length or time-aligned), but N and K must match true_traj.
    n_samples : int, optional
        Monte Carlo sample count per pair, passed through to each pairwise
        estimate. Defaults to min(T, 2000) for each pair.
    eps : float
        Numerical floor for bandwidths.
    random_state : int or np.random.Generator, optional
        Seed/generator for reproducible subsampling. The same generator
        (advancing across calls) is reused across all N pairs so results
        are reproducible as a whole, not just per-pair.
    return_individual : bool
        If True, also return the raw per-trajectory Dstsp values (useful
        for reporting median/IQR across trajectories rather than just the
        mean).
 
    Returns
    -------
    float
        Mean Dstsp across the N trajectory pairs (or the single pair's
        Dstsp, if unbatched).
    values : (N,) ndarray, optional
        Per-trajectory Dstsp values. Only returned if return_individual=True.
        For a single (unbatched) pair, this is a length-1 array.
    """
    true_traj = np.asarray(true_traj, dtype=float)
    pred_traj = np.asarray(pred_traj, dtype=float)
 
    rng = np.random.default_rng(random_state)
 
    # Single pair: (T, K) vs (T2, K)
    if true_traj.ndim <= 2 and pred_traj.ndim <= 2:
        value = _dstsp_pair(true_traj, pred_traj, n_samples=n_samples, eps=eps, rng=rng)
        values = np.array([value])
        return (value, values) if return_individual else value
 
    # Batched: (N, T, K) vs (N, T2, K)
    if true_traj.ndim != 3 or pred_traj.ndim != 3:
        raise ValueError(
            "Expected both inputs to be (T, K) or (N, T, K); got shapes "
            f"{true_traj.shape} and {pred_traj.shape}"
        )
    if true_traj.shape[0] != pred_traj.shape[0]:
        raise ValueError(
            "true_traj and pred_traj must have the same number of "
            f"trajectories N; got {true_traj.shape[0]} and {pred_traj.shape[0]}"
        )
    if true_traj.shape[-1] != pred_traj.shape[-1]:
        raise ValueError("true_traj and pred_traj must have the same dimensionality K")
 
    N = true_traj.shape[0]
    values = np.empty(N, dtype=float)
    for i in range(N):
        values[i] = _dstsp_pair(
            true_traj[i], pred_traj[i], n_samples=n_samples, eps=eps, rng=rng
        )
 
    mean_value = float(np.mean(values))
    return (mean_value, values) if return_individual else mean_value
 

# def dstsp(
#     true,
#     pred,
#     t_burn=500,
#     bandwidth=None,
#     symmetric=False,
#     max_samples=1000,
#     random_state=0,
#     eps=1e-12,
# ):
#     """
#     D_stsp: a Kullback-Leibler divergence between the true and predicted
#     trajectories' state-space (attractor) distributions, following the
#     GMM/kernel-density measure used for dynamical systems reconstruction
#     in Koppe et al. (2019), Brenner et al. (2022) and Hess et al. (2023).

#     Each pooled trajectory (all K trajectories, timesteps after `t_burn`)
#     is turned into a Gaussian-kernel-density estimate: `p_true` from
#     `true`, `p_gen` from `pred`. `D_stsp` is then a Monte Carlo estimate
#     of D_KL(p_true || p_gen) -- how much probability mass the true
#     attractor places in regions the predicted attractor does not cover --
#     using the (possibly subsampled) true states themselves as both the
#     kernel centers and the evaluation/integration points for `p_true`.

#     As in `wasserstein_distance`, each dimension is first standardized by
#     the standard deviation of the pooled true samples, so a single
#     isotropic bandwidth can be shared across dimensions of different
#     physical scale.

#     Parameters
#     ----------
#     true : ndarray, shape (K, T, D)
#     pred : ndarray, shape (K, T, D)
#     t_burn : int, default=500
#         Number of initial timesteps discarded before pooling samples.
#     bandwidth : float, optional
#         Standard deviation (in standardized units) of each Gaussian
#         kernel. If None, uses Scott's rule `N ** (-1 / (D + 4))`, with N
#         the number of samples the kernel density is built from -- a
#         standard KDE bandwidth default that shrinks as more samples
#         become available.
#     symmetric : bool, default=False
#         If True, return the average of D_KL(p_true || p_gen) and
#         D_KL(p_gen || p_true) instead of the one-directional divergence.
#     max_samples : int, default=1000
#         Both the number of kernel components and the number of Monte
#         Carlo evaluation points are capped at this many samples
#         (subsampled without replacement) per distribution, since the
#         exact computation is O(N_eval * N_components). Pass a value
#         larger than K * (T - t_burn) to disable subsampling.
#     random_state : int, default=0
#         Seed for subsampling when `max_samples` triggers it.
#     eps : float
#         Numerical floor added before divisions/logs, for stability.

#     Returns
#     -------
#     float
#     """
#     _check_shapes(true, pred)
#     D = true.shape[2]
#     true_pool = true[:, t_burn:, :].reshape(-1, D)
#     pred_pool = pred[:, t_burn:, :].reshape(-1, D)

#     std_d = np.std(true_pool, axis=0) + eps  # (D,)
#     true_pool = true_pool / std_d
#     pred_pool = pred_pool / std_d

#     # true_pool and pred_pool are always the same length (true/pred have the
#     # same shape), so a single shared set of indices subsamples both -- this
#     # both reduces Monte Carlo variance versus independent subsamples, and
#     # ensures identical true/pred trajectories yield an exact 0 divergence
#     # regardless of max_samples.
#     rng = np.random.default_rng(random_state)
#     if true_pool.shape[0] > max_samples:
#         idx = rng.choice(true_pool.shape[0], size=max_samples, replace=False)
#         true_pool = true_pool[idx]
#         pred_pool = pred_pool[idx]

#     def _kl(samples, kernel_means, other_means):
#         if bandwidth is None:
#             h = samples.shape[0] ** (-1.0 / (D + 4))
#         else:
#             h = bandwidth
#         log_p = _gmm_log_density(samples, kernel_means, h, eps)
#         log_q = _gmm_log_density(samples, other_means, h, eps)
#         return float(np.mean(log_p - log_q))

#     kl_true_gen = _kl(true_pool, true_pool, pred_pool)
#     if not symmetric:
#         return kl_true_gen

#     kl_gen_true = _kl(pred_pool, pred_pool, true_pool)
#     return 0.5 * (kl_true_gen + kl_gen_true)


def correlation_dimension(
    traj,
    t_burn=500,
    r_quantiles=(0.01, 0.2),
    n_radii=20,
    max_samples=2000,
    random_state=0,
    eps=1e-12,
):
    """
    Grassberger-Procaccia correlation dimension of a trajectory's
    attractor.

    Estimates the correlation sum C(r), the fraction of point pairs
    (pooled across the K trajectories, after discarding the first
    `t_burn` timesteps as transient) closer together than r, for
    `n_radii` log-spaced radii spanning the `r_quantiles` quantiles of
    the pairwise-distance distribution. In the scaling region, C(r) grows
    as r^D2, so D2 is estimated as the least-squares slope of log C(r)
    against log r.

    The radius range defaults to the [1st, 20th] percentile of pairwise
    distances rather than the full range: at very small r, C(r) is
    dominated by sampling noise / near-duplicate points (too few pairs
    that close to estimate a slope from), and at large r it saturates
    towards 1 (every pair counted), so the log-log relationship is no
    longer a power law -- both regimes bias the fitted slope away from
    the attractor's actual fractal dimension.

    Parameters
    ----------
    traj : ndarray, shape (K, T, D)
    t_burn : int, default=500
        Number of initial timesteps discarded before pooling samples.
    r_quantiles : (float, float), default=(0.01, 0.2)
        Lower and upper quantile of the pairwise-distance distribution
        used to set the scaling region [r_min, r_max].
    n_radii : int, default=20
        Number of log-spaced radii within the scaling region.
    max_samples : int, default=2000
        Points are subsampled to at most this many before computing
        pairwise distances, since both the distance computation and the
        correlation sum are O(N^2). Pass a value larger than
        K * (T - t_burn) to disable subsampling.
    random_state : int, default=0
        Seed for subsampling when `max_samples` triggers it.
    eps : float
        Numerical floor added inside logs, and used to discard
        (near-)duplicate points, for stability.

    Returns
    -------
    float
        Estimated correlation dimension D2.
    """
    D = traj.shape[2]
    pool = traj[:, t_burn:, :].reshape(-1, D)

    rng = np.random.default_rng(random_state)
    pool = _subsample(pool, max_samples, rng)

    dists = pdist(pool)
    dists = dists[dists > eps]

    r_min, r_max = np.quantile(dists, r_quantiles)
    radii = np.geomspace(r_min, r_max, n_radii)

    corr_sum = np.array([np.mean(dists < r) for r in radii])
    slope, _ = np.polyfit(np.log(radii), np.log(corr_sum + eps), 1)
    return float(slope)


def correlation_dimension_error(true, pred, t_burn=500, **corr_dim_kwargs):
    """
    Absolute difference between the Grassberger-Procaccia correlation
    dimension (see `correlation_dimension`) of the true and predicted
    attractors.

    Parameters
    ----------
    true : ndarray, shape (K, T, D)
    pred : ndarray, shape (K, T, D)
    t_burn : int, default=500
        Number of initial timesteps discarded before pooling samples.
    **corr_dim_kwargs
        Forwarded to `correlation_dimension` (e.g. `r_quantiles`,
        `n_radii`, `max_samples`, `random_state`, `eps`).

    Returns
    -------
    float
    """
    _check_shapes(true, pred)
    d2_true = correlation_dimension(true, t_burn=t_burn, **corr_dim_kwargs)
    d2_pred = correlation_dimension(pred, t_burn=t_burn, **corr_dim_kwargs)
    return float(np.abs(d2_true - d2_pred))


def _benettin_lyapunov_exponents(step_operators, dt):
    """
    Benettin / QR algorithm for Lyapunov exponents.

    Repeatedly applies each per-step linear tangent propagator to an
    orthonormal basis of tangent vectors, re-orthonormalizing via QR after
    every step and accumulating the log-growth of each basis vector. This
    is agnostic to whether a step operator came from a continuous-time
    Jacobian (via matrix exponential) or is itself a discrete-time
    one-step Jacobian.

    Parameters
    ----------
    step_operators : ndarray, shape (N, D, D)
        Linear map applied to the tangent basis at each of the N steps.
    dt : float or ndarray, shape (N,)
        Time elapsed at each step (scalar if uniform), used to convert
        accumulated log-growth into exponents per unit time.

    Returns
    -------
    ndarray, shape (D,)
        Lyapunov exponents, sorted from largest to smallest.
    """
    N, D, _ = step_operators.shape
    Q = np.eye(D)
    log_growth = np.zeros(D)
    for n in range(N):
        Q = step_operators[n] @ Q
        Q, R = np.linalg.qr(Q)
        log_growth += np.log(np.abs(np.diag(R)))
    total_time = np.sum(np.broadcast_to(dt, (N,)))
    return np.sort(log_growth / total_time)[::-1]


def true_lyapunov_exponents(
    system: Union[str, flows.DynSys],
    num_timesteps=2000,
    t_burn=500,
    make_trajectory_kwargs=None,
):
    """
    Lyapunov exponent spectrum of a `dysts` chaotic system computed from its
    analytical Jacobian (`system._jac`) via the Benettin/QR algorithm.

    A reference trajectory is generated with `system.make_trajectory`, the
    first `t_burn` steps are discarded as transient, and at each of the
    remaining `num_timesteps` points the Jacobian is evaluated (using the
    system's actual parameter values, read off the instance) and turned
    into a one-step tangent propagator via the matrix exponential
    `expm(J * dt)`.

    `dt` is taken from the exact sample timestamps returned by
    `system.make_trajectory(..., return_times=True)`, rather than from the
    system's `.dt` attribute: by default `make_trajectory` resamples its
    output to `period / pts_per_period` spacing (Fourier timescale), which
    differs from `.dt` (the raw integrator step) by roughly the number of
    integrator substeps per output sample -- using `.dt` instead silently
    produces the wrong tangent propagators and a wrong exponent spectrum,
    even though the sum of exponents (which only depends on the mean
    Jacobian trace) misleadingly still comes out right.

    Parameters
    ----------
    system : str or dysts.flows.DynSys
        Name of a `dysts.flows` system class, or an instance of one, that
        exposes an analytical Jacobian via `_jac`.
    num_timesteps : int, default=2000
        Number of post-burn-in steps used to estimate the exponents.
    t_burn : int, default=500
        Number of initial trajectory steps discarded as transient before
        settling onto the attractor.
    make_trajectory_kwargs : dict, optional
        Extra keyword arguments forwarded to `system.make_trajectory`
        (e.g. `resample`, `pts_per_period`, `timescale`). `return_times`
        is always forced to True internally.

    Returns
    -------
    ndarray, shape (D,)
        Lyapunov exponents, sorted from largest to smallest.

    Notes
    -----
    This assumes (matching the rest of `causaldynamics.systems`, e.g.
    `get_adjacency_matrix_from_jac`) that `system._jac`'s signature is
    `_jac(x_1, ..., x_D, t, *params)`, and that each parameter name in
    `*params` is also an attribute on `system` holding its current value.
    If that assumption doesn't hold for a given system, this will raise or
    silently use the wrong parameter values -- worth spot-checking against
    a known result (e.g. Lorenz's spectrum is approximately
    [0.906, 0, -14.57]) the first time it's used.
    """
    if isinstance(system, str):
        system = getattr(flows, system)()
    elif inspect.isclass(system):
        system = system()

    if not hasattr(system, "_jac") or system._jac is None:
        raise ValueError(
            f"{system.__class__.__name__} has no analytical Jacobian (_jac)."
        )

    make_trajectory_kwargs = dict(make_trajectory_kwargs or {})
    make_trajectory_kwargs["return_times"] = True
    tpts, trajectory = system.make_trajectory(
        t_burn + num_timesteps, **make_trajectory_kwargs
    )
    tpts = np.asarray(tpts)[t_burn:]  # (num_timesteps,)
    trajectory = np.asarray(trajectory)[t_burn:]  # (num_timesteps, D)
    D = trajectory.shape[1]

    jac_param_names = list(inspect.signature(system._jac).parameters)[D + 1 :]
    param_values = [getattr(system, name) for name in jac_param_names]

    step_dt = np.diff(tpts)  # (num_timesteps - 1,)
    step_operators = np.empty((len(step_dt), D, D))
    for n, dt_n in enumerate(step_dt):
        jac = np.array(
            system._jac(*trajectory[n], tpts[n], *param_values), dtype=float
        )
        step_operators[n] = expm(jac * dt_n)

    return _benettin_lyapunov_exponents(step_operators, step_dt)


def nn_lyapunov_exponents(model, x0, num_steps=2000, t_burn=500, dt=1.0):
    """
    Lyapunov exponent spectrum of a one-step autoregressive NN model
    (x_{t+1} = model(x_t)) via the Benettin/QR algorithm.

    The model is rolled out from `x0`, discarding the first `t_burn` steps
    as transient. For each of the remaining `num_steps` states, the
    model's Jacobian is computed with `torch.func.jacrev` and used
    directly as the one-step tangent propagator (no matrix exponential is
    needed since the model already maps one step to the next).

    Parameters
    ----------
    model : torch.nn.Module
        Differentiable model mapping a state x_t of shape (D,) to the next
        state x_{t+1} of shape (D,).
    x0 : torch.Tensor, shape (D,)
        Initial state to roll out from.
    num_steps : int, default=2000
        Number of post-burn-in steps used to estimate the exponents.
    t_burn : int, default=500
        Number of initial rollout steps discarded as transient.
    dt : float, default=1.0
        Physical time elapsed per model step. Set this to the true
        system's sampling `dt` to make the result comparable to
        `true_lyapunov_exponents`; leave at 1.0 to get exponents in units
        of "per model step".

    Returns
    -------
    ndarray, shape (D,)
        Lyapunov exponents, sorted from largest to smallest.
    """
    model.eval()
    x = x0
    D = x.shape[-1]

    with torch.no_grad():
        for _ in range(t_burn):
            x = model(x)

    jacobians = np.empty((num_steps, D, D))
    for n in range(num_steps):
        jacobians[n] = torch.func.jacrev(model)(x).detach().cpu().numpy()
        with torch.no_grad():
            x = model(x)

    return _benettin_lyapunov_exponents(jacobians, dt)


def nn_lyapunov_exponents_delta(
    model, traj, mean_grad, std_grad, dt, num_steps=2000, t_burn=500
):
    """
    Lyapunov exponent spectrum of a one-step autoregressive NN model that
    predicts a normalized increment, via the Benettin/QR algorithm.

    This is the delta-parameterized counterpart to `nn_lyapunov_exponents`,
    matching the one-step update used elsewhere in this codebase:

        diff = model.infer(x[None])[0]
        x_next = x + diff * std_grad + mean_grad

    i.e. `model.infer` predicts a *normalized* increment, which is
    un-normalized with `std_grad` / `mean_grad` before being added to the
    current state. The full one-step map's Jacobian is taken directly
    (via `torch.func.jacrev`) of that whole expression, so the elementwise
    `std_grad` scaling and the `+ x` term are both accounted for correctly
    -- unlike `nn_lyapunov_exponents`/its unnormalized delta counterpart,
    it is not simply `I + J_model(x)`.

    Rather than rolling the model out autoregressively -- which can drift
    off the attractor as one-step errors compound over `t_burn + num_steps`
    steps -- the base trajectory is taken directly from `traj`: the local
    Jacobian at step `n` is evaluated at `traj[t_burn + n]`, the true state,
    not at a model-predicted one.

    Internally this calls `model.infer_diff` rather than `model.infer`:
    `infer` wraps its body in `torch.no_grad()`, and `torch.func.jacrev`
    silently returns an all-zero Jacobian (no error) for a function whose
    body runs under `torch.no_grad()`, so `model` must expose a
    `infer_diff` method identical to `infer` but without that wrapping.

    Parameters
    ----------
    model : object
        Model exposing a differentiable `infer_diff` method mapping a batch
        of states, shape (1, D), to a batch of normalized increments, shape
        (1, D).
    traj : torch.Tensor, shape (t_burn + num_steps + 1, D)
        Reference trajectory to evaluate local Jacobians along.
    mean_grad, std_grad : torch.Tensor, shape (D,) or scalar
        Mean and standard deviation used to un-normalize `model.infer`'s
        output back into raw increment units, as
        `diff * std_grad + mean_grad`.
    dt : float
        Physical time elapsed per model step -- i.e. the true system's
        sampling `dt` (the same one used to generate `traj`, and the same
        one `true_lyapunov_exponents` would use). Required, with no
        default: exponents scale as `1/dt`, so silently leaving this at an
        arbitrary value (e.g. 1.0) produces numbers in the wrong units
        that are not comparable to `true_lyapunov_exponents`.
    num_steps : int, default=2000
        Number of post-burn-in steps used to estimate the exponents.
    t_burn : int, default=500
        Number of initial `traj` steps skipped as transient.

    Returns
    -------
    ndarray, shape (D,)
        Lyapunov exponents, sorted from largest to smallest.
    """
    D = traj.shape[-1]

    def step(x):
        diff = model.infer_diff(x[None])[0]
        return x + diff * std_grad + mean_grad

    jacobians = np.empty((num_steps, D, D))
    for n in range(num_steps):
        x = traj[t_burn + n]
        jacobians[n] = torch.func.jacrev(step)(x).detach().cpu().numpy()

    return _benettin_lyapunov_exponents(jacobians, dt)


def nn_jacobian_mse(system, traj, model, mean, std):
    """
    Mean squared error, averaged over a trajectory, between a `dysts`
    system's analytical Jacobian and the Jacobian implied by a trained
    one-step delta-prediction neural network model.

    `model` is assumed to predict a normalized state increment,
    `model(x) ~= (x_{t+1} - x_t - mean) / std`, so the model's estimate of
    the raw (un-normalized) increment is `g(x) = std * model(x) + mean`.
    At each point `x = traj[k]`, this computes the Jacobian of `g` with
    respect to `x` (via automatic differentiation, `torch.func.jacrev`)
    and compares it, entrywise, to the system's analytical Jacobian
    `system.jac(x, 0)` at that same point (matching the usage in
    `notebooks/3D_trajectories.ipynb`). The squared entrywise differences
    are averaged over the D x D Jacobian entries and then over the T
    points of `traj`.

    Note that `mean`, being a constant shift, does not itself affect the
    Jacobian of `g` (its derivative is 0) -- it's included in `g` only so
    that the un-normalization exactly matches the one used elsewhere in
    this codebase (e.g. `nn_lyapunov_exponents_delta`), and in case
    `model`'s own output happens to depend on `mean`.

    Parameters
    ----------
    system : dysts.flows.DynSys
        System exposing an analytical Jacobian via `system.jac(x, t)`.
    traj : torch.Tensor, shape (T, D)
        Trajectory of states to evaluate the Jacobian error at.
    model : callable
        Differentiable model mapping a state `x` of shape (D,) to a
        normalized increment prediction of shape (D,). Must not wrap its
        forward pass in `torch.no_grad()`, since `torch.func.jacrev`
        silently returns an all-zero Jacobian for a function that does.
    mean, std : torch.Tensor, shape (D,) or scalar
        Mean/std used to un-normalize the model's output, matching the
        normalization the model was trained with.

    Returns
    -------
    float
    """
    T = traj.shape[0]

    def g(x):
        return std * model(x) + mean

    sq_errs = np.empty(T)
    for k in range(T):
        x = traj[k]
        model_jac = torch.func.jacrev(g)(x).detach().cpu().numpy()
        true_jac = np.asarray(system.jac(x.detach().cpu().numpy(), 0), dtype=float)
        sq_errs[k] = np.mean((model_jac - true_jac) ** 2)

    return float(np.mean(sq_errs))


def kaplan_yorke_dimension(lyapunov_exponents):
    """
    Kaplan-Yorke (Lyapunov) dimension of an attractor, estimated from its
    Lyapunov exponent spectrum.

    Defined as D_KY = k + (sum_{i=1}^k lambda_i) / |lambda_{k+1}|, where
    the exponents are sorted from largest to smallest and k is the
    largest number of leading exponents whose cumulative sum stays
    non-negative (found by walking the sorted spectrum and stopping at
    the first exponent that would push the running sum negative, rather
    than assuming the cumulative sum is monotonic).

    If every exponent is negative, k = 0 and D_KY = 0. If the cumulative
    sum of the entire spectrum stays non-negative (no negative exponent
    left to provide lambda_{k+1}), D_KY is capped at the full dimension D.

    Parameters
    ----------
    lyapunov_exponents : array-like, shape (D,)
        Lyapunov exponents in any order; sorted descending internally.

    Returns
    -------
    float
    """
    exponents = np.sort(np.asarray(lyapunov_exponents, dtype=float))[::-1]
    D = len(exponents)

    k = 0
    running_sum = 0.0
    for lam in exponents:
        if running_sum + lam < 0:
            break
        running_sum += lam
        k += 1

    if k == 0:
        return 0.0
    if k >= D:
        return float(D)
    return float(k + running_sum / np.abs(exponents[k]))
