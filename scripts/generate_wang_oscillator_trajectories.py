"""
Wang (2001) Unified Oscillator for ENSO.

Reference
---------
Wang, C., 2001: A Unified Oscillator Model for the El Nino-Southern
Oscillation. J. Climate, 14, 98-115.
The four coupled equations below are Eqs. (7), (10), (13), (14) in that
paper, which the author explicitly states "form a unified oscillator
model of the coupled ocean-atmosphere system."

State vector (4-dimensional)
-----------------------------
T1   : Nino-3 SST anomaly (degC)
h    : Nino-6 off-equatorial (western Pacific) thermocline depth anomaly (m)
tau1 : Nino-4 zonal wind stress anomaly (N/m^2)   [equatorial westerlies]
tau2 : Nino-5 zonal wind stress anomaly (N/m^2)   [western equatorial easterlies]

Equations
---------
dT1/dt   = a*tau1(t) - b1*tau1(t-eta) + b2*tau2(t-delta) - eps*T1(t)^3
dh/dt    = -c*tau1(t-lambda) - R_h*h(t)
dtau1/dt = d*T1(t) - R_tau1*tau1(t)
dtau2/dt = e*h(t)  - R_tau2*tau2(t)

Three delays, all acting on the wind-stress variables:
eta    ~ 150 days : Rossby wave generated in the east, reflects at the
                     western boundary, returns as a Kelvin wave (the
                     classical Suarez-Schopf / Battisti-Hirst delay).
delta  ~ 30 days  : western Pacific wind-forced Kelvin wave reaching the
                     eastern Pacific.
lambda ~ 180 days : off-equatorial Rossby wave propagating to the far
                     western Pacific.

Parameter defaults
-------------------
The paper does not tabulate one combined parameter set for the full
4-equation system -- it gives parameters separately for reduced special
cases (Fig. 4 caption: 1-variable delayed oscillator; Fig. 6 caption:
western Pacific oscillator). The DEFAULT_PARAMS below merge those two
sets (they overlap in a, d, eps, R_tau1), covering all 10 coefficients
needed for the full model. Units in the paper are per-year rates; they
are converted to per-day internally since the integrator runs in days.
Treat these as a reasonable starting point, not a literal single-table
citation -- override anything via function arguments.
"""

import numpy as np
from tqdm import trange

DAYS_PER_YEAR = 365.25

# Parameters as given in Wang (2001), Figs. 4 and 6 captions (per-year units).
DEFAULT_PARAMS = dict(
    a=150.0,       # degC m^2 N^-1 yr^-1   (Bjerknes-type feedback, T1 <- tau1)
    b1=250.0,      # degC m^2 N^-1 yr^-1   (delayed wave-reflection feedback)
    b2=750.0,      # degC m^2 N^-1 yr^-1   (western Pacific wind-forced feedback)
    c=1500.0,      # m^3 N^-1 yr^-1        (tau1 -> h coupling)
    d=0.036,       # degC^-1 N m^-2 yr^-1  (T1 -> tau1 coupling)
    e=0.003,       # N m^-3 yr^-1          (h -> tau2 coupling)
    eps=1.2,       # degC^-2 yr^-1         (cubic damping on T1)
    R_h=5.0,       # yr^-1                 (damping of h)
    R_tau1=2.0,    # yr^-1
    R_tau2=2.0,    # yr^-1
)

DEFAULT_DELAYS_DAYS = dict(eta=150.0, delta=30.0, lam=180.0)


def simulate_unified_oscillator(
    n_days=100 * 365,
    dt=1.0,
    params=None,
    delays_days=None,
    ic=(0.5, 0.0, 0.0, 0.0),
    noise_std=0.0,
    seed=None,
):
    """
    Integrate Wang (2001)'s unified oscillator with fixed-step Euler.

    Parameters
    ----------
    n_days : total simulated length, in days.
    dt : integration step, in days. Must evenly divide all three delays
         (default delays 30/150/180 are all multiples of dt=1).
    params : dict overriding any of DEFAULT_PARAMS (per-year rates).
    delays_days : dict overriding any of DEFAULT_DELAYS_DAYS.
    ic : initial condition (T1, h, tau1, tau2), held constant as the
         "history" for t < 0.
    noise_std : if > 0, adds Gaussian white-noise forcing (Euler-Maruyama)
        to the tau1 tendency, representing unresolved atmospheric
        variability (e.g. westerly wind bursts). 0 = deterministic.
    seed : RNG seed for reproducibility.

    Returns
    -------
    t : (n_steps+1,) array, time in days.
    state : (4, n_steps+1) array, rows = [T1, h, tau1, tau2].
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    delays = {**DEFAULT_DELAYS_DAYS, **(delays_days or {})}

    # convert per-year rates to per-day
    a, b1, b2, c, d, e = (p[k] / DAYS_PER_YEAR for k in ("a", "b1", "b2", "c", "d", "e"))
    eps = p["eps"] / DAYS_PER_YEAR
    R_h, R_tau1, R_tau2 = (p[k] / DAYS_PER_YEAR for k in ("R_h", "R_tau1", "R_tau2"))

    eta_steps = int(round(delays["eta"] / dt))
    delta_steps = int(round(delays["delta"] / dt))
    lam_steps = int(round(delays["lam"] / dt))
    max_delay_steps = max(eta_steps, delta_steps, lam_steps)

    n_steps = int(round(n_days / dt))
    total_len = max_delay_steps + n_steps + 1

    T1 = np.empty(total_len)
    h = np.empty(total_len)
    tau1 = np.empty(total_len)
    tau2 = np.empty(total_len)

    # constant history for t < 0
    T1[: max_delay_steps + 1] = ic[0]
    h[: max_delay_steps + 1] = ic[1]
    tau1[: max_delay_steps + 1] = ic[2]
    tau2[: max_delay_steps + 1] = ic[3]

    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)

    for i in trange(max_delay_steps, max_delay_steps + n_steps):
        dT1 = a * tau1[i] - b1 * tau1[i - eta_steps] + b2 * tau2[i - delta_steps] - eps * T1[i] ** 3
        dh = -c * tau1[i - lam_steps] - R_h * h[i]
        dtau1 = d * T1[i] - R_tau1 * tau1[i]
        dtau2 = e * h[i] - R_tau2 * tau2[i]

        T1[i + 1] = T1[i] + dT1 * dt
        h[i + 1] = h[i] + dh * dt
        tau1[i + 1] = tau1[i] + dtau1 * dt
        if noise_std > 0:
            tau1[i + 1] += noise_std * sqrt_dt * rng.standard_normal()
        tau2[i + 1] = tau2[i] + dtau2 * dt

    # drop the constant-history warm-up, keep t=0..n_days
    T1, h, tau1, tau2 = (arr[max_delay_steps:] for arr in (T1, h, tau1, tau2))
    t = np.arange(n_steps + 1) * dt
    state = np.stack([T1, h, tau1, tau2], axis=0)
    return t, state


def subsample(t, state, interval_days, dt):
    """
    Take exact snapshots every `interval_days` (must be a multiple of dt),
    e.g. interval_days=30 for monthly points aligned with the delays.
    No averaging -- just clean subsampling, so lags stay exact integers.
    """
    step = int(round(interval_days / dt))
    return t[::step], state[:, ::step]


if __name__ == "__main__":
    # quick demo: 50 years, daily integration, subsampled to monthly
    t, state = simulate_unified_oscillator(n_days=100 * 365, dt=1.0, seed=0, noise_std=0.25)
    t_m, state_m = subsample(t, state, interval_days=30, dt=1.0)

    labels = ["T1 (Nino-3 SST, degC)", "h (Nino-6 thermocline, m)",
              "tau1 (Nino-4 wind stress)", "tau2 (Nino-5 wind stress)"]
    print(f"Daily series: {state.shape}, monthly-subsampled: {state_m.shape}")
    for lab, row in zip(labels, state_m):
        print(f"{lab}: min={row.min():.3g}, max={row.max():.3g}")

    np.save("wang_oscillator_daily.npy", state)    
    np.save("wang_oscillator_monthly.npy", state_m)
    np.save("wang_oscillator_daily_time.npy", t)    
    np.save("wang_oscillator_monthly_time.npy", t_m)