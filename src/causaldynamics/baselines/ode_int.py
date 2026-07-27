"""
custom_odeint.py
================
 
A minimal, from-scratch PyTorch reimplementation of torchdiffeq.odeint(),
written to show exactly what that function does internally.
 
What odeint(func, y0, t, method=...) actually does
----------------------------------------------------
Given:
    func(t, y) -> dy/dt      a function returning the instantaneous derivative
    y0                       the state at time t[0]
    t                        a 1-D tensor of times [t0, t1, ..., tN]
 
it numerically solves the initial value problem
 
    dy/dt = func(t, y),   y(t0) = y0
 
and returns y evaluated at every time in `t`. Mechanically it:
 
  1. Starts at y(t0) = y0.
  2. Walks forward through the requested times t0 -> t1 -> t2 -> ... -> tN.
  3. Between each pair of consecutive requested times (t_i, t_{i+1}), it
     advances the state using a single-step update rule built from one or
     more evaluations of `func` (a "Runge-Kutta step"). For FIXED-STEP
     methods (euler, midpoint, rk4) this interval can optionally be
     subdivided into smaller internal steps (`step_size`/`options`) for
     better accuracy. For ADAPTIVE methods (dopri5) the solver instead
     picks its own internal step sizes, shrinking or growing them based on
     an estimate of the local truncation error compared to `rtol`/`atol`,
     and lands exactly on t_{i+1}.
  4. Appends the resulting y(t_{i+1}) to the output and repeats until all
     requested times are covered.
  5. Stacks all the y(t_i) into a single tensor of shape (len(t), *y0.shape).
 
Because every step is just ordinary tensor arithmetic built out of calls to
`func` (which is itself a normal nn.Module / autograd-tracked function),
PyTorch's autograd can differentiate straight through the whole loop -- this
is "discretize-then-optimize" backprop (what we implement below). The
production torchdiffeq library *additionally* offers `odeint_adjoint`, which
instead solves a second, backward-in-time ODE for the gradients (the
"adjoint sensitivity method"). That uses O(1) memory instead of O(num steps),
but isn't needed to understand what odeint is doing -- it's a memory
optimization, not a different forward solve.
"""
 
import torch
 
 
# ---------------------------------------------------------------------------
# 1. Fixed-step solvers: each takes one step of size `dt` from (t, y).
#    These differ only in how many times they sample `func` and how they
#    combine the samples -- i.e. their Butcher tableau.
# ---------------------------------------------------------------------------
 
def _euler_step(func, t, dt, y):
    """Forward Euler. 1 function evaluation, locally O(dt^2) accurate."""
    return y + dt * func(t, y)
 
 
def _midpoint_step(func, t, dt, y):
    """Explicit midpoint (RK2). 2 function evaluations, O(dt^3) accurate."""
    k1 = func(t, y)
    k2 = func(t + dt / 2, y + dt / 2 * k1)
    return y + dt * k2
 
 
def _rk4_step(func, t, dt, y):
    """
    Classic 4th-order Runge-Kutta. 4 function evaluations, O(dt^5) accurate
    per step. This is the default "Runge-Kutta" method.
    """
    k1 = func(t, y)
    k2 = func(t + dt / 2, y + dt / 2 * k1)
    k3 = func(t + dt / 2, y + dt / 2 * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
 
 
_FIXED_STEP_SOLVERS = {
    "euler": _euler_step,
    "midpoint": _midpoint_step,
    "rk4": _rk4_step,
}
 
 
# ---------------------------------------------------------------------------
# 2. Adaptive solver: Dormand-Prince RK45 (same family as torchdiffeq's
#    default 'dopri5'). It evaluates func 7 times per step and combines
#    them in two different ways -- a 5th-order estimate (used as the actual
#    step) and an embedded 4th-order estimate (used only to measure error).
#    The gap between the two tells the solver how big its next step can be.
# ---------------------------------------------------------------------------
 
_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]       # 5th order weights
_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200,
       187 / 2100, 1 / 40]                                                  # 4th order weights
 
 
def _dopri5_step(func, t, dt, y):
    """One Dormand-Prince step. Returns (y5, y4): 5th- and 4th-order estimates."""
    ks = []
    for i in range(7):
        ti = t + _C[i] * dt
        yi = y
        for j, a in enumerate(_A[i]):
            if a != 0:
                yi = yi + dt * a * ks[j]
        ks.append(func(ti, yi))
    y5 = y + dt * sum(b * k for b, k in zip(_B5, ks) if b != 0)
    y4 = y + dt * sum(b * k for b, k in zip(_B4, ks) if b != 0)
    return y5, y4
 
 
def _error_ratio(y5, y4, y_prev, rtol, atol):
    """How many multiples of the allowed tolerance the step's error spans."""
    scale = atol + rtol * torch.maximum(y_prev.abs(), y5.abs())
    err = (y5 - y4) / scale
    return err.detach().pow(2).mean().sqrt().item()  # RMS error ratio, detached: never backprop through step control
 
 
def _dopri5_integrate(func, y0, t0, t1, rtol, atol):
    """Adaptively integrate from t0 to t1 (a single requested sub-interval)."""
    direction = 1.0 if t1 >= t0 else -1.0
    t = t0
    y = y0
    span = (t1 - t0).abs()
    dt = direction * (span * 0.05 if span > 0 else torch.tensor(1e-3))
    if dt == 0:
        return y
 
    max_iters = 10000  # safety net against runaway loops
    for _ in range(max_iters):
        if (direction > 0 and t >= t1) or (direction < 0 and t <= t1):
            break
        # don't overshoot the target time
        if (direction > 0 and t + dt > t1) or (direction < 0 and t + dt < t1):
            dt = t1 - t
 
        y5, y4 = _dopri5_step(func, t, dt, y)
        err_ratio = _error_ratio(y5, y4, y, rtol, atol)
 
        if err_ratio <= 1.0:           # step accepted
            t = t + dt
            y = y5
 
        # PI-style step size controller (standard safety factor 0.9, order 5)
        factor = 0.9 * (1.0 / max(err_ratio, 1e-10)) ** (1.0 / 5.0)
        factor = min(max(factor, 0.2), 5.0)
        dt = dt * factor
 
    return y
 
 
# ---------------------------------------------------------------------------
# 3. The public odeint() function -- mirrors torchdiffeq.odeint's signature.
# ---------------------------------------------------------------------------
 
def odeint(func, y0, t, method="rk4", rtol=1e-7, atol=1e-9, options=None):
    """
    Solve dy/dt = func(t, y), y(t[0]) = y0, and return y at every time in `t`.
 
    Parameters
    ----------
    func : callable(t, y) -> dy/dt
        t is a 0-D tensor (scalar time), y has shape (..., state_dim).
    y0 : tensor, shape (..., state_dim)
        Initial condition, value of y at t[0].
    t : 1-D tensor, shape (T,)
        Monotonically increasing (or decreasing) times to evaluate y at.
        t[0] is treated as the initial time (no integration happens for it).
    method : {'euler', 'midpoint', 'rk4', 'dopri5'}
        'euler'/'midpoint'/'rk4' are fixed-step explicit Runge-Kutta methods.
        'dopri5' is the adaptive Dormand-Prince Runge-Kutta 4(5) method.
    rtol, atol : float
        Relative / absolute error tolerances, used only by 'dopri5'.
    options : dict, optional
        For fixed-step methods, options={'step_size': h} subdivides each
        (t_i, t_{i+1}) interval into steps no larger than h. If omitted,
        a single step is taken directly from t_i to t_{i+1}.
 
    Returns
    -------
    Tensor of shape (T, *y0.shape): the solution evaluated at each time in t.
    """
    if method not in (*_FIXED_STEP_SOLVERS, "dopri5"):
        raise ValueError(f"Unknown method '{method}'")
 
    step_size = (options or {}).get("step_size", None)
 
    ys = [y0]
    y = y0
    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i + 1]
 
        if method == "dopri5":
            y = _dopri5_integrate(func, y, t0, t1, rtol, atol)
        else:
            step_fn = _FIXED_STEP_SOLVERS[method]
            dt_full = t1 - t0
            n_substeps = 1
            if step_size is not None:
                n_substeps = max(1, int(torch.ceil(dt_full.abs() / step_size).item()))
            dt = dt_full / n_substeps
            tt = t0
            for _ in range(n_substeps):
                y = step_fn(func, tt, dt, y)
                tt = tt + dt
 
        ys.append(y)
 
    return torch.stack(ys, dim=0)
 
 