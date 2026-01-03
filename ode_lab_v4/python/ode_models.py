
from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List

import numpy as np

_has_scipy = False
try:
    from scipy.integrate import solve_ivp  # type: ignore
    _has_scipy = True
except Exception:
    pass

_has_jax = False
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    import diffrax as dfx  # type: ignore # noqa: F401
    _has_jax = True
except Exception:
    pass

Array = np.ndarray
RHS = Callable[[float, Array], Array]

# ---------------- Integrators ----------------

def euler(f: RHS, y0: Array, t0: float, t1: float, dt: float):
    n = int(math.ceil((t1 - t0) / dt))
    ys = np.zeros((n + 1, len(np.atleast_1d(y0))), dtype=float)
    ts = np.linspace(t0, t1, n + 1)
    y = np.array(y0, dtype=float)
    ys[0] = y
    for k in range(n):
        y = y + dt * np.asarray(f(ts[k], y))
        ys[k + 1] = y
    return ts, ys

def rk4(f: RHS, y0: Array, t0: float, t1: float, dt: float):
    n = int(math.ceil((t1 - t0) / dt))
    ys = np.zeros((n + 1, len(np.atleast_1d(y0))), dtype=float)
    ts = np.linspace(t0, t1, n + 1)
    y = np.array(y0, dtype=float)
    ys[0] = y
    for k in range(n):
        t = ts[k]
        k1 = np.asarray(f(t, y))
        k2 = np.asarray(f(t + 0.5 * dt, y + 0.5 * dt * k1))
        k3 = np.asarray(f(t + 0.5 * dt, y + 0.5 * dt * k2))
        k4 = np.asarray(f(t + dt, y + dt * k3))
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[k + 1] = y
    return ts, ys

def _integrate_scipy(f: RHS, y0: Array, t0: float, t1: float, dt: Optional[float], stiff: bool,
                     rtol=1e-6, atol=1e-9, method: Optional[str] = None):
    if not _has_scipy:
        raise RuntimeError("SciPy not available")
    if method is None:
        method = ("Radau" if stiff else "DOP853")
    if dt is None:
        dt = (t1 - t0) / 1000.0 if t1 > t0 else 0.01
    t_eval = np.arange(t0, t1 + 1e-12, dt)

    def fun(t, y):
        return np.asarray(f(float(t), y))

    sol = solve_ivp(fun, (t0, t1), np.asarray(y0, dtype=float), method=method,
                    t_eval=t_eval, rtol=rtol, atol=atol, vectorized=False)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y.T

def _integrate_jax_diffrax(f: RHS, y0: Array, t0: float, t1: float, dt: Optional[float], stiff: bool,
                           rtol=1e-6, atol=1e-9, method: Optional[str] = None):
    if not _has_jax:
        raise RuntimeError("JAX/Diffrax not available")
    import jax.numpy as jnp
    import diffrax as dfx

    if method is None:
        solver = dfx.Kvaerno5() if stiff else dfx.Dopri5()
    else:
        solver = {"dopri5": dfx.Dopri5(), "tsit5": dfx.Tsit5(), "kvaerno5": dfx.Kvaerno5()}.get(method.lower(), dfx.Dopri5())

    if dt is None:
        steps = 1000
        dt = (t1 - t0) / steps if t1 > t0 else 0.01

    term = dfx.ODETerm(lambda t, y, args: jnp.array(f(float(t), np.array(y, dtype=float))))
    saveat = dfx.SaveAt(ts=jnp.arange(t0, t1 + 1e-12, dt))
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0=dt, y0=jnp.array(y0, dtype=float),
                          rtol=rtol, atol=atol, saveat=saveat)
    ts = np.array(sol.ts)
    ys = np.array(sol.ys)
    return ts, ys

def integrate(f: RHS, y0: Array, t0: float, t1: float, dt: Optional[float] = None,
              backend: str = "auto", stiff: bool = False, rtol=1e-6, atol=1e-9, method: Optional[str] = None):
    """
    Unified integration entrypoint.

    backend:
      - "auto": SciPy if available else JAX/Diffrax else RK4
      - "scipy": solve_ivp
      - "jax": Diffrax
      - "rk4": fixed step RK4
    """
    if backend == "scipy" or (backend == "auto" and _has_scipy):
        return _integrate_scipy(f, y0, t0, t1, dt, stiff, rtol, atol, method)
    if backend == "jax" or (backend == "auto" and _has_jax):
        return _integrate_jax_diffrax(f, y0, t0, t1, dt, stiff, rtol, atol, method)
    if backend == "rk4" or backend == "auto":
        if dt is None:
            dt = (t1 - t0) / 1000.0 if t1 > t0 else 0.01
        return rk4(f, y0, t0, t1, dt)
    raise ValueError(f"Unknown backend: {backend}")

# ---------------- Models (RHS factories) ----------------

def linear_system(A: np.ndarray) -> RHS:
    A = np.asarray(A, dtype=float)
    def f(t: float, y: Array) -> Array:
        return A @ y
    return f

def mass_spring_damper(m=1.0, c=0.2, k=1.0, forcing=None) -> RHS:
    if forcing is None:
        forcing = lambda t: 0.0
    def f(t, y):
        x, v = y
        return np.array([v, (forcing(t) - c*v - k*x)/m])
    return f

def rlc_series(R=1.0, L=1.0, C=1.0, E=None) -> RHS:
    if E is None:
        E = lambda t: 0.0
    def f(t, y):
        q, i = y
        return np.array([i, (E(t) - R*i - q/C)/L])
    return f

def pendulum_simple(g=9.81, L=1.0, damping=0.0, drive_A=0.0, drive_omega=1.0) -> RHS:
    def f(t, y):
        th, w = y
        return np.array([w, - (g/L)*math.sin(th) - damping*w + drive_A*math.sin(drive_omega*t)])
    return f

def duffing(delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2) -> RHS:
    def f(t, y):
        x, v = y
        return np.array([v, -delta*v - alpha*x - beta*x**3 + gamma*math.cos(omega*t)])
    return f

def vanderpol(mu=3.0) -> RHS:
    def f(t, y):
        x, v = y
        return np.array([v, mu*(1 - x*x)*v - x])
    return f

def lorenz63(sigma=10.0, rho=28.0, beta=8.0/3.0) -> RHS:
    def f(t, y):
        x, yy, z = y
        return np.array([sigma*(yy - x), x*(rho - z) - yy, x*yy - beta*z])
    return f

def rossler(a=0.2, b=0.2, c=5.7) -> RHS:
    def f(t, y):
        x, yy, z = y
        return np.array([-yy - z, x + a*yy, b + z*(x - c)])
    return f

def lotka_volterra_pred_prey(alpha=1.0, beta=0.1, delta=0.075, gamma=1.5) -> RHS:
    def f(t, y):
        x, p = y
        return np.array([alpha*x - beta*x*p, delta*x*p - gamma*p])
    return f

def competitive_lv(r1=1.0, r2=0.8, K1=1.0, K2=1.0, a12=0.5, a21=0.6) -> RHS:
    def f(t, y):
        x, yy = y
        return np.array([r1*x*(1.0 - (x + a12*yy)/K1),
                         r2*yy*(1.0 - (yy + a21*x)/K2)])
    return f

def sir(beta=0.5, gamma=0.2) -> RHS:
    def f(t, y):
        S, I, R = y
        N = S + I + R
        dS = -beta * S * I / (N + 1e-12)
        dI = beta * S * I / (N + 1e-12) - gamma * I
        dR = gamma * I
        return np.array([dS, dI, dR])
    return f

# ---------------- Registry + facade ----------------

def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")

MODEL_REGISTRY = {
    "chua":          lambda **p: chua(**p),
    "thomas":        lambda **p: thomas(**p),
    "chen":          lambda **p: chen(**p),
    "halvorsen":     lambda **p: halvorsen(**p),
    "hindmarshrose": lambda **p: hindmarsh_rose(**p),
    "robertson":     lambda **p: robertson(),

    "lorenz":        lambda **p: lorenz63(**p),
    "lorenz63":      lambda **p: lorenz63(**p),
    "rossler":       lambda **p: rossler(**p),
    "vanderpol":     lambda **p: vanderpol(**p),
    "duffing":       lambda **p: duffing(**p),
    "predatorprey":  lambda **p: lotka_volterra_pred_prey(**p),
    "competingspecies": lambda **p: competitive_lv(**p),
    "massspring":    lambda **p: mass_spring_damper(**p),
    "rlc":           lambda **p: rlc_series(**p),
    "pendulum":      lambda **p: pendulum_simple(**p),
    "sir":           lambda **p: sir(**p),
    "linear2x2":     lambda **p: linear_system(np.array(p["A"], dtype=float)),
}

def list_models() -> List[str]:
    return sorted(set(MODEL_REGISTRY.keys()) | {"linear2x2"})

def get_model(name: str, **params) -> RHS:
    key = _norm(name)
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name!r} (normalized: {key})")
    return MODEL_REGISTRY[key](**params)

def integrate_model(name: str,
                    y0,
                    t0: float,
                    t1: float,
                    dt: float | None = None,
                    backend: str = "auto",
                    stiff: bool = False,
                    rtol: float = 1e-6,
                    atol: float = 1e-9,
                    method: str | None = None,
                    **params):
    f = get_model(name, **params)
    y0_arr = np.array(y0, dtype=float)
    return integrate(f, y0_arr, t0, t1, dt=dt, backend=backend, stiff=stiff, rtol=rtol, atol=atol, method=method)

# ---------------- Presets (JSON) ----------------

def load_presets(path: str | Path) -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text())
    if isinstance(data, dict) and "name" in data:
        return {data["name"]: data}
    out: Dict[str, Dict[str, Any]] = {}
    for preset in data:
        out[preset["name"]] = preset
    return out

def run_preset(preset: Dict[str, Any]):
    return integrate_model(
        preset["model"],
        y0=preset["y0"],
        t0=preset["tspan"][0],
        t1=preset["tspan"][1],
        dt=preset.get("dt"),
        backend=preset.get("backend", "auto"),
        stiff=preset.get("stiff", False),
        **preset.get("params", {}),
    )


# ---------------- Advanced / "chaos zoo" models ----------------

def chua(alpha=15.6, beta=28.0, m0=-1.143, m1=-0.714) -> RHS:
    """
    Chua's circuit (dimensionless). State: (x,y,z).
    Piecewise-linear nonlinearity: f(x) = m1*x + 0.5*(m0-m1)*(|x+1|-|x-1|)
    """
    def f(t, y):
        x, yy, z = y
        fx = m1*x + 0.5*(m0-m1)*(abs(x+1.0) - abs(x-1.0))
        dx = alpha*(yy - x - fx)
        dy = x - yy + z
        dz = -beta*yy
        return np.array([dx, dy, dz])
    return f

def thomas(a=0.208186) -> RHS:
    """Thomas' cyclically symmetric attractor. State: (x,y,z)."""
    def f(t, y):
        x, yy, z = y
        return np.array([math.sin(yy) - a*x, math.sin(z) - a*yy, math.sin(x) - a*z])
    return f

def chen(a=35.0, b=3.0, c=28.0) -> RHS:
    """Chen system. State: (x,y,z)."""
    def f(t, y):
        x, yy, z = y
        return np.array([a*(yy - x), (c-a)*x - x*z + c*yy, x*yy - b*z])
    return f

def halvorsen(a=1.4) -> RHS:
    """Halvorsen attractor. State: (x,y,z)."""
    def f(t, y):
        x, yy, z = y
        return np.array([-a*x - 4*yy - 4*z - yy*yy,
                         -a*yy - 4*z - 4*x - z*z,
                         -a*z - 4*x - 4*yy - x*x])
    return f

def hindmarsh_rose(a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4.0, xR=-1.6, I=3.25) -> RHS:
    """
    Hindmarsh–Rose neuron model (3D). State: (x,y,z).
    """
    def f(t, y):
        x, yy, z = y
        dx = yy - a*x**3 + b*x**2 - z + I
        dy = c - d*x**2 - yy
        dz = r*(s*(x - xR) - z)
        return np.array([dx, dy, dz])
    return f

# ---------------- Classic stiff benchmark: Robertson problem ----------------

def robertson() -> RHS:
    """
    Robertson (stiff) chemical kinetics (3D).
    y1' = -0.04 y1 + 1e4 y2 y3
    y2' = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
    y3' = 3e7 y2^2
    Typically y0 = [1,0,0], t in [0, 1e5]
    """
    def f(t, y):
        y1, y2, y3 = y
        dy1 = -0.04*y1 + 1.0e4*y2*y3
        dy2 =  0.04*y1 - 1.0e4*y2*y3 - 3.0e7*y2*y2
        dy3 =  3.0e7*y2*y2
        return np.array([dy1, dy2, dy3])
    return f
