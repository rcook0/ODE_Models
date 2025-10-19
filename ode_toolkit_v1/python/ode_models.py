
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Optional
import numpy as np

_has_scipy = False
try:
    from scipy.integrate import solve_ivp  # type: ignore
    _has_scipy = True
except Exception:
    pass

_has_jax = False
try:
    import jax
    import jax.numpy as jnp
    import diffrax as dfx  # type: ignore
    _has_jax = True
except Exception:
    pass

Array = np.ndarray
RHS = Callable[[float, Array], Array]

def euler(f: RHS, y0: Array, t0: float, t1: float, dt: float):
    n = int(math.ceil((t1 - t0) / dt))
    ys = np.zeros((n+1, len(np.atleast_1d(y0))), dtype=float)
    ts = np.linspace(t0, t1, n+1)
    y = np.array(y0, dtype=float)
    ys[0] = y
    for k in range(n):
        y = y + dt * np.asarray(f(ts[k], y))
        ys[k+1] = y
    return ts, ys

def rk4(f: RHS, y0: Array, t0: float, t1: float, dt: float):
    n = int(math.ceil((t1 - t0) / dt))
    ys = np.zeros((n+1, len(np.atleast_1d(y0))), dtype=float)
    ts = np.linspace(t0, t1, n+1)
    y = np.array(y0, dtype=float)
    ys[0] = y
    for k in range(n):
        t = ts[k]
        k1 = np.asarray(f(t, y))
        k2 = np.asarray(f(t + 0.5*dt, y + 0.5*dt*k1))
        k3 = np.asarray(f(t + 0.5*dt, y + 0.5*dt*k2))
        k4 = np.asarray(f(t + dt,   y + dt*k3))
        y = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        ys[k+1] = y
    return ts, ys

def _integrate_scipy(f: RHS, y0: Array, t0: float, t1: float, dt: Optional[float], stiff: bool,
                     rtol=1e-6, atol=1e-9, method: Optional[str] = None):
    if not _has_scipy:
        raise RuntimeError("SciPy not available")
    if method is None:
        method = ("Radau" if stiff else "DOP853")
    if dt is None:
        dt = (t1 - t0)/1000.0 if t1 > t0 else 0.01
    t_eval = np.arange(t0, t1 + 1e-12, dt)
    def fun(t, y):
        return np.asarray(f(t, y))
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
        dt = (t1 - t0)/steps if t1 > t0 else 0.01
    term = dfx.ODETerm(lambda t, y, args: jnp.array(f(float(t), np.array(y, dtype=float))))
    saveat = dfx.SaveAt(ts=jnp.arange(t0, t1 + 1e-12, dt))
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0=dt, y0=jnp.array(y0, dtype=float),
                          rtol=rtol, atol=atol, saveat=saveat)
    ts = np.array(sol.ts); ys = np.array(sol.ys)
    return ts, ys

def integrate(f: RHS, y0: Array, t0: float, t1: float, dt: Optional[float] = None,
              backend: str = "auto", stiff: bool = False, rtol=1e-6, atol=1e-9, method: Optional[str]=None):
    if backend == "scipy" or (backend == "auto" and _has_scipy):
        return _integrate_scipy(f, y0, t0, t1, dt, stiff, rtol, atol, method)
    if backend == "jax" or (backend == "auto" and _has_jax):
        return _integrate_jax_diffrax(f, y0, t0, t1, dt, stiff, rtol, atol, method)
    if dt is None:
        dt = (t1 - t0)/1000.0 if t1 > t0 else 0.01
    return rk4(f, y0, t0, t1, dt)

def linear_system(A: np.ndarray) -> RHS:
    A = np.asarray(A, dtype=float)
    def f(t: float, y: Array) -> Array:
        return A @ y
    return f

def classify_linear_2x2(A: np.ndarray):
    A = np.asarray(A, dtype=float).reshape(2,2)
    tr = float(np.trace(A))
    det = float(np.linalg.det(A))
    disc = tr**2 - 4*det
    if det < 0:
        typ = "saddle"
    elif det == 0:
        typ = "degenerate (line of equilibria)"
    else:
        if disc < 0:
            typ = "spiral (focus): stable" if tr < 0 else "spiral (focus): unstable"
        elif disc == 0:
            typ = "repeated real: star/degenerate node"
        else:
            typ = "node: stable" if (tr < 0 and det > 0) else ("node: unstable" if tr > 0 else "improper/indeterminate")
    return {"trace": tr, "determinant": det, "discriminant": disc, "type": typ}

def mass_spring_damper(m=1.0, c=0.2, k=1.0, forcing=None):
    if forcing is None:
        forcing = lambda t: 0.0
    def f(t, y):
        x, v = y
        return np.array([v, (forcing(t) - c*v - k*x)/m])
    return f

def rlc_series(R=1.0, L=1.0, C=1.0, E=None):
    if E is None:
        E = lambda t: 0.0
    def f(t, y):
        q, i = y
        return np.array([i, (E(t) - R*i - q/C)/L])
    return f

def pendulum_simple(g=9.81, L=1.0, damping=0.0, drive_A=0.0, drive_omega=1.0):
    def f(t, y):
        th, w = y
        return np.array([w, - (g/L)*math.sin(th) - damping*w + drive_A*math.sin(drive_omega*t)])
    return f

def double_pendulum(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
    def f(t, y):
        th1, w1, th2, w2 = y
        d = th2 - th1
        den1 = (m1 + m2)*L1 - m2*L1*math.cos(d)*math.cos(d)
        den2 = (L2/L1)*den1
        a1 = (m2*L1*w1*w1*math.sin(d)*math.cos(d)
              + m2*g*math.sin(th2)*math.cos(d)
              + m2*L2*w2*w2*math.sin(d)
              - (m1+m2)*g*math.sin(th1)) / den1
        a2 = (-m2*L2*w2*w2*math.sin(d)*math.cos(d)
              + (m1+m2)*(g*math.sin(th1)*math.cos(d) - L1*w1*w1*math.sin(d) - g*math.sin(th2))) / den2
        return np.array([w1, a1, w2, a2])
    return f

def duffing(delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2):
    def f(t, y):
        x, v = y
        return np.array([v, -delta*v - alpha*x - beta*x**3 + gamma*math.cos(omega*t)])
    return f

def vanderpol(mu=3.0):
    def f(t, y):
        x, v = y
        return np.array([v, mu*(1 - x*x)*v - x])
    return f

def lorenz63(sigma=10.0, rho=28.0, beta=8.0/3.0):
    def f(t, y):
        x, yy, z = y
        return np.array([sigma*(yy - x), x*(rho - z) - yy, x*yy - beta*z])
    return f

def lotka_volterra_pred_prey(alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
    def f(t, y):
        x, p = y
        return np.array([alpha*x - beta*x*p, delta*x*p - gamma*p])
    return f

def competitive_lv(r1=1.0, r2=0.8, K1=1.0, K2=1.0, a12=0.5, a21=0.6):
    def f(t, y):
        x, yy = y
        return np.array([r1*x*(1.0 - (x + a12*yy)/K1),
                         r2*yy*(1.0 - (yy + a21*x)/K2)])
    return f

def oregonator(s=77.27, q=8.375e-6, f=0.161):
    def f_rhs(t, y):
        x, yv, z = y
        return np.array([ s*(y + x*(1 - q - x) - f*x),
                          ( -y - x*(1 - q - x) )/s,
                          f*(x - z) ])
    return f_rhs

@dataclass
class GliderParams:
    m: float = 5.0
    S: float = 0.5
    rho: float = 1.225
    g: float = 9.81
    CL_alpha: float = 5.0
    CD0: float = 0.02
    k: float = 0.05
    alpha_cmd: float = 5.0*math.pi/180

def glider2d(params: GliderParams):
    m,S,rho,g = params.m, params.S, params.rho, params.g
    CL_a, CD0, k = params.CL_alpha, params.CD0, params.k
    def f(t, y):
        X,Z, v, gam = y
        alpha = params.alpha_cmd
        CL = CL_a*alpha
        CD = CD0 + k*CL*CL
        q = 0.5*rho*v*v
        L = q*S*CL
        D = q*S*CD
        vdot   = (-D/m) - g*math.sin(gam)
        gamdot = (L/(m*v + 1e-9)) - (g*math.cos(gam)/(v + 1e-9))
        Xdot = v*math.cos(gam)
        Zdot = v*math.sin(gam)
        return np.array([Xdot, Zdot, vdot, gamdot])
    return f
