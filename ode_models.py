# Create a compact, dependency-free ODE playground with many classic models.
# We'll save it as /mnt/data/ode_models.py and include a small demo script.

from textwrap import dedent

code = dedent(r"""
# ode_models.py
# Lightweight ODE playground: classic right-hand-sides + simple integrators (Euler, RK4).
# No external deps beyond numpy & math (matplotlib optional for plotting).
# Usage examples are at bottom under __main__.
#
# Included RHS models:
# - Euler demo wrappers (1D & systems)
# - Linear system: x' = A x  (2x2 general, N×N also works)
# - Mass–spring–damper
# - RLC (series) circuit
# - Simple pendulum (damped, optionally driven)
# - Double pendulum (planar)
# - Duffing oscillator (damped, driven)
# - Van der Pol oscillator
# - Lorenz 1963
# - Lotka–Volterra predator–prey
# - Competitive Lotka–Volterra (two species)
# - Oregonator (BZ reaction surrogate)
# - 2D Glider point-mass model with lift/drag (simplified)
# Utilities:
# - RK4/Euler solvers
# - Phase line sampling for 1D y' = f(y)
# - Linear 2×2 classification via Trace–Determinant (TD) plane
# - Simple “target practice” shooting for BVP-like hits
#
# (c) 2025. MIT License.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, List, Optional, Dict
import numpy as np

Array = np.ndarray
RHS = Callable[[float, Array], Array]

# -----------------------------
# Generic time integrators
# -----------------------------

def euler(f: RHS, y0: Array, t0: float, t1: float, dt: float) -> Tuple[Array, Array]:
    """Fixed-step forward Euler for y' = f(t,y)."""
    n = int(math.ceil((t1 - t0) / dt))
    ys = np.zeros((n+1, len(np.atleast_1d(y0))), dtype=float)
    ts = np.linspace(t0, t1, n+1)
    y = np.array(y0, dtype=float)
    ys[0] = y
    for k in range(n):
        y = y + dt * np.asarray(f(ts[k], y))
        ys[k+1] = y
    return ts, ys

def rk4(f: RHS, y0: Array, t0: float, t1: float, dt: float) -> Tuple[Array, Array]:
    """Fixed-step classical Runge–Kutta 4 for y' = f(t,y)."""
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

# -----------------------------
# Linear systems
# -----------------------------

def linear_system(A: Array) -> RHS:
    """Return f(t,y)=A y for constant matrix A (any shape compatible with y)."""
    A = np.asarray(A, dtype=float)
    def f(t: float, y: Array) -> Array:
        return A @ y
    return f

def classify_linear_2x2(A: Array) -> Dict[str, float | str]:
    """Trace–Determinant classification for 2×2 A."""
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
            # real eigenvalues
            typ = "node: stable" if (tr < 0 and det > 0) else ("node: unstable" if tr > 0 else "improper/indeterminate")
    return {"trace": tr, "determinant": det, "discriminant": disc, "type": typ}

# -----------------------------
# Mechanics / circuits
# -----------------------------

def mass_spring_damper(m=1.0, c=0.2, k=1.0, forcing: Callable[[float], float] | None = None) -> RHS:
    """y = [x, v].  x'' + (c/m) x' + (k/m) x = (1/m)F(t)."""
    if forcing is None:
        forcing = lambda t: 0.0
    def f(t, y):
        x, v = y
        return np.array([v, (forcing(t) - c*v - k*x)/m])
    return f

def rlc_series(R=1.0, L=1.0, C=1.0, E: Callable[[float], float] | None = None) -> RHS:
    """Series RLC in charge–current state: y=[q,i].  q' = i,  i' = (E(t) - R i - q/C)/L"""
    if E is None:
        E = lambda t: 0.0
    def f(t, y):
        q, i = y
        return np.array([i, (E(t) - R*i - q/C)/L])
    return f

def pendulum_simple(g=9.81, L=1.0, damping=0.0, drive_A=0.0, drive_omega=1.0) -> RHS:
    """Planar pendulum with optional viscous damping and sinusoidal drive. y=[theta, theta_dot]."""
    def f(t, y):
        th, w = y
        return np.array([w, - (g/L)*math.sin(th) - damping*w + drive_A*math.sin(drive_omega*t)])
    return f

def double_pendulum(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81) -> RHS:
    """Planar double pendulum (point masses, massless rods). y=[th1,w1, th2,w2]."""
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

# -----------------------------
# Canonical nonlinear oscillators
# -----------------------------

def duffing(delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2) -> RHS:
    """Duffing: x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t). y=[x,v]."""
    def f(t, y):
        x, v = y
        return np.array([v, -delta*v - alpha*x - beta*x**3 + gamma*math.cos(omega*t)])
    return f

def vanderpol(mu=3.0) -> RHS:
    """Van der Pol: x'' - mu (1 - x^2) x' + x = 0.  y=[x,v]."""
    def f(t, y):
        x, v = y
        return np.array([v, mu*(1 - x*x)*v - x])
    return f

def lorenz63(sigma=10.0, rho=28.0, beta=8.0/3.0) -> RHS:
    """Lorenz 1963: x' = sigma(y-x); y' = x(rho - z) - y; z' = x y - beta z."""
    def f(t, y):
        x, yy, z = y
        return np.array([sigma*(yy - x), x*(rho - z) - yy, x*yy - beta*z])
    return f

# -----------------------------
# Population dynamics
# -----------------------------

def lotka_volterra_pred_prey(alpha=1.1, beta=0.4, delta=0.1, gamma=0.4) -> RHS:
    """Prey x, predator y: x' = alpha x - beta x y;  y' = delta x y - gamma y."""
    def f(t, y):
        x, p = y
        return np.array([alpha*x - beta*x*p, delta*x*p - gamma*p])
    return f

def competitive_lv(r1=1.0, r2=0.8, K1=1.0, K2=1.0, a12=0.5, a21=0.6) -> RHS:
    """Two-species competition with logistic terms.
       x' = r1 x (1 - (x + a12 y)/K1),  y' = r2 y (1 - (y + a21 x)/K2)."""
    def f(t, y):
        x, yy = y
        return np.array([r1*x*(1.0 - (x + a12*yy)/K1),
                         r2*yy*(1.0 - (yy + a21*x)/K2)])
    return f

# -----------------------------
# Chemical oscillator (Oregonator form)
# -----------------------------

def oregonator(s=77.27, q=8.375e-6, f=0.161) -> RHS:
    """Dimensionless Oregonator (Field–Körös–Noyes). y=[x,y,z]."""
    def f_rhs(t, y):
        x, yv, z = y
        return np.array([ s*(y + x*(1 - q - x) - f*x),
                          ( -y - x*(1 - q - x) )/s,
                          f*(x - z) ])
    return f_rhs

# -----------------------------
# Simplified 2D glider point-mass model
# -----------------------------

@dataclass
class GliderParams:
    m: float = 5.0          # kg
    S: float = 0.5          # m^2 (wing area)
    rho: float = 1.225      # air density
    g: float = 9.81
    CL_alpha: float = 5.0   # lift slope per rad
    CD0: float = 0.02
    k: float = 0.05         # induced-drag factor
    alpha_cmd: float = 5.0*math.pi/180  # command (rad), constant by default

def glider2d(params: GliderParams) -> RHS:
    """State y=[x, z, v, gamma]; level air, small-angle 2D model, control = alpha (AoA)."""
    m,S,rho,g = params.m, params.S, params.rho, params.g
    CL_a, CD0, k = params.CL_alpha, params.CD0, params.k
    def f(t, y):
        X,Z, v, gam = y  # Z up is positive
        alpha = params.alpha_cmd  # could be made time-varying
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

# -----------------------------
# Phase-line utility for 1D ODE y' = f(y)
# -----------------------------

def phase_line_1d(fy: Callable[[float], float], y_min: float, y_max: float, N: int = 400) -> Dict[str, Array]:
    ys = np.linspace(y_min, y_max, N)
    vals = np.array([fy(y) for y in ys])
    zeros = ys[np.isclose(vals, 0.0, atol=1e-8)]
    signs = np.sign(vals)
    return {"y": ys, "f": vals, "zeros": zeros, "sign": signs}

# -----------------------------
# Target practice (simple shooting): adjust parameter to hit target
# -----------------------------

def shoot_for_target(f: RHS, y0: Array, t0: float, t1: float, dt: float,
                     vary_index: int, vary_value: float,
                     target_index: int, target_value: float,
                     tol: float = 1e-3, max_iter: int = 25) -> Tuple[float, Array, Array]:
    """
    Basic secant shooting: vary one component of y0 to make y[target_index](t1) ~= target_value.
    Returns (optimal_initial_component, ts, ys).
    """
    def run_with(val):
        y0_mod = np.array(y0, dtype=float)
        y0_mod[vary_index] = val
        ts, ys = rk4(f, y0_mod, t0, t1, dt)
        return ts, ys, ys[-1, target_index]

    a = vary_value
    b = vary_value * (1.0 + 0.1 if vary_value != 0 else 0.1)
    ts, ys, Fa = run_with(a)
    _, _, Fb = run_with(b)

    for _ in range(max_iter):
        if abs(Fb - Fa) < 1e-12:
            break
        c = b + (target_value - Fb) * (b - a) / (Fb - Fa)
        _, _, Fc = run_with(c)
        if abs(Fc - target_value) < tol:
            tc, yc, _ = run_with(c)
            return c, tc, yc
        a, Fa = b, Fb
        b, Fb = c, Fc
    # fallback
    return b, ts, ys

# -----------------------------
# Helpers for vector fields (phase portraits)
# -----------------------------

def vector_field_2d(f: RHS, xlim: Tuple[float,float], ylim: Tuple[float,float], nx=25, ny=25, t=0.0):
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(ny):
        for j in range(nx):
            dy = f(t, np.array([X[i,j], Y[i,j]]))
            U[i,j], V[i,j] = dy[0], dy[1]
    return X, Y, U, V

# -----------------------------
# Minimal demo
# -----------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1) Lorenz
    f = lorenz63()
    t, y = rk4(f, y0=np.array([1.0, 1.0, 1.0]), t0=0.0, t1=30.0, dt=0.01)
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(y[:,0], y[:,1], y[:,2], lw=0.5)
    ax.set_title("Lorenz63 (σ=10, ρ=28, β=8/3)")
    plt.tight_layout()
    plt.show()

    # 2) Van der Pol
    f = vanderpol(mu=3.0)
    t, y = rk4(f, y0=np.array([1.0, 0.0]), t0=0.0, t1=40.0, dt=0.01)
    plt.figure()
    plt.plot(y[:,0], y[:,1])
    plt.title("Van der Pol phase portrait (μ=3)")
    plt.xlabel("x"); plt.ylabel("x'")
    plt.tight_layout()
    plt.show()

    # 3) TD-plane classify
    A = np.array([[ -1.0, -5.0],
                  [  2.0,  -3.0]])
    info = classify_linear_2x2(A)
    print("TD-plane:", info)
""")

with open("/mnt/data/ode_models.py", "w") as f:
    f.write(code)

print("Wrote /mnt/data/ode_models.py")
