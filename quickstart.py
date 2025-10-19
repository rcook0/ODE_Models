# Lorenz attractor quick run (inside a notebook/REPL)
from ode_models import rk4, lorenz63
import numpy as np
t, y = rk4(lorenz63(), y0=np.array([1.,1.,1.]), t0=0., t1=10., dt=0.01)

# Van der Pol phase portrait
from ode_models import vanderpol
t, y = rk4(vanderpol(mu=3.0), y0=np.array([1.,0.]), t0=0., t1=20., dt=0.01)

# TD-plane classification of a 2×2 linear system
from ode_models import classify_linear_2x2
info = classify_linear_2x2([[ -1., -5.],[ 2., -3.]])  # returns trace, det, discriminant, type
