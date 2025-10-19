import numpy as np
from ode_models import integrate, vanderpol
ts, ys = integrate(vanderpol(mu=3.0), np.array([1.,0.]), 0., 20., dt=0.01, backend='jax')
print(ts.shape, ys.shape)
