import numpy as np
from ode_models import integrate, lorenz63
ts, ys = integrate(lorenz63(), np.array([1.,1.,1.]), 0., 20., dt=0.01, backend='auto')
print(ts.shape, ys.shape)
