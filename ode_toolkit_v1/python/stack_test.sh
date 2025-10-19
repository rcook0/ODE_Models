//unzip ode_toolkit_v1.zip && cd ode_toolkit_v1/python
python3 - <<'PY'
import numpy as np
from ode_models import integrate, lorenz63, vanderpol, classify_linear_2x2
# Non-stiff
t,y = integrate(lorenz63(), np.array([1.,1.,1.]), 0., 20., dt=0.01, backend="auto")
print("Lorenz:", y.shape)
# Stiff-ish
t,y = integrate(vanderpol(mu=25.), np.array([1.,0.]), 0., 60., dt=0.005, backend="auto", stiff=True)
print("VdP:", y.shape)
print("TD:", classify_linear_2x2([[-1,-5],[2,-3]]))
PY
