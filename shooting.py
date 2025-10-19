# Hit x(T)=0 with a damped mass–spring by tuning initial velocity
from ode_models import mass_spring_damper, shoot_for_target, np
f = mass_spring_damper(m=1.0, c=0.2, k=1.0)
best_v0, ts, ys = shoot_for_target(f, y0=np.array([1.0, 0.0]), t0=0.0, t1=10.0, dt=0.01,
                                   vary_index=1, vary_value=0.0, target_index=0, target_value=0.0)
print(best_v0)  # initial velocity that lands x(10)≈0
