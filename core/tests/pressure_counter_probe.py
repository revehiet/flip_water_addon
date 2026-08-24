import sys
import numpy as np

sys.path.insert(0, r"C:\Users\revehiet")
from flip_water_addon import solver_bridge

core, err = solver_bridge.load()
print("load ok:", core is not None, "| err:", err)
if core is not None:
    print("core file:", getattr(core, "__file__", "?"))
if core is None:
    sys.exit(1)
s = core.FlipSolver()
st = core.SolverSettings()
st.solver_backend = core.SolverBackend.CPU
st.resolution = 24
st.max_particles = 2000000
s.init_domain(np.array([0, 0, 0], dtype=np.float32),
              np.array([2, 2, 2], dtype=np.float32), st)
s.add_particles_box(np.array([0.65, 0.65, 1.05], dtype=np.float32),
                    np.array([1.35, 1.35, 1.75], dtype=np.float32), 2, None, 1)
print("count before:", core.pressure_call_count())
s.step(1.0 / 24.0)
print("count after :", core.pressure_call_count())
