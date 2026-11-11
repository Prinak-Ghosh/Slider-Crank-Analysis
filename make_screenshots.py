"""Save still images of the simulator window (no display needed).

Run:  python make_screenshots.py
"""
import os
import matplotlib
matplotlib.use("Agg")

import slider_crank_simulator as sim

os.makedirs("outputs", exist_ok=True)


def shot(name, rpm, theta_deg):
    sim.s_rpm.set_val(rpm)          # triggers on_change -> redraws the curves
    sim.state["theta"] = theta_deg * 3.141592653589793 / 180
    sim.state["paused"] = True      # draw exactly this angle, do not advance
    sim.update(0)
    sim.fig.savefig(os.path.join("outputs", name), dpi=130)


shot("screenshot_simulator_600rpm.png", 600, 40)
shot("screenshot_simulator_3000rpm.png", 3000, 20)
print("saved screenshots to outputs/")
