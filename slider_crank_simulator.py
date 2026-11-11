"""
Slider-Crank Mechanism Simulator
--------------------------------
Mechanical concept : kinematics + dynamics of the mechanism used in IC engines
                     and compressors.
Python tools       : NumPy, Matplotlib (animation, sliders, button)

What it shows
  - Live animation of crank, connecting rod and piston
  - Piston position, velocity and acceleration vs crank angle
    (exact equations, from slider_crank_core.py)
  - Reciprocating inertia force, connecting-rod force and side thrust
    on the cylinder wall, for the inertia load alone
  - Sliders to change RPM, crank radius, rod length and piston mass
  - Pause / play button

The gas-pressure loads, torque curve and flywheel sizing are in analysis.py.

Run:  python slider_crank_simulator.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

from slider_crank_core import kinematics, rod_angle

# ---------------------------------------------------------------- defaults
R0 = 50.0      # crank radius (mm)
L0 = 150.0     # connecting rod length (mm)
RPM0 = 600.0   # crank speed (rev/min)
M0 = 0.5       # piston mass (kg)

SLOW_MOTION = 0.15     # animation runs at 15 % of real speed so it is watchable
FRAME_DT = 0.02        # seconds between animation frames
PISTON_W = 30.0        # drawn piston length along the axis (mm)
BORE_HALF = 25.0       # half the drawn cylinder width (mm)


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(13, 8))
fig.suptitle("Slider-Crank Mechanism Simulator", fontsize=16, fontweight="bold")

ax_mech = fig.add_axes([0.05, 0.38, 0.40, 0.52])
ax_pos = fig.add_axes([0.55, 0.77, 0.40, 0.15])
ax_vel = fig.add_axes([0.55, 0.59, 0.40, 0.15], sharex=ax_pos)
ax_acc = fig.add_axes([0.55, 0.41, 0.40, 0.15], sharex=ax_pos)

# sliders
ax_s_rpm = fig.add_axes([0.10, 0.17, 0.35, 0.025])
ax_s_r = fig.add_axes([0.10, 0.12, 0.35, 0.025])
ax_s_l = fig.add_axes([0.10, 0.07, 0.35, 0.025])
ax_s_m = fig.add_axes([0.10, 0.02, 0.35, 0.025])
s_rpm = Slider(ax_s_rpm, "Speed (RPM)", 60, 3000, valinit=RPM0, valstep=10)
s_r = Slider(ax_s_r, "Crank r (mm)", 20, 80, valinit=R0, valstep=1)
s_l = Slider(ax_s_l, "Rod l (mm)", 100, 300, valinit=L0, valstep=1)
s_m = Slider(ax_s_m, "Piston m (kg)", 0.1, 2.0, valinit=M0, valstep=0.05)

# pause / play button
ax_btn = fig.add_axes([0.10, 0.25, 0.09, 0.04])
btn_pause = Button(ax_btn, "Pause")

# info panel (bottom right)
ax_info = fig.add_axes([0.55, 0.0, 0.42, 0.31])
ax_info.axis("off")
info_text = ax_info.text(0, 1, "", va="top", family="monospace", fontsize=9.5)

# ---------------------------------------------------------------- state
state = {"theta": 0.0, "paused": False}
curves = {}


def params():
    r, l, m = s_r.val, s_l.val, s_m.val
    omega = s_rpm.val * 2 * np.pi / 60.0
    return r, l, m, omega


def geometry_ok(r, l):
    """The rod must be longer than the crank, otherwise the mechanism locks."""
    return l > r


def setup_plots():
    r, l, m, omega = params()
    if not geometry_ok(r, l):
        return
    th = np.linspace(0, 2 * np.pi, 721)
    x, v, a = kinematics(th, r, l, omega)
    deg = np.degrees(th)

    for ax in (ax_pos, ax_vel, ax_acc):
        ax.clear()
        ax.grid(True, alpha=0.3)
    ax_pos.plot(deg, x, color="tab:blue")
    ax_vel.plot(deg, v / 1000, color="tab:green")
    ax_acc.plot(deg, a / 1000, color="tab:red")
    ax_pos.set_ylabel("Position (mm)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_acc.set_ylabel("Accel (m/s²)")
    ax_acc.set_xlabel("Crank angle (deg)")
    ax_acc.set_xlim(0, 360)
    ax_acc.set_xticks(range(0, 361, 60))
    plt.setp(ax_pos.get_xticklabels(), visible=False)
    plt.setp(ax_vel.get_xticklabels(), visible=False)

    curves["m_pos"], = ax_pos.plot([], [], "o", color="k")
    curves["m_vel"], = ax_vel.plot([], [], "o", color="k")
    curves["m_acc"], = ax_acc.plot([], [], "o", color="k")

    # mechanism axes
    ax_mech.clear()
    reach = r + l
    ax_mech.set_xlim(-r - 20, reach + 60)
    ax_mech.set_ylim(-r - 50, r + 50)
    ax_mech.set_aspect("equal")
    ax_mech.set_title("Mechanism")
    ax_mech.grid(True, alpha=0.2)
    ax_mech.set_xlabel("mm")
    # cylinder walls
    ax_mech.plot([r, reach + 50], [BORE_HALF, BORE_HALF], "k-", lw=3)
    ax_mech.plot([r, reach + 50], [-BORE_HALF, -BORE_HALF], "k-", lw=3)
    ax_mech.plot(0, 0, "ks", ms=10)  # crank centre

    curves["crank"], = ax_mech.plot([], [], "-", color="tab:blue", lw=6)
    curves["rod"], = ax_mech.plot([], [], "-", color="tab:orange", lw=4)
    curves["pin"], = ax_mech.plot([], [], "o", color="k", ms=8)
    t_c = np.linspace(0, 2 * np.pi, 200)
    curves["circle"], = ax_mech.plot(r * np.cos(t_c), r * np.sin(t_c),
                                     "--", color="gray", lw=0.8)
    curves["piston"] = Rectangle((0, -BORE_HALF + 3), PISTON_W,
                                 2 * BORE_HALF - 6, color="tab:red", alpha=0.8)
    ax_mech.add_patch(curves["piston"])


def update(_frame):
    r, l, m, omega = params()
    if not geometry_ok(r, l):
        info_text.set_text("Invalid geometry: rod length must be\n"
                           "greater than crank radius (l > r).")
        return []

    if not state["paused"]:
        state["theta"] = (state["theta"] + omega * FRAME_DT * SLOW_MOTION) % (2 * np.pi)
    th = state["theta"]

    # geometry
    cx, cy = r * np.cos(th), r * np.sin(th)
    x, v, a = kinematics(np.array([th]), r, l, omega)
    x, v, a = x[0], v[0], a[0]

    curves["crank"].set_data([0, cx], [0, cy])
    curves["rod"].set_data([cx, x], [cy, 0])
    curves["pin"].set_data([cx, x], [cy, 0])
    curves["piston"].set_xy((x, -BORE_HALF + 3))

    deg = np.degrees(th)
    curves["m_pos"].set_data([deg], [x])
    curves["m_vel"].set_data([deg], [v / 1000])
    curves["m_acc"].set_data([deg], [a / 1000])

    # loads from the reciprocating inertia alone (no gas pressure here)
    a_si = a / 1000.0                          # m/s^2
    phi = rod_angle(th, r, l)
    f_inertia = -m * a_si                      # d'Alembert force, + = away from crank
    f_axis = m * a_si                          # force sent to the crank along the axis
    rod_force = f_axis / np.cos(phi)
    side_thrust = f_axis * np.tan(phi)
    info_text.set_text(
        f"Crank angle      : {deg:7.1f} deg\n"
        f"Rod angle (phi)  : {np.degrees(phi):7.1f} deg\n"
        f"Piston position  : {x:7.1f} mm\n"
        f"Piston velocity  : {v/1000:7.2f} m/s\n"
        f"Piston accel.    : {a_si:7.1f} m/s²\n"
        f"Inertia force    : {f_inertia:7.1f} N  (+ = away from crank)\n"
        f"Rod force        : {rod_force:7.1f} N  (inertia only)\n"
        f"Side thrust      : {side_thrust:7.1f} N  (inertia only)\n"
        f"Stroke           : {2*r:7.1f} mm   (l/r = {l/r:.2f})"
    )
    return []


def on_change(_val):
    setup_plots()
    fig.canvas.draw_idle()


def on_pause(_event):
    state["paused"] = not state["paused"]
    btn_pause.label.set_text("Play" if state["paused"] else "Pause")
    fig.canvas.draw_idle()


for s in (s_rpm, s_r, s_l, s_m):
    s.on_changed(on_change)
btn_pause.on_clicked(on_pause)

setup_plots()
anim = FuncAnimation(fig, update, interval=int(FRAME_DT * 1000),
                     blit=False, cache_frame_data=False)

if __name__ == "__main__":
    plt.show()
