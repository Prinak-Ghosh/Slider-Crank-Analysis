"""
analysis.py - batch study of the slider-crank / single-cylinder engine.

Produces (in outputs/):
  fig_*.png            all figures used in the report
  cycle_data.csv       kinematics, pressure, forces, torque every 1 deg of the 720 deg cycle
  results_summary.json every headline number quoted in the report

Run:  python analysis.py
"""

import os
import json
import csv
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from slider_crank_core import (Engine, kinematics, kinematics_approx, rod_angle,
                               loads, flywheel_inertia, indicated_work)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

eng = Engine()
RPM0 = 1500
W0 = RPM0 * 2 * np.pi / 60
res = {}


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=180)
    plt.close()


# =========================================================== 1. mechanism sketch
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
r_mm, l_mm = eng.r * 1000, eng.l * 1000
for ax, deg in zip(axes, (0, 90, 180, 270)):
    th = np.radians(deg)
    cx, cy = r_mm * np.cos(th), r_mm * np.sin(th)
    x = kinematics(th, r_mm, l_mm, 1.0)[0]
    ax.plot([r_mm, r_mm + l_mm + 60], [40, 40], "k-", lw=2)
    ax.plot([r_mm, r_mm + l_mm + 60], [-40, -40], "k-", lw=2)
    ax.add_patch(Rectangle((x, -34), 30, 68, color="tab:red", alpha=0.8))
    ax.plot([0, cx], [0, cy], color="tab:blue", lw=5)
    ax.plot([cx, x], [cy, 0], color="tab:orange", lw=3)
    ax.plot([0, cx, x], [0, cy, 0], "ko", ms=5)
    ax.set_aspect("equal")
    ax.set_xlim(-70, r_mm + l_mm + 70)
    ax.set_ylim(-70, 70)
    ax.set_title(f"crank angle = {deg}°")
    ax.set_xticks([]); ax.set_yticks([])
save("fig_mechanism.png")

# ================================================ 2. kinematics: exact vs approx
th = np.linspace(0, 2 * np.pi, 7201)
x, v, a = kinematics(th, eng.r, eng.l, W0)
xa, va, aa = kinematics_approx(th, eng.r, eng.l, W0)

# independent check: differentiate the exact position numerically
t = th / W0
v_fd = np.gradient(x, t, edge_order=2)
a_fd = np.gradient(v_fd, t, edge_order=2)
sl = slice(5, -5)
res["fd_err_v_pct"] = 100 * np.max(np.abs(v_fd[sl] - v[sl])) / np.max(np.abs(v))
res["fd_err_a_pct"] = 100 * np.max(np.abs(a_fd[sl] - a[sl])) / np.max(np.abs(a))
res["approx_err_a_pct"] = 100 * np.max(np.abs(aa - a)) / np.max(np.abs(a))
res["approx_err_a_at_deg"] = float(np.degrees(th[np.argmax(np.abs(aa - a))]))
res["a_tdc"] = float(a[0]); res["a_bdc"] = float(a[3600])
res["a_tdc_formula"] = float(-eng.r * W0**2 * (1 + eng.r / eng.l))
res["a_bdc_formula"] = float(eng.r * W0**2 * (1 - eng.r / eng.l))
res["v_max"] = float(np.max(np.abs(v)))
res["v_max_at_deg"] = float(np.degrees(th[np.argmax(np.abs(v))]))
res["v_mean"] = float(2 * (2 * eng.r) * RPM0 / 60)  # = 2 * stroke * N[rev/s]

fig, ax = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
d = np.degrees(th)
for axi, ex, ap, lab in ((ax[0], x * 1000, xa * 1000, "Position (mm)"),
                         (ax[1], v, va, "Velocity (m/s)"),
                         (ax[2], a, aa, "Acceleration (m/s²)")):
    axi.plot(d, ex, color="tab:blue", label="exact")
    axi.plot(d, ap, "--", color="tab:orange", label="2-harmonic approximation")
    axi.set_ylabel(lab)
ax[1].plot(d[::240], v_fd[::240], "k.", ms=5, label="numerical derivative of x(t)")
ax[2].plot(d[::240], a_fd[::240], "k.", ms=5, label="numerical 2nd derivative")
ax[0].legend(loc="lower left", fontsize=8)
ax[1].legend(loc="lower right", fontsize=8)
ax[2].set_xlabel("Crank angle (deg)")
ax[2].set_xticks(range(0, 361, 60)); ax[2].set_xlim(0, 360)
save("fig_kinematics.png")

# =============================================================== 3. l/r study
ratios = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
th4 = np.linspace(0, 4 * np.pi, 14400, endpoint=False)
th2 = np.linspace(0, 2 * np.pi, 7201)
rows = []
for k in ratios:
    e = replace(eng, l=k * eng.r)
    _, _, ae = kinematics(th2, e.r, e.l, W0)
    _, _, aap = kinematics_approx(th2, e.r, e.l, W0)
    L = loads(th4, e, RPM0)
    rows.append(dict(l_over_r=k,
                     approx_err_pct=100 * np.max(np.abs(aap - ae)) / np.max(np.abs(ae)),
                     a_tdc=float(-ae[0]),
                     peak_side_thrust=float(np.max(np.abs(L["N"]))),
                     peak_rod_force=float(np.max(np.abs(L["R"]))),
                     max_rod_angle_deg=float(np.degrees(np.max(np.abs(L["phi"]))))))
res["lr_table"] = rows

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(ratios, [q["approx_err_pct"] for q in rows], "o-")
ax[0].set_xlabel("l / r"); ax[0].set_ylabel("max acceleration error (%)")
ax[0].set_title("Error of the 2-term approximation")
ax[1].plot(ratios, [q["peak_side_thrust"] / 1000 for q in rows], "s-", color="tab:red")
ax[1].set_xlabel("l / r"); ax[1].set_ylabel("peak side thrust (kN)")
ax[1].set_title(f"Side thrust at {RPM0} rpm")
save("fig_lr_study.png")

# ================================================ 4. full cycle at the base speed
L = loads(th4, eng, RPM0)
deg4 = np.degrees(th4)
res["rpm"] = RPM0
res["p_max_bar"] = float(L["p"].max() / 1e5)
res["F_gas_max"] = float(L["F_gas"].max())
res["F_net_max"] = float(L["F_net"].max())
res["F_net_min"] = float(L["F_net"].min())
res["F_net_min_at_deg"] = float(deg4[np.argmin(L["F_net"])])
res["R_max"] = float(L["R"].max())
res["N_max"] = float(np.max(np.abs(L["N"])))
res["N_max_at_deg"] = float(deg4[np.argmax(np.abs(L["N"]))])
res["T_max"] = float(L["T"].max()); res["T_min"] = float(L["T"].min())
res["T_mean"] = float(L["T"].mean())
res["power_kW"] = float(L["T"].mean() * W0 / 1000)

dth = th4[1] - th4[0]
W_torque = float(np.sum(L["T"]) * dth)
W_ind = float(indicated_work(L["V"], L["p"], eng.p_atm))
res["W_torque"] = W_torque; res["W_indicated"] = W_ind
res["W_rel_diff_pct"] = 100 * abs(W_torque - W_ind) / W_ind
T_vw = -L["F_net"] * L["v"] / L["omega"]          # torque from virtual work
res["T_formula_diff"] = float(np.max(np.abs(T_vw - L["T"])))
res["A_cm2"] = eng.area * 1e4; res["Vs_cc"] = eng.swept * 1e6; res["Vc_cc"] = eng.clearance * 1e6

# p-V diagram
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(L["V"] * 1e6, L["p"] / 1e5, color="tab:blue")
ax.set_xlabel("Cylinder volume (cm³)"); ax.set_ylabel("Pressure (bar)")
for lab, i in (("intake", 900), ("compression", 4500), ("power", 9000), ("exhaust", 12600)):
    ax.annotate(lab, (L["V"][i] * 1e6, L["p"][i] / 1e5), fontsize=8,
                xytext=(6, 6), textcoords="offset points")
save("fig_pv.png")

# forces vs crank angle
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(deg4, L["F_gas"] / 1000, label="gas force", color="tab:blue")
ax.plot(deg4, L["F_rec"] / 1000, label="reciprocating term  m·a", color="tab:green")
ax.plot(deg4, L["F_net"] / 1000, label="net force", color="k", lw=1.2)
ax.set_xlabel("Crank angle over the 4-stroke cycle (deg)"); ax.set_ylabel("Force (kN)")
ax.set_xticks(range(0, 721, 180)); ax.set_xlim(0, 720)
for xv in (180, 360, 540):
    ax.axvline(xv, color="gray", ls=":", lw=0.8)
ax.legend(loc="upper right")
save("fig_forces.png")

# torque + flywheel
Tm, dE, I = flywheel_inertia(th4, L["T"], W0, 0.05)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(deg4, L["T"], color="tab:purple", label="crank torque")
ax.axhline(Tm, color="k", ls="--", label=f"mean torque = {Tm:.1f} N·m")
ax.fill_between(deg4, L["T"], Tm, where=L["T"] > Tm, color="tab:purple", alpha=0.2)
ax.set_xlabel("Crank angle over the 4-stroke cycle (deg)"); ax.set_ylabel("Torque (N·m)")
ax.set_xticks(range(0, 721, 180)); ax.set_xlim(0, 720)
ax.legend(loc="upper right")
save("fig_torque.png")

# side thrust + rod force
fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
ax[0].plot(deg4, L["R"] / 1000, color="tab:orange")
ax[0].set_ylabel("Connecting-rod force (kN)")
ax[1].plot(deg4, L["N"] / 1000, color="tab:red")
ax[1].set_ylabel("Side thrust (kN)")
ax[1].set_xlabel("Crank angle over the 4-stroke cycle (deg)")
ax[1].set_xticks(range(0, 721, 180)); ax[1].set_xlim(0, 720)
save("fig_side_thrust.png")

# ====================================================== 5. flywheel at 3 speeds
rho, t_fw = 7850.0, 0.030
fw = []
for rpm in (1000, 1500, 3000):
    Lr = loads(th4, eng, rpm)
    w = rpm * 2 * np.pi / 60
    Tm_r, dE_r, I_r = flywheel_inertia(th4, Lr["T"], w, 0.05)
    R_fw = (2 * I_r / (rho * np.pi * t_fw)) ** 0.25
    fw.append(dict(rpm=rpm, T_mean=float(Tm_r), dE=float(dE_r), I=float(I_r),
                   disc_diameter_mm=float(2 * R_fw * 1000),
                   disc_mass_kg=float(rho * np.pi * R_fw**2 * t_fw)))
res["flywheel"] = fw
res["flywheel_Cs"] = 0.05; res["flywheel_t_mm"] = t_fw * 1000

# ===================================================== 6. speed (inertia) study
rpms = np.arange(250, 8001, 250)
Fmax, Fmin = [], []
for rpm in rpms:
    Lr = loads(th4, eng, rpm)
    Fmax.append(Lr["F_net"].max()); Fmin.append(Lr["F_net"].min())
Fmax, Fmin = np.array(Fmax), np.array(Fmin)

w_grid = rpms * 2 * np.pi / 60
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for m in (0.3, 0.5, 0.8):
    ax[0].plot(rpms, m * eng.r * w_grid**2 * (1 + eng.r / eng.l) / 1000, label=f"m = {m} kg")
ax[0].axhline(res["F_gas_max"] / 1000, color="k", ls="--", label="peak gas force")
ax[0].set_xlabel("Engine speed (rpm)"); ax[0].set_ylabel("Peak inertia force at TDC (kN)")
ax[0].set_title("Inertia grows with speed²"); ax[0].legend(fontsize=8)
ax[1].plot(rpms, Fmax / 1000, label="max net force (compression on rod)")
ax[1].plot(rpms, Fmin / 1000, label="min net force (tension in rod)")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("Engine speed (rpm)"); ax[1].set_ylabel("Net piston force (kN)")
ax[1].set_title("Load range over the cycle"); ax[1].legend(fontsize=8)
save("fig_speed_study.png")

w_cross = np.sqrt(res["F_gas_max"] / (eng.m_rec * eng.r * (1 + eng.r / eng.l)))
res["rpm_inertia_equals_gas"] = float(w_cross * 60 / (2 * np.pi))
i3000 = int(np.where(rpms == 3000)[0][0]); i6000 = int(np.where(rpms == 6000)[0][0])
res["F_net_min_3000"] = float(Fmin[i3000]); res["F_net_min_6000"] = float(Fmin[i6000])
res["F_net_max_6000"] = float(Fmax[i6000])
res["rec_peak_1500"] = float(eng.m_rec * eng.r * W0**2 * (1 + eng.r / eng.l))

# ================================================================== CSV + JSON
ang = np.arange(0, 720, 1.0)
Lc = loads(np.radians(ang), eng, RPM0)
with open(os.path.join(OUT, "cycle_data.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["crank_angle_deg", "piston_x_mm", "velocity_m_s", "accel_m_s2", "rod_angle_deg",
                "pressure_bar", "F_gas_N", "F_rec_N", "F_net_N", "rod_force_N",
                "side_thrust_N", "torque_Nm"])
    for i in range(len(ang)):
        w.writerow([f"{ang[i]:.0f}", f"{Lc['x'][i]*1000:.3f}", f"{Lc['v'][i]:.4f}",
                    f"{Lc['a'][i]:.3f}", f"{np.degrees(Lc['phi'][i]):.4f}",
                    f"{Lc['p'][i]/1e5:.4f}", f"{Lc['F_gas'][i]:.2f}", f"{Lc['F_rec'][i]:.2f}",
                    f"{Lc['F_net'][i]:.2f}", f"{Lc['R'][i]:.2f}", f"{Lc['N'][i]:.2f}",
                    f"{Lc['T'][i]:.4f}"])

with open(os.path.join(OUT, "results_summary.json"), "w") as f:
    json.dump(res, f, indent=2)

# ================================================================ console output
print(f"Engine: bore {eng.bore*1000:.0f} mm, stroke {2*eng.r*1000:.0f} mm, l/r = {eng.l/eng.r:.1f}, "
      f"CR = {eng.comp_ratio}, m_rec = {eng.m_rec} kg, base speed {RPM0} rpm")
print(f"Swept volume {res['Vs_cc']:.0f} cc, clearance {res['Vc_cc']:.1f} cc, piston area {res['A_cm2']:.1f} cm2")
print("\n-- checks --")
print(f"numerical derivative vs exact : velocity {res["fd_err_v_pct"]:.2e} %, accel {res["fd_err_a_pct"]:.2e} %")
print(f"a(TDC) exact {res['a_tdc']:.1f} vs formula {res['a_tdc_formula']:.1f}; "
      f"a(BDC) exact {res['a_bdc']:.1f} vs formula {res['a_bdc_formula']:.1f} m/s2")
print(f"work per cycle: integral of T dtheta = {W_torque:.2f} J, integral of (p-p0) dV = {W_ind:.2f} J "
      f"(diff {res['W_rel_diff_pct']:.5f} %)")
print(f"torque: geometric formula vs virtual work, max difference {res['T_formula_diff']:.2e} N.m")
print("\n-- results at base speed --")
for k in ("p_max_bar", "F_gas_max", "F_net_max", "F_net_min", "R_max", "N_max", "N_max_at_deg",
          "T_max", "T_min", "T_mean", "power_kW", "v_max", "v_max_at_deg", "v_mean",
          "approx_err_a_pct", "approx_err_a_at_deg"):
    print(f"{k:22s}: {res[k]:.3f}")
print("\n-- l/r study --")
for q in rows:
    print({k: round(v, 3) for k, v in q.items()})
print("\n-- flywheel (Cs = 0.05) --")
for q in fw:
    print({k: round(v, 3) for k, v in q.items()})
print(f"\ninertia = gas force at {res['rpm_inertia_equals_gas']:.0f} rpm; "
      f"min net force at 3000 rpm {res['F_net_min_3000']:.0f} N, at 6000 rpm {res['F_net_min_6000']:.0f} N")
