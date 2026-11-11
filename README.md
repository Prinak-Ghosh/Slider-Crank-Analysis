# Slider-Crank Mechanism: Kinematics and Load Analysis in Python

Python Programming course project - Prinak Ghosh

An interactive simulator and a batch analysis of the slider-crank mechanism, the
linkage inside every piston engine and reciprocating compressor. The project
answers a practical design question: **for a single-cylinder 4-stroke engine,
how do the piston motion, the forces in the connecting rod, the side thrust on
the cylinder wall and the crank torque change with crank angle, speed and
connecting-rod length - and how big a flywheel does the torque curve need?**

## Objectives

1. Model the exact kinematics of the mechanism (position, velocity, acceleration).
2. Compare the exact result with the usual two-term textbook approximation.
3. Add an idealised gas-pressure cycle and compute rod force, side thrust and torque.
4. Size a flywheel from the torque fluctuation.
5. Verify the model with independent checks (numerical differentiation, energy balance).
6. Provide an interactive simulator to explore the effect of speed, r, l and piston mass.

## Repository contents

| File | What it does |
|---|---|
| `slider_crank_core.py` | All the physics: kinematics, `Engine` dataclass, gas-pressure model, loads, flywheel sizing |
| `slider_crank_simulator.py` | Interactive Matplotlib animation with sliders and a pause button |
| `analysis.py` | Batch study: runs the checks, writes every figure, the CSV and the JSON summary |
| `make_screenshots.py` | Saves still images of the simulator window (no display needed) |
| `outputs/` | Generated figures, `cycle_data.csv` (every 1 degree of the 720 degree cycle), `results_summary.json` |
| `requirements.txt` | Python dependencies |

## Requirements

Python 3.9 or newer, NumPy, Matplotlib.

```
pip install -r requirements.txt
```

## How to run

```
python slider_crank_simulator.py   # interactive window (needs a display)
python analysis.py                 # regenerates all figures, CSV and JSON in outputs/
python make_screenshots.py         # regenerates the simulator screenshots
```

In the simulator, the sliders change crank speed, crank radius, rod length and
piston mass. The curves and the information panel update immediately. The
information panel shows the inertia-only loads; the gas-pressure loads are in
`analysis.py`.

## Default engine used in the analysis

Bore 80 mm, stroke 100 mm (r = 50 mm), rod length 150 mm (l/r = 3), compression
ratio 8, reciprocating mass 0.5 kg, polytropic index 1.3, pressure rise ratio 3
at TDC, base speed 1500 rpm. Change them in the `Engine` dataclass in
`slider_crank_core.py`.

## Main results (1500 rpm)

* Peak cylinder pressure 40.3 bar, peak gas force 19.8 kN.
* Peak piston acceleration at TDC: 1645 m/s^2; mean piston speed 5.0 m/s.
* Peak side thrust 1.65 kN, about 28 degrees after firing TDC.
* Mean torque 23.2 N.m (3.64 kW indicated); peak torque is about 14 times the mean.
* Flywheel for Cs = 0.05: I = 0.27 kg.m^2 (a 328 mm steel disc, 30 mm thick, about 20 kg).
* Reciprocating inertia equals the peak gas force at about 7350 rpm for a 0.5 kg piston assembly.

## Verification built into `analysis.py`

* Position differentiated numerically twice agrees with the analytic velocity and
  acceleration (difference below 0.0001 %).
* Closed-form values at TDC and BDC match: a = -r w^2 (1 + r/l) and a = r w^2 (1 - r/l).
* Work per cycle from integrating torque over crank angle equals the indicated
  work from the p-V loop (difference about 0.00002 %).
* The geometric torque formula equals the virtual-work formula (difference below 1e-12 N.m).

## Assumptions and limitations

Rigid links, constant crank speed, instantaneous combustion at TDC (this
exaggerates the torque spike), cylinder pressures independent of speed, no
friction, no heat transfer, rod mass lumped into the reciprocating mass, no
crank balancing.

## Credits

Slider-crank formulation follows standard machine-dynamics texts (see the report
for references). Built with NumPy and Matplotlib.
